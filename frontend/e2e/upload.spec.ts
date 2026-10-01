import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";

// The e2e server runs with every model call disabled, so these specs cover the
// gates that are free by design (FILE) and what the UI does when the paid path
// is unavailable (TRIAGE). Acceptance needs a model, so it is the eval harness's
// job, not this suite's.

const GATES = ["FILE", "DEDUPE", "TRIAGE", "EXTRACT", "SUM_CHECK", "OWNERSHIP"];
const HOLDOUT = fileURLToPath(
  new URL("../../backend/data/upload_samples/holdout_carrier_invoice.pdf", import.meta.url),
);

test.beforeEach(async ({ page }) => {
  await page.goto("/");
});

async function upload(page: import("@playwright/test").Page, name: string, bytes: string | Buffer) {
  await page.locator(".drop input[type=file]").setInputFiles({
    name,
    mimeType: "application/pdf",
    buffer: Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes),
  });
}

async function expectStoppedAt(page: import("@playwright/test").Page, gate: string) {
  const strip = page.locator(".gates .gate");
  await expect(strip).toHaveCount(GATES.length);
  const stop = GATES.indexOf(gate);
  for (let i = 0; i < GATES.length; i++) {
    const cls = i < stop ? /\bpass\b/ : i === stop ? /\bfail\b/ : /\bskip\b/;
    await expect(strip.nth(i)).toHaveClass(cls);
  }
}

test("a renamed non-PDF is refused at FILE, before anything costs money", async ({ page }) => {
  await upload(page, "invoice.pdf", "PK\u0003\u0004 this is a zip wearing a .pdf extension");

  const verdict = page.locator(".verdict.bad");
  await expect(verdict.locator(".head")).toHaveText("Refused at FILE");
  await expect(verdict).toContainText("This is not a PDF.");
  await expect(verdict).toContainText("$0.00 — no model was called");
  await expectStoppedAt(page, "FILE");
});

test("an empty file is refused at FILE", async ({ page }) => {
  await upload(page, "empty.pdf", "");
  await expect(page.locator(".verdict.bad")).toContainText("The file is empty.");
  await expectStoppedAt(page, "FILE");
});

test("an encrypted PDF is refused at FILE", async ({ page }) => {
  await upload(page, "locked.pdf", "%PDF-1.7\n1 0 obj << /Encrypt 2 0 R >> endobj\n%%EOF\n");
  await expect(page.locator(".verdict.bad")).toContainText("encrypted");
  await expectStoppedAt(page, "FILE");
});

test("with no model configured, a real invoice clears the free gates and stops at TRIAGE as a notification", async ({
  page,
  request,
}) => {
  await page.locator(".drop input[type=file]").setInputFiles(HOLDOUT);

  const toast = page.locator(".toast[role=status]");
  await expect(toast).toBeVisible();
  await expect(toast.locator(".t-gate")).toHaveText("TRIAGE");
  await expect(toast).toContainText("No model is configured");
  await expectStoppedAt(page, "TRIAGE");

  // TRIAGE refusals are toast-only: no explanation panel is left on the page.
  await expect(page.locator(".verdict")).toHaveCount(0);

  await toast.getByRole("button", { name: "Dismiss" }).click();
  await expect(toast).toHaveCount(0);

  // Every attempt is recorded, refused or not.
  const hist = await (await request.get("/api/upload/history")).json();
  const last = hist.uploads[0];
  expect(last.filename).toBe("holdout_carrier_invoice.pdf");
  expect(last.rejected_at).toBe("TRIAGE");
  expect(last.accepted).toBeFalsy();
});
