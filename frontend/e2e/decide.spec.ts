import { expect, test } from "@playwright/test";
import { queue, summary } from "./fixtures";

// The one spec that writes. It runs against the throwaway server, and the
// assertions are relative to the state it finds, so it holds on a re-run too.

test("a reviewer's decision is logged, counted, and survives a reload", async ({
  page,
  request,
}) => {
  const before = await summary(request);
  const open = (await queue(request, "?disposition=NEEDS_REVIEW")).find((l) => !l._decision);
  test.skip(!open, "no undecided line left in needs review");

  await page.goto("/");
  const row = page.locator(".queue .row", {
    has: page.locator(".pro", { hasText: open!.pro_number }),
  }).filter({ hasText: open!.invoice_number });
  await row.click();

  const detail = page.locator(".detail");
  await expect(detail.locator("h3")).toHaveText(open!.pro_number);
  await detail.getByRole("button", { name: "Escalate" }).click();

  // The buttons give way to the logged action: a line is decided once.
  await expect(detail.locator(".actions .pill")).toHaveText("Logged: escalate");
  await expect(detail.getByRole("button", { name: "Approve" })).toHaveCount(0);

  const n = before.human_decisions + 1;
  await expect(
    page.locator(".tile", { hasText: "Human override rate" }).locator(".note"),
  ).toHaveText(`${n} decision${n === 1 ? "" : "s"} logged`);

  // Stored server-side, not just in component state.
  const after = (await queue(request, "?disposition=NEEDS_REVIEW")).find(
    (l) => l.pro_number === open!.pro_number && l.invoice_number === open!.invoice_number,
  );
  expect(after?._decision?.human_action).toBe("ESCALATE");

  await page.reload();
  await expect(row).toContainText("decided");
  await row.click();
  await expect(page.locator(".detail .actions .pill")).toHaveText("Logged: escalate");
});
