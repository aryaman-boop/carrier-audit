import { expect, test } from "@playwright/test";
import { queue, summary } from "./fixtures";

const BUCKETS = [
  { chip: "needs review", qs: "?disposition=NEEDS_REVIEW" },
  { chip: "⚑ abstained", qs: "?category=UNKNOWN" },
  { chip: "auto disputed", qs: "?disposition=AUTO_DISPUTED" },
  { chip: "auto approved", qs: "?disposition=AUTO_APPROVED" },
  { chip: "all", qs: "" },
];

const chip = (page: import("@playwright/test").Page, label: string) =>
  page.locator(".filters button.chip", { hasText: new RegExp(`^${label} \\d+$`) });

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".queue .row").first()).toBeVisible();
});

test("header names the current run and the demo banner disclaims production", async ({
  page,
  request,
}) => {
  const s = await summary(request);
  await expect(page.locator("header.top .sub")).toContainText(s.run_id);
  await expect(page.locator(".banner").first()).toContainText(
    "Not production, not deployed, no real carrier data",
  );
});

test("every filter chip shows the count the API reports", async ({ page, request }) => {
  const s = await summary(request);
  await expect(chip(page, "needs review")).toHaveText(`needs review ${s.by_disposition.NEEDS_REVIEW ?? 0}`);
  await expect(chip(page, "⚑ abstained")).toHaveText(`⚑ abstained ${s.by_category.UNKNOWN ?? 0}`);
  await expect(chip(page, "auto disputed")).toHaveText(`auto disputed ${s.by_disposition.AUTO_DISPUTED ?? 0}`);
  await expect(chip(page, "auto approved")).toHaveText(`auto approved ${s.by_disposition.AUTO_APPROVED ?? 0}`);
  await expect(chip(page, "all")).toHaveText(`all ${s.lines}`);
});

test("each bucket lists exactly the lines the API puts in it, in the same order", async ({
  page,
  request,
}) => {
  // Opens on "needs review": the queue a reviewer acts on comes first.
  await expect(chip(page, "needs review")).toHaveClass(/\bon\b/);

  for (const b of BUCKETS) {
    await chip(page, b.chip).click();
    await expect(chip(page, b.chip)).toHaveClass(/\bon\b/);
    const expected = (await queue(request, b.qs)).map((l) => l.pro_number);
    await expect(page.locator(".queue .row .pro")).toHaveText(expected);
  }
});

test("selecting a line shows billed against recomputed, with the policy reason", async ({
  page,
}) => {
  const second = page.locator(".queue .row").nth(1);
  const pro = (await second.locator(".pro").textContent())!.trim();
  await second.click();

  await expect(second).toHaveClass(/\bsel\b/);
  const detail = page.locator(".detail");
  await expect(detail.locator("h3")).toHaveText(pro);
  await expect(detail.locator("table.charges th").first()).toHaveText("Carrier invoice");
  await expect(detail.locator("table.charges").nth(1).locator("th").first()).toHaveText("Our recompute");
  await expect(detail.getByText("Policy — always code, never the model")).toBeVisible();
});

test("auto-disputed lines are duplicate PROs, settled by code with no model call", async ({
  page,
  request,
}) => {
  const disputed = await queue(request, "?disposition=AUTO_DISPUTED");
  test.skip(disputed.length === 0, "seed produced no auto-disputed lines");

  await chip(page, "auto disputed").click();
  for (let i = 0; i < disputed.length; i++) {
    await page.locator(".queue .row").nth(i).click();
    const detail = page.locator(".detail");
    await expect(detail.locator("h3")).toHaveText(disputed[i].pro_number);
    await expect(detail.locator(".variance-bar .pill").first()).toHaveText("AUTO DISPUTED");
    await expect(
      detail.locator("ul.facts li", { hasText: "PRO already invoiced" }),
    ).toContainText("YES");
    await expect(
      detail.getByText("Code — resolved deterministically; no model was called"),
    ).toBeVisible();
  }
});

test("oldest-first sorts the queue by invoice date, the ageing order", async ({ page }) => {
  await chip(page, "all").click();
  // Wait for the re-sorted queue itself; reading the rows straight after the
  // select reads the previous order.
  await Promise.all([
    page.waitForResponse((r) => r.url().includes("/api/queue") && r.url().includes("sort=oldest")),
    page.locator("select.sortsel").selectOption("oldest"),
  ]);
  const expected = await page.evaluate(async () => {
    const r = await fetch("/api/queue?sort=oldest");
    return (await r.json()).items.map((l: { pro_number: string }) => l.pro_number);
  });
  await expect(page.locator(".queue .row .pro")).toHaveText(expected);

  const dates = await page
    .locator(".queue .row .meta")
    .evaluateAll((els) =>
      els.map((e) => (e.textContent ?? "").match(/\d{4}-\d{2}-\d{2}/)?.[0] ?? ""),
    );
  const dated = dates.filter(Boolean);
  expect(dated.length).toBeGreaterThan(1);
  expect(dated).toEqual([...dated].sort());
});

test("a slow, stale queue response cannot overwrite a newer one", async ({ page, request }) => {
  // Regression for a race this suite found: switch bucket, then sort, and the
  // first request can finish last. Hold it back on purpose so the order of
  // arrival is the bad one every time, not one run in three.
  await page.route(/\/api\/queue$/, async (route) => {
    const response = await route.fetch();
    await new Promise((r) => setTimeout(r, 800));
    await route.fulfill({ response });
  });

  await chip(page, "all").click();
  await page.locator("select.sortsel").selectOption("oldest");
  const expected = (await queue(request, "?sort=oldest")).map((l) => l.pro_number);

  await expect(page.locator(".queue .row .pro")).toHaveText(expected);
  // Let the held response land, then check it was ignored.
  await page.waitForTimeout(1200);
  await expect(page.locator(".queue .row .pro")).toHaveText(expected);
  await expect(page.locator("select.sortsel")).toHaveValue("oldest");
});
