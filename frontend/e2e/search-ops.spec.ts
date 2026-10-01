import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.goto("/");
});

const searchPanel = (page: import("@playwright/test").Page) =>
  page.locator(".panel", { hasText: "Tariff search" });

test("tariff search returns the governing clause, quoted in full", async ({ page }) => {
  const panel = searchPanel(page);
  await panel.locator("input").fill("they billed a liftgate at a dock-equipped warehouse");
  await panel.locator("input").press("Enter");

  const first = panel.locator(".clause").first();
  await expect(first.locator(".ctitle")).toHaveText("Liftgate Service at Delivery");
  await expect(first.locator(".cid")).toContainText("MFS-200");
  await expect(first.locator(".ctext")).not.toBeEmpty();
});

test("a one-character query is not sent", async ({ page }) => {
  const sent: string[] = [];
  page.on("request", (r) => {
    if (r.url().includes("/api/search")) sent.push(r.url());
  });

  const panel = searchPanel(page);
  await panel.locator("input").fill("x");
  await panel.getByRole("button", { name: "Search" }).click();

  await expect(panel.getByText("The rate table is a join.")).toBeVisible();
  await expect(panel.locator(".clause")).toHaveCount(0);
  expect(sent).toEqual([]);
});

test("the Operations tab shows the run's stage telemetry, and the queue comes back", async ({
  page,
  request,
}) => {
  const obs = await (await request.get("/api/observability")).json();

  await page.getByRole("button", { name: "Operations" }).click();
  await expect(page.getByRole("heading", { name: "Per stage — latency, outcomes, cost" })).toBeVisible();
  await expect(
    page.locator(".tile", { hasText: "Traces" }).locator(".value"),
  ).toHaveText(String(obs.traces));
  await expect(page.locator(".queue")).toHaveCount(0);

  await page.getByRole("button", { name: "Review queue" }).click();
  await expect(page.locator(".queue .row").first()).toBeVisible();
});
