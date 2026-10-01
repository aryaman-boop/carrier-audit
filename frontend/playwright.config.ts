import { defineConfig, devices } from "@playwright/test";

// Browser tests against the built UI, served by FastAPI from a throwaway,
// freshly seeded database with every model call disabled (backend/e2e_server.py).
// A run costs nothing and never touches backend/data/carrier_audit.db.
//
// One worker, on purpose: the specs share one server, and the decision spec
// writes to it. Parallel workers would make the summary counts a race.

const PORT = 8765;
const PYTHON = process.env.PYTHON ?? "../.venv/bin/python";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `${PYTHON} ../backend/e2e_server.py --port ${PORT}`,
    url: `http://127.0.0.1:${PORT}/api/health`,
    // Never reuse: a reused server could be the real demo database.
    reuseExistingServer: false,
    timeout: 120_000,
    stdout: "pipe",
  },
});
