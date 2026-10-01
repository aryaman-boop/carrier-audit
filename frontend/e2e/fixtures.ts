import type { APIRequestContext } from "@playwright/test";
import type { AuditedLine, Summary } from "../src/types";

// The specs assert that the UI shows what the API says, rather than hard-coding
// counts. The seed is deterministic, but a test pinned to "31 lines" breaks on
// every honest change to the seed and proves nothing a comparison does not.

export async function summary(request: APIRequestContext): Promise<Summary> {
  const r = await request.get("/api/summary");
  if (!r.ok()) throw new Error(`summary ${r.status()}`);
  return r.json();
}

export async function queue(request: APIRequestContext, qs = ""): Promise<AuditedLine[]> {
  const r = await request.get(`/api/queue${qs}`);
  if (!r.ok()) throw new Error(`queue ${r.status()}`);
  return (await r.json()).items;
}
