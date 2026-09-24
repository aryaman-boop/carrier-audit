import { useEffect, useState } from "react";

/** Stage-level telemetry, not a trace tree.
 *
 *  This is a workflow: a fixed sequence of named stages. What it needs is
 *  timings and outcome distributions. An agent — where control flow is decided
 *  at runtime — would need nested spans per tool call and loop iteration, which
 *  is a much larger problem. Keeping it small is a payoff of not building one. */
interface StageStat {
  calls: number;
  ok: number;
  rejected: number;
  error: number;
  error_rate: number;
  p50_ms: number;
  p95_ms: number;
  cost_usd: number;
}

interface AlertItem {
  severity: "ACTION" | "WATCH" | "INFO";
  kind: string;
  title: string;
  detail: string;
  suggested_action: string;
  evidence: Record<string, unknown>;
}

interface AlertReport {
  action: number;
  watch: number;
  info: number;
  alerts: AlertItem[];
}

interface Report {
  window_days?: number;
  traces?: number;
  events?: number;
  by_stage?: Record<string, StageStat>;
  extraction_method_mix?: Record<string, { n: number; share: number }>;
  schema_conformance_rate?: number | null;
  total_cost_usd?: number;
  note?: string;
}

const ms = (n: number) => (n >= 1000 ? `${(n / 1000).toFixed(1)} s` : `${n.toFixed(1)} ms`);

export default function Dashboard() {
  const [r, setR] = useState<Report | null>(null);
  const [a, setA] = useState<AlertReport | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/observability").then((x) => x.json()).then(setR).catch((e) => setErr(String(e)));
    fetch("/api/alerts").then((x) => x.json()).then(setA).catch(() => undefined);
  }, []);

  if (err) return <div className="banner">{err}</div>;
  if (!r) return <div className="empty">Loading telemetry…</div>;
  if (!r.traces) return <div className="empty">{r.note ?? "No telemetry yet."}</div>;

  const stages = Object.entries(r.by_stage ?? {});
  const mix = Object.entries(r.extraction_method_mix ?? {});
  const free = mix.filter(([m]) => m === "text-layer" || m === "stub")
    .reduce((a, [, v]) => a + v.share, 0);

  return (
    <>
      {/* Alerts first. Telemetry tells you what happened; this tells you what to
          do about it. An abstention that recurs is a rule nobody has written --
          the point of alerting on patterns rather than on lines. */}
      {a && a.alerts.length > 0 && (
        <div className="panel" style={{ marginBottom: 16 }}>
          <h2>
            Needs your attention — {a.action} to action, {a.watch} to watch,{" "}
            {a.info} for information
          </h2>
          <div style={{ padding: 14 }}>
            {a.alerts.map((al, i) => (
              <div key={`${al.kind}-${i}`} className={`alert ${al.severity}`}>
                <div className="a-top">
                  <span className="a-sev">{al.severity}</span>
                  <span className="a-title">{al.title}</span>
                </div>
                <div className="a-detail">{al.detail}</div>
                <div className="a-do">
                  <b>Do this</b>
                  {al.suggested_action}
                </div>
                {"occurrences" in al.evidence && (
                  <div className="a-ev">
                    {String(al.evidence.occurrences)} occurrence(s) ·{" "}
                    ${Number(al.evidence.total_usd ?? 0).toFixed(2)}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="tiles" style={{ marginBottom: 16 }}>
        <div className="tile">
          <div className="label">Traces</div>
          <div className="value">{r.traces}</div>
          <div className="note">{r.events} stage events, {r.window_days}d window</div>
        </div>
        <div className="tile">
          <div className="label">Schema conformance</div>
          <div className="value">
            {r.schema_conformance_rate === null || r.schema_conformance_rate === undefined
              ? "—"
              : `${(r.schema_conformance_rate * 100).toFixed(1)}%`}
          </div>
          <div className="note">the drift alarm</div>
        </div>
        <div className="tile">
          <div className="label">Read without a model</div>
          <div className="value">{(free * 100).toFixed(0)}%</div>
          <div className="note">extraction on the free rung</div>
        </div>
        <div className="tile">
          <div className="label">Spend</div>
          <div className="value">${(r.total_cost_usd ?? 0).toFixed(4)}</div>
          <div className="note">across the window</div>
        </div>
      </div>

      <div className="panel" style={{ marginBottom: 16 }}>
        <h2>Per stage — latency, outcomes, cost</h2>
        <div style={{ padding: 14 }}>
          <table className="charges">
            <thead>
              <tr>
                <th>Stage</th>
                <th style={{ textAlign: "right" }}>Calls</th>
                <th style={{ textAlign: "right" }}>Errors</th>
                <th style={{ textAlign: "right" }}>p50</th>
                <th style={{ textAlign: "right" }}>p95</th>
                <th style={{ textAlign: "right" }}>Cost</th>
              </tr>
            </thead>
            <tbody>
              {stages.map(([name, s]) => (
                <tr key={name}>
                  <td style={{ fontWeight: 600 }}>{name}</td>
                  <td className="num">{s.calls}</td>
                  <td className="num" style={{ color: s.error ? "var(--bad)" : undefined }}>
                    {s.error}
                  </td>
                  <td className="num">{ms(s.p50_ms)}</td>
                  <td className="num">{ms(s.p95_ms)}</td>
                  <td className="num">${s.cost_usd.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {stages.some(([n]) => n === "classify") && (
            <div className="reason" style={{ marginTop: 14 }}>
              <div className="who">Read the classify row</div>
              The gap between p50 and p95 is the design. The median line is decided by
              code in under a millisecond and costs nothing; the p95 is the model, reached
              only when code has found a variance it cannot explain.
            </div>
          )}
        </div>
      </div>

      <div className="panel">
        <h2>Extraction ladder — which rung answered</h2>
        <div style={{ padding: 14 }}>
          {mix.length === 0 && <div className="empty">No extraction events yet.</div>}
          {mix
            .sort((a, b) => b[1].n - a[1].n)
            .map(([m, v]) => (
              <div key={m} style={{ marginBottom: 10 }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12.5 }}>
                  <span style={{ fontWeight: 600 }}>
                    {m}
                    {m === "text-layer" && " — deterministic parse, free"}
                    {m === "vision" && " — model call, ~$0.07"}
                    {m === "stub" && " — generator output, no extraction"}
                  </span>
                  <span style={{ fontFamily: "var(--mono)" }}>
                    {v.n} · {(v.share * 100).toFixed(0)}%
                  </span>
                </div>
                <div style={{ height: 6, background: "#eef0f3", borderRadius: 4, marginTop: 4 }}>
                  <div
                    style={{
                      width: `${v.share * 100}%`,
                      height: "100%",
                      borderRadius: 4,
                      background: m === "vision" ? "var(--warn)" : "var(--ok)",
                    }}
                  />
                </div>
              </div>
            ))}
          <div className="reason" style={{ marginTop: 12 }}>
            <div className="who">The number to watch</div>
            If the free rung's share falls, carriers have started sending scans and the
            bill is about to rise. That shows up here before it shows up on an invoice.
          </div>
        </div>
      </div>
    </>
  );
}
