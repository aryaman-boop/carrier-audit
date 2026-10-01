import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Dashboard from "./Dashboard";
import LiveClock from "./LiveClock";
import Toasts, { type ToastMsg } from "./Toast";
import type { AuditedLine, Clause, Summary, UploadResult } from "./types";

const money = (n: number) =>
  n.toLocaleString("en-CA", { style: "currency", currency: "USD", minimumFractionDigits: 2 });
const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

const DISPOSITION_TONE: Record<string, string> = {
  AUTO_APPROVED: "ok",
  AUTO_DISPUTED: "bad",
  NEEDS_REVIEW: "warn",
  BLOCKED_EXTRACTION: "bad",
};

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`);
  return r.json() as Promise<T>;
}

/* ------------------------------------------------------------------ */

function Tiles({ s }: { s: Summary }) {
  return (
    <div className="tiles">
      <div className="tile">
        <div className="label">Automation rate</div>
        <div className="value">{pct(s.automation_rate)}</div>
        <div className="note">reached a decision with no person</div>
      </div>
      <div className="tile">
        <div className="label">Escalation rate</div>
        <div className="value">{pct(s.escalation_rate)}</div>
        <div className="note">
          {s.by_disposition.NEEDS_REVIEW ?? 0} queued
          {(s.by_category?.UNKNOWN ?? 0) > 0 && `, ${s.by_category.UNKNOWN} abstained`}
        </div>
      </div>
      <div className="tile">
        <div className="label">Recoverable</div>
        <div className="value">{money(s.recoverable_usd)}</div>
        <div className="note">of {money(s.total_billed_usd)} billed</div>
      </div>
      <div className="tile">
        <div className="label">Auto-disputed</div>
        <div className="value">{s.by_disposition.AUTO_DISPUTED ?? 0}</div>
        <div className="note">documentary evidence only</div>
      </div>
      <div className="tile">
        <div className="label">Human override rate</div>
        <div className="value">
          {s.human_override_rate === null ? "—" : pct(s.human_override_rate)}
        </div>
        <div className="note">
          {s.human_decisions === 0
            ? "no decisions yet"
            : `${s.human_decisions} decision${s.human_decisions === 1 ? "" : "s"} logged`}
        </div>
      </div>
    </div>
  );
}

function ChargeTable({
  title,
  rows,
  total,
  flagged,
}: {
  title: string;
  rows: { code: string; label: string; basis?: string; amount: number }[];
  total: number;
  flagged?: Set<string>;
}) {
  return (
    <div>
      <table className="charges">
        <thead>
          <tr>
            <th>{title}</th>
            <th style={{ textAlign: "right" }}>Amount</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={`${r.code}-${i}`}>
              <td className={flagged?.has(r.code) ? "flag" : undefined}>
                {r.label}
                {r.basis && <span className="basis">{r.basis}</span>}
              </td>
              <td className="num">{money(r.amount)}</td>
            </tr>
          ))}
          <tr className="total">
            <td>Total</td>
            <td className="num">{money(total)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

function Detail({
  line,
  runId,
  onDecided,
}: {
  line: AuditedLine;
  runId: string;
  onDecided: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const r = line.recomputation;
  const flagged = useMemo(() => new Set(r.unsupported_accessorials), [r]);

  const decide = async (action: string) => {
    setBusy(true);
    try {
      await api("/decide", {
        method: "POST",
        body: JSON.stringify({
          run_id: runId,
          invoice_number: line.invoice_number,
          pro_number: line.pro_number,
          action,
          category: line.classification.category,
        }),
      });
      onDecided();
    } finally {
      setBusy(false);
    }
  };

  const correct = line._ground_truth === line.classification.category;

  return (
    <div className="detail">
      <div className="head">
        <h3>{line.pro_number}</h3>
        <div className="line2">
          {line.carrier_name} · invoice {line.invoice_number} ·{" "}
          {line.extracted.ship_date ?? "date not read"} ·{" "}
          {line.extracted.origin_zip} → {line.extracted.dest_zip}{" "}
          <a href={`/api/invoice/${line.invoice_number}.pdf`} target="_blank" rel="noreferrer">
            view PDF
          </a>
        </div>
      </div>

      <div className="section">
        <div className="variance-bar">
          <span className={`big ${r.variance > 0 ? "pos" : "neg"}`}>
            {r.variance > 0 ? "+" : ""}
            {money(r.variance)}
          </span>
          <span className={`pill ${DISPOSITION_TONE[line.disposition]}`}>
            {line.disposition.replace(/_/g, " ")}
          </span>
          <span className="pill neutral">{line.classification.category.replace(/_/g, " ")}</span>
        </div>
        <div className="reason">
          <div className="who">Policy — always code, never the model</div>
          {line.policy_reason}
        </div>
      </div>

      <div className="section">
        <h4>Carrier billed vs. what the rate card says</h4>
        <div className="cols">
          <ChargeTable
            title="Carrier invoice"
            rows={line.extracted.charges.map((c) => ({
              code: c.code,
              label: c.description,
              amount: c.amount,
            }))}
            total={r.billed_total}
            flagged={flagged}
          />
          <ChargeTable
            title="Our recompute"
            rows={r.expected_charges.map((c) => ({
              code: c.code,
              label: c.code.replace(/_/g, " "),
              basis: c.basis,
              amount: c.amount,
            }))}
            total={r.expected_total}
          />
        </div>
      </div>

      <div className="section">
        <h4>Machine-checked facts</h4>
        <ul className="facts">
          <li>
            <span>Matched to shipment</span>
            <span>
              {r.matched_shipment_id ?? "no match"} ({r.match_method.toLowerCase()},{" "}
              {r.match_confidence.toFixed(2)})
            </span>
          </li>
          <li className={r.weight_delta_lb ? "hot" : ""}>
            <span>Billed weight vs our BOL</span>
            <span>
              {r.weight_delta_lb === null
                ? "—"
                : `${r.weight_delta_lb > 0 ? "+" : ""}${r.weight_delta_lb.toLocaleString()} lb`}
            </span>
          </li>
          <li className={r.class_delta ? "hot" : ""}>
            <span>Billed class vs our BOL</span>
            <span>
              {r.class_delta === null
                ? "—"
                : `${r.class_delta > 0 ? "+" : ""}${r.class_delta}`}
            </span>
          </li>
          <li className={r.unsupported_accessorials.length ? "hot" : ""}>
            <span>Accessorials with no BOL or facility support</span>
            <span>
              {r.unsupported_accessorials.length
                ? r.unsupported_accessorials.join(", ").replace(/_/g, " ")
                : "none"}
            </span>
          </li>
          {r.fuel_pct_billed !== null && r.fuel_pct_expected !== null && (
            <li
              className={
                Math.abs(r.fuel_pct_billed - r.fuel_pct_expected) > 0.002 ? "hot" : ""
              }
            >
              <span>Fuel % billed vs published for pickup week</span>
              <span>
                {(r.fuel_pct_billed * 100).toFixed(2)}% vs{" "}
                {(r.fuel_pct_expected * 100).toFixed(2)}%
              </span>
            </li>
          )}
          <li className={r.is_duplicate_pro ? "hot" : ""}>
            <span>PRO already invoiced</span>
            <span>{r.is_duplicate_pro ? "YES" : "no"}</span>
          </li>
        </ul>
      </div>

      <div className="section">
        <h4>Why — classifier</h4>
        <div className="reason">
          <div className="who">
            {line.decided_by === "code"
              ? "Code — resolved deterministically; no model was called"
              : "Model — constrained to a closed category set, given the numbers above"}
          </div>
          {line.classification.reasoning}
        </div>
        {line._ground_truth && (
          <div className="truth">
            Demo check — injected defect: <b>{line._ground_truth.replace(/_/g, " ")}</b>{" "}
            <span className={correct ? "match" : "miss"}>
              {correct ? "· system agreed" : "· system said something else"}
            </span>
            {line._ground_truth_note && <div style={{ marginTop: 3 }}>{line._ground_truth_note}</div>}
          </div>
        )}
      </div>

      {line.cited_clauses.length > 0 && (
        <div className="section">
          <h4>Governing tariff — retrieved, quoted in full, never paraphrased</h4>
          {line.cited_clauses.map((c: Clause) => (
            <div className="clause" key={c.id}>
              <div className="cid">{c.id}</div>
              <div className="ctitle">{c.title}</div>
              <div className="ctext">{c.text}</div>
            </div>
          ))}
        </div>
      )}

      <div className="actions">
        {line._decision ? (
          <span className="pill ok">
            Logged: {line._decision.human_action.toLowerCase()}
          </span>
        ) : (
          <>
            <button className="act primary" disabled={busy} onClick={() => decide("APPROVE")}>
              Approve
            </button>
            <button className="act danger" disabled={busy} onClick={() => decide("DISPUTE")}>
              Dispute
            </button>
            <button className="act" disabled={busy} onClick={() => decide("ESCALATE")}>
              Escalate
            </button>
            <span style={{ color: "var(--muted)", fontSize: 12 }}>
              Every decision is logged against what the system said. That delta is the
              override rate.
            </span>
          </>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */


/** The six gates, in the order the server runs them. Rendered as a strip so a
 *  rejection shows exactly how far the document got before it was stopped. */
const GATES = ["FILE", "DEDUPE", "TRIAGE", "EXTRACT", "SUM_CHECK", "OWNERSHIP"] as const;

/** Refusals that are ordinary operator mistakes rather than findings: the same
 *  file twice, or a document that is not an invoice. These get a floating
 *  notification and nothing else — the gate strip already shows how far the
 *  document got, and a panel of explanation left sitting on the page is noise.
 *
 *  Everything else (a failed sum check, an invoice that is not ours) keeps the
 *  inline panel, because those are things a person has to act on. */
const TOAST_ONLY_GATES = new Set(["DEDUPE", "TRIAGE"]);

function Upload({
  onAccepted,
  notify,
}: {
  onAccepted: () => void;
  notify: (t: Omit<ToastMsg, "id">) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [over, setOver] = useState(false);
  const [res, setRes] = useState<UploadResult | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const send = async (f: File) => {
    setBusy(true);
    setErr(null);
    setRes(null);
    try {
      const body = new FormData();
      body.append("file", f);
      const r = await fetch("/api/upload", { method: "POST", body });
      const j: UploadResult = await r.json();
      setRes(j);
      if (j.accepted) {
        onAccepted();
      } else if (j.gate.rejected_at && TOAST_ONLY_GATES.has(j.gate.rejected_at)) {
        notify({
          tone: j.gate.rejected_at === "DEDUPE" ? "warn" : "bad",
          gate: j.gate.rejected_at,
          title: j.gate.reason ?? "Refused",
          body: j.gate.detail ?? undefined,
          cost: j.gate.cost_usd,
        });
      }
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  const stoppedAt = res?.gate.rejected_at;
  const stopIdx = stoppedAt ? GATES.indexOf(stoppedAt as (typeof GATES)[number]) : -1;

  return (
    <div className="panel">
      <h2>Upload an invoice — six gates, cheapest first</h2>
      <div style={{ padding: 14 }}>
        <label
          className={`drop ${over ? "over" : ""}`}
          style={{ display: "block" }}
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            const f = e.dataTransfer.files?.[0];
            if (f) void send(f);
          }}
        >
          <input
            type="file"
            accept="application/pdf"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void send(f);
            }}
          />
          <div className="big">{busy ? "Running the gates…" : "Drop a PDF here, or click to choose"}</div>
          <div className="small">
            Nothing is sent to a model until the free checks pass.
          </div>
        </label>

        {err && <div className="verdict bad" style={{ marginTop: 12 }}>{err}</div>}

        {res && (
          <>
            <div className="gates">
              {GATES.map((g, i) => {
                const passed = res.gate.gates_passed.includes(g);
                const failed = stoppedAt === g;
                const skipped = stopIdx >= 0 && i > stopIdx;
                return (
                  <span
                    key={g}
                    className={`gate ${passed ? "pass" : failed ? "fail" : skipped ? "skip" : ""}`}
                  >
                    {passed ? "✓ " : failed ? "✗ " : ""}
                    {g}
                  </span>
                );
              })}
            </div>

            {!(res.gate.rejected_at && TOAST_ONLY_GATES.has(res.gate.rejected_at)) && (
            <div className={`verdict ${res.accepted ? "ok" : "bad"}`}>
              <div className="head">
                {res.accepted
                  ? `Accepted — ${res.invoice_number}, ${res.lines} lines`
                  : `Refused at ${res.gate.rejected_at}`}
              </div>
              <div>{res.accepted ? res.triage?.reason : res.gate.reason}</div>
              {res.gate.detail && <div className="detail">{res.gate.detail}</div>}

              <dl>
                <dt>cost of this decision</dt>
                <dd>
                  {res.gate.cost_usd === 0
                    ? "$0.00 — no model was called"
                    : `$${res.gate.cost_usd.toFixed(4)}`}
                </dd>
                {res.extraction_method && (
                  <>
                    <dt>read by</dt>
                    <dd>
                      {res.extraction_method}
                      {res.extraction_method === "text-layer" ? " (deterministic parse, free)" : ""}
                    </dd>
                  </>
                )}
                {res.ownership && (
                  <>
                    <dt>our shipments</dt>
                    <dd>
                      {res.ownership.matched}/{res.ownership.lines} PRO numbers matched
                    </dd>
                  </>
                )}
                {res.accepted && (
                  <>
                    <dt>recoverable</dt>
                    <dd>{money(res.recoverable_usd ?? 0)}</dd>
                  </>
                )}
              </dl>
            </div>
            )}
          </>
        )}

      </div>
    </div>
  );
}

export default function App() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [items, setItems] = useState<AuditedLine[]>([]);
  const [runId, setRunId] = useState<string>("");
  // Either a disposition ("NEEDS_REVIEW"), the sentinel "ALL", or a category
  // filter written as "CAT:UNKNOWN".
  const [filter, setFilter] = useState<string>("NEEDS_REVIEW");
  // "invoice" is the order the pipeline produced -- grouped by document. The date
  // orders are for working the
  // ageing end of the book: MFS-400 runs 180 days from the INVOICE date, so oldest
  // first is the order that puts the lines nearest to losing the dispute right on top.
  const [sort, setSort] = useState<"invoice" | "oldest" | "newest">("invoice");
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<Clause[] | null>(null);
  const [view, setView] = useState<"queue" | "ops">("queue");
  const [toasts, setToasts] = useState<ToastMsg[]>([]);

  const notify = useCallback((t: Omit<ToastMsg, "id">) => {
    setToasts((prev) => [...prev, { ...t, id: Date.now() + Math.random() }]);
  }, []);
  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  // Changing the filter and then the sort starts two loads, and nothing makes the
  // older one finish first. Without this guard a slow, stale response landed last
  // and overwrote the newer one: the dropdown said "oldest first" while the rows
  // were in invoice order. Found by the Playwright sort spec; only the latest
  // load may write state.
  const loadSeq = useRef(0);

  const load = useCallback(async () => {
    const seq = ++loadSeq.current;
    try {
      const s = await api<Summary>("/summary");
      const sortQs = sort === "invoice" ? "" : `sort=${sort}`;
      const qs0 = filter === "ALL"
        ? ""
        : filter.startsWith("CAT:")
          ? `?category=${filter.slice(4)}`
          : `?disposition=${filter}`;
      const qs = qs0 ? (sortQs ? `${qs0}&${sortQs}` : qs0) : sortQs ? `?${sortQs}` : "";
      const q = await api<{ items: AuditedLine[] }>(`/queue${qs}`);
      if (seq !== loadSeq.current) return;
      setSummary(s);
      setRunId(s.run_id);
      setItems(q.items);
      setSelected((prev) =>
        prev && q.items.some((i) => i.pro_number + i.invoice_number === prev)
          ? prev
          : q.items[0]
            ? q.items[0].pro_number + q.items[0].invoice_number
            : null,
      );
      setError(null);
    } catch (e) {
      if (seq === loadSeq.current) setError(String(e));
    }
  }, [filter, sort]);

  useEffect(() => {
    void load();
  }, [load]);

  const runSearch = async () => {
    if (query.trim().length < 2) return;
    const r = await api<{ hits: Clause[] }>(`/search?q=${encodeURIComponent(query)}&k=4`);
    setHits(r.hits);
  };

  const current = items.find((i) => i.pro_number + i.invoice_number === selected) ?? null;

  return (
    <div className="app">
      <Toasts items={toasts} onClose={dismiss} />

      <header className="top">
        <h1>Carrier Invoice Audit</h1>
        <span className="sub">
          A/P reconciliation against carrier billing · Meridian Freight Systems ·{" "}
          {runId || "no run"}
        </span>
        <nav style={{ marginLeft: "auto", display: "flex", gap: 7, alignItems: "center" }}>
          <button className={`chip ${view === "queue" ? "on" : ""}`} onClick={() => setView("queue")}>
            Review queue
          </button>
          <button className={`chip ${view === "ops" ? "on" : ""}`} onClick={() => setView("ops")}>
            Operations
          </button>
          <LiveClock />
        </nav>
      </header>

      <div className="banner">
        <b>Demo build.</b> Synthetic carrier, synthetic rate card, synthetic invoices with a
        known set of injected billing defects. Not production, not deployed, no real carrier
        data. The injected defect is shown on each line so you can check the system's work —
        a real queue has no such column.
      </div>

      {error && <div className="banner">{error}</div>}

      {view === "ops" && <Dashboard />}

      {view === "queue" && (
       <>
      {/* Metrics and ingestion sit together: the override-rate tile is what the
          queue produces, and the upload panel is what feeds it. Reading them side
          by side makes the loop visible. */}
      <div className="topgrid">
        {summary ? <Tiles s={summary} /> : <div />}
        <Upload onAccepted={load} notify={notify} />
      </div>

      <div className="split">
        <div className="panel">
          <h2>Review queue</h2>
          {/* Ordered by how a reviewer works, not alphabetically: the queue you
              act on, then the subset of it that needs the most thought, then the
              two the system already settled, then everything. Abstentions sit
              directly after needs-review because they ARE needs-review -- just the
              part where the system declined rather than exceeded a threshold. */}
          <div className="filters">
            {[
              { key: "NEEDS_REVIEW", label: "needs review" },
              { key: "CAT:UNKNOWN", label: "⚑ abstained" },
              { key: "AUTO_DISPUTED", label: "auto disputed" },
              { key: "AUTO_APPROVED", label: "auto approved" },
              { key: "ALL", label: "all" },
            ].map(({ key, label }) => {
              const count =
                key === "ALL"
                  ? summary?.lines
                  : key === "CAT:UNKNOWN"
                    ? summary?.by_category?.UNKNOWN
                    : summary?.by_disposition?.[key];
              return (
                <button
                  key={key}
                  className={`chip ${filter === key ? "on" : ""}`}
                  title={
                    key === "CAT:UNKNOWN"
                      ? "Lines the model declined to classify. Already inside needs-review; shown separately because deciding one is a different job from confirming a threshold."
                      : undefined
                  }
                  onClick={() => setFilter(key)}
                >
                  {label}
                  {count !== undefined ? ` ${count}` : ""}
                </button>
              );
            })}
            <select
              className="sortsel"
              value={sort}
              onChange={(e) => setSort(e.target.value as typeof sort)}
              title="Oldest first is the ageing order: MFS-400 allows 180 days from the invoice date to dispute, so the oldest unreviewed line is the closest to losing that right."
            >
              <option value="invoice">by invoice</option>
              <option value="oldest">oldest first (ageing)</option>
              <option value="newest">newest first</option>
            </select>
          </div>
          <div className="queue">
            {items.length === 0 && <div className="empty">Nothing in this bucket.</div>}
            {items.map((i) => {
              const key = i.pro_number + i.invoice_number;
              return (
                <div
                  key={key}
                  className={`row ${selected === key ? "sel" : ""}`}
                  onClick={() => setSelected(key)}
                >
                  <div className="pro">{i.pro_number}</div>
                  <div className="amt">
                    {i.recomputation.variance > 0 ? "+" : ""}
                    {money(i.recomputation.variance)}
                  </div>
                  <div className="meta">
                    <span
                      className={`pill ${
                        i.classification.category === "UNKNOWN"
                          ? "neutral"
                          : DISPOSITION_TONE[i.disposition]
                      }`}
                    >
                      {i.classification.category === "UNKNOWN" ? "⚑ " : ""}
                      {i.classification.category.replace(/_/g, " ").toLowerCase()}
                    </span>{" "}
                    {i.invoice_number} · {i._invoice_date ?? i.extracted.ship_date}
                    {typeof i._dispute_days_left === "number" && (
                      <span
                        className={i._dispute_days_left <= 30 ? "aged" : undefined}
                        title="Days left in the MFS-400 dispute window, counted from the invoice date."
                      >
                        {" · "}
                        {i._dispute_days_left < 0
                          ? "window closed"
                          : `${i._dispute_days_left}d left`}
                      </span>
                    )}
                    {i._decision ? " · decided" : ""}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div>
          <div className="panel" style={{ marginBottom: 16 }}>
            {current ? (
              <Detail line={current} runId={runId} onDecided={load} />
            ) : (
              <div className="empty">Select a line from the queue.</div>
            )}
          </div>

          <div className="panel">
            <h2>Tariff search — hybrid BM25 + dense, fused with RRF</h2>
            <div className="searchbar">
              <input
                value={query}
                placeholder="e.g. they billed a liftgate at a dock-equipped warehouse"
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && void runSearch()}
              />
              <button className="act" onClick={() => void runSearch()}>
                Search
              </button>
            </div>
            <div style={{ padding: 14 }}>
              {hits === null && (
                <div style={{ color: "var(--muted)", fontSize: 12.5 }}>
                  The rate table is a join. The tariff rules are prose. Only the second one is
                  a retrieval problem — this searches the second.
                </div>
              )}
              {hits?.map((h) => (
                <div className="clause" key={h.id}>
                  <div className="cid">
                    {h.id} · {h.source} {h.score.toFixed(4)}
                  </div>
                  <div className="ctitle">{h.title}</div>
                  <div className="ctext">{h.text}</div>
                </div>
              ))}
              {hits?.length === 0 && <div className="empty">No clause matched.</div>}
            </div>
          </div>
        </div>
      </div>
       </>
      )}
    </div>
  );
}
