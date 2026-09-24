"""The API behind the review queue.

The interface is not decoration on this project. The posting this was built for
puts it in the requirements -- ship "the interface the ops team will actually use"
-- and the reason is that an audit system's real output is not a classification, it
is a decision a person makes in about twenty seconds.

So the queue is designed around that twenty seconds. Every line arrives with the
carrier's charges and our recomputed charges side by side, the specific numbers
that disagree, and the governing tariff clause quoted in full rather than
summarised. The reviewer should never have to open the rate card, and never have
to trust a paraphrase of a contract term they are about to quote at a carrier.

`POST /api/decide` is append-only and records what the system said alongside what
the human did. The delta between those two columns is the **human override rate**,
which is the single best signal for whether an automated workflow is working -- and
the override log is the eval set for the next iteration.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import alerts as alerts_mod
from . import db, extract, intake, llm, observe, pipeline, policy, recompute, retrieve, textlayer
from . import classify as classify_mod

app = FastAPI(title="Carrier Invoice Audit", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _con():
    return db.connect()


def _latest_run(con) -> str | None:
    r = db.one(con, "SELECT run_id FROM audit_results ORDER BY id DESC LIMIT 1")
    return r["run_id"] if r else None


@app.get("/api/health")
def health() -> dict:
    con = _con()
    n_inv = db.one(con, "SELECT COUNT(*) c FROM invoices")["c"]
    run = _latest_run(con)
    n_lines = db.one(con, "SELECT COUNT(*) c FROM audit_results WHERE run_id=?", (run,))["c"] if run else 0
    con.close()
    return {
        "ok": True,
        "invoices": n_inv,
        "latest_run": run,
        "lines_in_run": n_lines,
        "model_configured": llm.available(),
        "embedder": retrieve.embedder().name,
    }


# How a reviewer may order the queue. `invoice` is the order the pipeline produced:
# grouped by invoice, lines in the order they appear on the document. It is named for
# what it gives a reviewer -- one document cleared at a time, which is the unit a
# dispute letter and the sum check both work in -- and not for the insertion order it
# happens to be.
# The date orders exist for the other way people actually work: MFS-400 gives the
# shipper 180 days from the INVOICE date to dispute, so the oldest unreviewed line
# is the one closest to losing the right to dispute at all. That is why the sort key
# is the invoice date and not the ship date -- the clock in the contract starts on
# the invoice.
SORTS = ("invoice", "oldest", "newest")
DISPUTE_WINDOW_DAYS = 180


def _days_left(invoice_date: str | None) -> int | None:
    """Days remaining in the MFS-400 dispute window, or None if the date is unknown.

    Negative means the window has closed and the charge can no longer be disputed,
    which is worth showing rather than hiding: a reviewer should see that the line
    is now a write-off, not a task.
    """
    if not invoice_date:
        return None
    try:
        issued = date.fromisoformat(invoice_date)
    except ValueError:
        return None
    return DISPUTE_WINDOW_DAYS - (date.today() - issued).days


@app.get("/api/queue")
def queue(
    run_id: str | None = None,
    disposition: str | None = None,
    category: str | None = None,
    sort: str = "invoice",
) -> dict:
    con = _con()
    run_id = run_id or _latest_run(con)
    if not run_id:
        con.close()
        raise HTTPException(404, "No audit run on file. Run: python -m app.cli run --stub")

    if sort not in SORTS:
        con.close()
        raise HTTPException(400, f"sort must be one of {', '.join(SORTS)}")

    rows = db.rows(con, "SELECT payload FROM audit_results WHERE run_id=? ORDER BY id", (run_id,))
    invoice_dates = {
        r["invoice_number"]: r["invoice_date"]
        for r in db.rows(con, "SELECT invoice_number, invoice_date FROM invoices")
    }
    truth = {
        (r["invoice_number"], r["pro_number"]): r
        for r in db.rows(con, "SELECT * FROM ground_truth")
    }
    decided = {
        (r["invoice_number"], r["pro_number"]): r
        for r in db.rows(con,
                         "SELECT * FROM decisions WHERE run_id=? ORDER BY id", (run_id,))
    }
    con.close()

    items = []
    for r in rows:
        p = json.loads(r["payload"])
        if disposition and p["disposition"] != disposition:
            continue
        if category and p["classification"]["category"] != category:
            continue
        key = (p["invoice_number"], p["pro_number"])
        # Ground truth is shown in the UI ONLY because this is a demo over synthetic
        # data and the point is to let a viewer check the system's work. A production
        # queue has no such column, because nothing knows the answer.
        p["_ground_truth"] = truth.get(key, {}).get("category")
        p["_ground_truth_note"] = truth.get(key, {}).get("note")
        p["_decision"] = decided.get(key)
        # The invoice date is on the invoice, not on the audited line, so it is joined
        # here rather than duplicated into every payload. Days left is derived from it
        # and never stored: a stored countdown is wrong the next morning.
        inv_date = invoice_dates.get(p["invoice_number"])
        p["_invoice_date"] = inv_date
        p["_dispute_days_left"] = _days_left(inv_date)
        items.append(p)

    if sort != "invoice":
        # Undated lines sort last in BOTH directions, which is why this is not one
        # `reverse=` flag. An unknown date is not old and it is not new; floating it to
        # the top of an ageing queue would be a false alarm, and a plain reverse would
        # do exactly that.
        dated = sorted((p for p in items if p["_invoice_date"]),
                       key=lambda p: p["_invoice_date"])
        undated = [p for p in items if not p["_invoice_date"]]
        items = (dated if sort == "oldest" else list(reversed(dated))) + undated

    return {"run_id": run_id, "count": len(items), "sort": sort, "items": items}


@app.get("/api/summary")
def summary(run_id: str | None = None) -> dict:
    """Operational numbers for a run, recomputed from stored results."""
    con = _con()
    run_id = run_id or _latest_run(con)
    if not run_id:
        con.close()
        raise HTTPException(404, "No audit run on file.")
    rows = [json.loads(r["payload"]) for r in
            db.rows(con, "SELECT payload FROM audit_results WHERE run_id=?", (run_id,))]
    n_dec = db.one(con, "SELECT COUNT(*) c FROM decisions WHERE run_id=?", (run_id,))["c"]
    overrides = db.one(
        con,
        "SELECT COUNT(*) c FROM decisions WHERE run_id=? AND human_category IS NOT NULL "
        "AND human_category != system_category",
        (run_id,),
    )["c"]
    con.close()

    by_disp: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    recoverable = 0.0
    billed = 0.0
    for p in rows:
        by_disp[p["disposition"]] = by_disp.get(p["disposition"], 0) + 1
        c = p["classification"]["category"]
        by_cat[c] = by_cat.get(c, 0) + 1
        billed += p["recomputation"]["billed_total"]
        if p["disposition"] in ("AUTO_DISPUTED", "NEEDS_REVIEW") and p["recomputation"]["variance"] > 0:
            recoverable += p["recomputation"]["variance"]

    n = len(rows)
    esc = by_disp.get("NEEDS_REVIEW", 0)
    return {
        "run_id": run_id,
        "lines": n,
        "by_disposition": by_disp,
        "by_category": by_cat,
        "automation_rate": round((n - esc) / n, 4) if n else 0,
        "escalation_rate": round(esc / n, 4) if n else 0,
        "total_billed_usd": round(billed, 2),
        "recoverable_usd": round(recoverable, 2),
        "human_decisions": n_dec,
        # The number that matters most once this is live.
        "human_override_rate": round(overrides / n_dec, 4) if n_dec else None,
    }


@app.get("/api/eval")
def eval_report() -> dict:
    path = db.DATA_DIR / "eval_report.json"
    if not path.exists():
        raise HTTPException(404, "No eval report. Run: python eval/run_eval.py --stub")
    return json.loads(path.read_text())


class Decision(BaseModel):
    run_id: str
    invoice_number: str
    pro_number: str
    action: str            # APPROVE | DISPUTE | ESCALATE
    category: str | None = None
    note: str | None = None


@app.post("/api/decide")
def decide(d: Decision) -> dict:
    con = _con()
    row = db.one(
        con,
        "SELECT payload FROM audit_results WHERE run_id=? AND invoice_number=? AND pro_number=? "
        "ORDER BY id DESC LIMIT 1",
        (d.run_id, d.invoice_number, d.pro_number),
    )
    if not row:
        con.close()
        raise HTTPException(404, "No such audited line in that run.")
    p = json.loads(row["payload"])
    con.execute(
        "INSERT INTO decisions (run_id,invoice_number,pro_number,system_disposition,"
        "system_category,human_action,human_category,note) VALUES (?,?,?,?,?,?,?,?)",
        (d.run_id, d.invoice_number, d.pro_number, p["disposition"],
         p["classification"]["category"], d.action, d.category, d.note),
    )
    con.commit()
    con.close()
    return {"ok": True, "overrode_category": bool(d.category and d.category != p["classification"]["category"])}


@app.get("/api/search")
def search(q: str = Query(min_length=2), k: int = 5, strategy: str = "hybrid") -> dict:
    """Tariff search. Retrieval only -- the clause is returned, never a paraphrase."""
    con = _con()
    hits = retrieve.search(con, q, k=k, strategy=strategy)
    con.close()
    return {
        "query": q, "strategy": strategy, "embedder": retrieve.embedder().name,
        "hits": [h.__dict__ for h in hits],
    }


@app.get("/api/invoice/{invoice_number}.pdf")
def invoice_pdf(invoice_number: str):
    con = _con()
    row = db.one(con, "SELECT pdf_path FROM invoices WHERE invoice_number=?", (invoice_number,))
    con.close()
    if not row or not Path(row["pdf_path"]).exists():
        raise HTTPException(404, "No PDF for that invoice.")
    return FileResponse(row["pdf_path"], media_type="application/pdf")


class RunRequest(BaseModel):
    stub: bool = True
    strategy: str = "hybrid"
    limit: int | None = None


@app.post("/api/run")
def trigger_run(req: RunRequest) -> dict:
    """Re-audit from the UI.

    Synchronous, which is the honest shape for nine invoices. A real deployment puts
    this on a queue -- a vision extraction run is minutes of work and does not belong
    in a request handler.
    """
    con = _con()
    try:
        return pipeline.run(con, stub=req.stub, strategy=req.strategy,
                            limit=req.limit, verbose=False)
    finally:
        con.close()




# --------------------------------------------------------------------------
# Upload
# --------------------------------------------------------------------------


# What each sample is FOR. The previous version had one ternary -- holdout, or
# "a design studio bill" -- so the judgment invoice, which is a real invoice meant
# to be accepted, was described in the UI as a decoy that would be refused at
# triage. A sample panel that lies about what a sample proves is worse than no
# panel, because it is read aloud during a demo.
SAMPLE_EXPECTATIONS = {
    "holdout": "accepted --- a real carrier invoice that was never ingested. Proves the "
               "whole chain end to end on a document the system has not seen.",
    "decoy": "rejected at TRIAGE --- invoice-shaped but not freight. Proves a document "
             "that is not ours is refused, and refused for nothing.",
    "judgment": "accepted --- and every line needs a human. Proves the cases arithmetic "
                "cannot decide: charges that are correctly priced and contractually "
                "impermissible.",
}


@app.get("/api/upload/samples")
def upload_samples() -> dict:
    """The sample PDFs, and what each one is meant to prove."""
    d = db.DATA_DIR / "upload_samples"
    if not d.exists():
        return {"directory": str(d), "files": []}

    def expect(name: str) -> str:
        for key, text in SAMPLE_EXPECTATIONS.items():
            if key in name:
                return text
        return "unlabelled sample --- run it and see which gate answers"

    return {
        "directory": str(d),
        "files": [
            {"name": f.name, "bytes": f.stat().st_size, "expect": expect(f.name)}
            for f in sorted(d.glob("*.pdf"))
        ],
    }


@app.get("/api/upload/history")
def upload_history(limit: int = 20) -> dict:
    con = _con()
    rows = db.rows(con, "SELECT * FROM uploads ORDER BY id DESC LIMIT ?", (limit,))
    con.close()
    return {"count": len(rows), "uploads": rows}


@app.post("/api/upload")
async def upload_invoice(file: UploadFile = File(...)) -> dict:
    """Run one PDF through the ingestion gates, and audit it if it survives.

    The chain is ordered cheapest-first so that a document which is not an invoice
    is refused for a fifth of a cent rather than seven cents, and one that is not a
    PDF at all is refused for nothing. Every attempt is recorded, accepted or not.
    """
    raw = await file.read()
    sha = intake.sha256(raw)
    usage = llm.Usage()
    g = intake.GateResult(filename=file.filename or "upload.pdf", sha256=sha, size_bytes=len(raw))
    con = _con()

    try:
        # 1 + 2: free, deterministic. Nothing has cost anything yet.
        intake.gate_file(raw, g.filename)
        g.ok("FILE")
        intake.gate_duplicate_file(con, sha)
        g.ok("DEDUPE")

        if not llm.available():
            raise intake.Rejected(
                "TRIAGE", "No model is configured, so the document cannot be read.",
                "Set ANTHROPIC_API_KEY. The deterministic gates above passed.")

        # 3: triage. Free from the text layer when it can decide; a cheap model
        # call only when it cannot (a scan, or genuinely ambiguous signals).
        path = intake.save_pdf(raw, sha)
        text = textlayer.extract_text(path)
        triage = intake.gate_triage_free(text) if textlayer.has_text_layer(text) else None
        if triage is None:
            import base64
            triage = intake.gate_triage(base64.standard_b64encode(raw).decode(), usage)
            g.cost_usd = usage.cost_usd
        g.ok("TRIAGE")

        # 4: extraction, via the ladder -- deterministic parse first, model only if
        # that fails or the document is a scan.
        result = extract.extract_invoice(path, usage)
        g.cost_usd = usage.cost_usd
        if result.invoice is None:
            raise intake.Rejected("EXTRACT", "Extraction failed.", result.error)
        g.ok("EXTRACT")

        # 5: arithmetic. The model's transcription has to add up.
        if not result.sum_check_passed:
            raise intake.Rejected(
                "SUM_CHECK", "The transcription does not reconcile.", result.error)
        g.ok("SUM_CHECK")

        # 2b + 6: is it new, and is it ours?
        intake.gate_duplicate_invoice(con, result.invoice.invoice_number)
        ownership = intake.gate_ownership(con, result.invoice)
        g.ok("OWNERSHIP")

    except intake.Rejected as r:
        g.rejected_at, g.reason, g.detail = r.stage, r.reason, r.detail
        intake.record(con, g)
        con.close()
        return {"accepted": False, "gate": g.as_dict(),
                "usage": usage.summary() if usage.calls else None}

    # Accepted. Persist the invoice and audit its lines.
    inv = result.invoice
    con.execute(
        "INSERT INTO invoices (invoice_number,carrier_name,invoice_date,pdf_path,"
        "stated_total,seed_payload,source) VALUES (?,?,?,?,?,?,'upload')",
        (inv.invoice_number, inv.carrier_name, str(inv.invoice_date or ""),
         str(path), inv.invoice_total, "{}"),
    )
    con.commit()

    run_id = f"upload-{sha[:8]}"
    # Seed the duplicate check from every PRO already invoiced on a DIFFERENT
    # invoice. Without this the upload path starts from an empty set, so a carrier
    # re-billing a shipment we have already paid for lands as a clean line -- the
    # single defect this system exists to catch, silently missed on the one path a
    # person actually drives.
    seen: set[str] = {
        r["pro_number"] for r in db.rows(
            con,
            "SELECT DISTINCT pro_number FROM audit_results WHERE invoice_number != ?",
            (inv.invoice_number,),
        )
    }
    audited = []
    for line in inv.lines:
        rec = recompute.recompute(con, line, seen)
        seen.add(line.pro_number)
        cls, clauses, decided_by = classify_mod.classify(
            con, rec, usage=usage, billed_charges=line.charges)
        disposition, reason = policy.decide(rec, cls)
        payload = {
            "pro_number": line.pro_number, "invoice_number": inv.invoice_number,
            "carrier_name": inv.carrier_name, "extracted": line.model_dump(mode="json"),
            "recomputation": rec.model_dump(mode="json"),
            "classification": cls.model_dump(mode="json"),
            "disposition": disposition.value, "policy_reason": reason,
            "cited_clauses": clauses, "decided_by": decided_by, "ground_truth": None,
        }
        db.save_audit(con, run_id, payload)
        audited.append(payload)

    g.cost_usd = usage.cost_usd
    intake.record(con, g, inv.invoice_number)
    con.close()

    recoverable = round(sum(a["recomputation"]["variance"] for a in audited
                            if a["recomputation"]["variance"] > 0
                            and a["disposition"] != "AUTO_APPROVED"), 2)
    return {
        "accepted": True,
        "gate": g.as_dict(),
        "triage": triage.model_dump(),
        "ownership": ownership,
        "invoice_number": inv.invoice_number,
        "extraction_method": result.method,
        "run_id": run_id,
        "lines": len(audited),
        "recoverable_usd": recoverable,
        "by_disposition": {d: sum(1 for a in audited if a["disposition"] == d)
                           for d in {a["disposition"] for a in audited}},
        "usage": usage.summary(),
    }


@app.get("/api/alerts")
def alerts(run_id: str | None = None) -> dict:
    """What the owner should look at, most actionable first.

    Not one alert per abstention -- a single declined line is a reviewer's job.
    These fire on PATTERNS: a charge recurring with no contracted rate, several
    abstentions sharing a shape, a rate climbing, extraction drifting. Each carries
    a suggested action, because an alert that only describes a problem hands the
    work of deciding what to do back to the reader every time it fires.
    """
    con = _con()
    try:
        return alerts_mod.summary(con, run_id)
    finally:
        con.close()


@app.get("/api/observability")
def observability(days: int = 30) -> dict:
    """Stage telemetry: latency, outcomes, cost, and the drift alarm.

    Stage-level rather than span-based because this is a workflow, not an agent --
    a fixed sequence of named steps needs timings and outcome distributions, not a
    trace tree.
    """
    con = _con()
    try:
        return observe.report(con, limit_days=days)
    finally:
        con.close()


@app.get("/api/observability/trace/{trace_id}")
def observability_trace(trace_id: str) -> dict:
    """Every stage one document passed through, in order."""
    con = _con()
    try:
        rows = observe.trace_detail(con, trace_id)
        if not rows:
            raise HTTPException(404, "No such trace.")
        return {"trace_id": trace_id, "stages": rows}
    finally:
        con.close()


# ---------------------------------------------------------------------------
# The UI mount goes LAST, and that is load-bearing.
#
# Starlette matches routes in registration order. A catch-all mount at "/"
# registered before an API route will swallow it and answer 405. That happened:
# appending the upload endpoints below the mount made POST /api/upload return
# "Method Not Allowed" with no other symptom.
# ---------------------------------------------------------------------------
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
