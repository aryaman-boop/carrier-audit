"""Orchestration.

Six stages, and the interesting thing about the shape is how little of it is the
model:

    ingest -> extract (ladder) -> guard -> match -> recompute -> classify (code, then
    LLM) -> decide

At most two model calls per invoice, and usually fewer. Extraction is a ladder whose
model rung is reached only for scans and unknown layouts -- on the current set it is
never reached at all. Classification short-circuits in code for duplicates, unmatched
lines and in-band variances, which resolves roughly half the lines before any call.
Everything between is arithmetic and a join.

That is not modesty about what models can do. It is that this is a *workflow*, not
an agent: the steps are known in advance, they run in a fixed order, and there is
nothing for a planner to decide. An agent loop here would add non-determinism,
cost and failure modes to a process that has none of those problems, and being the
person who says so is worth more than being the person who wires up a loop.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from pathlib import Path

from . import classify as classify_mod
from . import db, extract, llm, observe, policy, recompute
from .schema import AuditedLine, Disposition, VarianceCategory, VarianceClassification

# Default hard ceiling on what one run may spend. A full 8-invoice run on the most
# expensive model pairing is about $1.80, so $2.50 clears it with headroom while
# still stopping a runaway -- a bad loop, a retry storm, or a much larger invoice
# batch than expected. Raise it deliberately with --max-spend; it is not a
# suggestion the run may ignore.
DEFAULT_MAX_SPEND_USD = 2.50


class BudgetExceeded(RuntimeError):
    """Raised when a run hits its spend ceiling. Work already done is kept."""


def run(
    con: sqlite3.Connection,
    *,
    stub: bool = False,
    strategy: str = "hybrid",
    limit: int | None = None,
    run_id: str | None = None,
    verbose: bool = True,
    max_spend_usd: float | None = None,
    reuse_run: str | None = None,
) -> dict:
    """Audit every invoice on file. Returns a run summary.

    `max_spend_usd` is a hard ceiling checked after every invoice. It is enforced
    rather than advisory: when it trips, the run stops, keeps everything already
    audited, and says so in the summary. Spend that can only be discovered on the
    bill is not a design.
    """
    run_id = run_id or f"run-{uuid.uuid4().hex[:8]}"
    usage = llm.Usage()
    started = time.time()
    budget = DEFAULT_MAX_SPEND_USD if max_spend_usd is None else max_spend_usd
    stopped_on_budget = False

    # Replaying a previous run's extraction. Vision is the expensive stage and it is
    # deterministic input to everything after it, so iterating on the classifier or
    # the policy layer should not re-pay for reading the same PDFs. This is what
    # makes hill-climbing the classifier affordable: ~$0.46 a pass instead of ~$1.00.
    replay: dict[str, list] = {}
    if reuse_run:
        import json as _json
        from .schema import ExtractedInvoiceLine
        prior = db.rows(
            con, "SELECT payload FROM audit_results WHERE run_id=? ORDER BY id", (reuse_run,)
        )
        if not prior:
            raise ValueError(f"no stored extraction for run {reuse_run!r}")
        for row in prior:
            p = _json.loads(row["payload"])
            replay.setdefault(p["invoice_number"], []).append(
                ExtractedInvoiceLine.model_validate(p["extracted"])
            )

    invoices = db.rows(con, "SELECT * FROM invoices ORDER BY invoice_number")
    if limit:
        invoices = invoices[:limit]

    seen_pros: set[str] = set()
    audited: list[AuditedLine] = []
    blocked: list[dict] = []
    methods: dict[str, int] = {}

    for inv in invoices:
        # Checked BEFORE the next invoice, not after -- so the ceiling is a ceiling
        # rather than a number the run is allowed to overshoot once.
        if not stub and budget is not None and usage.cost_usd >= budget:
            stopped_on_budget = True
            if verbose:
                print(f"\n  STOPPED: spend ${usage.cost_usd:.4f} reached the "
                      f"${budget:.2f} ceiling. {len(audited)} lines audited so far "
                      f"are kept. Re-run with --max-spend to raise it.")
            break

        if verbose:
            print(f"  {inv['invoice_number']} ... ", end="", flush=True)

        trace = observe.new_trace()

        if reuse_run:
            from .schema import ExtractedInvoice, ExtractionResult
            lines_ = replay.get(inv["invoice_number"], [])
            result = ExtractionResult(
                invoice=ExtractedInvoice(
                    carrier_name=inv["carrier_name"], invoice_number=inv["invoice_number"],
                    invoice_date=None, lines=lines_,
                    invoice_total=round(sum(l.line_total for l in lines_), 2),
                ),
                sum_check_passed=bool(lines_),
                error=None if lines_ else "no stored lines for this invoice",
            )
        else:
            with observe.stage(con, trace, "extract", subject=inv["invoice_number"],
                               run_id=run_id) as ev:
                before = usage.cost_usd
                result = (
                    extract.extract_stub(con, inv["invoice_number"]) if stub
                    else extract.extract_invoice(Path(inv["pdf_path"]), usage)
                )
                ev["method"] = result.method
                ev["cost_usd"] = round(usage.cost_usd - before, 6)
                if not result.sum_check_passed:
                    ev["outcome"] = "rejected"
                    ev["detail"] = result.error

        if not result.sum_check_passed or result.invoice is None:
            # The guard fired. The whole invoice goes to a human -- we do not audit
            # lines we are not confident we read correctly.
            blocked.append({
                "invoice_number": inv["invoice_number"],
                "reason": result.error or "sum check failed",
                "delta": result.sum_check_delta,
            })
            if verbose:
                print(f"BLOCKED ({result.error})")
            continue

        for line in result.invoice.lines:
            rec = recompute.recompute(con, line, seen_pros)
            seen_pros.add(line.pro_number)

            try:
                with observe.stage(con, trace, "classify", subject=line.pro_number,
                                   run_id=run_id) as ev:
                    before = usage.cost_usd
                    cls, clauses, decided_by = classify_mod.classify(
                        con, rec, usage=usage, strategy=strategy,
                        use_model=not stub or llm.available(),
                        billed_charges=line.charges,
                    )
                    ev["cost_usd"] = round(usage.cost_usd - before, 6)
                    ev["method"] = "deterministic" if usage.cost_usd == before else "model"
            except Exception as e:
                cls = VarianceClassification(
                    category=VarianceCategory.UNKNOWN,
                    reasoning=f"Classification failed and the line was routed to a human: {e}",
                )
                clauses, decided_by = [], "code"

            disposition, reason = policy.decide(rec, cls)
            audited.append(AuditedLine(
                pro_number=line.pro_number,
                invoice_number=inv["invoice_number"],
                carrier_name=result.invoice.carrier_name,
                extracted=line,
                recomputation=rec,
                classification=cls,
                disposition=disposition,
                policy_reason=reason,
                cited_clauses=[c for c in clauses if c["id"] in set(cls.cited_clause_ids)] or clauses,
                decided_by=decided_by,
            ))

        methods[result.method] = methods.get(result.method, 0) + 1
        if verbose:
            print(f"{len(result.invoice.lines)} lines  [{result.method}]")

    for a in audited:
        db.save_audit(con, run_id, a.model_dump(mode="json"))

    requested = ("replay" if reuse_run else "stub" if stub else "ladder")
    # Report what actually happened, not what was asked for.
    mode = f"{requested} -> " + ", ".join(f"{k} x{v}" for k, v in sorted(methods.items())) \
        if methods else requested
    return summarize(run_id, audited, blocked, usage, time.time() - started, mode, strategy,
                     budget=budget, stopped_on_budget=stopped_on_budget,
                     invoices_total=len(invoices), methods=methods)


def summarize(
    run_id: str,
    audited: list[AuditedLine],
    blocked: list[dict],
    usage: llm.Usage,
    seconds: float,
    mode: str,
    strategy: str,
    budget: float | None = None,
    stopped_on_budget: bool = False,
    invoices_total: int | None = None,
    methods: dict[str, int] | None = None,
) -> dict:
    """The operational numbers. These are what the business is actually buying."""
    n = len(audited)
    by_disp: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    for a in audited:
        by_disp[a.disposition.value] = by_disp.get(a.disposition.value, 0) + 1
        by_cat[a.classification.category.value] = by_cat.get(a.classification.category.value, 0) + 1

    recoverable = round(sum(
        a.recomputation.variance for a in audited
        if a.recomputation.variance > 0
        and a.disposition in (Disposition.AUTO_DISPUTED, Disposition.NEEDS_REVIEW)
    ), 2)
    billed = round(sum(a.recomputation.billed_total for a in audited), 2)
    escalated = by_disp.get("NEEDS_REVIEW", 0)

    return {
        "run_id": run_id,
        "mode": mode,
        "retrieval_strategy": strategy,
        "lines_audited": n,
        "invoices_blocked_by_sum_check": len(blocked),
        "blocked_detail": blocked,
        "by_disposition": by_disp,
        "by_category": by_cat,
        # Automation rate is the headline number: the share of lines that reached a
        # decision without a person. Escalation rate is its complement and is the
        # number that must not silently drop -- a sudden fall usually means something
        # broke open, not that the system got better.
        "automation_rate": round((n - escalated) / n, 4) if n else 0.0,
        "escalation_rate": round(escalated / n, 4) if n else 0.0,
        "total_billed_usd": billed,
        "recoverable_usd": recoverable,
        "recoverable_pct_of_billed": round(recoverable / billed * 100, 2) if billed else 0.0,
        "wall_seconds": round(seconds, 1),
        "usage": usage.summary(),
        "cost_per_line_usd": round(usage.cost_usd / n, 5) if n else 0.0,
        "extraction_methods": methods or {},
        "budget_usd": budget,
        "stopped_on_budget": stopped_on_budget,
        "invoices_total": invoices_total,
    }
