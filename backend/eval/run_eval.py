"""The eval harness.

This is the part of the project that would survive contact with production, and
it is the part most demos do not have. A pipeline that produces plausible output
is not evidence of anything. A pipeline with a fixed golden set, per-category
precision and recall, and a business metric attached is something you can make a
decision with.

Three things are measured, because they fail independently:

  1. **Classification quality** -- per-category precision, recall and F1 against the
     injected ground truth, plus the confusion pairs. Macro-averaged, because the
     classes are heavily imbalanced and a micro average would let the system score
     well by being right about the common ones.

  2. **Retrieval quality** -- recall@k of the governing clause for each defect,
     measured separately for bm25, dense, hybrid and hybrid+rerank. This is what
     turns "we use hybrid search" from a claim into a number, and it is what tells
     you whether the rerank stage is worth a model call per query.

  3. **Business outcome** -- dollars correctly identified as recoverable, dollars
     missed, and the false-dispute rate. A false dispute is tracked separately from
     ordinary error because it costs carrier goodwill rather than money, and those
     are not interchangeable.

The eval set is generated from a fixed seed, so a change in the score is a change
in the system rather than a change in the data.

**Every metric's full justification -- why this one rather than the obvious
alternative, what it costs to be wrong on it, and what it fails to capture -- is in
`eval/METRICS.md`, including the list of things none of these numbers measure.**
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import classify as classify_mod  # noqa: E402
from app import db, llm, pipeline, retrieve  # noqa: E402
from app.schema import Disposition, VarianceCategory  # noqa: E402

# The tariff rule that governs each defect category -- the retrieval target.
GOVERNING_RULE = {
    "REWEIGH": "MFS-110",
    "RECLASS": "MFS-120",
    "UNAUTHORIZED_ACCESSORIAL": "MFS-200",
    "DUPLICATE_BILLING": "MFS-300",
    "FUEL_SURCHARGE_ERROR": "MFS-130",
    "RATE_CARD_MISMATCH": "MFS-100",
}


# --------------------------------------------------------------------------
# Classification metrics
# --------------------------------------------------------------------------


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 for one category.

    Read in this system's terms: precision is "of the lines we called REWEIGH, how
    many were", recall is "of the reweighs, how many did we find". **Precision
    matters more here**, because code already caught the variance and escalated it
    -- the classifier is naming the defect, not detecting it. A wrong name sends a
    reviewer to build the wrong dispute; a missing name costs them context on a
    line they were going to see anyway.
    """
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return round(p, 4), round(r, 4), round(f, 4)


def score_classification(con, run_id: str) -> dict:  # noqa: C901
    """Per-category precision/recall/F1, macro-averaged, plus directed confusions.

    **Macro, not micro or weighted.** The classes are heavily imbalanced (21
    NO_VARIANCE against 4 DUPLICATE_BILLING). A weighted average lets a system look
    excellent by being right about the common easy case and wrong about every rare
    expensive one. Macro makes an error on DUPLICATE_BILLING cost exactly what an
    error on NO_VARIANCE costs, which is closer to how it costs in reality.

    Caveat that matters at this sample size: macro over-weights small categories.
    With n=4, one error moves that category's F1 by 0.25 and macro F1 by ~0.03. Do
    not read a 2-point macro move on 66 lines as a real change.

    **Accuracy is reported but is the weakest number here** -- labelling everything
    NO_VARIANCE scores 35% on this set while being useless. A large gap between
    accuracy and macro F1 means performance is concentrated in the easy classes.

    **Confusions are recorded with direction, and that is what found the real bug.**
    `REWEIGH -> UNKNOWN` and `REWEIGH -> RATE_CARD_MISMATCH` are one error each to
    accuracy and completely different events: the first is the model abstaining
    (safe, routed to a human, working as designed), the second is a confident wrong
    answer that could reach a carrier. Three reweighs going to UNKNOWN rather than
    to a wrong label said the model was abstaining rather than confused, which
    pointed at incomplete evidence rather than a weak prompt.
    """
    gt = db.rows(con, "SELECT * FROM ground_truth")
    # Lines the architecture structurally cannot catch are excluded from the score
    # and counted separately. Scoring them as model errors would blame the
    # classifier for a design limitation; hiding them would be worse.
    blind = {(r["invoice_number"], r["pro_number"]) for r in gt if r["blind_spot"]}
    truth = {(r["invoice_number"], r["pro_number"]): r["category"]
             for r in gt if not r["blind_spot"]}
    rows = db.rows(con, "SELECT payload FROM audit_results WHERE run_id = ?", (run_id,))
    preds = [json.loads(r["payload"]) for r in rows]
    n_blind = sum(1 for p in preds if (p["invoice_number"], p["pro_number"]) in blind)

    labels = sorted({*truth.values(), *(p["classification"]["category"] for p in preds)})
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    confusion: dict[str, int] = defaultdict(int)
    correct = 0

    # A duplicate PRO appears twice in the book; the ground-truth key is unique per
    # (invoice, PRO), so scoring walks predictions and looks the truth up.
    for p in preds:
        key = (p["invoice_number"], p["pro_number"])
        gold = truth.get(key)
        if gold is None:
            continue
        got = p["classification"]["category"]
        if got == gold:
            tp[gold] += 1
            correct += 1
        else:
            fp[got] += 1
            fn[gold] += 1
            confusion[f"{gold} -> {got}"] += 1

    per_label = {}
    for lab in labels:
        p_, r_, f_ = prf(tp[lab], fp[lab], fn[lab])
        support = sum(1 for k, v in truth.items() if v == lab)
        per_label[lab] = {"precision": p_, "recall": r_, "f1": f_, "support": support,
                          "tp": tp[lab], "fp": fp[lab], "fn": fn[lab]}

    scored = [l for l in labels if per_label[l]["support"] > 0]
    macro_f1 = round(sum(per_label[l]["f1"] for l in scored) / len(scored), 4) if scored else 0.0

    # Accuracy is over SCORED lines, not over every prediction. An uploaded invoice
    # has no injected ground truth, so including it in the denominator silently
    # deflates the score -- which is exactly what happened when the holdout invoice
    # was ingested through the upload gates and joined the next run: 66 predictions,
    # 60 of them scoreable, and accuracy reported 87.9% when it was 96.7%.
    n_scored = sum(1 for p in preds if (p["invoice_number"], p["pro_number"]) in truth)
    return {
        "n_scored": n_scored,
        "n_unscored": len(preds) - n_scored - n_blind,
        "n_blind_spot": n_blind,
        "accuracy": round(correct / n_scored, 4) if n_scored else 0.0,
        "macro_f1": macro_f1,
        "per_category": per_label,
        "confusion": dict(sorted(confusion.items(), key=lambda kv: -kv[1])),
    }


# --------------------------------------------------------------------------
# Business outcome
# --------------------------------------------------------------------------


def score_business(con, run_id: str) -> dict:
    """Dollars, not lines -- plus false disputes, which are a different currency.

    **Dollar recall is dollar-weighted on purpose.** Line counts and money do not
    correlate: a $1,900 duplicate and a $4 accessorial are one line each. Freight
    audit exists to recover money, so the metric is denominated in money. A system
    catching 90% of lines while missing the one large one has failed at the job.
    Missed dollars have no recovery path -- the invoice was paid -- which is why
    this is the metric to optimise if forced to choose one.

    It says nothing about effort, though: escalating every line scores 100% and
    saves nobody any work. **Read it against the escalation rate or not at all.**

    **False disputes are tracked separately rather than folded into precision**,
    because they cost carrier relationship rather than money or minutes, and that
    does not convert to dollars. Averaging it into a precision figure would let a
    good dollar number pay for a damaged relationship. Target is zero and it is a
    release gate, not a score to improve -- which is why the policy layer
    auto-disputes exactly one category, the one whose evidence is documentary
    rather than interpretive.
    """
    truth = {
        (r["invoice_number"], r["pro_number"]): r["category"]
        for r in db.rows(con, "SELECT * FROM ground_truth")
    }
    preds = [json.loads(r["payload"])
             for r in db.rows(con, "SELECT payload FROM audit_results WHERE run_id = ?", (run_id,))]

    clean = {"NO_VARIANCE", "WITHIN_TOLERANCE"}
    caught = missed = 0.0
    false_disputes = 0
    caught_n = missed_n = 0

    for p in preds:
        gold = truth.get((p["invoice_number"], p["pro_number"]))
        if gold is None:
            continue
        var = p["recomputation"]["variance"]
        flagged = p["disposition"] in (Disposition.AUTO_DISPUTED.value, Disposition.NEEDS_REVIEW.value)

        if gold not in clean and var > 0:
            if flagged:
                caught += var
                caught_n += 1
            else:
                missed += var
                missed_n += 1
        # A false dispute is a clean line the system sent to the carrier on its own.
        if gold in clean and p["disposition"] == Disposition.AUTO_DISPUTED.value:
            false_disputes += 1

    total = caught + missed
    return {
        "recoverable_identified_usd": round(caught, 2),
        "recoverable_missed_usd": round(missed, 2),
        "dollar_recall": round(caught / total, 4) if total else 0.0,
        "defect_lines_caught": caught_n,
        "defect_lines_missed": missed_n,
        # Tracked separately: this one costs goodwill, not money.
        "false_disputes": false_disputes,
        "false_dispute_rate": round(false_disputes / len(preds), 4) if preds else 0.0,
    }


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def score_retrieval(con, strategies: list[str], k: int = 4) -> dict:
    """Recall@k of the governing clause, per strategy, with MRR.

    The query is built the same way the pipeline builds it -- from the computed
    evidence -- so this measures the retrieval the system actually performs rather
    than a hand-written question that flatters it.

    **Recall rather than precision**, because both consumers (a model that reads all
    k results, and a human shown them) tolerate an extra irrelevant clause and
    neither tolerates the governing rule being absent.

    **k=4 because that is what the pipeline passes to the classifier.** Measuring
    recall@10 while shipping k=4 measures a system nobody runs.

    **MRR alongside it because recall@4 saturates** -- every strategy scores 100%
    here, so it cannot rank them. MRR asks *where* the right clause landed, which is
    what separates BM25 (1.000, always first) from dense (0.917, sometimes second).
    Position matters: the model attends more to earlier context, and a reviewer
    reads top-down.

    **What this measurement bought: it killed a feature.** The LLM rerank scored
    identically to BM25 alone on both numbers while costing a call per query, so it
    is off by default.

    **What it misses:** 14 chunks from one synthetic tariff. 100% here means
    "retrieval is not the bottleneck", never "retrieval is solved".
    """
    from app.schema import Recomputation

    cases = []
    for cat, rule in GOVERNING_RULE.items():
        rec = Recomputation(
            pro_number="EVAL", matched_shipment_id="EVAL", match_method="PRO_EXACT",
            match_confidence=1.0, expected_charges=[], expected_total=100.0,
            billed_total=150.0, variance=50.0, variance_pct=50.0,
            weight_delta_lb=180.0 if cat == "REWEIGH" else None,
            class_delta=15.0 if cat == "RECLASS" else None,
            unsupported_accessorials=["LIFTGATE_DELIVERY"] if cat == "UNAUTHORIZED_ACCESSORIAL" else [],
            fuel_pct_billed=0.2540 if cat == "FUEL_SURCHARGE_ERROR" else None,
            fuel_pct_expected=0.2375 if cat == "FUEL_SURCHARGE_ERROR" else None,
            is_duplicate_pro=(cat == "DUPLICATE_BILLING"),
        )
        cases.append((cat, rule, classify_mod.build_query(rec)))

    out: dict[str, dict] = {}
    for strat in strategies:
        if strat == "hybrid_rerank" and not llm.available():
            out[strat] = {"skipped": "needs ANTHROPIC_API_KEY"}
            continue
        hits_at = 0
        detail = {}
        for cat, rule, q in cases:
            hits = retrieve.search(con, q, k=k, strategy=strat)
            found = any(h.rule_id == rule for h in hits)
            rank = next((i + 1 for i, h in enumerate(hits) if h.rule_id == rule), None)
            hits_at += int(found)
            detail[cat] = {"target": rule, "found": found, "rank": rank,
                           "returned": [h.id for h in hits]}
        out[strat] = {
            f"recall_at_{k}": round(hits_at / len(cases), 4),
            "mrr": round(sum(1 / d["rank"] for d in detail.values() if d["rank"]) / len(cases), 4),
            "per_case": detail,
        }
    return {"embedder": retrieve.embedder().name, "k": k, "strategies": out}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def render(report: dict) -> str:
    L: list[str] = []
    add = L.append
    run = report["run"]
    cls = report["classification"]
    biz = report["business"]

    add("=" * 78)
    add("  CARRIER INVOICE AUDIT -- EVAL REPORT")
    add("=" * 78)
    add(f"  run           {run['run_id']}")
    add(f"  extraction    {run['mode']}")
    if run.get("extraction_methods"):
        free = sum(v for k, v in run["extraction_methods"].items() if k in ("text-layer", "stub"))
        tot = sum(run["extraction_methods"].values())
        add(f"                {free}/{tot} invoices read WITHOUT a model call")
    add(f"  retrieval     {run['retrieval_strategy']}  (embedder: {report['retrieval']['embedder']})")
    for stage, mid in run["usage"].get("models", {}).items():
        add(f"  model.{stage:<8}{mid}")
    if run["mode"].startswith("stub"):
        add("  NOTE          stub extraction is 100% accurate by construction. These")
        add("                numbers measure the classifier and policy layer ONLY.")
    add("")
    add("-- OPERATIONS " + "-" * 63)
    add(f"  lines audited            {run['lines_audited']}")
    add(f"  invoices blocked (guard) {run['invoices_blocked_by_sum_check']}")
    add(f"  automation rate          {run['automation_rate'] * 100:.1f}%")
    add(f"  escalation rate          {run['escalation_rate'] * 100:.1f}%")
    add(f"  dispositions             {run['by_disposition']}")
    add("")
    add("-- CLASSIFICATION " + "-" * 59)
    add(f"  scored                   {cls['n_scored']}"
        + (f"   ({cls['n_unscored']} unlabelled)" if cls.get("n_unscored") else ""))
    if cls.get("n_blind_spot"):
        add(f"  known blind spots        {cls['n_blind_spot']}   EXCLUDED from the score --- "
            f"lines this architecture cannot catch")
        add(f"                           (correctly priced but contractually impermissible "
            f"charges have $0 variance and never escalate)")
    add(f"  accuracy                 {cls['accuracy'] * 100:.1f}%")
    add(f"  macro F1                 {cls['macro_f1']:.3f}")
    add("")
    add(f"  {'category':<28}{'prec':>7}{'rec':>7}{'F1':>7}{'n':>6}")
    for lab, m in sorted(cls["per_category"].items(), key=lambda kv: -kv[1]["support"]):
        if m["support"] == 0 and m["fp"] == 0:
            continue
        add(f"  {lab:<28}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}{m['support']:>6}")
    if cls["confusion"]:
        add("")
        add("  confusions (gold -> predicted):")
        for pair, n in list(cls["confusion"].items())[:8]:
            add(f"    {n:>3}  {pair}")
    add("")
    add("-- BUSINESS OUTCOME " + "-" * 57)
    add(f"  recoverable identified   ${biz['recoverable_identified_usd']:>12,.2f}   "
        f"({biz['defect_lines_caught']} lines)")
    add(f"  recoverable missed       ${biz['recoverable_missed_usd']:>12,.2f}   "
        f"({biz['defect_lines_missed']} lines)")
    add(f"  dollar recall            {biz['dollar_recall'] * 100:>12.1f}%")
    add(f"  false disputes           {biz['false_disputes']:>12}   "
        f"({biz['false_dispute_rate'] * 100:.1f}% of lines)")
    add("")
    add("-- RETRIEVAL (recall@k of the governing tariff clause) " + "-" * 23)
    for strat, m in report["retrieval"]["strategies"].items():
        if "skipped" in m:
            add(f"  {strat:<18} skipped: {m['skipped']}")
            continue
        k = report["retrieval"]["k"]
        add(f"  {strat:<18} recall@{k} {m[f'recall_at_{k}'] * 100:>5.1f}%    MRR {m['mrr']:.3f}")
    add("")
    add("-- COST " + "-" * 69)
    u = run["usage"]
    add(f"  model calls              {u['calls']}   {u['calls_by_stage']}")
    for stage, c in u.get("cost_by_stage", {}).items():
        add(f"    {stage:<22} {c['model']:<20} {c['calls']:>3} calls  ${c['cost_usd']:.5f}")
    add(f"  tokens in / out          {u['input_tokens']:,} / {u['output_tokens']:,}")
    add(f"  prompt cache read        {u['cache_read_tokens']:,}   "
        f"(cache {'HIT' if u['cache_hit'] else 'MISS'})")
    add(f"  run cost                 ${u['cost_usd']:.4f}"
        + (f"   of a ${run['budget_usd']:.2f} ceiling" if run.get("budget_usd") else ""))
    if run.get("stopped_on_budget"):
        add(f"  *** RUN STOPPED ON BUDGET -- {run['lines_audited']} lines audited of "
            f"{run.get('invoices_total', '?')} invoices. Scores below are PARTIAL. ***")
    add(f"  cost per line            ${run['cost_per_line_usd']:.5f}")
    add(f"  wall time                {run['wall_seconds']}s")
    add("=" * 78)
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the carrier-audit eval.")
    ap.add_argument("--stub", action="store_true",
                    help="Skip vision extraction; measures the classifier only.")
    ap.add_argument("--strategy", default="hybrid",
                    choices=["bm25", "dense", "hybrid", "hybrid_rerank"])
    ap.add_argument("--limit", type=int, default=None, help="Audit only the first N invoices.")
    ap.add_argument("--json", type=Path, default=None, help="Also write the report as JSON.")
    ap.add_argument("--model", help="Set every model stage at once, e.g. claude-sonnet-5.")
    ap.add_argument("--extract-model", help="Override the vision extraction stage only.")
    ap.add_argument("--classify-model", help="Override the classifier stage only.")
    ap.add_argument("--rerank-model", help="Override the reranker stage only.")
    ap.add_argument("--max-spend", type=float, default=None,
                    help="Hard USD ceiling for this run. The run stops when it is hit.")
    ap.add_argument("--reuse-extraction", metavar="RUN_ID", default=None,
                    help="Replay a prior run's extraction ('last' for the most recent). "
                         "Costs only the classifier -- use it to iterate on classification.")
    args = ap.parse_args()

    import os
    for flag, env in (("model", "CARRIER_AUDIT_MODEL"),
                      ("extract_model", "CARRIER_AUDIT_MODEL_EXTRACT"),
                      ("classify_model", "CARRIER_AUDIT_MODEL_CLASSIFY"),
                      ("rerank_model", "CARRIER_AUDIT_MODEL_RERANK")):
        if getattr(args, flag):
            os.environ[env] = getattr(args, flag)
    try:
        from app import models as _m
        _m.describe()
    except _m.UnknownModel as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    con = db.connect()
    if not db.rows(con, "SELECT 1 FROM invoices LIMIT 1"):
        print("No data. Run:  python -m app.cli seed", file=sys.stderr)
        return 1

    print(f"Running pipeline ({'stub' if args.stub else 'vision'} extraction, "
          f"{args.strategy} retrieval)...")
    run = pipeline.run(con, stub=args.stub, strategy=args.strategy, limit=args.limit,
                       max_spend_usd=args.max_spend,
                       reuse_run=(db.one(con, "SELECT run_id FROM audit_results ORDER BY id "
                                              "DESC LIMIT 1")["run_id"]
                                  if args.reuse_extraction == "last" else args.reuse_extraction))

    report = {
        "run": run,
        "classification": score_classification(con, run["run_id"]),
        "business": score_business(con, run["run_id"]),
        "retrieval": score_retrieval(con, ["bm25", "dense", "hybrid", "hybrid_rerank"]),
    }
    text = render(report)
    print("\n" + text)

    out = args.json or (Path(__file__).resolve().parent.parent / "data" / "eval_report.json")
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nJSON report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
