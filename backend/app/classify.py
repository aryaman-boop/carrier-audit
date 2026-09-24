"""Stage 3 -- name the variance.

This is the one genuinely judgment-light step in the workflow, and it is the only
place a model makes a call that reaches a decision. Everything about how it is set
up is aimed at making it a narrow call rather than an open one.

**Code decides first, and often.** A duplicate PRO is a set membership test. A
variance of two cents is arithmetic. A variance inside the auto-approve band is a
threshold. None of those need a model and none of them get one -- roughly half the
lines in a run are resolved before any call is made, which is both cheaper and
more auditable. The model is reached only when code has found a real variance and
cannot say *why*.

**The query is constructed, not embedded.** The retrieval query is built from the
machine-checked evidence -- weight delta, class delta, which accessorials lack
support, whether the fuel percentage matches the pickup week. It is not the invoice
text and it is not the user's question, because there is no user. Building the
query from the facts is what makes the retrieved clause the one that actually
governs, rather than the one that shares the most vocabulary with a PDF.

**The model is given the evidence and the rules, and asked only to label.** It gets
a closed enum, the numbers code computed, and the tariff clauses retrieved for
those numbers. It does not get the ability to invent a category, and its citations
are validated against the ids that were actually retrieved -- a clause id it made
up is dropped rather than shown to a reviewer as a reason to dispute.
"""

from __future__ import annotations

import sqlite3

from . import llm, retrieve
from .schema import Recomputation, VarianceCategory, VarianceClassification

# Auto-resolve bands. Both must hold for a line to count as rounding noise.
TOLERANCE_ABS = 2.00
TOLERANCE_PCT = 1.0

SYSTEM = """You classify freight billing variances for an accounts-payable audit desk.

You are given: a variance that has already been computed by our rating engine, the
machine-checked facts behind it, and the tariff clauses that govern those facts.

Your only job is to choose the ONE category that best explains why the carrier billed
more or less than the contracted rate card says they should have.

Category guidance:
- REWEIGH: the carrier billed a different weight than our bill of lading.
- RECLASS: the carrier billed a different NMFC class than our bill of lading.
- UNAUTHORIZED_ACCESSORIAL: an accessorial was billed that the bill of lading did not
  request and the facility master contradicts.
- DUPLICATE_BILLING: this shipment was already invoiced.
- FUEL_SURCHARGE_ERROR: the fuel percentage applied does not match the percentage
  published for the pickup week.
- RATE_CARD_MISMATCH: weight, class, accessorials and fuel all reconcile, but the
  linehaul charge still does not match the contracted rate.
- NO_VARIANCE / WITHIN_TOLERANCE: the amounts agree.
- UNKNOWN: the evidence does not clearly support any single category above.

Rules:
- Choose UNKNOWN rather than guessing. An UNKNOWN goes to a human, which is cheap. A
  confident wrong category can produce a wrong dispute with a carrier, which is not.
- When more than one category could apply, choose the one carrying the larger dollar
  effect, and say in your reasoning that the other is also present.
- Cite only clause ids that appear in the RETRIEVED CLAUSES section.
- Reasoning is at most two sentences and must reference the actual numbers."""


# Vocabulary each charge code's governing clause would use. Consulted when the
# evidence does not explain the variance, so retrieval still reaches the right rule.
CODE_TERMS = {
    "LIFTGATE_DELIVERY": "liftgate service delivery power liftgate loading dock forklift "
                         "one liftgate charge per delivery regardless of piece count",
    "RESIDENTIAL_DELIVERY": "residential delivery private dwelling commercial facility "
                            "master limited access higher of the two waived",
    "LIMITED_ACCESS": "limited access delivery restricted site appointment escort "
                      "security school construction residential not both assessed",
    "INSIDE_DELIVERY": "inside delivery beyond the tailgate accessorial service",
    "APPOINTMENT": "appointment notification charge consignee once per shipment redelivery",
    "DETENTION": "detention with vehicle free time two hours arrival delivery receipt "
                 "half-hour increments shall not exceed eight hours in any single day",
    "REDELIVERY": "redelivery second attempt appointment window consignee attributable",
    "LUMPER": "lumper unloading service charge",
    "REWEIGH_FEE": "reweigh fee certified weight certificate valid upward adjustment",
    "RECLASSIFICATION_FEE": "reclassification fee inspection certificate density dimensions",
    "FUEL_SURCHARGE": "fuel surcharge percentage of linehaul charge does not apply to "
                      "accessorial service charges or detention pickup week",
    "LINEHAUL": "application of rates minimum charge weight break class factor",
    "OTHER": "charges not provided for in this tariff supporting documentation burden",
}


def build_query(rec: Recomputation, billed: list | None = None) -> str:
    """Turn the computed evidence into a retrieval query.

    Deterministic and evidence-driven. Each condition contributes the vocabulary a
    tariff clause about that condition would use, which is why BM25 does most of
    the work on this corpus.

    **And when the evidence explains nothing, fall back to the charges themselves.**
    That gap was a real defect: on an invoice where every machine-checked fact
    reconciled and a variance remained -- a liftgate billed twice, detention past
    the daily cap -- no condition fired, the query fell through to a generic
    "application of rates" string, and the governing clause was never retrieved.
    The model then diagnosed the arithmetic correctly and abstained to UNKNOWN,
    because it had been handed the wrong rules to reason from.

    The lesson is the same one the evidence bug taught, one layer up: **an
    unexplained variance is exactly when retrieval matters most, and it was exactly
    when the query carried the least information.** So a residual variance now pulls
    the vocabulary of every accessorial actually on the invoice.
    """
    parts: list[str] = []
    if rec.is_duplicate_pro:
        parts.append("duplicate billing same PRO number invoiced twice corrected invoice")
    if rec.weight_delta_lb:
        parts.append(
            "reweigh reweighed weight bill of lading weight governs tolerance "
            "certified weight certificate reweigh fee"
        )
    if rec.class_delta:
        parts.append(
            "freight classification reclassification NMFC class inspection certificate "
            "reclassification fee rerate"
        )
    for code in rec.unsupported_accessorials:
        parts.append({
            "LIFTGATE_DELIVERY": "liftgate service delivery loading dock forklift "
                                 "requested bill of lading delivery receipt",
            "RESIDENTIAL_DELIVERY": "residential delivery private dwelling commercial "
                                    "facility master limited access",
            "INSIDE_DELIVERY": "inside delivery beyond the tailgate accessorial service",
            "REDELIVERY": "redelivery second attempt appointment window consignee",
            "APPOINTMENT": "appointment notification charge consignee scheduling",
            "LIMITED_ACCESS": "limited access delivery restricted site school "
                              "construction military appointment",
            "LUMPER": "lumper unloading service charge",
            "DETENTION": "detention with vehicle free time driver arrival half-hour increments",
        }.get(code, code.replace("_", " ").lower()))
    if (rec.fuel_pct_billed is not None and rec.fuel_pct_expected is not None
            and abs(rec.fuel_pct_billed - rec.fuel_pct_expected) > 0.002):
        parts.append(
            "fuel surcharge percentage weekly published pickup week linehaul charge applies"
        )
    # Nothing in the evidence accounts for the variance. Retrieve on what was
    # actually billed instead of guessing at a generic rates clause.
    if not parts and billed:
        seen: set[str] = set()
        for ch in billed:
            code = getattr(ch, "code", None) or (ch.get("code") if isinstance(ch, dict) else None)
            if code and code not in seen and code in CODE_TERMS:
                seen.add(code)
                parts.append(CODE_TERMS[code])

    if not parts:
        parts.append(
            "application of rates minimum charge weight break class factor contracted rate table"
        )
    parts.append("billing dispute supporting documentation burden carrier")
    return " ".join(parts)


def _evidence(rec: Recomputation, billed: list | None = None) -> str:
    """Assemble what the classifier is allowed to see.

    The first version of this function showed the expected charges and the computed
    deltas but NOT the charges the carrier actually billed. That is an evidence gap
    rather than a cosmetic one: on a reweigh, the weight delta explains part of the
    variance and a separate REWEIGH_FEE line explains the rest, so the model was
    asked to account for $44.94 while being shown facts covering $6 of it. It
    abstained to UNKNOWN, which was the correct response to incomplete evidence and
    which the eval scored as a miss.

    The lesson generalises: when a classifier abstains, check whether the evidence
    actually supports the answer you wanted before touching the prompt.

    So the variance is now reconciled line by line, per charge code, and every
    dollar of it is attributed to something visible.
    """
    lines = [
        f"PRO: {rec.pro_number}",
        f"Match: {rec.match_method} (confidence {rec.match_confidence})",
        f"Billed total:   ${rec.billed_total:,.2f}",
        f"Expected total: ${rec.expected_total:,.2f}",
        f"Variance:       ${rec.variance:,.2f}  ({rec.variance_pct:+.2f}%)",
        "",
        "Rating engine derivation of the expected total:",
    ]
    lines += [f"  {e.code:<24} ${e.amount:>10,.2f}   {e.basis}" for e in rec.expected_charges]

    if billed is not None:
        # Side-by-side reconciliation per charge code, so every dollar of the
        # variance is attributable to a line the model can see.
        exp_by_code: dict[str, float] = {}
        for e in rec.expected_charges:
            exp_by_code[e.code] = exp_by_code.get(e.code, 0.0) + e.amount
        bill_by_code: dict[str, float] = {}
        for c in billed:
            bill_by_code[c.code] = bill_by_code.get(c.code, 0.0) + c.amount

        lines.append("")
        lines.append("Charge-by-charge reconciliation (billed vs expected):")
        lines.append(f"  {'CODE':<24}{'BILLED':>12}{'EXPECTED':>12}{'DIFF':>12}")
        for code in sorted(set(exp_by_code) | set(bill_by_code)):
            b = bill_by_code.get(code, 0.0)
            e = exp_by_code.get(code, 0.0)
            note = ""
            if e == 0 and b:
                note = "  <- billed but NOT in the contracted expectation"
            elif b == 0 and e:
                note = "  <- expected but not billed"
            lines.append(f"  {code:<24}{b:>12,.2f}{e:>12,.2f}{b - e:>12,.2f}{note}")

    lines.append("")
    lines.append("Machine-checked facts:")
    lines.append(f"  weight delta vs our BOL: {rec.weight_delta_lb:+,.1f} lb"
                 if rec.weight_delta_lb is not None else "  weight delta: n/a")
    if rec.reweigh_governs is not None and rec.weight_delta_lb:
        lines.append(
            f"  MFS-110 tolerance band: {rec.reweigh_tolerance_lb:,.1f} lb "
            f"(2% or 50 lb, whichever is greater)"
        )
        lines.append(
            "  MFS-110 verdict: the reweighed weight GOVERNS; charges adjust to it"
            if rec.reweigh_governs else
            "  MFS-110 verdict: the delta falls INSIDE the tolerance, so the BOL weight "
            "governs and no adjustment is due. The expected charges above are rated on "
            "the BOL weight accordingly."
        )
    lines.append(f"  class delta vs our BOL:  {rec.class_delta:+g}"
                 if rec.class_delta is not None else "  class delta: n/a")
    if rec.unsupported_accessorials:
        lines.append("  accessorials the bill of lading did not request and the tariff does "
                     "not establish:")
        for code in rec.unsupported_accessorials:
            why = rec.unsupported_reasons.get(code, "not requested on the bill of lading")
            lines.append(f"    {code}: {why}")
    else:
        lines.append("  accessorials with no support in the facility master or BOL: none")
    if rec.fuel_pct_billed is not None and rec.fuel_pct_expected is not None:
        lines.append(f"  fuel % billed: {rec.fuel_pct_billed * 100:.2f}%   "
                     f"fuel % published for the pickup week: {rec.fuel_pct_expected * 100:.2f}%")
    lines.append(f"  PRO already seen on an earlier invoice: {rec.is_duplicate_pro}")
    return "\n".join(lines)


def classify(
    con: sqlite3.Connection,
    rec: Recomputation,
    usage: llm.Usage | None = None,
    strategy: str = "hybrid",
    use_model: bool = True,
    billed_charges: list | None = None,
) -> tuple[VarianceClassification, list[dict], str]:
    """Return the classification, the cited clauses, and WHO decided.

    The third value is `"code"` or `"model"`. It is returned rather than carried on
    `VarianceClassification` because that class is the LLM's output schema --- the
    model must not be the thing that tells us whether the model ran.

    It exists because the review UI was labelling every line "decided by the model",
    including the roughly half that never reached one. A caption that is false half
    the time is worse than no caption.
    """

    # ---- deterministic short-circuits: no model call ----
    if rec.is_duplicate_pro:
        hits = retrieve.search(con, build_query(rec, billed_charges), k=2, strategy=strategy)
        return (
            VarianceClassification(
                category=VarianceCategory.DUPLICATE_BILLING,
                reasoning=(f"PRO {rec.pro_number} was already invoiced on an earlier invoice in "
                           f"this batch; the full ${rec.billed_total:,.2f} is billed twice."),
                cited_clause_ids=[h.id for h in hits],
            ),
            [h.__dict__ for h in hits],
            "code",
        )

    if rec.match_method == "UNMATCHED":
        return (
            VarianceClassification(
                category=VarianceCategory.UNKNOWN,
                reasoning=("No shipment in our book matches this PRO, and lane, date and weight "
                           "did not resolve a candidate above the confidence floor."),
            ),
            [],
            "code",
        )

    if abs(rec.variance) <= 0.02:
        return VarianceClassification(
            category=VarianceCategory.NO_VARIANCE,
            reasoning=f"Billed ${rec.billed_total:,.2f} against an expected "
                      f"${rec.expected_total:,.2f}; the amounts agree.",
        ), [], "code"

    if abs(rec.variance) < TOLERANCE_ABS and abs(rec.variance_pct) < TOLERANCE_PCT:
        return VarianceClassification(
            category=VarianceCategory.WITHIN_TOLERANCE,
            reasoning=f"Variance of ${rec.variance:,.2f} ({rec.variance_pct:+.2f}%) is inside "
                      f"the ${TOLERANCE_ABS:.2f} / {TOLERANCE_PCT:.1f}% rounding band.",
        ), [], "code"

    # ---- the model is reached only here ----
    hits = retrieve.search(con, build_query(rec, billed_charges), k=4, strategy=strategy)
    clauses = [h.__dict__ for h in hits]

    if not use_model or not llm.available():
        return VarianceClassification(
            category=VarianceCategory.UNKNOWN,
            reasoning="Classification stage skipped (no model configured); routed to a human.",
            cited_clause_ids=[h.id for h in hits],
        ), clauses, "code"

    retrieved = "\n\n".join(f"[{h.id}] {h.title}\n{h.text}" for h in hits)
    out = llm.structured(
        VarianceClassification,
        system=SYSTEM,
        content=f"EVIDENCE\n{_evidence(rec, billed_charges)}\n\nRETRIEVED CLAUSES\n{retrieved}",
        usage=usage,
        stage="classify",
        max_tokens=2000,
        effort="medium",
    )

    # A citation the model invented is not evidence. Drop it.
    valid = {h.id for h in hits}
    out.cited_clause_ids = [c for c in out.cited_clause_ids if c in valid]
    return out, clauses, "model"
