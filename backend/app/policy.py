"""Stage 4 -- decide what to do about it.

Pure code, deliberately, and separate from the classifier on purpose. What a
variance *is* and what we *do* about it are different questions with different
owners: the first is a technical judgement, the second is a business policy that
Finance should be able to change without anyone touching a prompt.

The asymmetry that sets the thresholds is worth stating plainly, because it is the
whole reason the numbers are not symmetric:

    A false escalation costs a reviewer about a minute.
    A false auto-approve costs the disputed amount, permanently.
    A false auto-dispute costs carrier goodwill, which is not recoverable
    with a credit memo.

So the bands are not tuned for accuracy. They are tuned so that the expensive
mistakes require a human and the cheap ones do not. Auto-dispute is reserved for
the single category where the evidence is documentary rather than interpretive --
a duplicate PRO is a fact, not an opinion.

**Autonomy is earned, not assumed.** The intended rollout is shadow mode (run
alongside the humans, ship nothing), then suggest-only, then auto-approve inside a
narrow band, then widen it as the eval numbers hold. `AUTO_APPROVE_CEILING` is
deliberately low for that reason: it is a starting position, not a final one.
"""

from __future__ import annotations

from .schema import Disposition, Recomputation, VarianceCategory, VarianceClassification

AUTO_APPROVE_CEILING = 25.00   # dollars of variance we will absorb without a human
AUTO_DISPUTE_FLOOR = 5.00      # not worth a carrier conversation below this
FUZZY_MATCH_FLOOR = 0.85       # a weaker match never auto-decides anything

# The only category whose evidence is documentary rather than interpretive.
AUTO_DISPUTABLE = {VarianceCategory.DUPLICATE_BILLING}


def decide(rec: Recomputation, cls: VarianceClassification) -> tuple[Disposition, str]:
    """Return the disposition and the reason, in the reviewer's language."""

    if rec.match_method == "UNMATCHED":
        return Disposition.NEEDS_REVIEW, (
            "No shipment matched this PRO. A charge we cannot tie to a shipment is never "
            "auto-approved."
        )

    if rec.match_method == "FUZZY_LANE_DATE_WEIGHT" and rec.match_confidence < FUZZY_MATCH_FLOOR:
        return Disposition.NEEDS_REVIEW, (
            f"Matched on lane, date and weight at {rec.match_confidence:.2f} confidence, below "
            f"the {FUZZY_MATCH_FLOOR:.2f} floor for an automated decision."
        )

    if cls.category is VarianceCategory.UNKNOWN:
        return Disposition.NEEDS_REVIEW, (
            "The classifier could not attribute the variance to a known billing behaviour. "
            "An UNKNOWN is a supported answer and routes here by design."
        )

    if cls.category in (VarianceCategory.NO_VARIANCE, VarianceCategory.WITHIN_TOLERANCE):
        return Disposition.AUTO_APPROVED, (
            f"Variance of ${rec.variance:,.2f} is inside the rounding band and the shipment "
            f"reconciles to the rate card."
        )

    if cls.category in AUTO_DISPUTABLE and rec.variance >= AUTO_DISPUTE_FLOOR:
        return Disposition.AUTO_DISPUTED, (
            f"Duplicate billing is documentary evidence, not a judgement call: "
            f"${rec.variance:,.2f} on a PRO already invoiced. Disputed automatically under "
            f"MFS-300."
        )

    if rec.variance < 0:
        return Disposition.AUTO_APPROVED, (
            f"Carrier billed ${abs(rec.variance):,.2f} UNDER the contracted rate. Nothing to "
            f"recover and nothing to dispute."
        )

    if rec.variance <= AUTO_APPROVE_CEILING:
        return Disposition.AUTO_APPROVED, (
            f"${rec.variance:,.2f} overbilled, below the ${AUTO_APPROVE_CEILING:,.2f} ceiling. "
            f"Chasing it costs more in review time than it recovers."
        )

    return Disposition.NEEDS_REVIEW, (
        f"${rec.variance:,.2f} overbilled, attributed to {cls.category.value}. Above the "
        f"${AUTO_APPROVE_CEILING:,.2f} ceiling and the category needs a human to confirm "
        f"before it goes to the carrier."
    )
