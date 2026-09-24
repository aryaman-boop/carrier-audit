"""The output contract.

Every value an LLM is allowed to produce in this system is defined here, and every
one of them is either a closed vocabulary or a value that gets re-validated against
a deterministic source before it is used.

The design rule this file encodes: *the model reads the invoice, it does not price
it*. Extraction produces transcribed values only. Classification produces one label
from a fixed set. Neither one is ever the source of a dollar figure that reaches a
decision -- those come from `recompute.py`, which is arithmetic over the rate card.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# --------------------------------------------------------------------------
# Closed vocabularies
# --------------------------------------------------------------------------


class VarianceCategory(str, Enum):
    """Why an invoiced amount differs from what the rate card says it should be.

    Closed set on purpose. A category outside this list is unrepresentable rather
    than caught downstream -- the classifier is given these values and nothing
    else, so it cannot invent `PARTIAL_RECLASS_MAYBE` at 2am on a Tuesday.

    UNKNOWN is not a failure mode, it is a supported answer. A system that can
    say "I don't know" is what makes the other labels trustworthy.
    """

    NO_VARIANCE = "NO_VARIANCE"
    WITHIN_TOLERANCE = "WITHIN_TOLERANCE"
    REWEIGH = "REWEIGH"
    RECLASS = "RECLASS"
    UNAUTHORIZED_ACCESSORIAL = "UNAUTHORIZED_ACCESSORIAL"
    DUPLICATE_BILLING = "DUPLICATE_BILLING"
    FUEL_SURCHARGE_ERROR = "FUEL_SURCHARGE_ERROR"
    RATE_CARD_MISMATCH = "RATE_CARD_MISMATCH"
    UNKNOWN = "UNKNOWN"


class Disposition(str, Enum):
    """What the system decided to do. Set by policy.py, never by a model."""

    AUTO_APPROVED = "AUTO_APPROVED"
    AUTO_DISPUTED = "AUTO_DISPUTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    BLOCKED_EXTRACTION = "BLOCKED_EXTRACTION"


class AccessorialCode(str, Enum):
    """The accessorials this carrier contract recognises."""

    LIFTGATE_DELIVERY = "LIFTGATE_DELIVERY"
    RESIDENTIAL_DELIVERY = "RESIDENTIAL_DELIVERY"
    INSIDE_DELIVERY = "INSIDE_DELIVERY"
    LIMITED_ACCESS = "LIMITED_ACCESS"
    APPOINTMENT = "APPOINTMENT"
    DETENTION = "DETENTION"
    REWEIGH_FEE = "REWEIGH_FEE"
    RECLASSIFICATION_FEE = "RECLASSIFICATION_FEE"
    REDELIVERY = "REDELIVERY"
    LUMPER = "LUMPER"


# NMFC freight classes actually used by this contract.
FREIGHT_CLASSES = (50.0, 55.0, 60.0, 65.0, 70.0, 77.5, 85.0, 92.5, 100.0, 110.0, 125.0, 150.0)


# --------------------------------------------------------------------------
# Extraction -- what the vision model is allowed to return
# --------------------------------------------------------------------------


class ExtractedCharge(BaseModel):
    """One charge line transcribed from the invoice PDF.

    `code` is constrained to the accessorial vocabulary plus the two base
    categories, so a charge description the carrier invented maps to a known
    bucket or fails loudly.
    """

    code: Literal[
        "LINEHAUL",
        "FUEL_SURCHARGE",
        "LIFTGATE_DELIVERY",
        "RESIDENTIAL_DELIVERY",
        "INSIDE_DELIVERY",
        "LIMITED_ACCESS",
        "APPOINTMENT",
        "DETENTION",
        "REWEIGH_FEE",
        "RECLASSIFICATION_FEE",
        "REDELIVERY",
        "LUMPER",
        "OTHER",
    ]
    description: str = Field(description="The charge description exactly as printed.")
    amount: float = Field(description="The charge amount in dollars, as printed.")


class ExtractedInvoiceLine(BaseModel):
    """One shipment's worth of billing on the invoice."""

    pro_number: str = Field(description="Carrier PRO / tracking number, exactly as printed.")
    bol_number: str | None = None
    ship_date: date | None = None
    origin_zip: str | None = None
    dest_zip: str | None = None
    billed_weight_lb: float = Field(description="The weight the CARRIER billed, not the BOL weight.")
    billed_class: float = Field(description="The NMFC class the CARRIER billed.")
    charges: list[ExtractedCharge]
    line_total: float

    @field_validator("pro_number")
    @classmethod
    def _strip_pro(cls, v: str) -> str:
        return v.strip().upper().replace(" ", "").replace("-", "")


class ExtractedInvoice(BaseModel):
    """The complete transcription of one carrier invoice PDF.

    Note what is NOT here: no computed fields, no opinions, no "expected" amounts.
    This is a transcription contract. Everything derived happens in code.
    """

    carrier_name: str
    invoice_number: str
    invoice_date: date | None = None
    lines: list[ExtractedInvoiceLine]
    invoice_total: float


class ExtractionResult(BaseModel):
    """Extraction plus the arithmetic guard that decides whether to trust it.

    The guard is the cheapest quality control in the whole system and it needs no
    eval set to work: the line items must sum to the printed invoice total. If the
    model misreads a digit, the sum breaks, and the invoice goes to a human. That
    single rule catches most transcription error without any model self-assessment.
    """

    invoice: ExtractedInvoice | None
    sum_check_passed: bool
    sum_check_delta: float | None = None
    error: str | None = None
    # Which rung of the extraction ladder produced this: "text-layer" (free,
    # deterministic), "vision" (a model call), or "stub". Reported in the run
    # summary so the saving is measured rather than asserted.
    method: str = "unknown"


# --------------------------------------------------------------------------
# Recompute -- deterministic, produced by code only
# --------------------------------------------------------------------------


class ExpectedCharge(BaseModel):
    code: str
    basis: str = Field(description="Human-readable derivation, e.g. '2,150 lb @ M2M $18.40/cwt'.")
    amount: float


class Recomputation(BaseModel):
    """What the rate card says this shipment should have cost.

    Every number here comes from arithmetic over the contracted rate card and the
    internal shipment record. No model contributes to any field.
    """

    pro_number: str
    matched_shipment_id: str | None
    match_method: Literal["PRO_EXACT", "FUZZY_LANE_DATE_WEIGHT", "UNMATCHED"]
    match_confidence: float
    expected_charges: list[ExpectedCharge]
    expected_total: float
    billed_total: float
    variance: float
    variance_pct: float
    # Machine-checkable facts the classifier is given as evidence.
    weight_delta_lb: float | None = None
    class_delta: float | None = None
    # MFS-110 verdict, computed by the rating engine rather than left for the
    # classifier to derive from the retrieved clause. `reweigh_governs` is True when
    # the delta exceeds the tolerance band and the carrier's weight would govern.
    reweigh_tolerance_lb: float | None = None
    reweigh_governs: bool | None = None
    unsupported_accessorials: list[str] = Field(default_factory=list)
    # Why each flagged accessorial is unsupported, keyed by code. The reason matters
    # as much as the flag: "the facility master contradicts it" and "the carrier owes
    # documentation it did not supply" are different dispute arguments.
    unsupported_reasons: dict[str, str] = Field(default_factory=dict)
    fuel_pct_billed: float | None = None
    fuel_pct_expected: float | None = None
    is_duplicate_pro: bool = False


# --------------------------------------------------------------------------
# Classification -- the one genuinely judgment-light LLM step
# --------------------------------------------------------------------------


class VarianceClassification(BaseModel):
    """The model's label for a variance, plus its evidence.

    `category` is a closed enum. `cited_clause_ids` must reference chunks that were
    actually retrieved -- validated in classify.py, so the model cannot cite a
    tariff rule that does not exist.
    """

    category: VarianceCategory
    reasoning: str = Field(description="Two sentences maximum, referencing the numbers.")
    cited_clause_ids: list[str] = Field(default_factory=list)


class AuditedLine(BaseModel):
    """The complete audited record for one invoice line -- what the reviewer sees."""

    pro_number: str
    invoice_number: str
    carrier_name: str
    extracted: ExtractedInvoiceLine
    recomputation: Recomputation
    classification: VarianceClassification
    disposition: Disposition
    policy_reason: str
    cited_clauses: list[dict] = Field(default_factory=list)
    # "code" when a deterministic short-circuit resolved the line, "model" when the
    # classifier ran. Shown in the UI, because labelling every line as the model's
    # work when half never reach one is a caption that is false half the time.
    decided_by: str = "model"
    # Populated only in eval runs.
    ground_truth: VarianceCategory | None = None
