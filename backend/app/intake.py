"""Ingestion gates for uploaded PDFs.

An upload endpoint with no gate is how a resume ends up audited against a freight
tariff, and how a 400-page scan costs forty dollars of vision tokens because
nobody checked. This is the door, and it is deliberately built as a chain that
gets more expensive as it goes:

    1  FILE       magic bytes, size, page count, encryption      free
    2  DEDUPE     content hash, invoice number already seen      free
    3  TRIAGE     "is this even a carrier freight invoice?"      ~$0.002
    4  EXTRACT    full vision transcription                      ~$0.07
    5  SUM CHECK  line items must sum to the printed totals      free
    6  OWNERSHIP  our carrier? our shipments?                    free

**The ordering is the design.** Every gate that can reject a document for free
runs before any gate that costs money, and the cheap model call (stage 3) exists
specifically to stop the expensive one (stage 4) from ever seeing a document that
is not an invoice. Rejecting a CV costs a fifth of a cent instead of seven cents,
and rejecting a 400-page scan costs nothing at all.

**Stage 6 is the one people forget.** A document can be a perfectly valid freight
invoice, pass every technical check, and still not be ours -- a different shipper's
invoice, or our carrier billing a different customer. The test is whether the PRO
numbers resolve against our own shipment book. If none of them do, the document is
someone else's problem and auditing it would produce confident nonsense.

Every rejection is recorded with its stage and reason. A gate that silently drops
things is indistinguishable from a gate that is broken.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

from . import db, llm, textlayer
from .ratecard import CARRIER_NAME

# --- stage 1 limits -------------------------------------------------------
MAX_BYTES = 12 * 1024 * 1024      # 12 MB. Base64 inflates ~33%, and the API caps at 32 MB.
MAX_PAGES = 25                    # a carrier invoice batch; anything larger is not one
PDF_MAGIC = b"%PDF-"

# --- stage 6 thresholds ---------------------------------------------------
MIN_PRO_MATCH_RATE = 0.25         # at least a quarter of lines must be our shipments


class Rejected(Exception):
    """A document that will not be processed, with the stage that stopped it."""

    def __init__(self, stage: str, reason: str, detail: str | None = None):
        self.stage, self.reason, self.detail = stage, reason, detail
        super().__init__(f"[{stage}] {reason}")


@dataclass
class GateResult:
    """The audit trail of one upload, pass or fail."""

    filename: str
    sha256: str
    size_bytes: int
    passed: list[str] = field(default_factory=list)
    rejected_at: str | None = None
    reason: str | None = None
    detail: str | None = None
    cost_usd: float = 0.0

    def ok(self, stage: str) -> None:
        self.passed.append(stage)

    def as_dict(self) -> dict:
        return {
            "filename": self.filename,
            "sha256": self.sha256[:16],
            "size_bytes": self.size_bytes,
            "gates_passed": self.passed,
            "rejected_at": self.rejected_at,
            "reason": self.reason,
            "detail": self.detail,
            "cost_usd": round(self.cost_usd, 5),
            "accepted": self.rejected_at is None,
        }


# --------------------------------------------------------------------------
# Stage 1 -- the file itself
# --------------------------------------------------------------------------


def count_pages(raw: bytes) -> int:
    """Approximate page count without a PDF library.

    Counts `/Type /Page` objects, excluding `/Pages` tree nodes. It is an
    approximation and it is used only as an upper-bound guard, never as a fact
    reported anywhere -- a document that lies about its page count still has to
    pass every later gate.
    """
    return len(re.findall(rb"/Type\s*/Page[^s]", raw))


def gate_file(raw: bytes, filename: str) -> None:
    if not raw:
        raise Rejected("FILE", "The file is empty.")

    if not raw.lstrip()[:5].startswith(PDF_MAGIC):
        raise Rejected(
            "FILE", "This is not a PDF.",
            "The file does not begin with the %PDF- signature. A renamed .docx, "
            "image or executable will fail here regardless of its extension.",
        )

    if len(raw) > MAX_BYTES:
        raise Rejected(
            "FILE", f"Too large: {len(raw) / 1e6:.1f} MB (limit {MAX_BYTES / 1e6:.0f} MB).",
            "Carrier invoices are small. A file this size is a scan batch or the "
            "wrong document, and sending it to a vision model would be expensive.",
        )

    if b"/Encrypt" in raw:
        raise Rejected(
            "FILE", "The PDF is encrypted or password-protected.",
            "Nothing downstream can read it. Ask the sender for an unprotected copy.",
        )

    pages = count_pages(raw)
    if pages > MAX_PAGES:
        raise Rejected(
            "FILE", f"Too many pages: about {pages} (limit {MAX_PAGES}).",
            "Vision extraction is billed per page. A document this long is not a "
            "single carrier invoice.",
        )


# --------------------------------------------------------------------------
# Stage 2 -- have we seen it
# --------------------------------------------------------------------------


def gate_duplicate_file(con: sqlite3.Connection, sha: str) -> None:
    row = db.one(con, "SELECT filename, created_at FROM uploads WHERE sha256=? AND accepted=1", (sha,))
    if row:
        raise Rejected(
            "DEDUPE", "This exact file has already been processed.",
            f"Uploaded as {row['filename']} on {row['created_at']}. Byte-identical, "
            f"so re-auditing it would double-count any recovery.",
        )


def gate_duplicate_invoice(con: sqlite3.Connection, invoice_number: str) -> None:
    if db.one(con, "SELECT 1 FROM invoices WHERE invoice_number=?", (invoice_number,)):
        raise Rejected(
            "DEDUPE", f"Invoice {invoice_number} is already on file.",
            "A different file claiming an invoice number we already hold. That is "
            "either a re-send or a corrected invoice; under MFS-300 a corrected "
            "invoice must reference the original, so this needs a human.",
        )


# --------------------------------------------------------------------------
# Stage 3 -- the cheap model gate
# --------------------------------------------------------------------------


class Triage(BaseModel):
    """Deliberately tiny. This call exists to be cheap, not to be clever."""

    is_freight_invoice: bool = Field(
        description="True only if this document is an invoice from a freight carrier "
                    "billing for transportation services."
    )
    document_kind: str = Field(
        description="What the document actually is, in three words or fewer: "
                    "'carrier freight invoice', 'resume', 'purchase order', "
                    "'bank statement', 'unclear', etc."
    )
    carrier_name: str | None = Field(
        default=None, description="The billing carrier's name, exactly as printed, or null."
    )
    reason: str = Field(description="One sentence justifying the classification.")


TRIAGE_SYSTEM = """You are a document gate for an accounts-payable freight audit system.

You see one document. Decide whether it is an invoice from a freight carrier billing
for transportation services.

Say true ONLY for a carrier's freight invoice. Say false for everything else,
including documents that are adjacent but different: bills of lading, rate quotes,
purchase orders, packing slips, delivery receipts, statements of account, and any
document that is not about freight at all.

Be decisive and be brief. A wrong 'true' sends a document into an expensive pipeline
that will produce confident nonsense from it."""


def gate_triage_free(text: str) -> Triage | None:
    """Decide for nothing, using the PDF's own text layer.

    Returns a verdict when the signals are unambiguous in either direction, and
    None when they are not -- in which case the caller pays for the model.

    This is the rung below the cheap model call, and on a document with a text
    layer it usually removes that call entirely. The decoy statement of account is
    refused here for zero cost and zero latency; a real carrier invoice is admitted
    the same way.
    """
    v = textlayer.triage_text(text)
    if not v.confident:
        return None
    if not v.is_freight_invoice:
        raise Rejected(
            "TRIAGE", f"Not a carrier freight invoice. Best guess: {v.document_kind}.",
            f"{v.reason} Decided from the PDF text layer at zero cost; no model was called.",
        )
    return Triage(is_freight_invoice=True, document_kind=v.document_kind,
                  carrier_name=None, reason=v.reason + " (decided without a model call)")


def gate_triage(raw_b64: str, usage: llm.Usage) -> Triage:
    """One cheap call that decides whether the expensive call ever happens.

    Reached only when `gate_triage_free` could not decide: a scanned PDF with no
    text layer, or a document whose signals genuinely conflict.
    """
    out = llm.structured(
        Triage,
        system=TRIAGE_SYSTEM,
        content=[
            {"type": "document",
             "source": {"type": "base64", "media_type": "application/pdf", "data": raw_b64}},
            {"type": "text", "text": "What is this document?"},
        ],
        usage=usage,
        stage="triage",
        max_tokens=1500,
        effort="low",
        timeout=180.0,
    )
    if not out.is_freight_invoice:
        raise Rejected(
            "TRIAGE", f"Not a carrier freight invoice --- this looks like a {out.document_kind}.",
            out.reason,
        )
    return out


# --------------------------------------------------------------------------
# Stage 6 -- is it OURS
# --------------------------------------------------------------------------


def gate_ownership(con: sqlite3.Connection, invoice) -> dict:
    """A valid invoice is not necessarily our invoice.

    Two independent checks. The carrier name is a soft signal -- carriers rebrand,
    and a subsidiary may bill under a different name -- so a mismatch is reported
    but does not reject on its own. The PRO match rate is the hard one: if none of
    the shipments on this invoice are in our book, the document belongs to someone
    else and every number we computed about it would be fiction.
    """
    carrier_ok = CARRIER_NAME.lower().split()[0] in (invoice.carrier_name or "").lower()

    pros = [l.pro_number for l in invoice.lines]
    if not pros:
        raise Rejected("OWNERSHIP", "The invoice has no shipment lines.")

    placeholders = ",".join("?" * len(pros))
    found = db.rows(con, f"SELECT pro_number FROM shipments WHERE pro_number IN ({placeholders})",
                    tuple(pros))
    rate = len(found) / len(pros)

    if rate < MIN_PRO_MATCH_RATE:
        raise Rejected(
            "OWNERSHIP",
            f"Only {len(found)} of {len(pros)} shipments on this invoice are in our book "
            f"({rate:.0%}).",
            "This is a freight invoice, but it does not appear to be billed against our "
            "shipments. Auditing it would compare a stranger's freight to our rate card "
            "and produce confident nonsense.",
        )

    return {
        "carrier_name_matches_contract": carrier_ok,
        "pro_match_rate": round(rate, 3),
        "lines": len(pros),
        "matched": len(found),
    }


# --------------------------------------------------------------------------
# Persistence of the attempt
# --------------------------------------------------------------------------


def record(con: sqlite3.Connection, g: GateResult, invoice_number: str | None = None) -> None:
    con.execute(
        "INSERT INTO uploads (filename, sha256, size_bytes, accepted, rejected_at, reason, "
        "detail, invoice_number, cost_usd) VALUES (?,?,?,?,?,?,?,?,?)",
        (g.filename, g.sha256, g.size_bytes, int(g.rejected_at is None), g.rejected_at,
         g.reason, g.detail, invoice_number, g.cost_usd),
    )
    con.commit()


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save_pdf(raw: bytes, sha: str) -> Path:
    path = db.INVOICE_DIR / f"upload-{sha[:12]}.pdf"
    path.write_bytes(raw)
    return path
