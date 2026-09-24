"""Stage 1 -- read the invoice PDF.

This is the only place where a model is allowed to *transcribe* the invoice, and it
is allowed to do exactly that one thing. (The upload triage gate in `intake.py` also
shows a model the document, but only to answer "is this a freight invoice at all";
it reads nothing off the page.) It produces no expected amounts, no
opinions and no arithmetic. Everything it returns is re-validated before use.

**Vision-first, not OCR-then-parse.** The model reads the page as a document rather
than reading text a parser flattened out of it, which is why it can associate a
charge line with the shipment header above it. Published benchmarks put native
image reading around 93% on scanned invoices against about 64% for parsed-text
pipelines. The tradeoff is real and worth naming: vision costs more and runs
slower per document. For a workflow that processes invoices in a nightly batch,
that is the correct side of the tradeoff. For something interactive it would not be.

**The sum check is the cheapest quality control in the system.** Line items must
add to the printed invoice total. If the model drops a digit, transposes an
amount, or misses a charge line, the arithmetic breaks and the whole invoice is
routed to a human. It needs no eval set, no confidence score and no second model
call -- and it catches most transcription error on its own. A model's own
confidence in its answer is the weakest signal available; an arithmetic identity
the document must satisfy is one of the strongest.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from pathlib import Path

from . import db, llm, textlayer
from .schema import ExtractedInvoice, ExtractionResult

SYSTEM = """You transcribe freight invoices. You are a transcription instrument, not an auditor.

Rules:
- Transcribe only what is printed. Never compute, correct, infer or normalise an amount.
- billed_weight_lb and billed_class are the values THE CARRIER billed, printed in the
  shipment header line. They are frequently different from the bill of lading values.
  Transcribe what is on this document.
- Map each charge description to the closest code in the schema. Use OTHER only when no
  code fits; do not force a match.
- The remittance block after the totals contains bank and reference numbers. Those are
  not charges. Do not transcribe them as charges.
- Invoices may run to several pages with a repeated column header. Transcribe every
  shipment on every page exactly once.
- line_total is the "Shipment total" printed under each shipment's charges.
- invoice_total is the "TOTAL DUE" printed once at the end."""

SUM_TOLERANCE = 0.02  # cents-level rounding only


def extract_invoice(
    pdf_path: Path,
    usage: llm.Usage | None = None,
    allow_text_layer: bool = True,
) -> ExtractionResult:
    """Read one invoice, using the cheapest method that actually works.

    A ladder, not a single strategy:

      1. **Deterministic parse of the PDF text layer** (`textlayer.py`). Free and
         instant when the carrier sent a digital PDF in a layout we know.
      2. **The arithmetic guard**, applied to that parse. This is what makes rung 1
         safe to trust: a regex that mis-associates an amount fails with total
         confidence, so it is not believed until it reconciles to the printed
         totals.
      3. **Vision extraction.** Reached when there is no text layer (a scan), when
         the layout is one the parser does not know, or when the parse fails the
         arithmetic.

    Most demos would stop at rung 3 and call the whole thing "AI-powered". The
    interesting engineering is the two rungs underneath it, because they are where
    the cost goes and where the determinism lives.
    """
    if allow_text_layer:
        text = textlayer.extract_text(pdf_path)
        if textlayer.has_text_layer(text):
            parsed = textlayer.parse_invoice(text)
            if parsed is not None:
                checked = _guard(parsed)
                if checked.sum_check_passed:
                    checked.method = "text-layer"
                    return checked
                # Parsed, but the numbers do not reconcile. Do not trust it; the
                # model gets the document instead. This is the whole reason the
                # cheap rung is allowed to exist.

    return _extract_with_vision(pdf_path, usage)


def _extract_with_vision(pdf_path: Path, usage: llm.Usage | None = None) -> ExtractionResult:
    """Rung 3. The expensive, general path."""
    try:
        data = base64.standard_b64encode(pdf_path.read_bytes()).decode()
    except OSError as e:
        return ExtractionResult(invoice=None, sum_check_passed=False, error=f"unreadable pdf: {e}")

    try:
        invoice = llm.structured(
            ExtractedInvoice,
            system=SYSTEM,
            content=[
                {"type": "document",
                 "source": {"type": "base64", "media_type": "application/pdf", "data": data}},
                {"type": "text", "text": "Transcribe every shipment and every charge on this invoice."},
            ],
            usage=usage,
            stage="extract",
            max_tokens=16000,
            effort="high",
            timeout=600.0,
        )
    except Exception as e:
        return ExtractionResult(invoice=None, sum_check_passed=False, error=str(e))

    out = _guard(invoice)
    out.method = "vision"
    return out


def _guard(invoice: ExtractedInvoice) -> ExtractionResult:
    """The sum check. Line charges must add to line totals, and line totals to the
    invoice total. Either failure blocks the whole invoice."""
    for line in invoice.lines:
        charge_sum = round(sum(c.amount for c in line.charges), 2)
        if abs(charge_sum - line.line_total) > SUM_TOLERANCE:
            return ExtractionResult(
                invoice=invoice, sum_check_passed=False,
                sum_check_delta=round(charge_sum - line.line_total, 2),
                error=(f"line {line.pro_number}: charges sum to {charge_sum:.2f} but the "
                       f"printed shipment total is {line.line_total:.2f}"),
            )

    line_sum = round(sum(l.line_total for l in invoice.lines), 2)
    delta = round(line_sum - invoice.invoice_total, 2)
    if abs(delta) > SUM_TOLERANCE:
        return ExtractionResult(
            invoice=invoice, sum_check_passed=False, sum_check_delta=delta,
            error=(f"shipment totals sum to {line_sum:,.2f} but the printed invoice total "
                   f"is {invoice.invoice_total:,.2f}"),
        )
    return ExtractionResult(invoice=invoice, sum_check_passed=True, sum_check_delta=delta)


def extract_stub(con: sqlite3.Connection, invoice_number: str) -> ExtractionResult:
    """Perfect extraction, read from the generator's own output.

    For running the rest of the pipeline with no API key. This is NOT an eval mode:
    it makes extraction accuracy exactly 100% by construction, so any eval run
    using it is measuring the classifier and the policy layer only. The eval report
    labels runs that used it, so the number can never be quoted as if the model had
    read the page.
    """
    row = db.one(con, "SELECT seed_payload FROM invoices WHERE invoice_number = ?", (invoice_number,))
    if not row or not row["seed_payload"]:
        return ExtractionResult(invoice=None, sum_check_passed=False, error="no seed payload")

    # An UPLOADED invoice is stored with an empty payload -- it was never generated,
    # so there is no generator output to replay. That is not an error the caller
    # should crash on: a stub run simply cannot cover uploads, and it says so.
    try:
        payload = json.loads(row["seed_payload"])
        invoice = ExtractedInvoice.model_validate(payload)
    except Exception:
        return ExtractionResult(
            invoice=None, sum_check_passed=False,
            error=(f"{invoice_number} was uploaded rather than generated, so it has no stored "
                   f"transcription to replay. Stub mode covers seeded invoices only; re-run "
                   f"without --stub to read it."),
        )
    out = _guard(invoice)
    out.method = "stub"
    return out
