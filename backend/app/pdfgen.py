"""Render a synthetic carrier invoice to PDF.

The layout matters more than it looks like it should. Real LTL invoices are dense,
multi-line-per-shipment documents where the charge breakdown sits under the
shipment header rather than in tidy columns, and the totals are visually separated
from the detail. A vision model that reads a clean two-column table proves nothing
about a document that looks like this one.

Deliberately included, because each one is a real extraction hazard:
  - charge detail indented under a shipment header, not in its own table
  - weight and class printed as free text inside a header line
  - a page break mid-invoice with a repeated column header
  - `$` and thousands separators on every amount
  - a remittance block after the totals containing numbers that are NOT charges
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas

W, H = LETTER
LEFT = 0.6 * inch
RIGHT = W - 0.6 * inch


def _money(x: float) -> str:
    return f"${x:,.2f}"


def _header(c: canvas.Canvas, inv: dict, page: int) -> float:
    y = H - 0.7 * inch
    c.setFont("Helvetica-Bold", 15)
    c.drawString(LEFT, y, inv["carrier_name"].upper())
    c.setFont("Helvetica", 7.5)
    c.drawString(LEFT, y - 12, "Mississauga ON  ·  billing@meridianfreight.example")

    c.setFont("Helvetica-Bold", 12)
    c.drawRightString(RIGHT, y, "FREIGHT INVOICE")
    c.setFont("Helvetica", 8.5)
    c.drawRightString(RIGHT, y - 14, f"Invoice  {inv['invoice_number']}")
    c.drawRightString(RIGHT, y - 25, f"Invoice Date  {inv['invoice_date']}")
    c.drawRightString(RIGHT, y - 36, f"Page {page}")

    y -= 52
    c.setFont("Helvetica", 8.5)
    c.drawString(LEFT, y, "BILL TO:  Northwind Logistics Inc.  ·  Account MFS-NWL-2026-A")
    c.drawString(LEFT, y - 11, "Terms: Net 30  ·  Contract rates apply per agreement dated 2026-01-01")

    y -= 26
    c.setStrokeColor(colors.black)
    c.setLineWidth(0.8)
    c.line(LEFT, y, RIGHT, y)
    y -= 12
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(LEFT, y, "PRO NUMBER / SHIPMENT DETAIL")
    c.drawRightString(RIGHT, y, "AMOUNT")
    y -= 4
    c.setLineWidth(0.4)
    c.line(LEFT, y, RIGHT, y)
    return y - 14


def render_invoice_pdf(inv: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.setTitle(f"{inv['carrier_name']} invoice {inv['invoice_number']}")
    page = 1
    y = _header(c, inv, page)

    for line in inv["lines"]:
        needed = 26 + 11 * len(line["charges"])
        if y - needed < 1.3 * inch:
            c.setFont("Helvetica-Oblique", 7.5)
            c.drawRightString(RIGHT, 0.9 * inch, "continued ...")
            c.showPage()
            page += 1
            y = _header(c, inv, page)

        # Shipment header. Weight and class live inside this free-text run, which
        # is where a naive text-parse pipeline loses them.
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(LEFT, y, line["pro_number"])
        c.setFont("Helvetica", 7.5)
        c.drawString(
            LEFT + 1.15 * inch, y,
            f"BOL {line['bol_number']}   Shipped {line['ship_date']}   "
            f"{line['origin_zip']} to {line['dest_zip']}   "
            f"Billed Wt {line['billed_weight_lb']:,.0f} lb   Class {line['billed_class']:g}",
        )
        y -= 12

        for ch in line["charges"]:
            c.setFont("Helvetica", 8)
            c.drawString(LEFT + 0.3 * inch, y, ch["description"])
            c.drawRightString(RIGHT, y, _money(ch["amount"]))
            y -= 10.5

        c.setFont("Helvetica-Bold", 8)
        c.drawString(LEFT + 0.3 * inch, y, "Shipment total")
        c.drawRightString(RIGHT, y, _money(line["line_total"]))
        y -= 6
        c.setStrokeColor(colors.HexColor("#bbbbbb"))
        c.setLineWidth(0.3)
        c.line(LEFT + 0.3 * inch, y, RIGHT, y)
        c.setStrokeColor(colors.black)
        y -= 14

    if y < 1.9 * inch:
        c.showPage()
        page += 1
        y = _header(c, inv, page)

    y -= 6
    c.setLineWidth(0.8)
    c.line(LEFT + 3.6 * inch, y, RIGHT, y)
    y -= 15
    c.setFont("Helvetica-Bold", 10.5)
    c.drawString(LEFT + 3.6 * inch, y, "TOTAL DUE")
    c.drawRightString(RIGHT, y, _money(inv["invoice_total"]))

    # Remittance block. Contains numbers that are deliberately not charges.
    y -= 34
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(LEFT, y, "REMITTANCE")
    c.setFont("Helvetica", 7)
    c.drawString(LEFT, y - 10, "EFT: Transit 55501  Institution 999  Account 1029384756")
    c.drawString(LEFT, y - 19, "Reference the invoice number above. Disputes: 180 days per tariff MFS-400.")

    c.showPage()
    c.save()
    return path


def render_decoy_pdf(path: Path) -> Path:
    """A well-formed PDF that is emphatically not a carrier invoice.

    Exists so the ingestion gates can be demonstrated refusing something, not only
    accepting something. A gate nobody has watched reject a document is a gate
    nobody should trust.

    Deliberately adversarial in a small way: it carries an invoice-shaped header, a
    table of amounts and a total, so a shallow "does it have numbers and the word
    invoice" heuristic would wave it through. Only reading the document tells you
    it is a consulting bill for design work with no freight on it anywhere.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.setTitle("Statement of account")
    y = H - 0.9 * inch

    c.setFont("Helvetica-Bold", 16)
    c.drawString(LEFT, y, "NORTHSIDE DESIGN STUDIO")
    c.setFont("Helvetica", 8.5)
    c.drawString(LEFT, y - 14, "Statement of account  ·  Ref NDS-2026-0841  ·  2026-08-31")
    y -= 44

    c.setFont("Helvetica", 9)
    for line in ["Brand identity workshop, two sessions", "Logotype revisions, round three",
                 "Print-ready artwork preparation", "Photography licensing, 12 months"]:
        c.drawString(LEFT, y, line)
        c.drawRightString(RIGHT, y, "$1,250.00")
        y -= 14

    y -= 8
    c.setLineWidth(0.8)
    c.line(LEFT + 3.5 * inch, y, RIGHT, y)
    y -= 16
    c.setFont("Helvetica-Bold", 11)
    c.drawString(LEFT + 3.5 * inch, y, "TOTAL DUE")
    c.drawRightString(RIGHT, y, "$5,000.00")

    y -= 40
    c.setFont("Helvetica", 8)
    c.drawString(LEFT, y, "Payable within 30 days. Thank you for your business.")
    c.showPage()
    c.save()
    return path
