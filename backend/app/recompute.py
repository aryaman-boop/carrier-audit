"""Stage 2 -- match the invoice line to our shipment, then price it ourselves.

Two deterministic steps and no model in either.

**Matching.** PRO number first, because it is the carrier's own key and an exact
hit is unambiguous. When the PRO does not resolve -- a mis-keyed digit, a PRO
reissued after a re-delivery -- fall back to blocking on lane and ship-date window
and scoring candidates on weight proximity. Every fuzzy match carries a confidence
and anything under the floor is routed to a human rather than guessed at. This is
entity resolution, and it is the same discipline as reconciling records across
systems that were never designed to be joined.

**Recompute.** Price the shipment from the contracted rate card and our own bill of
lading, and compare. The comparison surfaces machine-checkable facts -- weight
delta, the MFS-110 tolerance verdict on that delta, class delta, accessorials the
bill of lading did not request and the tariff does not establish (each with the
reason it fails), fuel percentage applied against fuel percentage published -- which
are then handed to the classifier as evidence.

Note what "unsupported" does and does not mean. A liftgate billed to a dock-equipped
site is flatly invalid under MFS-200. A liftgate billed to a site with no dock is a
different argument: the condition is real, but MFS-200 allows an unrequested liftgate
only where the carrier documents that condition on the delivery receipt. Both are
disputable and they are not disputable for the same reason, so the reason travels
with the flag rather than being assumed.

That last point is the design that makes the classifier reliable. It is not asked
"is this invoice wrong?" It is asked to name which of a fixed set of known billing
behaviours produced a variance that code has already found and quantified.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta

from . import db, pricing
from .ratecard import ACCESSORIAL_PRICES, LANE_RATES_CWT
from .schema import ExpectedCharge, ExtractedInvoiceLine, Recomputation

FUZZY_FLOOR = 0.72
DATE_WINDOW_DAYS = 5


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def match_shipment(
    con: sqlite3.Connection, line: ExtractedInvoiceLine
) -> tuple[dict | None, str, float]:
    exact = db.one(con, "SELECT * FROM shipments WHERE pro_number = ?", (line.pro_number,))
    if exact:
        return exact, "PRO_EXACT", 1.0

    if not (line.ship_date and line.origin_zip and line.dest_zip):
        return None, "UNMATCHED", 0.0

    lo = (line.ship_date - timedelta(days=DATE_WINDOW_DAYS)).isoformat()
    hi = (line.ship_date + timedelta(days=DATE_WINDOW_DAYS)).isoformat()
    cands = db.rows(
        con,
        "SELECT * FROM shipments WHERE origin_zone=? AND dest_zone=? AND ship_date BETWEEN ? AND ?",
        (line.origin_zip, line.dest_zip, lo, hi),
    )
    if not cands:
        return None, "UNMATCHED", 0.0

    best, best_score = None, 0.0
    for c in cands:
        # Weight is the discriminating field once lane and date have blocked.
        w_ratio = min(c["bol_weight_lb"], line.billed_weight_lb) / max(
            c["bol_weight_lb"], line.billed_weight_lb, 1.0
        )
        days = abs((date.fromisoformat(c["ship_date"]) - line.ship_date).days)
        d_score = 1.0 - days / (DATE_WINDOW_DAYS + 1)
        score = 0.7 * w_ratio + 0.3 * d_score
        if score > best_score:
            best, best_score = c, score

    if best is None or best_score < FUZZY_FLOOR:
        return None, "UNMATCHED", round(best_score, 3)
    return best, "FUZZY_LANE_DATE_WEIGHT", round(best_score, 3)


# --------------------------------------------------------------------------
# Recompute
# --------------------------------------------------------------------------


def recompute(
    con: sqlite3.Connection, line: ExtractedInvoiceLine, seen_pros: set[str]
) -> Recomputation:
    shipment, method, confidence = match_shipment(con, line)
    billed_total = round(sum(c.amount for c in line.charges), 2)
    is_dup = line.pro_number in seen_pros

    if shipment is None:
        return Recomputation(
            pro_number=line.pro_number, matched_shipment_id=None, match_method="UNMATCHED",
            match_confidence=confidence, expected_charges=[], expected_total=0.0,
            billed_total=billed_total, variance=billed_total, variance_pct=100.0,
            is_duplicate_pro=is_dup,
        )

    sd = date.fromisoformat(shipment["ship_date"])
    facility = db.one(con, "SELECT * FROM facilities WHERE id = ?",
                      (shipment["consignee_facility_id"],))
    requested = set(json.loads(shipment["requested_accessorials"]))

    expected: list[ExpectedCharge] = []

    # Linehaul from OUR bill of lading weight and class, not the carrier's.
    if (shipment["origin_zone"], shipment["dest_zone"]) in LANE_RATES_CWT:
        lh, basis = pricing.linehaul(
            shipment["origin_zone"], shipment["dest_zone"],
            shipment["bol_weight_lb"], shipment["bol_class"],
        )
    else:
        lh, basis = 0.0, "no contracted rate for this lane"
    expected.append(ExpectedCharge(code="LINEHAUL", basis=basis, amount=lh))

    fpct = pricing.fuel_pct(sd)
    fuel = round(lh * fpct, 2) if fpct else 0.0
    expected.append(ExpectedCharge(
        code="FUEL_SURCHARGE",
        basis=(f"pickup week {pricing.fuel_week(sd)} @ {fpct * 100:.2f}% of linehaul"
               if fpct else "pickup week outside published fuel index"),
        amount=fuel,
    ))

    for code in sorted(requested):
        expected.append(ExpectedCharge(
            code=code, basis="requested on the bill of lading",
            amount=ACCESSORIAL_PRICES[code],
        ))
    if shipment["detention_hours"]:
        expected.append(ExpectedCharge(
            code="DETENTION",
            basis=f"{shipment['detention_hours']:g} hr beyond 2 hr free time (MFS-220)",
            amount=pricing.accessorial_price("DETENTION", shipment["detention_hours"]),
        ))

    expected_total = round(sum(e.amount for e in expected), 2)

    # A shipment already invoiced is owed NOTHING on a second invoice. Rating it a
    # second time and comparing like-for-like produces a variance of zero, which is
    # arithmetically true and commercially nonsense -- the whole re-billed amount is
    # recoverable, and MFS-300 credits a duplicate in full without proration. So the
    # expected total for a duplicate is zero, carried as an explicit credit line so a
    # reviewer can still see what the shipment legitimately cost the first time.
    if is_dup:
        expected.append(ExpectedCharge(
            code="ALREADY_INVOICED",
            basis=f"PRO {line.pro_number} was invoiced previously; nothing further is owed "
                  f"on this shipment (MFS-300)",
            amount=-expected_total,
        ))
        expected_total = 0.0

    # ---- machine-checkable evidence for the classifier ----
    weight_delta = round(line.billed_weight_lb - shipment["bol_weight_lb"], 1)
    class_delta = round(line.billed_class - shipment["bol_class"], 1)

    # MFS-110, applied HERE rather than left to the classifier to derive from the
    # retrieved clause. The band is 2% or 50 lb, whichever is GREATER; inside it the
    # BOL weight governs and no adjustment is due, which is what the expected charges
    # above already assume. Outside it the carrier's reweigh governs and the expected
    # linehaul would have to be recomputed on the billed weight -- no shipment in this
    # dataset falls outside the band, so that repricing is deliberately not written
    # rather than written untested.
    reweigh_tolerance = round(max(shipment["bol_weight_lb"] * 0.02, 50.0), 1)
    reweigh_governs = pricing.reweigh_is_valid(
        shipment["bol_weight_lb"], line.billed_weight_lb
    )

    # An accessorial is unsupported when the bill of lading did not request it AND the
    # tariff does not otherwise establish it. Those are two different arguments and a
    # reviewer needs to know which one applies:
    #
    #   - the facility master CONTRADICTS the condition billed for (a liftgate at a
    #     dock-equipped site) -- MFS-200/MFS-210 make the charge flatly invalid;
    #   - the facility master SUPPORTS the condition but the carrier owes documentation
    #     on the delivery receipt and none accompanies the invoice -- still unsupported,
    #     but the argument is the carrier's evidentiary burden, not a contradiction;
    #   - the facility master says NOTHING about this accessorial, so "not requested on
    #     the bill of lading" is the whole of the argument.
    #
    # The previous version flagged all three and reported the first reason for every
    # one of them, which put a false sentence in front of a reviewer about to dispute.
    unsupported: list[str] = []
    reasons: dict[str, str] = {}
    loc = facility["location_type"] if facility else None
    dock = bool(facility["dock_equipped"]) if facility else None

    def _flag(code: str, reason: str) -> None:
        unsupported.append(code)
        reasons[code] = reason

    for ch in line.charges:
        code = ch.code
        if code in ("LINEHAUL", "FUEL_SURCHARGE", "OTHER"):
            continue
        if code in requested:
            continue
        if code == "DETENTION" and shipment["detention_hours"]:
            continue
        if code in ("REWEIGH_FEE", "RECLASSIFICATION_FEE"):
            continue  # judged by the weight/class deltas, not by support

        if code == "LIFTGATE_DELIVERY":
            if dock:
                _flag(code, "facility master records this consignee as dock-equipped and the "
                            "bill of lading did not request a liftgate; MFS-200 makes the "
                            "charge invalid")
            else:
                _flag(code, "consignee lacks a dock, so the condition exists -- but MFS-200 "
                            "allows an unrequested liftgate only where carrier documents that "
                            "condition on the delivery receipt, and no such documentation "
                            "accompanies this invoice")
        elif code == "RESIDENTIAL_DELIVERY":
            if loc != "RESIDENTIAL":
                _flag(code, f"facility master records this consignee as {loc}, not residential, "
                            f"and the bill of lading did not request it; MFS-210 bars the charge "
                            f"absent documentation on the delivery receipt")
            else:
                _flag(code, "consignee is residential, but the bill of lading did not request "
                            "the charge and no delivery-receipt documentation accompanies it")
        elif code == "LIMITED_ACCESS":
            if loc != "LIMITED_ACCESS":
                _flag(code, f"facility master records this consignee as {loc}, which is not a "
                            f"limited-access location, and the bill of lading did not request it")
            else:
                _flag(code, "consignee is limited-access, but the bill of lading did not "
                            "request the charge")
        else:
            _flag(code, "not requested on the bill of lading; the facility master does not "
                        "record a condition that would establish this accessorial")

    billed_fuel = next((c.amount for c in line.charges if c.code == "FUEL_SURCHARGE"), None)
    billed_lh = next((c.amount for c in line.charges if c.code == "LINEHAUL"), None)
    fuel_billed_pct = (
        round(billed_fuel / billed_lh, 4) if billed_fuel and billed_lh else None
    )

    variance = round(billed_total - expected_total, 2)
    return Recomputation(
        pro_number=line.pro_number,
        matched_shipment_id=shipment["id"],
        match_method=method,
        match_confidence=confidence,
        expected_charges=expected,
        expected_total=expected_total,
        billed_total=billed_total,
        variance=variance,
        variance_pct=round(variance / expected_total * 100, 2) if expected_total else 100.0,
        weight_delta_lb=weight_delta,
        class_delta=class_delta,
        reweigh_tolerance_lb=reweigh_tolerance,
        reweigh_governs=reweigh_governs,
        unsupported_accessorials=unsupported,
        unsupported_reasons=reasons,
        fuel_pct_billed=fuel_billed_pct,
        fuel_pct_expected=fpct,
        is_duplicate_pro=is_dup,
    )
