"""Synthetic data generation.

This builds a facility master, a book of shipments, and a set of carrier invoices
in which a known set of billing defects has been deliberately injected. The
injected defect for each line is written to `ground_truth`, which the pipeline
never reads and the eval harness always does.

Everything is generated from a fixed seed, so the eval set is stable across runs
and a change in the score means a change in the system rather than a change in the
data. That property is the whole reason to synthesize rather than hand-write.

The defect taxonomy is not invented: reweighs, reclassifications, unauthorised
accessorials, duplicate billing and misapplied fuel surcharges are the categories
a freight-audit desk actually works, and each one maps to a specific tariff rule
in `ratecard.py`.
"""

from __future__ import annotations

import json
import random
from datetime import date, timedelta

from . import db, pricing
from .ratecard import (
    ACCESSORIAL_PRICES,
    CARRIER_NAME,
    CLASS_FACTORS,
    FUEL_SURCHARGE_BY_WEEK,
    LANE_RATES_CWT,
    TARIFF_RULES,
    WEIGHT_BREAKS,
)

SEED = 20260910
DEST_ZONES = [z[1] for z in LANE_RATES_CWT]

FACILITIES = [
    # id,     name,                                zone,  type,             dock
    ("FAC-01", "Metro Electronics DC",             "M5V", "COMMERCIAL",     1),
    ("FAC-02", "Northline Retail Warehouse",       "N2L", "COMMERCIAL",     1),
    ("FAC-03", "Capital Supply Depot",             "K1A", "COMMERCIAL",     1),
    ("FAC-04", "Rue Sherbrooke Storefront",        "H3B", "COMMERCIAL",     0),
    ("FAC-05", "Harbourside Fulfilment",           "V6B", "COMMERCIAL",     1),
    ("FAC-06", "Foothills Distribution",           "T2P", "COMMERCIAL",     1),
    ("FAC-07", "Atlantic Cross-Dock",              "B3J", "COMMERCIAL",     1),
    ("FAC-08", "Lakeshore Residence 44",           "M5V", "RESIDENTIAL",    0),
    ("FAC-09", "Grandview Academy Receiving",      "N2L", "LIMITED_ACCESS", 0),
    ("FAC-10", "Parkdale Clinic Stores",           "K1A", "LIMITED_ACCESS", 0),
    ("FAC-11", "Vieux-Port Popup",                 "H3B", "RESIDENTIAL",    0),
    ("FAC-12", "Seymour Street Studio",            "V6B", "COMMERCIAL",     0),
]

# How many invoice lines carry each defect. Sums to 60.
DEFECT_PLAN = {
    "NO_VARIANCE": 21,
    "WITHIN_TOLERANCE": 6,
    "REWEIGH": 7,
    "RECLASS": 5,
    "UNAUTHORIZED_ACCESSORIAL": 7,
    "DUPLICATE_BILLING": 4,
    "FUEL_SURCHARGE_ERROR": 5,
    "RATE_CARD_MISMATCH": 5,
}

DEFECT_NOTES = {
    "NO_VARIANCE": "Billed exactly as contracted.",
    "WITHIN_TOLERANCE": "Sub-dollar rounding difference; inside the auto-approve band.",
    "REWEIGH": "Carrier billed a reweighed weight that falls INSIDE the MFS-110 tolerance "
               "(2% or 50 lb, whichever is greater), so the BOL weight should govern. A "
               "reweigh fee was also assessed, which MFS-110 permits only on a valid "
               "upward adjustment.",
    "RECLASS": "Carrier rerated at a higher NMFC class than the BOL and assessed a "
               "reclassification fee. MFS-120 requires an inspection certificate; none "
               "accompanies the invoice.",
    # The injector picks the first candidate accessorial the BOL did not request; it
    # does not consult the facility master, so this note must not claim it did. Both
    # shapes are disputable and they are disputable for different reasons: either the
    # facility master contradicts the condition (MFS-200/MFS-210 make the charge
    # invalid outright), or the condition is real but the carrier owes delivery-receipt
    # documentation that does not accompany the invoice.
    "UNAUTHORIZED_ACCESSORIAL": "An accessorial was billed that the bill of lading did "
                                "not request and the tariff does not otherwise establish "
                                "-- either the facility master contradicts the condition, "
                                "or the carrier owes delivery-receipt documentation that "
                                "is absent.",
    "DUPLICATE_BILLING": "This PRO was already invoiced on an earlier invoice. MFS-300.",
    "FUEL_SURCHARGE_ERROR": "Fuel surcharge applied at the INVOICE week's percentage "
                            "rather than the PICKUP week's, contrary to MFS-130.",
    "RATE_CARD_MISMATCH": "Linehaul was rated off a weight break the shipment does not "
                          "qualify for, so the cwt rate is wrong.",
}


# --------------------------------------------------------------------------
# Tariff chunking + indexing
# --------------------------------------------------------------------------


def chunk_tariff() -> list[tuple[str, str, str, str, int]]:
    """Chunk the tariff prose by rule, then by sentence group within a rule.

    Structural chunking, not fixed-width. These clauses are already authored as
    self-contained rules, and a rule cut in half is a rule that retrieves as a
    half-truth -- MFS-110's "whichever is greater" landing in a different chunk
    from the tolerance it modifies would be exactly the failure this project is
    about. Rules longer than ~700 characters are split on sentence boundaries with
    the rule title repeated into each chunk so the chunk stands alone.
    """
    out: list[tuple[str, str, str, str, int]] = []
    for rule_id, title, body in TARIFF_RULES:
        text = " ".join(body.split())
        sentences = [s.strip() + "." for s in text.split(". ") if s.strip()]
        groups, buf = [], ""
        for s in sentences:
            if len(buf) + len(s) > 700 and buf:
                groups.append(buf.strip())
                buf = s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            groups.append(buf.strip())
        for i, g in enumerate(groups):
            cid = f"{rule_id}#{i}" if len(groups) > 1 else rule_id
            out.append((cid, rule_id, title, g, i))
    return out


# --------------------------------------------------------------------------
# Shipment book
# --------------------------------------------------------------------------


def build_shipments(rng: random.Random) -> list[dict]:
    shipments = []
    start = date(2026, 6, 1)
    for i in range(78):   # 60 invoiced, 6 judgment, 6 holdout, 6 uploadable judgment
        fac = FACILITIES[rng.randrange(len(FACILITIES))]
        ship_date = start + timedelta(days=rng.randrange(0, 88))
        # Skip weekends -- pickups are weekday events, and it keeps fuel weeks clean.
        while ship_date.weekday() >= 5:
            ship_date += timedelta(days=1)

        weight = float(rng.choice([
            380, 460, 540, 620, 780, 950, 1120, 1400, 1680, 1950,
            2200, 2650, 3100, 3800, 4400, 5200, 6100, 7400, 8800, 11200,
        ]))
        fclass = rng.choice([50.0, 55.0, 60.0, 65.0, 70.0, 77.5, 85.0, 92.5, 100.0, 110.0])

        # FACILITIES rows are (id, name, zone, location_type, dock_equipped).
        fac_id, _fac_name, fac_zone, fac_type, fac_dock = fac

        requested: list[str] = []
        if fac_dock == 0 and rng.random() < 0.65:
            requested.append("LIFTGATE_DELIVERY")
        if fac_type == "RESIDENTIAL":
            requested.append("RESIDENTIAL_DELIVERY")
        if fac_type == "LIMITED_ACCESS":
            requested.append("LIMITED_ACCESS")
        appointment = 1 if (fac_type == "LIMITED_ACCESS" or rng.random() < 0.2) else 0
        if appointment:
            requested.append("APPOINTMENT")
        detention = rng.choice([0, 0, 0, 0, 0, 0.5, 1.0, 1.5]) if rng.random() < 0.18 else 0.0

        shipments.append({
            "id": f"SHP-{2600 + i}",
            "pro_number": f"MFS{rng.randrange(10**8, 10**9)}",
            "bol_number": f"BOL-{74000 + i}",
            "ship_date": ship_date.isoformat(),
            "origin_zone": "L6T",
            "dest_zone": fac_zone,
            "bol_weight_lb": weight,
            "bol_class": fclass,
            "consignee_facility_id": fac_id,
            "requested_accessorials": json.dumps(requested),
            "detention_hours": detention,
            "appointment_required": appointment,
        })
    return shipments


# --------------------------------------------------------------------------
# Charge construction
# --------------------------------------------------------------------------


def correct_charges(s: dict) -> tuple[list[dict], float, float]:
    """What this shipment SHOULD be billed. Returns (charges, weight, class)."""
    sd = date.fromisoformat(s["ship_date"])
    lh, _ = pricing.linehaul(s["origin_zone"], s["dest_zone"], s["bol_weight_lb"], s["bol_class"])
    fpct = pricing.fuel_pct(sd) or 0.245
    charges = [
        {"code": "LINEHAUL", "description": "Linehaul transportation charge", "amount": lh},
        {"code": "FUEL_SURCHARGE",
         "description": f"Fuel surcharge @ {fpct * 100:.2f}%",
         "amount": round(lh * fpct, 2)},
    ]
    for code in json.loads(s["requested_accessorials"]):
        charges.append({
            "code": code,
            "description": code.replace("_", " ").title(),
            "amount": pricing.accessorial_price(code),
        })
    if s["detention_hours"]:
        charges.append({
            "code": "DETENTION",
            "description": f"Detention {s['detention_hours']:g} hr beyond free time",
            "amount": pricing.accessorial_price("DETENTION", s["detention_hours"]),
        })
    return charges, s["bol_weight_lb"], s["bol_class"]


def apply_defect(defect: str, s: dict, rng: random.Random) -> tuple[list[dict], float, float]:
    """Return the charges the carrier ACTUALLY billed, with `defect` injected."""
    charges, weight, fclass = correct_charges(s)
    sd = date.fromisoformat(s["ship_date"])

    def relinehaul(w: float, c: float) -> None:
        lh, _ = pricing.linehaul(s["origin_zone"], s["dest_zone"], w, c)
        fpct = pricing.fuel_pct(sd) or 0.245
        charges[0]["amount"] = lh
        charges[1]["amount"] = round(lh * fpct, 2)
        charges[1]["description"] = f"Fuel surcharge @ {fpct * 100:.2f}%"

    if defect == "WITHIN_TOLERANCE":
        charges[0]["amount"] = round(charges[0]["amount"] + rng.choice([0.31, -0.42, 0.77, 1.10]), 2)

    elif defect == "REWEIGH":
        # Inside the MFS-110 tolerance band, so the BOL weight should have governed.
        band = max(weight * 0.02, 50.0)
        weight = round(weight + band * rng.uniform(0.45, 0.92), 0)
        relinehaul(weight, fclass)
        charges.append({"code": "REWEIGH_FEE", "description": "Reweigh fee",
                        "amount": ACCESSORIAL_PRICES["REWEIGH_FEE"]})

    elif defect == "RECLASS":
        ladder = sorted(CLASS_FACTORS)
        idx = ladder.index(fclass)
        fclass = ladder[min(idx + rng.choice([1, 2]), len(ladder) - 1)]
        relinehaul(weight, fclass)
        charges.append({"code": "RECLASSIFICATION_FEE", "description": "Reclassification fee",
                        "amount": ACCESSORIAL_PRICES["RECLASSIFICATION_FEE"]})

    elif defect == "UNAUTHORIZED_ACCESSORIAL":
        requested = set(json.loads(s["requested_accessorials"]))
        for cand in ("LIFTGATE_DELIVERY", "RESIDENTIAL_DELIVERY", "INSIDE_DELIVERY", "REDELIVERY"):
            if cand not in requested:
                charges.append({"code": cand, "description": cand.replace("_", " ").title(),
                                "amount": ACCESSORIAL_PRICES[cand]})
                break

    elif defect == "FUEL_SURCHARGE_ERROR":
        # The invoice week's percentage instead of the pickup week's.
        weeks = sorted(FUEL_SURCHARGE_BY_WEEK)
        cur = pricing.fuel_week(sd)
        nxt = weeks[min(weeks.index(cur) + rng.choice([2, 3]), len(weeks) - 1)] if cur in weeks else weeks[-1]
        wrong = FUEL_SURCHARGE_BY_WEEK[nxt]
        charges[1]["amount"] = round(charges[0]["amount"] * wrong, 2)
        charges[1]["description"] = f"Fuel surcharge @ {wrong * 100:.2f}%"

    elif defect == "RATE_CARD_MISMATCH":
        # Rated off the weight break BELOW the one the shipment qualifies for,
        # which raises the cwt rate and so the charge.
        lane = LANE_RATES_CWT[(s["origin_zone"], s["dest_zone"])]
        brk = pricing.weight_break(weight)
        names = [b[0] for b in WEIGHT_BREAKS]
        i = max(names.index(brk) - 1, 0)
        wrong_rate = lane[names[i]]
        factor = CLASS_FACTORS[fclass]
        charges[0]["amount"] = round(weight / 100.0 * wrong_rate * factor, 2)
        fpct = pricing.fuel_pct(sd) or 0.245
        charges[1]["amount"] = round(charges[0]["amount"] * fpct, 2)

    return charges, weight, fclass


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def build_invoices(shipments: list[dict], rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Group shipments into invoices and inject the planned defects.

    Returns (invoices, ground_truth_rows).
    """
    plan: list[str] = []
    for defect, n in DEFECT_PLAN.items():
        plan.extend([defect] * n)
    rng.shuffle(plan)

    # Duplicates are not a property of a line, they are a property of a pair, so
    # they are handled by re-billing an earlier line rather than by perturbing this one.
    dup_slots = [i for i, d in enumerate(plan) if d == "DUPLICATE_BILLING"]

    lines = []
    for i, s in enumerate(shipments):
        defect = plan[i]
        if defect == "DUPLICATE_BILLING":
            continue  # filled in below
        charges, w, c = apply_defect(defect, s, rng)
        lines.append({"shipment": s, "defect": defect, "charges": charges,
                      "billed_weight": w, "billed_class": c})

    # Re-bill some already-invoiced lines verbatim, later in the sequence.
    for k, _slot in enumerate(dup_slots):
        src = lines[k * 7 + 3]
        lines.append({"shipment": src["shipment"], "defect": "DUPLICATE_BILLING",
                      "charges": [dict(ch) for ch in src["charges"]],
                      "billed_weight": src["billed_weight"], "billed_class": src["billed_class"]})

    invoices, truth = [], []
    per_invoice = 8
    for n in range(0, len(lines), per_invoice):
        batch = lines[n:n + per_invoice]
        inv_no = f"MFS-INV-{9100 + n // per_invoice}"
        latest = max(date.fromisoformat(b["shipment"]["ship_date"]) for b in batch)
        inv_date = latest + timedelta(days=rng.randrange(3, 10))
        inv_lines = []
        for b in batch:
            total = round(sum(ch["amount"] for ch in b["charges"]), 2)
            inv_lines.append({
                "pro_number": b["shipment"]["pro_number"],
                "bol_number": b["shipment"]["bol_number"],
                "ship_date": b["shipment"]["ship_date"],
                "origin_zip": b["shipment"]["origin_zone"],
                "dest_zip": b["shipment"]["dest_zone"],
                "billed_weight_lb": b["billed_weight"],
                "billed_class": b["billed_class"],
                "charges": b["charges"],
                "line_total": total,
            })
            truth.append({"invoice_number": inv_no, "pro_number": b["shipment"]["pro_number"],
                          "category": b["defect"], "note": DEFECT_NOTES[b["defect"]]})
        invoices.append({
            "invoice_number": inv_no,
            "carrier_name": CARRIER_NAME,
            "invoice_date": inv_date.isoformat(),
            "lines": inv_lines,
            "invoice_total": round(sum(l["line_total"] for l in inv_lines), 2),
        })
    return invoices, truth


def build_judgment_invoice(shipments: list[dict], rng: random.Random) -> tuple[dict, list[dict]]:
    """The invoice where CODE CANNOT DECIDE and the model has to read the tariff.

    Every other defect in this book is mechanically detectable: a weight delta, a
    class delta, a duplicate PRO, a fuel percentage that does not match the table.
    Code finds those and the classifier only has to name them. That makes for a
    flattering demo and a dishonest one, because it never shows the model doing
    work that could not have been done with a join.

    These two cases cannot be resolved by arithmetic:

    1. **Residential AND limited-access billed on the same shipment.** Both were
       requested on the bill of lading, so every machine-checked fact says
       "supported" -- `unsupported_accessorials` is empty and the facility master
       agrees. Only MFS-210 says they *shall not both be assessed*, and that the
       lower is waived. **Code says fine; the tariff says not fine.** The only way
       to that answer is retrieving the clause and applying it.

    2. **A charge the tariff does not provide for at all.** "Winter operations
       surcharge" maps to no contracted accessorial, so there is no rate to compare
       against and no rule that forbids it either. The correct answer is UNKNOWN and
       a human, which is the behaviour worth demonstrating: the system declining
       rather than inventing a category.
    """
    lines, truth = [], []
    inv_no = "MFS-INV-9300"

    # --- case 1: two accessorials that individually check out ---
    s1 = shipments[0]
    charges, w, c = correct_charges(s1)
    codes = {ch["code"] for ch in charges}
    for extra in ("RESIDENTIAL_DELIVERY", "LIMITED_ACCESS"):
        if extra not in codes:
            charges.append({"code": extra, "description": extra.replace("_", " ").title(),
                            "amount": ACCESSORIAL_PRICES[extra]})
    lines.append({
        "pro_number": s1["pro_number"], "bol_number": s1["bol_number"],
        "ship_date": s1["ship_date"], "origin_zip": s1["origin_zone"],
        "dest_zip": s1["dest_zone"], "billed_weight_lb": w, "billed_class": c,
        "charges": charges, "line_total": round(sum(x["amount"] for x in charges), 2),
    })
    truth.append({"invoice_number": inv_no, "pro_number": s1["pro_number"],
                  "category": "UNAUTHORIZED_ACCESSORIAL", "blind_spot": 1,
                  "note": "KNOWN BLIND SPOT. Residential delivery AND limited access are "
                          "both billed, both requested on the BOL, and both priced at the "
                          "contracted rate -- so the variance is $0.00 and nothing "
                          "escalates. MFS-210 says they shall not both be assessed and "
                          "the lower is waived, which means the carrier owes a credit of "
                          "$76.50 that this architecture cannot see. The pipeline "
                          "escalates on DOLLAR VARIANCE; a charge that is correctly priced "
                          "but contractually impermissible has no variance and is "
                          "invisible. Catching it needs a separate permissibility pass "
                          "over billed accessorial COMBINATIONS, not a better classifier."})

    # --- case 2: a charge the contract has no rate for ---
    s2 = shipments[1]
    charges, w, c = correct_charges(s2)
    charges.append({"code": "OTHER", "description": "Winter operations surcharge",
                    "amount": 185.00})
    lines.append({
        "pro_number": s2["pro_number"], "bol_number": s2["bol_number"],
        "ship_date": s2["ship_date"], "origin_zip": s2["origin_zone"],
        "dest_zip": s2["dest_zone"], "billed_weight_lb": w, "billed_class": c,
        "charges": charges, "line_total": round(sum(x["amount"] for x in charges), 2),
    })
    truth.append({"invoice_number": inv_no, "pro_number": s2["pro_number"],
                  "category": "UNKNOWN", "blind_spot": 0,
                  "note": "A charge the contracted tariff does not provide for. There is "
                          "no rate to compare against and no clause forbidding it. UNKNOWN "
                          "and a human is the correct answer, not an invented category."})

    # --- case 3: the same accessorial billed twice on one delivery ---
    # MFS-200: "Only one liftgate charge may be assessed per delivery regardless of
    # piece count." Liftgate IS on the bill of lading, so `unsupported_accessorials`
    # stays empty and every fact says supported. Only the clause caps the count.
    s3 = shipments[2]
    charges, w, c = correct_charges(s3)
    if not any(ch["code"] == "LIFTGATE_DELIVERY" for ch in charges):
        charges.append({"code": "LIFTGATE_DELIVERY", "description": "Liftgate Delivery",
                        "amount": ACCESSORIAL_PRICES["LIFTGATE_DELIVERY"]})
    charges.append({"code": "LIFTGATE_DELIVERY", "description": "Liftgate Delivery (2nd pallet)",
                    "amount": ACCESSORIAL_PRICES["LIFTGATE_DELIVERY"]})
    lines.append({
        "pro_number": s3["pro_number"], "bol_number": s3["bol_number"],
        "ship_date": s3["ship_date"], "origin_zip": s3["origin_zone"],
        "dest_zip": s3["dest_zone"], "billed_weight_lb": w, "billed_class": c,
        "charges": charges, "line_total": round(sum(x["amount"] for x in charges), 2),
    })
    truth.append({"invoice_number": inv_no, "pro_number": s3["pro_number"],
                  "category": "UNAUTHORIZED_ACCESSORIAL", "blind_spot": 0,
                  "note": "Liftgate billed twice for one delivery. The service WAS "
                          "requested, so nothing is flagged as unsupported and the only "
                          "machine-visible fact is a $92 variance. MFS-200 caps it at one "
                          "per delivery regardless of piece count -- a conclusion reachable "
                          "only from the clause."})

    # --- case 4: detention beyond the daily cap ---
    # MFS-220 allows detention but caps it at eight hours in any single day. Our
    # record has detention, so code treats the charge as supported; only the
    # arithmetic of the cap makes the excess invalid.
    s4 = dict(shipments[3])
    s4["detention_hours"] = 8.0        # the cap; our record agrees up to it
    charges, w, c = correct_charges(s4)
    charges = [ch for ch in charges if ch["code"] != "DETENTION"]
    charges.append({"code": "DETENTION", "description": "Detention 11 hr beyond free time",
                    "amount": round(ACCESSORIAL_PRICES["DETENTION"] * 11, 2)})
    lines.append({
        "pro_number": s4["pro_number"], "bol_number": s4["bol_number"],
        "ship_date": s4["ship_date"], "origin_zip": s4["origin_zone"],
        "dest_zip": s4["dest_zone"], "billed_weight_lb": w, "billed_class": c,
        "charges": charges, "line_total": round(sum(x["amount"] for x in charges), 2),
    })
    truth.append({"invoice_number": inv_no, "pro_number": s4["pro_number"],
                  "category": "UNAUTHORIZED_ACCESSORIAL", "blind_spot": 0,
                  "note": "Eleven hours of detention billed. Detention itself is "
                          "legitimate and on our record for eight hours, so nothing is "
                          "flagged as unsupported. MFS-220 caps detention at eight hours "
                          "in any single day, making the three excess hours ($195) "
                          "unassessable."})

    # --- case 5: fuel surcharge computed on the wrong base ---
    # MFS-130: the surcharge applies to LINEHAUL, not to accessorials. The
    # percentage itself is right and the week is right; only the base is wrong.
    # Code sees a fuel discrepancy and cannot say why -- the model has to.
    s5 = shipments[4]
    charges, w, c = correct_charges(s5)
    lh = next(ch["amount"] for ch in charges if ch["code"] == "LINEHAUL")
    acc = sum(ch["amount"] for ch in charges
              if ch["code"] not in ("LINEHAUL", "FUEL_SURCHARGE"))
    fpct = pricing.fuel_pct(date.fromisoformat(s5["ship_date"])) or 0.245
    for ch in charges:
        if ch["code"] == "FUEL_SURCHARGE":
            ch["amount"] = round((lh + acc) * fpct, 2)
            ch["description"] = f"Fuel surcharge @ {fpct * 100:.2f}%"
    lines.append({
        "pro_number": s5["pro_number"], "bol_number": s5["bol_number"],
        "ship_date": s5["ship_date"], "origin_zip": s5["origin_zone"],
        "dest_zip": s5["dest_zone"], "billed_weight_lb": w, "billed_class": c,
        "charges": charges, "line_total": round(sum(x["amount"] for x in charges), 2),
    })
    truth.append({"invoice_number": inv_no, "pro_number": s5["pro_number"],
                  "category": "FUEL_SURCHARGE_ERROR", "blind_spot": 0,
                  "note": "The fuel PERCENTAGE and the WEEK are both correct. The base is "
                          "not: the surcharge was computed on linehaul plus accessorials, "
                          "and MFS-130 applies it to linehaul only. The right category for "
                          "the wrong-looking reason -- a test of whether the model reads the "
                          "clause or pattern-matches 'fuel mismatch' to 'wrong week'."})

    # --- case 6: the SAME unmapped charge, a second time ---
    # One unmapped charge is a reviewer's job. The same one twice is a rule nobody
    # has written: either negotiate a rate for it or dispute it as a class. This
    # second occurrence is what turns the alert from INFO into ACTION, and it is
    # the whole argument for alerting on patterns rather than on lines.
    s6 = shipments[5]
    charges, w, c = correct_charges(s6)
    charges.append({"code": "OTHER", "description": "WINTER OPERATIONS SURCHARGE",
                    "amount": 210.00})
    lines.append({
        "pro_number": s6["pro_number"], "bol_number": s6["bol_number"],
        "ship_date": s6["ship_date"], "origin_zip": s6["origin_zone"],
        "dest_zip": s6["dest_zone"], "billed_weight_lb": w, "billed_class": c,
        "charges": charges, "line_total": round(sum(x["amount"] for x in charges), 2),
    })
    truth.append({"invoice_number": inv_no, "pro_number": s6["pro_number"],
                  "category": "UNKNOWN", "blind_spot": 0,
                  "note": "The same uncontracted charge as the other OTHER line, billed "
                          "again at a different amount and in different case. Individually "
                          "it is another abstention; together the two make a pattern the "
                          "owner should resolve once rather than a reviewer resolving "
                          "twice."})

    latest = max(date.fromisoformat(l["ship_date"]) for l in lines)
    return {
        "invoice_number": inv_no, "carrier_name": CARRIER_NAME,
        "invoice_date": (latest + timedelta(days=4)).isoformat(),
        "lines": lines,
        "invoice_total": round(sum(l["line_total"] for l in lines), 2),
    }, truth


def build_holdout_invoice(shipments: list[dict], rng: random.Random) -> dict:
    """One never-ingested invoice, for exercising the upload gates.

    Separate from `build_invoices` on purpose: that function injects duplicate
    billing by index and assumes the full 60-shipment book, so it cannot be handed
    a six-line slice. This builds a plain invoice and injects a small, realistic
    spread of defects so an audit of it shows something worth looking at.
    """
    defects = ["NO_VARIANCE", "REWEIGH", "UNAUTHORIZED_ACCESSORIAL",
               "NO_VARIANCE", "FUEL_SURCHARGE_ERROR", "RECLASS"]
    lines = []
    for s_, defect in zip(shipments, defects):
        charges, w, c = apply_defect(defect, s_, rng)
        lines.append({
            "pro_number": s_["pro_number"], "bol_number": s_["bol_number"],
            "ship_date": s_["ship_date"], "origin_zip": s_["origin_zone"],
            "dest_zip": s_["dest_zone"], "billed_weight_lb": w, "billed_class": c,
            "charges": charges,
            "line_total": round(sum(ch["amount"] for ch in charges), 2),
        })
    latest = max(date.fromisoformat(l["ship_date"]) for l in lines)
    return {
        "invoice_number": "MFS-INV-9200",
        "carrier_name": CARRIER_NAME,
        "invoice_date": (latest + timedelta(days=5)).isoformat(),
        "lines": lines,
        "invoice_total": round(sum(l["line_total"] for l in lines), 2),
    }


def seed_all(with_pdfs: bool = True) -> dict:
    from .pdfgen import render_decoy_pdf, render_invoice_pdf
    from .retrieve import index_clauses

    rng = random.Random(SEED)
    db.reset()
    con = db.connect()

    con.executemany(
        "INSERT INTO facilities (id,name,zone,location_type,dock_equipped) VALUES (?,?,?,?,?)",
        FACILITIES,
    )
    shipments = build_shipments(rng)
    con.executemany(
        """INSERT INTO shipments (id,pro_number,bol_number,ship_date,origin_zone,dest_zone,
           bol_weight_lb,bol_class,consignee_facility_id,requested_accessorials,
           detention_hours,appointment_required)
           VALUES (:id,:pro_number,:bol_number,:ship_date,:origin_zone,:dest_zone,
           :bol_weight_lb,:bol_class,:consignee_facility_id,:requested_accessorials,
           :detention_hours,:appointment_required)""",
        shipments,
    )

    for cid, rule_id, title, text, seq in chunk_tariff():
        con.execute("INSERT INTO clauses (id,rule_id,title,text,seq) VALUES (?,?,?,?,?)",
                    (cid, rule_id, title, text, seq))
        con.execute("INSERT INTO clauses_fts (id,title,text) VALUES (?,?,?)", (cid, title, text))

    # The last six shipments are deliberately NOT invoiced. They exist so there is
    # a genuine, never-ingested carrier invoice to upload through the gates --
    # demonstrating the real path rather than re-processing something already on file.
    invoiced, judgment = shipments[:60], shipments[60:66]
    holdout, upload_judgment = shipments[66:72], shipments[72:78]
    invoices, truth = build_invoices(invoiced, rng)
    for inv in invoices:
        pdf_path = db.INVOICE_DIR / f"{inv['invoice_number']}.pdf"
        if with_pdfs:
            render_invoice_pdf(inv, pdf_path)
        con.execute(
            "INSERT INTO invoices (invoice_number,carrier_name,invoice_date,pdf_path,"
            "stated_total,seed_payload) VALUES (?,?,?,?,?,?)",
            (inv["invoice_number"], inv["carrier_name"], inv["invoice_date"],
             str(pdf_path), inv["invoice_total"], json.dumps(inv)),
        )
    # The judgment invoice: cases code cannot resolve. Ingested like any other, so
    # it appears in the queue beside the mechanical ones.
    #
    # Case 1 only works if BOTH accessorials were genuinely requested on the bill of
    # lading. That is the whole point: every machine-checked fact then says
    # "supported", and only MFS-210 says the pair is not billable together. If they
    # were unrequested, code would flag them and the model would have nothing to do.
    # Each judgment case only works if the shipment record AGREES with the invoice
    # on everything except the contractual point at issue. Otherwise code flags the
    # discrepancy mechanically and the model has nothing left to decide -- which is
    # exactly what happened on the first attempt at cases 3 and 4.
    def _align(book: list[dict], idx: int, **cols) -> None:
        sets = ", ".join(f"{k}=?" for k in cols)
        con.execute(f"UPDATE shipments SET {sets} WHERE pro_number=?",
                    (*cols.values(), book[idx]["pro_number"]))
        book[idx].update(cols)

    def _align_judgment(book: list[dict]) -> None:
        """Apply the same alignment to any set used for a judgment invoice.

        Applied to BOTH the seeded judgment invoice and the uploadable one --
        otherwise the uploaded copy has unaligned shipment records, code flags the
        discrepancies mechanically, and the demo claim "the model is deciding this"
        is false on the very invoice used to demonstrate it.
        """
        # case 1: both accessorials genuinely requested, so nothing reads unsupported
        _align(book, 0, requested_accessorials=json.dumps(
            ["RESIDENTIAL_DELIVERY", "LIMITED_ACCESS"]))
        # case 3: liftgate genuinely requested, so the SECOND one is the only issue
        _align(book, 2, requested_accessorials=json.dumps(["LIFTGATE_DELIVERY"]))
        # case 4: detention genuinely occurred; only the hours past the cap are wrong
        _align(book, 3, detention_hours=8.0, requested_accessorials=json.dumps([]))
        # case 5: real accessorials, so the wrong fuel base clears the approve ceiling
        _align(book, 4, requested_accessorials=json.dumps(
            ["LIFTGATE_DELIVERY", "LIMITED_ACCESS", "APPOINTMENT"]))

    _align_judgment(judgment)
    _align_judgment(upload_judgment)
    con.commit()

    judgment_inv, judgment_truth = build_judgment_invoice(judgment, rng)
    jpath = db.INVOICE_DIR / f"{judgment_inv['invoice_number']}.pdf"
    if with_pdfs:
        render_invoice_pdf(judgment_inv, jpath)
    con.execute(
        "INSERT INTO invoices (invoice_number,carrier_name,invoice_date,pdf_path,"
        "stated_total,seed_payload,source) VALUES (?,?,?,?,?,?,'seed')",
        (judgment_inv["invoice_number"], judgment_inv["carrier_name"],
         judgment_inv["invoice_date"], str(jpath), judgment_inv["invoice_total"],
         json.dumps(judgment_inv)),
    )
    truth = truth + judgment_truth

    con.executemany(
        "INSERT INTO ground_truth (invoice_number,pro_number,category,note,blind_spot) "
        "VALUES (:invoice_number,:pro_number,:category,:note,:blind_spot)",
        [{**t, "blind_spot": t.get("blind_spot", 0)} for t in truth],
    )
    con.commit()

    # Holdout invoice: rendered to disk, deliberately absent from `invoices`. Its
    # shipments ARE in the book, so it passes the ownership gate the way a real
    # unprocessed invoice would.
    if with_pdfs:
        samples = db.DATA_DIR / "upload_samples"
        render_invoice_pdf(build_holdout_invoice(holdout, rng),
                           samples / "holdout_carrier_invoice.pdf")
        render_decoy_pdf(samples / "decoy_not_an_invoice.pdf")

        # A SECOND judgment invoice, never ingested, so the cases where the model
        # does the deciding can be demonstrated through the upload path rather than
        # only being visible on a pre-seeded invoice. Uploading the seeded one would
        # be refused at DEDUPE, which shows the gates working and the classifier not.
        uj, _ = build_judgment_invoice(upload_judgment, rng)
        uj["invoice_number"] = "MFS-INV-9301"
        render_invoice_pdf(uj, samples / "judgment_carrier_invoice.pdf")

    index_clauses(con)
    stats = {
        "upload_samples": str(db.DATA_DIR / "upload_samples"),
        "holdout_invoice": "MFS-INV-9200 (rendered, deliberately NOT ingested)",
        "facilities": len(FACILITIES),
        "shipments": len(shipments),
        "invoices": len(invoices),
        "invoice_lines": sum(len(i["lines"]) for i in invoices) + 6,
        "judgment_cases": "MFS-INV-9300 --- 6 lines: 5 code cannot resolve + 1 blind spot",
        "clauses": len(chunk_tariff()),
        "defects": DEFECT_PLAN,
    }
    con.close()
    return stats
