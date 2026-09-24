"""The pricing engine. Pure arithmetic over the contracted rate card.

No model is invoked anywhere in this file, and nothing in this file is allowed to
depend on anything a model produced. This is the deterministic core: the same
shipment always prices to the same number, which is what makes the output usable
as audit evidence rather than as a suggestion.

If you remember one sentence from this project: *the model reads the invoice, it
does not price it.* This is the half that prices it.
"""

from __future__ import annotations

from datetime import date, timedelta

from .ratecard import (
    ACCESSORIAL_PRICES,
    CLASS_FACTORS,
    FUEL_SURCHARGE_BY_WEEK,
    LANE_RATES_CWT,
    MINIMUM_CHARGE,
    WEIGHT_BREAKS,
)


def weight_break(weight_lb: float) -> str:
    """Return the rate-table weight break for a gross weight."""
    for name, lo, hi in WEIGHT_BREAKS:
        if lo <= weight_lb <= hi:
            return name
    return WEIGHT_BREAKS[-1][0]


def fuel_week(ship_date: date) -> str:
    """The Monday of the week the shipment was PICKED UP.

    Tariff rule MFS-130: the pickup week governs, not the invoice week. This
    one-line function is the entire difference between a correct fuel surcharge
    and the FUEL_SURCHARGE_ERROR category.
    """
    monday = ship_date - timedelta(days=ship_date.weekday())
    return monday.isoformat()


def fuel_pct(ship_date: date) -> float | None:
    """Published fuel surcharge percentage for the pickup week, or None if the
    week is outside the published index."""
    return FUEL_SURCHARGE_BY_WEEK.get(fuel_week(ship_date))


def linehaul(
    origin_zone: str, dest_zone: str, weight_lb: float, freight_class: float
) -> tuple[float, str]:
    """Rated linehaul charge and a human-readable derivation of it.

    The basis string is not decoration. It goes in front of a reviewer who has 20
    seconds to decide whether to dispute, and it is what lets them check the
    system's arithmetic without opening the rate card.
    """
    lane = LANE_RATES_CWT.get((origin_zone, dest_zone))
    if lane is None:
        raise KeyError(f"No contracted rate for lane {origin_zone} -> {dest_zone}")

    brk = weight_break(weight_lb)
    rate_cwt = lane[brk]
    factor = CLASS_FACTORS.get(freight_class)
    if factor is None:
        raise KeyError(f"Class {freight_class} is not in the contracted class factor table")

    cwt = weight_lb / 100.0
    computed = round(cwt * rate_cwt * factor, 2)

    if computed < MINIMUM_CHARGE:
        basis = (
            f"{weight_lb:,.0f} lb @ {brk} ${rate_cwt:.2f}/cwt x class-{freight_class:g} "
            f"factor {factor:.2f} = ${computed:,.2f}, below minimum -> minimum charge applied"
        )
        return MINIMUM_CHARGE, basis

    basis = (
        f"{weight_lb:,.0f} lb @ {brk} ${rate_cwt:.2f}/cwt "
        f"x class-{freight_class:g} factor {factor:.2f}"
    )
    return computed, basis


def accessorial_price(code: str, units: float = 1.0) -> float:
    """Contracted price for an accessorial. `units` carries detention hours."""
    base = ACCESSORIAL_PRICES.get(code)
    if base is None:
        raise KeyError(f"{code} is not a contracted accessorial")
    return round(base * units, 2)


def reweigh_is_valid(bol_weight: float, reweighed: float) -> bool:
    """Tariff rule MFS-110: a reweigh governs only outside the tolerance band.

    The band is 2% OR 50 lb, *whichever is greater* -- and that "whichever is
    greater" is the part that gets read wrong. On a 600 lb shipment the 50 lb
    floor is what binds; on a 9,000 lb shipment the 2% is. Implementing it as
    max() rather than min() is the difference between a valid dispute and an
    embarrassing one.
    """
    delta = abs(reweighed - bol_weight)
    tolerance = max(bol_weight * 0.02, 50.0)
    return delta > tolerance
