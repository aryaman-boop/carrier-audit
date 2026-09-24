"""The carrier contract, in its two halves.

The split in this file is the whole architectural argument of the project.

  STRUCTURED half -- rate tables, weight breaks, class factors, the weekly fuel
  index, accessorial prices. These are numbers with keys. Looking one up is a
  join. There is no retrieval problem here and no reason for a model to touch it.

  PROSE half -- TARIFF_RULES below. When a liftgate charge is valid, how much
  free time precedes detention, what counts as limited access, how a reweigh
  tolerance works. These are genuinely unstructured, they are what a human
  auditor actually argues with the carrier about, and they are the only part of
  the contract that is a retrieval problem.

The pricing engine reads the first half. The retriever indexes the second half.
Keeping them apart is what stops this from becoming a chatbot over a database.

All figures are synthetic. `Meridian Freight Systems` is not a real carrier.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Structured: the contracted rate table
# --------------------------------------------------------------------------

CARRIER_NAME = "Meridian Freight Systems"
CONTRACT_ID = "MFS-NWL-2026-A"
CONTRACT_EFFECTIVE = "2026-01-01"

# Weight breaks, in pounds. LTL rates fall as weight rises.
WEIGHT_BREAKS = [
    ("L5C", 0, 499),        # less than 500 lb
    ("M5C", 500, 999),      # minimum 500
    ("M1M", 1000, 1999),
    ("M2M", 2000, 4999),
    ("M5M", 5000, 9999),
    ("M10M", 10000, 99999),
]

# Lane base rates in dollars per hundredweight (CWT), keyed by
# (origin zone, destination zone, weight break). Zones are ZIP prefixes.
LANE_RATES_CWT = {
    ("L6T", "M5V"): {"L5C": 41.20, "M5C": 32.75, "M1M": 26.40, "M2M": 21.85, "M5M": 17.30, "M10M": 14.05},
    ("L6T", "N2L"): {"L5C": 38.60, "M5C": 30.10, "M1M": 24.20, "M2M": 19.95, "M5M": 15.85, "M10M": 12.90},
    ("L6T", "K1A"): {"L5C": 47.90, "M5C": 38.45, "M1M": 31.10, "M2M": 25.60, "M5M": 20.40, "M10M": 16.55},
    ("L6T", "H3B"): {"L5C": 52.30, "M5C": 42.15, "M1M": 34.05, "M2M": 28.10, "M5M": 22.35, "M10M": 18.20},
    ("L6T", "V6B"): {"L5C": 71.80, "M5C": 58.40, "M1M": 47.25, "M2M": 39.00, "M5M": 31.10, "M10M": 25.35},
    ("L6T", "T2P"): {"L5C": 64.50, "M5C": 52.30, "M1M": 42.30, "M2M": 34.90, "M5M": 27.80, "M10M": 22.65},
    ("L6T", "B3J"): {"L5C": 68.20, "M5C": 55.35, "M1M": 44.80, "M2M": 36.95, "M5M": 29.45, "M10M": 24.00},
}

# Class factor. The rate table is quoted at class 70; other classes scale.
CLASS_FACTORS = {
    50.0: 0.78, 55.0: 0.84, 60.0: 0.89, 65.0: 0.94, 70.0: 1.00, 77.5: 1.09,
    85.0: 1.18, 92.5: 1.27, 100.0: 1.38, 110.0: 1.52, 125.0: 1.71, 150.0: 2.04,
}

# Absolute minimum charge per shipment, before fuel and accessorials.
MINIMUM_CHARGE = 118.50

# Weekly fuel surcharge as a percentage of linehaul, indexed to the published
# diesel average. Effective-dated: the week the shipment MOVED governs, not the
# week it was invoiced. Getting this wrong is a real and common billing error,
# which is why FUEL_SURCHARGE_ERROR is its own variance category.
FUEL_SURCHARGE_BY_WEEK = {
    "2026-06-01": 0.2350, "2026-06-08": 0.2375, "2026-06-15": 0.2410,
    "2026-06-22": 0.2395, "2026-06-29": 0.2440, "2026-07-06": 0.2485,
    "2026-07-13": 0.2520, "2026-07-20": 0.2495, "2026-07-27": 0.2460,
    "2026-08-03": 0.2425, "2026-08-10": 0.2450, "2026-08-17": 0.2505,
    "2026-08-24": 0.2540, "2026-08-31": 0.2575,
}

# Contracted accessorial prices.
ACCESSORIAL_PRICES = {
    "LIFTGATE_DELIVERY": 92.00,
    "RESIDENTIAL_DELIVERY": 76.50,
    "INSIDE_DELIVERY": 88.00,
    "LIMITED_ACCESS": 105.00,
    "APPOINTMENT": 42.00,
    "DETENTION": 65.00,          # per hour after free time
    "REWEIGH_FEE": 38.00,
    "RECLASSIFICATION_FEE": 45.00,
    "REDELIVERY": 118.00,
    "LUMPER": 150.00,
}

# --------------------------------------------------------------------------
# Prose: the rules half. This is what gets chunked and indexed.
# --------------------------------------------------------------------------

TARIFF_RULES = [
    (
        "MFS-100",
        "Application of Rates and Minimum Charge",
        """Rates published in the contracted rate table apply to shipments moving between the
        origin and destination postal zones shown, subject to the weight break applicable to
        the actual gross weight tendered. Where the computed linehaul charge for a shipment
        falls below the absolute minimum charge stated in the contract, the minimum charge
        applies in place of the computed charge. The minimum charge is assessed before fuel
        surcharge and before any accessorial service charge; fuel surcharge is then computed
        on the minimum charge as though it were linehaul. Rates are quoted at NMFC class 70
        and are adjusted by the published class factor for shipments of any other class.""",
    ),
    (
        "MFS-110",
        "Weight Determination, Reweigh and Reweigh Tolerance",
        """The weight shown on the bill of lading governs the charge unless the carrier
        reweighs the shipment. Carrier reserves the right to reweigh any shipment. Where a
        reweigh is performed and the reweighed gross weight differs from the bill of lading
        weight by MORE THAN two percent (2%) OR by more than fifty (50) pounds, whichever is
        GREATER, the reweighed weight shall govern and charges shall be adjusted accordingly.
        Where the difference falls at or within that tolerance, the bill of lading weight
        governs and no adjustment shall be made. A reweigh fee shall be assessed ONLY where
        the reweigh results in a valid upward adjustment under this rule. A reweigh that
        confirms the bill of lading weight, or that falls within tolerance, is performed at
        carrier expense and no reweigh fee is due. Carrier must furnish the certified weight
        certificate upon shipper request within ten (10) business days; failure to furnish
        voids the adjustment.""",
    ),
    (
        "MFS-120",
        "Freight Classification and Reclassification",
        """Freight is classified in accordance with the National Motor Freight Classification
        in effect on the date of shipment. Where carrier inspects a shipment and determines
        that the class shown on the bill of lading is incorrect, carrier may reclassify the
        shipment and rerate the charges. A reclassification is valid ONLY where carrier
        furnishes an inspection certificate stating the density, dimensions and commodity
        description observed. Absent that certificate the class shown on the bill of lading
        governs. A reclassification fee shall be assessed only in connection with a valid
        upward reclassification supported by an inspection certificate. Reclassification of a
        shipment whose bill of lading class was furnished from the shipper's published
        product master and whose dimensions are unchanged shall not be assessed.""",
    ),
    (
        "MFS-130",
        "Fuel Surcharge",
        """A fuel surcharge is assessed as a percentage of the linehaul charge, including any
        applicable minimum charge, and is published weekly. The surcharge percentage in
        effect for the week in which the shipment was PICKED UP shall apply. The week of
        invoicing, the week of delivery and the week of rating shall not govern. Weeks
        commence on Monday. Fuel surcharge does not apply to accessorial service charges,
        detention, or to any charge assessed under the reweigh or reclassification rules.""",
    ),
    (
        "MFS-200",
        "Liftgate Service at Delivery",
        """Liftgate service is assessed where carrier supplies a power liftgate to lower
        freight from the vehicle at a location having no loading dock and no forklift
        available to receive the shipment. Liftgate service shall be assessed ONLY where it
        has been requested on the bill of lading in advance, OR where the consignee location
        is confirmed at delivery to lack dock or mechanical unloading capability and carrier
        documents that condition on the delivery receipt. A liftgate charge assessed to a
        consignee location recorded as dock-equipped in the shipper's facility master, and
        not requested on the bill of lading, is not valid and shall be removed on request.
        Only one liftgate charge may be assessed per delivery regardless of piece count.""",
    ),
    (
        "MFS-210",
        "Residential and Limited Access Delivery",
        """A residential delivery charge applies where the delivery address is a private
        dwelling, including a dwelling at which a business is operated. A limited access
        charge applies where delivery is made to a location that restricts carrier access by
        appointment requirement, escort requirement, security clearance, or restricted hours,
        including but not limited to military installations, correctional facilities, mines,
        schools, places of worship, and construction sites. Residential delivery and limited
        access shall not both be assessed on the same shipment; where both conditions are
        documented the higher of the two charges applies and the lower is waived. A location
        classified as commercial in the shipper's facility master shall not be assessed a
        residential delivery charge absent documentation on the delivery receipt.""",
    ),
    (
        "MFS-220",
        "Detention With Vehicle and Free Time",
        """Detention is assessed where carrier's vehicle and driver are delayed at the
        shipper or consignee premises beyond the free time allowed. Free time is TWO (2)
        HOURS at each of origin and destination, computed from the driver's arrival as
        recorded on the delivery receipt or the appointment time, whichever is LATER.
        Detention is assessed in half-hour increments after free time expires, at the rate
        stated in the contract, and shall not exceed eight (8) hours in any single day.
        Detention shall not be assessed where the delay is caused by carrier's arrival
        outside a scheduled appointment window. Fuel surcharge does not apply to detention.""",
    ),
    (
        "MFS-230",
        "Appointment, Notification and Redelivery",
        """Where the consignee requires an appointment prior to delivery, an appointment
        charge is assessed once per shipment. Redelivery is assessed where carrier attempts
        delivery, is unable to complete it for reasons attributable to the consignee, and
        returns on a subsequent day. Redelivery shall not be assessed where the first
        attempt was made outside an agreed appointment window, nor where the shipment was
        tendered to the wrong address by carrier. An appointment charge and a redelivery
        charge arising from carrier's own failure to schedule the appointment shall not both
        be assessed.""",
    ),
    (
        "MFS-300",
        "Duplicate Billing and Corrected Invoices",
        """A shipment identified by a single PRO number shall be invoiced once. Where a
        corrected invoice is issued, it shall reference the original invoice number and shall
        state the amount of the correction; a corrected invoice that does not reference the
        original is treated as a duplicate. Payment of a duplicate invoice does not
        constitute acceptance of the charge. Duplicate billing identified within the dispute
        window shall be credited in full without proration.""",
    ),
    (
        "MFS-400",
        "Billing Disputes, Dispute Window and Supporting Documentation",
        """Shipper may dispute any charge within one hundred and eighty (180) days of the
        invoice date. Carrier shall respond to a dispute within thirty (30) days. Carrier
        bears the burden of furnishing supporting documentation for any charge assessed
        above the rated amount, including certified weight certificates for reweighs,
        inspection certificates for reclassifications, and signed delivery receipts
        documenting accessorial conditions. Where carrier fails to furnish supporting
        documentation within thirty (30) days of a written dispute, the disputed charge
        shall be credited in full. Disputed amounts may be withheld from payment pending
        resolution without incurring a late payment charge.""",
    ),
]
