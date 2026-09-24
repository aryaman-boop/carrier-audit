"""Alerts --- turning abstentions into work you can actually do.

An `UNKNOWN` sitting in a review queue is a line a human will resolve once. The
same `UNKNOWN` appearing three times is a **rule somebody should write**, and the
difference between those two is the difference between a workflow that stays the
same and one that gets better.

So the rule here is not "page on every abstention". A single one is a reviewer's
job and interrupting the owner about it is noise. What earns an interruption is a
**pattern**: the same unmapped charge description recurring, the abstention rate
climbing, a run where extraction stopped conforming. Those are the ones where the
answer is a change to the system rather than a decision about one invoice.

Every alert carries a **suggested action**, because an alert that only describes a
problem transfers the work of deciding what to do back to the reader every single
time it fires.

Three severities, and they mean different things about your evening:

  ACTION  something recurring that a change would fix. Look at it this week.
  WATCH   a number moving in the wrong direction. Not yet a problem.
  INFO    worth knowing, no action implied.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from . import db

# A single abstention is a reviewer's job. This many of the same KIND is a rule.
RECURRENCE_THRESHOLD = 2
# Share of lines the system may decline before the taxonomy is the problem.
ABSTENTION_RATE_WATCH = 0.05
ABSTENTION_RATE_ACTION = 0.12
# Extraction must read this share of documents into a reconciling record.
CONFORMANCE_FLOOR = 0.95


@dataclass
class Alert:
    severity: str          # ACTION | WATCH | INFO
    kind: str
    title: str
    detail: str
    suggested_action: str
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _norm(description: str) -> str:
    """Normalise a charge description so wording variants group together.

    Carriers are not consistent: "Winter operations surcharge",
    "WINTER OPERATIONS SURCHARGE" and "Winter ops surcharge (Dec)" are the same
    charge. Lowercasing, stripping punctuation, digits and parenthetical suffixes
    gets most of that for free -- and this only has to be good enough to spot a
    recurrence, not to be a canonical key.
    """
    d = re.sub(r"\([^)]*\)", " ", description.lower())
    d = re.sub(r"[^a-z ]+", " ", d)
    return " ".join(d.split())


def _latest_run(con: sqlite3.Connection) -> str | None:
    r = db.one(con, "SELECT run_id FROM audit_results ORDER BY id DESC LIMIT 1")
    return r["run_id"] if r else None


def evaluate(con: sqlite3.Connection, run_id: str | None = None) -> list[Alert]:
    """Everything worth telling the owner about, most actionable first."""
    run_id = run_id or _latest_run(con)
    if not run_id:
        return []

    rows = [json.loads(r["payload"]) for r in
            db.rows(con, "SELECT payload FROM audit_results WHERE run_id=?", (run_id,))]
    if not rows:
        return []

    out: list[Alert] = []
    abstentions = [p for p in rows if p["classification"]["category"] == "UNKNOWN"]

    # ---- 1. The one that matters: a charge we cannot map, more than once -----
    # These are the recurring gaps. Each is either a rate to negotiate into the
    # contract or a charge class to dispute as a standing policy -- both are
    # decisions the owner makes once and never repeats.
    unmapped: dict[str, list[dict]] = defaultdict(list)
    for p in rows:
        for ch in p["extracted"]["charges"]:
            if ch["code"] == "OTHER":
                unmapped[_norm(ch["description"])].append(
                    {"pro": p["pro_number"], "invoice": p["invoice_number"],
                     "description": ch["description"], "amount": ch["amount"]}
                )

    for key, hits in sorted(unmapped.items(), key=lambda kv: -sum(h["amount"] for h in kv[1])):
        total = round(sum(h["amount"] for h in hits), 2)
        label = hits[0]["description"]
        if len(hits) >= RECURRENCE_THRESHOLD:
            out.append(Alert(
                "ACTION", "recurring_unmapped_charge",
                f'"{label}" billed {len(hits)} times --- ${total:,.2f} and no contracted rate',
                f"This charge has no rate in the contract, so every occurrence becomes an "
                f"abstention and a reviewer has to decide it from scratch. It is now "
                f"recurring, which makes it a policy question rather than a per-invoice one.",
                f"Decide it once: either negotiate a rate for it into the contract, or add "
                f"it to the tariff as a charge that is not assessable and let the system "
                f"dispute it automatically. Either way the next occurrence stops needing you.",
                {"occurrences": len(hits), "total_usd": total,
                 "lines": hits[:6], "normalised": key},
            ))
        else:
            out.append(Alert(
                "INFO", "new_unmapped_charge",
                f'First sighting: "{label}" (${total:,.2f})',
                "A charge the contract has no rate for. One occurrence is a reviewer's "
                "job; if it appears again it becomes an ACTION.",
                "No action yet. Watch for a second occurrence.",
                {"occurrences": 1, "total_usd": total, "lines": hits},
            ))

    # ---- 2. Abstentions that share a shape -----------------------------------
    # Several abstentions with the same charge-code signature usually mean one
    # missing rule, not several hard invoices.
    sig = Counter(
        tuple(sorted({c["code"] for c in p["extracted"]["charges"]})) for p in abstentions
    )
    for codes, n in sig.items():
        if n >= RECURRENCE_THRESHOLD:
            out.append(Alert(
                "ACTION", "abstention_cluster",
                f"{n} abstentions share the same charge shape: {', '.join(codes)}",
                "Several declined lines carry an identical combination of charge codes. "
                "That usually means one missing rule rather than several hard invoices.",
                "Look at one of them, work out the rule, and add it to the tariff or the "
                "recompute. Fixing one fixes the cluster.",
                {"charge_codes": list(codes), "count": n},
            ))

    # ---- 3. Abstention rate ---------------------------------------------------
    rate = len(abstentions) / len(rows)
    if rate >= ABSTENTION_RATE_ACTION:
        out.append(Alert(
            "ACTION", "abstention_rate_high",
            f"The system declined {rate:.1%} of lines ({len(abstentions)} of {len(rows)})",
            "A high abstention rate means the category taxonomy no longer covers what "
            "carriers are billing. Reviewers are doing classification work the system "
            "was supposed to do.",
            "Read ten abstentions together. If they share a cause, add a category; if "
            "they are genuinely varied, the invoices changed and extraction needs a look.",
            {"rate": round(rate, 4), "abstentions": len(abstentions), "lines": len(rows)},
        ))
    elif rate >= ABSTENTION_RATE_WATCH:
        out.append(Alert(
            "WATCH", "abstention_rate_rising",
            f"Abstention rate is {rate:.1%} ({len(abstentions)} of {len(rows)})",
            "Above the quiet threshold but not yet a problem. Worth a glance if it keeps "
            "climbing.",
            "No action. Check again next run.",
            {"rate": round(rate, 4)},
        ))

    # ---- 4. Extraction drift --------------------------------------------------
    # The alarm that fires before accuracy moves: a layout change breaks structure
    # long before it starts being subtly wrong.
    try:
        from . import observe
        rep = observe.report(con, limit_days=7)
        scr = rep.get("schema_conformance_rate")
        if scr is not None and scr < CONFORMANCE_FLOOR:
            out.append(Alert(
                "ACTION", "schema_conformance_drop",
                f"Extraction conformance fell to {scr:.1%}",
                "Documents are failing to read into a complete, reconciling record. This "
                "degrades BEFORE accuracy does --- it is usually a carrier changing their "
                "invoice layout.",
                "Open a blocked invoice and compare it to the parser's expectations. If "
                "the layout changed, the deterministic rung needs updating or that carrier "
                "moves to the vision path.",
                {"conformance": scr, "floor": CONFORMANCE_FLOOR},
            ))
    except Exception:
        pass

    # ---- 5. Money sitting in the abstention queue -----------------------------
    at_stake = round(sum(p["recomputation"]["variance"] for p in abstentions
                         if p["recomputation"]["variance"] > 0), 2)
    if abstentions:
        out.append(Alert(
            "INFO", "abstention_value",
            f"${at_stake:,.2f} is sitting in the abstention queue",
            f"{len(abstentions)} line(s) the system declined to classify, carrying real "
            f"money. They are escalated and a human will see them, but nobody has decided "
            f"them yet.",
            "Work the abstention queue before the auto-approved one. It is where both the "
            "unrecovered money and the next improvement live.",
            {"amount_usd": at_stake, "lines": len(abstentions)},
        ))

    order = {"ACTION": 0, "WATCH": 1, "INFO": 2}
    return sorted(out, key=lambda a: (order[a.severity], -len(a.evidence)))


def summary(con: sqlite3.Connection, run_id: str | None = None) -> dict:
    alerts = evaluate(con, run_id)
    counts = Counter(a.severity for a in alerts)
    return {
        "run_id": run_id or _latest_run(con),
        "action": counts.get("ACTION", 0),
        "watch": counts.get("WATCH", 0),
        "info": counts.get("INFO", 0),
        "alerts": [a.as_dict() for a in alerts],
    }
