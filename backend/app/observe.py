"""Observability.

**First, a correction to the question this usually gets asked as.** This is not an
agentic system and instrumenting it as one would be the wrong shape. An agent needs
a *trace tree* -- spans nested per tool call, per loop iteration, with the branching
recorded, because the control flow is decided at runtime and you cannot know in
advance what happened. A workflow has a fixed sequence of named stages, so what it
needs is *stage-level telemetry*: how long each stage took, what it decided, what it
cost, and how those distributions are moving. That is a far smaller problem, and
getting to keep it small is one of the payoffs of not building an agent.

Four things are recorded, and each exists because something specific would
otherwise fail silently:

  1. **A trace id per document.** Every stage a document passes through shares one
     id, so "what happened to invoice 9102" is one query rather than a hunt. Without
     it, a failure is visible in aggregate and untraceable in particular.

  2. **Per-stage duration and outcome.** Latency regressions and rising failure
     rates both show up here first. The extraction stage additionally records
     *which rung of the ladder* answered, which is the number that says whether the
     free path is still carrying the load or whether carriers have started sending
     scans.

  3. **Schema-conformance rate.** The share of documents where extraction produced a
     structurally complete record that reconciled arithmetically. `METRICS.md` calls
     this the drift alarm and the reason is that **it degrades before accuracy
     does**: when a carrier changes their invoice layout, extraction starts failing
     structurally long before it starts being subtly wrong. Watching accuracy alone
     means finding out a month late from a reviewer.

  4. **Cost and tokens per stage.** Already collected by `llm.Usage`; persisted here
     so it can be trended rather than read once at the end of a run.

**What is deliberately not here:** no vendor tracing SDK, no OpenTelemetry, no
spans. At this size those would be ceremony. The honest production answer is that
this table is what you would replace with Langfuse, Braintrust or Phoenix on day
one -- and knowing what you would ask of them is the part that transfers.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager

from . import db

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id     TEXT NOT NULL,
    run_id       TEXT,
    subject      TEXT,              -- invoice number or PRO, whatever the stage is about
    stage        TEXT NOT NULL,
    outcome      TEXT NOT NULL,     -- ok | rejected | error
    detail       TEXT,
    method       TEXT,              -- for extract: text-layer | vision | stub
    duration_ms  REAL NOT NULL,
    cost_usd     REAL NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_traces_trace ON traces(trace_id);
CREATE INDEX IF NOT EXISTS idx_traces_stage ON traces(stage, created_at);
"""


def init(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    con.commit()


def new_trace() -> str:
    return f"t-{uuid.uuid4().hex[:12]}"


@contextmanager
def stage(con: sqlite3.Connection, trace_id: str, name: str, *,
          subject: str | None = None, run_id: str | None = None):
    """Time one stage and record how it ended.

    Yields a dict the caller mutates to attach outcome, method, cost or detail.
    An exception is recorded as `error` and re-raised -- instrumentation must never
    swallow a failure, because a monitoring layer that hides errors is worse than
    none.
    """
    rec = {"outcome": "ok", "detail": None, "method": None, "cost_usd": 0.0}
    t0 = time.perf_counter()
    try:
        yield rec
    except Exception as e:
        rec["outcome"] = "error"
        rec["detail"] = f"{type(e).__name__}: {e}"[:400]
        raise
    finally:
        try:
            init(con)
            con.execute(
                "INSERT INTO traces (trace_id, run_id, subject, stage, outcome, detail, "
                "method, duration_ms, cost_usd) VALUES (?,?,?,?,?,?,?,?,?)",
                (trace_id, run_id, subject, name, rec["outcome"], rec["detail"],
                 rec["method"], (time.perf_counter() - t0) * 1000, rec["cost_usd"]),
            )
            con.commit()
        except Exception:
            pass          # telemetry must never break the pipeline it observes


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(int(q * len(s)), len(s) - 1)
    return round(s[i], 1)


def report(con: sqlite3.Connection, limit_days: int = 30) -> dict:
    """What an on-call engineer would want on one screen."""
    init(con)
    rows = db.rows(
        con,
        "SELECT * FROM traces WHERE created_at >= datetime('now', ?) ORDER BY id",
        (f"-{limit_days} days",),
    )
    if not rows:
        return {"traces": 0, "note": "No telemetry recorded yet. Run the pipeline."}

    stages: dict[str, dict] = {}
    for r in rows:
        st = stages.setdefault(r["stage"], {"n": 0, "ok": 0, "rejected": 0,
                                            "error": 0, "ms": [], "cost": 0.0})
        st["n"] += 1
        st[r["outcome"]] = st.get(r["outcome"], 0) + 1
        st["ms"].append(r["duration_ms"])
        st["cost"] += r["cost_usd"] or 0

    out_stages = {
        name: {
            "calls": s["n"],
            "ok": s["ok"], "rejected": s["rejected"], "error": s["error"],
            "error_rate": round(s["error"] / s["n"], 4) if s["n"] else 0,
            "p50_ms": _pct(s["ms"], 0.50),
            "p95_ms": _pct(s["ms"], 0.95),
            "cost_usd": round(s["cost"], 5),
        }
        for name, s in stages.items()
    }

    # Which rung of the extraction ladder answered. The number to watch: if
    # text-layer share falls, carriers have started sending scans and the bill
    # is about to rise.
    methods: dict[str, int] = {}
    for r in rows:
        if r["stage"] == "extract" and r["method"]:
            methods[r["method"]] = methods.get(r["method"], 0) + 1
    total_ext = sum(methods.values())

    # Schema conformance: extraction produced a structurally complete, arithmetically
    # reconciling record. Degrades BEFORE accuracy does when a layout changes.
    ext = [r for r in rows if r["stage"] == "extract"]
    conforming = sum(1 for r in ext if r["outcome"] == "ok")

    return {
        "window_days": limit_days,
        "traces": len({r["trace_id"] for r in rows}),
        "events": len(rows),
        "by_stage": out_stages,
        "extraction_method_mix": {
            k: {"n": v, "share": round(v / total_ext, 4)} for k, v in methods.items()
        } if total_ext else {},
        "schema_conformance_rate": round(conforming / len(ext), 4) if ext else None,
        "schema_conformance_note": (
            "Share of documents extraction read into a complete, reconciling record. "
            "This is the drift alarm: it degrades before accuracy does when a carrier "
            "changes their invoice layout."
        ),
        "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 5),
    }


def trace_detail(con: sqlite3.Connection, trace_id: str) -> list[dict]:
    """Every stage one document passed through, in order."""
    init(con)
    return db.rows(con, "SELECT * FROM traces WHERE trace_id=? ORDER BY id", (trace_id,))
