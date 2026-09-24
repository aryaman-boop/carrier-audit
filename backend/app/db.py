"""Storage.

SQLite, deliberately. The whole corpus here is one carrier contract and a few
hundred shipments, and at that size SQLite's FTS5 gives real BM25 ranking with no
infrastructure at all. The retrieval interface in `retrieve.py` is written against
two functions, so the production swap is Postgres with `pg_search` (or `tsvector`)
plus `pgvector` -- the same two indexes on the database they already run, rather
than a vector database nobody asked for.

Every table that records a decision is append-only. An audit workflow that
overwrites its own history is not an audit workflow.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "carrier_audit.db"
INVOICE_DIR = DATA_DIR / "invoices"

SCHEMA = """
CREATE TABLE IF NOT EXISTS facilities (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    zone            TEXT NOT NULL,
    location_type   TEXT NOT NULL CHECK (location_type IN ('COMMERCIAL','RESIDENTIAL','LIMITED_ACCESS')),
    dock_equipped   INTEGER NOT NULL
);

-- The internal record of what we actually tendered. This is the source of truth
-- the carrier's invoice is audited against.
CREATE TABLE IF NOT EXISTS shipments (
    id                      TEXT PRIMARY KEY,
    pro_number              TEXT NOT NULL UNIQUE,
    bol_number              TEXT NOT NULL,
    ship_date               TEXT NOT NULL,
    origin_zone             TEXT NOT NULL,
    dest_zone               TEXT NOT NULL,
    bol_weight_lb           REAL NOT NULL,
    bol_class               REAL NOT NULL,
    consignee_facility_id   TEXT NOT NULL REFERENCES facilities(id),
    requested_accessorials  TEXT NOT NULL DEFAULT '[]',
    detention_hours         REAL NOT NULL DEFAULT 0,
    appointment_required    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ship_pro  ON shipments(pro_number);
CREATE INDEX IF NOT EXISTS idx_ship_lane ON shipments(origin_zone, dest_zone, ship_date);

CREATE TABLE IF NOT EXISTS invoices (
    invoice_number  TEXT PRIMARY KEY,
    carrier_name    TEXT NOT NULL,
    invoice_date    TEXT NOT NULL,
    pdf_path        TEXT NOT NULL,
    stated_total    REAL NOT NULL,
    -- Generated invoice content. Read ONLY by --stub mode so the pipeline can be
    -- exercised without an API key. The live extraction path never touches it.
    seed_payload    TEXT NOT NULL DEFAULT '{}',
    -- 'seed' for the generated book, 'upload' for anything that came through the
    -- ingestion gates. Kept so a reviewer can always tell which is which.
    source          TEXT NOT NULL DEFAULT 'seed'
);

-- Every upload attempt, accepted or rejected, with the gate that stopped it.
-- A gate that silently drops documents is indistinguishable from a broken one,
-- so refusals are recorded as carefully as acceptances.
CREATE TABLE IF NOT EXISTS uploads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    accepted        INTEGER NOT NULL,
    rejected_at     TEXT,
    reason          TEXT,
    detail          TEXT,
    invoice_number  TEXT,
    cost_usd        REAL NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_uploads_sha ON uploads(sha256);

-- Tariff prose, chunked. The only corpus in the system that is a retrieval problem.
CREATE TABLE IF NOT EXISTS clauses (
    id          TEXT PRIMARY KEY,
    rule_id     TEXT NOT NULL,
    title       TEXT NOT NULL,
    text        TEXT NOT NULL,
    seq         INTEGER NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS clauses_fts USING fts5(
    id UNINDEXED, title, text, tokenize = 'porter unicode61'
);
CREATE TABLE IF NOT EXISTS clause_vectors (
    id   TEXT PRIMARY KEY REFERENCES clauses(id),
    dim  INTEGER NOT NULL,
    vec  BLOB NOT NULL
);

-- Output of a pipeline run. Append-only: a re-run inserts a new row.
CREATE TABLE IF NOT EXISTS audit_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    invoice_number  TEXT NOT NULL,
    pro_number      TEXT NOT NULL,
    payload         TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_audit_run ON audit_results(run_id);

-- Injected defects, for the eval harness only. Never read by the pipeline.
CREATE TABLE IF NOT EXISTS ground_truth (
    invoice_number  TEXT NOT NULL,
    pro_number      TEXT NOT NULL,
    category        TEXT NOT NULL,
    note            TEXT,
    -- A line the architecture CANNOT catch, flagged deliberately. Scored
    -- separately so a known structural gap is never mistaken for a model error,
    -- and never quietly improves the headline number either.
    blind_spot      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (invoice_number, pro_number)
);

-- What a human did. The override log is next month's eval set.
CREATE TABLE IF NOT EXISTS decisions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    invoice_number      TEXT NOT NULL,
    pro_number          TEXT NOT NULL,
    system_disposition  TEXT NOT NULL,
    system_category     TEXT NOT NULL,
    human_action        TEXT NOT NULL,
    human_category      TEXT,
    note                TEXT,
    decided_at          TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INVOICE_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


def init(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    con.commit()


def reset() -> None:
    """Drop and rebuild. Seeding is idempotent this way."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = connect()
    init(con)
    con.close()


def rows(con: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in con.execute(sql, params).fetchall()]


def one(con: sqlite3.Connection, sql: str, params: tuple = ()) -> dict[str, Any] | None:
    r = con.execute(sql, params).fetchone()
    return dict(r) if r else None


def save_audit(con: sqlite3.Connection, run_id: str, payload: dict) -> None:
    con.execute(
        "INSERT INTO audit_results (run_id, invoice_number, pro_number, payload) VALUES (?,?,?,?)",
        (run_id, payload["invoice_number"], payload["pro_number"], json.dumps(payload)),
    )
    con.commit()
