"""Test isolation.

**The defect this fixes:** the tests called `db.connect()`, which resolves
`db.DB_PATH` at call time and therefore opened the *real* database. Every
`pytest` run appended four rows-worth of audit runs to production data. After a
day's development the run history held sixty-four runs, fifty-five of which were
test artifacts, and the review UI — which shows the most recent run — was
displaying whichever fake-client run pytest had written last.

That is worse than untidy. The one table whose whole purpose is to be an
append-only record of what the system decided had been quietly filled with
fiction, and the failure was invisible until someone opened the UI and wondered
why every line said UNKNOWN.

The fix is the ordinary one: point the storage layer at a throwaway directory for
the duration of the session, seed it once, and put it back afterwards. Tests get a
real, fully-seeded database — clauses indexed, invoices rendered to PDF, ground
truth present — and production data is never touched.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import db, seed  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def isolated_database():
    """Redirect all storage into a temporary directory for the whole session.

    Session-scoped because seeding renders eight invoice PDFs and indexing the
    tariff costs a second or so; doing that per test would be wasteful and would
    not catch anything extra.
    """
    original = (db.DATA_DIR, db.DB_PATH, db.INVOICE_DIR)

    with tempfile.TemporaryDirectory(prefix="carrier-audit-tests-") as tmp:
        root = Path(tmp)
        db.DATA_DIR = root
        db.DB_PATH = root / "carrier_audit.db"
        db.INVOICE_DIR = root / "invoices"

        # A real seed, not a stub: the pipeline tests read actual PDFs off disk
        # and the retrieval tests need the tariff genuinely indexed.
        seed.seed_all(with_pdfs=True)

        try:
            yield root
        finally:
            db.DATA_DIR, db.DB_PATH, db.INVOICE_DIR = original


@pytest.fixture(autouse=True)
def _assert_not_touching_production():
    """Belt and braces: fail loudly if anything re-points at the real database.

    A guard rather than a test. If a future change reintroduces the bug, this
    turns a silent data-pollution problem back into a red test.
    """
    yield
    assert "carrier-audit-tests-" in str(db.DB_PATH), (
        f"tests are writing to {db.DB_PATH}, which is not an isolated temporary "
        f"database. The session fixture in conftest.py has been bypassed."
    )
