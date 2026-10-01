"""Serve the API and the built UI against a throwaway database, for browser tests.

    python e2e_server.py [--port 8765]

The Playwright suite in `frontend/e2e/` starts this. It exists for the same
reason `tests/conftest.py` does: a test that opens the real database writes
fiction into an append-only audit record (that bug happened once already). So
storage is redirected to a temporary directory, seeded from scratch, audited
once in stub mode. The directory is fixed per port and wiped at startup rather
than at exit, because Playwright stops the server by killing it and an exit hook
never runs; this way a killed run leaves at most one directory behind, and the
next run clears it.

It is also fully offline. `llm.go_offline()` refuses every model call for the
life of the process, so a browser run costs nothing and is deterministic even on
a machine with ANTHROPIC_API_KEY set. The consequence is visible in the tests
and is intended: with no model, an uploaded PDF is refused at TRIAGE, so the
upload specs exercise only the gates that are free by design.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from app import db, llm, pipeline, seed


def prepare(root: Path) -> str:
    db.DATA_DIR = root
    db.DB_PATH = root / "carrier_audit.db"
    db.INVOICE_DIR = root / "invoices"
    llm.go_offline()

    seed.seed_all(with_pdfs=True)
    con = db.connect()
    try:
        summary = pipeline.run(con, stub=True, strategy="hybrid", verbose=False)
    finally:
        con.close()
    return summary.get("run_id", "?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    static = Path(__file__).resolve().parent / "static" / "index.html"
    if not static.exists():
        raise SystemExit("No UI build. Run: cd frontend && npm run build")

    root = Path(tempfile.gettempdir()) / f"carrier-audit-e2e-{args.port}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    run_id = prepare(root)
    print(f"e2e: isolated data in {root}, stub run {run_id}, model calls disabled",
          flush=True)

    import uvicorn
    from app.api import app

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
