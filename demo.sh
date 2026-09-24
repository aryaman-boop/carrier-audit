#!/usr/bin/env bash
#
# Start the demo.
#
#   ./demo.sh            verify and serve. Fast, free, no network. USE THIS.
#   ./demo.sh --fresh    rebuild everything from scratch (~3 min, ~$0.45)
#
# The default deliberately does NOT reseed. The first version of this script
# reseeded on every run, which wipes the audit run and the eval report and then
# spends three minutes and real money rebuilding them. Run that five minutes
# before a call on bad wifi and there is no demo -- the one job this script has.
#
# So: the default verifies a good state exists and serves it. Rebuilding is
# something you ask for, on purpose, with time to spare.
set -euo pipefail
cd "$(dirname "$0")"

FRESH=0
[ "${1:-}" = "--fresh" ] && FRESH=1

# ---- dependencies (idempotent; no network if already present) ---------------
if [ ! -d .venv ]; then
  echo "==> creating virtualenv"
  python3 -m venv .venv
  ./.venv/bin/pip -q install -r requirements.txt
fi

if [ ! -f backend/static/index.html ]; then
  echo "==> building the UI"
  ( cd frontend && npm install --silent && npm run build )
fi

cd backend

# ---- rebuild only when asked ------------------------------------------------
if [ "$FRESH" = "1" ]; then
  echo "==> reseeding (this wipes the current audit run and eval report)"
  ../.venv/bin/python -m app.cli seed
  echo "==> running the eval (~2.5 min, ~\$0.45)"
  ../.venv/bin/python eval/run_eval.py --max-spend 1.50
fi

# ---- always: make the upload demo repeatable --------------------------------
# The gates dedupe on content hash, so a rehearsal makes the live run show
# DEDUPE instead of the six-gate accept. Clearing uploads costs nothing.
../.venv/bin/python -m app.cli reset-demo

# ---- always: verify before serving ------------------------------------------
echo
if ! ../.venv/bin/python -m app.cli preflight; then
  echo
  echo "Preflight failed. Fix the FAIL lines above, or run:  ./demo.sh --fresh"
  exit 1
fi

# ---- free the port, then serve ----------------------------------------------
if lsof -ti tcp:8000 >/dev/null 2>&1; then
  echo "==> port 8000 is busy; stopping the old server"
  lsof -ti tcp:8000 | xargs kill 2>/dev/null || true
  sleep 1
fi

cat <<'BANNER'

  ---------------------------------------------------------------
   Review queue      http://127.0.0.1:8000
   Operations tab    top right -- alerts, stage latency, ladder mix

   Demo order
     1  the eval report                the numbers
     2  a NEEDS_REVIEW line            billed vs recomputed, clause quoted
     3  the abstained chip             where the system declined
     4  upload decoy_not_an_invoice    refused at TRIAGE, $0.00
     5  upload judgment_carrier_...    5 of 6 lines decided by the model
     6  Operations tab                 the ACTION alert

   Samples: backend/data/upload_samples/
   Stop: Ctrl-C
  ---------------------------------------------------------------

BANNER

exec ../.venv/bin/python -m app.cli serve --port 8000
