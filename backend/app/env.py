"""Load a `.env` file, if there is one.

Twelve lines instead of a dependency, and the semantics are the ones that matter
for a credential:

  - **A real environment variable always wins.** `export ANTHROPIC_API_KEY=...`
    overrides the file, never the other way round. A file that can silently replace
    an exported credential is a debugging nightmare and a security footgun -- you
    would have no way to temporarily use a different key.
  - **Searched upward from this package**, so it works whether you run from the
    project root, from `backend/`, or from an editor with some other working
    directory. Nothing is more annoying than a key that loads from one shell and
    not another.
  - **Never logged, never echoed.** `describe()` reports only whether a value is
    present and its last four characters, which is enough to tell two keys apart
    and not enough to use one.
"""

from __future__ import annotations

import os
from pathlib import Path

CANDIDATE_NAMES = (".env", ".env.local")


def find_env_files() -> list[Path]:
    """Every .env from this package upward to the filesystem root, nearest last.

    Nearest last so a file closer to the project wins over one further away.
    """
    found: list[Path] = []
    here = Path(__file__).resolve().parent
    for directory in [*reversed(here.parents), here]:
        for name in CANDIDATE_NAMES:
            candidate = directory / name
            if candidate.is_file():
                found.append(candidate)
    return found


def _parse(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        # Strip one matching pair of quotes; a key pasted with quotes is common.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            out[key] = value
    return out


def collect() -> tuple[dict[str, str], dict[str, Path], list[Path]]:
    """Merge every .env found, nearest winning.

    Two precedence rules, and they are not the same rule:

      1. Among .env FILES, the one nearest the package wins. `backend/.env`
         overrides a stale `carrier-audit/.env`.
      2. A REAL environment variable beats every file, always.

    Getting (1) backwards is a genuinely nasty bug -- it silently ignores the file
    you just edited in favour of one you forgot about, and the only symptom is an
    authentication error that points at the wrong thing. It cost a debugging round
    here, which is why the winning source is now tracked and reported.
    """
    merged: dict[str, str] = {}
    source: dict[str, Path] = {}
    files = find_env_files()          # farthest first, nearest last
    for path in files:
        try:
            values = _parse(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        for key, value in values.items():
            # A placeholder left in from .env.example is not a credential.
            if not value or value.endswith("..."):
                continue
            merged[key] = value       # nearest file, applied last, wins
            source[key] = path
    return merged, source, files


def load(override: bool = False) -> list[Path]:
    """Populate os.environ from any .env found. Returns the files applied."""
    merged, _source, files = collect()
    for key, value in merged.items():
        if override or not os.environ.get(key):
            os.environ[key] = value
    return files


def provenance(key: str) -> str:
    """Where the value in os.environ for `key` actually came from."""
    merged, source, _ = collect()
    current = os.environ.get(key)
    if current is None:
        return "not set"
    if key in merged and merged[key] == current:
        return f"from {source[key]}"
    return "from the shell environment (overrides every .env file)"


def describe() -> dict[str, str]:
    """Redacted view of the credentials in play, for diagnostics."""
    out = {}
    for key in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY"):
        value = os.environ.get(key)
        out[key] = f"set (...{value[-4:]})" if value else "not set"
    return out
