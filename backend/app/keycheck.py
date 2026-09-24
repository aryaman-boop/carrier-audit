"""Diagnose a credential without ever revealing it.

A 401 tells you the key is wrong and nothing else. Almost every 401 on a
freshly-pasted key is one of a small number of shapes, and all of them are
detectable locally, for free, before another request is spent:

  - truncated by a partial copy (an Anthropic key is ~100+ characters)
  - a stray space, tab, or newline inside the value
  - smart quotes or a non-ASCII character from a rich-text paste
  - the wrong credential entirely: an OAuth token, an Admin key, or a key copied
    from a different provider
  - the placeholder from `.env.example`, never edited

So the failure mode to avoid is guessing. This reports the *shape* of what was
loaded -- lengths, prefixes, character classes -- and never the value.
"""

from __future__ import annotations

import os
import re
import unicodedata

EXPECTED_PREFIX = "sk-ant-"
# The full format prefix, e.g. "sk-ant-api03-". Checking only "sk-ant-" is too
# loose: it accepts "sk-ant-ap103-", which is what an `i` mistyped as a `1` looks
# like, and that is a real 401 this check exists to catch. The version digits are
# matched as a pattern so a future "api04" is not falsely flagged.
FORMAT_RE = re.compile(r"^sk-ant-api\d{2}-")
# Anthropic keys are long. Anything much under this is a partial paste.
MIN_PLAUSIBLE_LEN = 90


def inspect(value: str | None, name: str = "ANTHROPIC_API_KEY") -> list[str]:
    """Return a list of findings, most important first. Empty means it looks fine."""
    out: list[str] = []
    if not value:
        return [f"{name} is not set at all."]

    if value != value.strip():
        out.append(
            "VALUE HAS LEADING OR TRAILING WHITESPACE. This is the most common cause "
            "of a 401 on a pasted key -- it is sent verbatim, spaces included."
        )

    stripped = value.strip()

    if re.search(r"\s", stripped):
        out.append(
            "VALUE CONTAINS WHITESPACE IN THE MIDDLE. A key wrapped across two lines "
            "by the terminal or an editor gets pasted with a space or newline in it."
        )

    if not stripped.isascii():
        bad = {c: unicodedata.name(c, "?") for c in stripped if not c.isascii()}
        out.append(
            f"VALUE CONTAINS NON-ASCII CHARACTERS: {bad}. Usually smart quotes or a "
            f"dash substituted by a rich-text editor. Retype or paste as plain text."
        )

    if stripped[:1] in "\"'" or stripped[-1:] in "\"'":
        out.append("VALUE STILL HAS QUOTE CHARACTERS in it after parsing.")

    if stripped.endswith("..."):
        out.append("VALUE IS THE .env.example PLACEHOLDER, not a real key.")

    if stripped.startswith(EXPECTED_PREFIX) and not FORMAT_RE.match(stripped) \
            and not stripped.startswith("sk-ant-admin"):
        head = stripped[:13]
        hint = ""
        # The classic transcription failures, in the segment that should read "api".
        segment = stripped[7:10]
        if segment in ("ap1", "ap!", "apl", "4pi", "ep1"):
            hint = (" The character after 'ap' should be the LETTER 'i', not the digit "
                    "'1' -- this is a transcription error, so the key was typed or read "
                    "from an image rather than pasted.")
        out.append(
            f"PREFIX IS {head!r}. An Anthropic API key begins 'sk-ant-api' followed by "
            f"two version digits, e.g. 'sk-ant-api03-'.{hint} Re-copy the key with the "
            f"Copy button in the console."
        )
    elif not stripped.startswith(EXPECTED_PREFIX):
        head = stripped[:8]
        if stripped.startswith("sk-"):
            out.append(
                f"PREFIX IS {head!r}, not {EXPECTED_PREFIX!r}. That looks like a key "
                f"for a different provider (OpenAI keys start 'sk-proj-' or 'sk-')."
            )
        else:
            out.append(
                f"PREFIX IS {head!r}, not {EXPECTED_PREFIX!r}. An Anthropic API key "
                f"begins with 'sk-ant-'. An OAuth token or an Admin key "
                f"('sk-ant-admin') will not work for the Messages API."
            )
    elif stripped.startswith("sk-ant-admin"):
        out.append(
            "THIS IS AN ADMIN KEY. Admin keys work only on the Admin API and are "
            "rejected by the Messages API. Create a normal API key instead."
        )

    if len(stripped) < MIN_PLAUSIBLE_LEN:
        out.append(
            f"VALUE IS ONLY {len(stripped)} CHARACTERS. Anthropic keys are typically "
            f"100+. This looks truncated -- copy it again with the Copy button rather "
            f"than selecting the text."
        )

    return out


def shape(value: str | None) -> dict:
    """Non-revealing description, safe to print anywhere."""
    if not value:
        return {"present": False}
    s = value.strip()
    return {
        "present": True,
        "length": len(s),
        "prefix": s[:11] + "..." if len(s) > 11 else "(short)",
        "suffix": "..." + s[-4:],
        "ascii_only": s.isascii(),
        "has_inner_whitespace": bool(re.search(r"\s", s)),
        "had_outer_whitespace": value != s,
    }


def report(name: str = "ANTHROPIC_API_KEY") -> tuple[bool, list[str], dict]:
    raw = os.environ.get(name)
    findings = inspect(raw, name)
    return (not findings), findings, shape(raw)
