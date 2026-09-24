"""Which model runs which stage, and what each model's API actually accepts.

The pipeline has three model stages and they are not the same job:

    extract    read a dense, multi-page invoice PDF and transcribe it exactly.
               Vision, long output, and the stage where a mistake is most
               expensive -- everything downstream trusts this transcription.

    classify   pick one label from a closed set, given numbers that code already
               computed and clauses that retrieval already found. Short input,
               short output, and most of the reasoning has been done for it.

    rerank     order 10 candidate clauses. Trivial.

Running all three on the same frontier model is the safe default and the wrong
default at volume. The reason to make them independently configurable is not
tidiness -- it is that "which stage actually needs the expensive model?" is a
question you answer by measuring, and you cannot measure it if the stages are
welded together.

    CARRIER_AUDIT_MODEL           sets all three at once
    CARRIER_AUDIT_MODEL_EXTRACT   overrides the extraction stage
    CARRIER_AUDIT_MODEL_CLASSIFY  overrides the classifier
    CARRIER_AUDIT_MODEL_RERANK    overrides the reranker

**The compatibility half of this file is load-bearing, not bookkeeping.** The
request shape is not the same across the family, and getting it wrong is a 400
rather than a degraded answer:

  - Haiku 4.5 does not take `output_config.effort` at all, and does not support
    adaptive thinking -- it uses the older `budget_tokens` form or no thinking.
    Sending an Opus-shaped request to Haiku fails outright.
  - Opus 4.8 and 4.7 run without thinking unless `adaptive` is set explicitly,
    where Opus 5 is adaptive by default.
  - `xhigh` and `max` effort exist on Opus 5 / 4.8 / 4.7 and Sonnet 5, but not on
    Opus 4.6 or Sonnet 4.6, which stop at `high`.
  - Haiku 4.5 has a 200K context window against 1M for the rest, which matters for
    a long invoice sent as an image.

So `request_params()` builds the correct request for whichever model a stage is
pointed at, and an unknown model id is rejected loudly rather than sent with an
Opus-shaped body and a guessed price.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

STAGES = ("extract", "classify", "rerank")


@dataclass(frozen=True)
class ModelSpec:
    """What one model accepts, and what it costs.

    Prices are Anthropic first-party list, USD per million tokens (input, output).
    They exist only for the cost report -- an unpriced model would silently report
    someone else's cost, which is worse than reporting none.
    """

    id: str
    price_in: float
    price_out: float
    context: int
    thinking: str          # "adaptive" | "budget" | "none"
    supports_effort: bool
    max_effort: str        # highest accepted effort level
    vision: bool = True


CATALOG: dict[str, ModelSpec] = {
    "claude-fable-5-1": ModelSpec("claude-fable-5-1", 10.00, 50.00, 1_000_000, "adaptive", True, "max"),
    "claude-fable-5":   ModelSpec("claude-fable-5", 10.00, 50.00, 1_000_000, "adaptive", True, "max"),
    "claude-opus-5":    ModelSpec("claude-opus-5", 5.00, 25.00, 1_000_000, "adaptive", True, "max"),
    "claude-opus-4-8":  ModelSpec("claude-opus-4-8", 5.00, 25.00, 1_000_000, "adaptive", True, "max"),
    "claude-opus-4-7":  ModelSpec("claude-opus-4-7", 5.00, 25.00, 1_000_000, "adaptive", True, "xhigh"),
    "claude-opus-4-6":  ModelSpec("claude-opus-4-6", 5.00, 25.00, 1_000_000, "adaptive", True, "high"),
    "claude-sonnet-5":  ModelSpec("claude-sonnet-5", 2.00, 10.00, 1_000_000, "adaptive", True, "max"),
    "claude-sonnet-4-6": ModelSpec("claude-sonnet-4-6", 3.00, 15.00, 1_000_000, "adaptive", True, "high"),
    # Haiku 4.5 is the one that breaks an Opus-shaped request. It takes neither
    # adaptive thinking nor `effort`.
    "claude-haiku-4-5": ModelSpec("claude-haiku-4-5", 1.00, 5.00, 200_000, "budget", False, "none"),
}

EFFORT_ORDER = ["low", "medium", "high", "xhigh", "max"]

DEFAULT = "claude-opus-5"


class UnknownModel(ValueError):
    """An id not in CATALOG. Refused rather than guessed at."""


def for_stage(stage: str) -> ModelSpec:
    """Resolve the model for one stage from the environment."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    chosen = (
        os.environ.get(f"CARRIER_AUDIT_MODEL_{stage.upper()}")
        or os.environ.get("CARRIER_AUDIT_MODEL")
        or DEFAULT
    )
    spec = CATALOG.get(chosen)
    if spec is None:
        raise UnknownModel(
            f"{chosen!r} is not in the model catalog, so its request shape and price are "
            f"unknown and it will not be sent blind. Known ids: {', '.join(sorted(CATALOG))}. "
            f"Add it to CATALOG in app/models.py with its real pricing and capabilities."
        )
    if stage == "extract" and not spec.vision:
        raise UnknownModel(f"{chosen} has no vision support and cannot run the extract stage.")
    return spec


def request_params(spec: ModelSpec, effort: str, max_tokens: int) -> dict:
    """The model-specific half of a request body.

    Returns only the keys that vary by model. Anything the model does not accept is
    omitted rather than sent and hoped for.
    """
    params: dict = {"model": spec.id, "max_tokens": min(max_tokens, 64_000)}

    if spec.thinking == "adaptive":
        params["thinking"] = {"type": "adaptive"}
    elif spec.thinking == "budget":
        # Older form. Must be below max_tokens, minimum 1024.
        budget = max(1024, min(4096, max_tokens - 1024))
        if budget >= 1024:
            params["thinking"] = {"type": "enabled", "budget_tokens": budget}

    if spec.supports_effort:
        ceiling = EFFORT_ORDER.index(spec.max_effort)
        want = EFFORT_ORDER.index(effort) if effort in EFFORT_ORDER else ceiling
        params["output_config"] = {"effort": EFFORT_ORDER[min(want, ceiling)]}

    return params


def describe() -> dict[str, str]:
    """What each stage is pointed at right now, for the run report."""
    return {s: for_stage(s).id for s in STAGES}
