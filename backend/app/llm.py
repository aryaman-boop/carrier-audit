"""The single place this system talks to a model.

One file, one client, one call shape. Every model call in the pipeline goes
through `structured()` and comes back as a validated Pydantic object -- there is no
path in this codebase where free text from a model reaches a decision.

Three things here are deliberate and worth defending:

  1. **Structured outputs, not prompt-and-parse.** The schema is passed as
     `output_format`, so the response is constrained to it rather than coaxed
     toward it and re-parsed with a regex. An invalid category is unrepresentable
     instead of caught downstream.

  2. **Prompt caching on the stable prefix.** The system prompt carries the tariff
     policy and the output contract and is identical across every call in a run;
     it is marked cacheable so a 60-line invoice run pays for it once. Cache hits
     are asserted in the usage log, because a cache you never verify is a cache
     that silently stopped working.

  3. **Usage is accumulated and reported in dollars.** Cost per audited line is a
     number the business cares about, so the system measures it rather than
     estimating it later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from . import models

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(RuntimeError):
    """Raised when a model stage is reached with no credentials configured."""


@dataclass
class StageUsage:
    """Tokens and calls for one stage, on one model."""

    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        spec = models.CATALOG[self.model]
        # Cache reads bill at a fraction of input; cache writes at a premium.
        billed_in = (
            self.input_tokens
            + self.cache_write_tokens * 1.25
            + self.cache_read_tokens * 0.10
        )
        return round(
            billed_in / 1e6 * spec.price_in + self.output_tokens / 1e6 * spec.price_out, 6
        )


@dataclass
class Usage:
    """Running token and dollar total for one pipeline run.

    Accounted PER STAGE and PER MODEL, because the whole point of letting stages
    run on different models is being able to see which stage is actually costing
    the money. A single global total cannot answer that.
    """

    stages: dict[str, StageUsage] = field(default_factory=dict)

    def add(self, stage: str, model: str, u: Any) -> None:
        su = self.stages.get(stage)
        if su is None or su.model != model:
            su = self.stages.setdefault(stage, StageUsage(model=model))
        su.calls += 1
        su.input_tokens += getattr(u, "input_tokens", 0) or 0
        su.output_tokens += getattr(u, "output_tokens", 0) or 0
        su.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        su.cache_write_tokens += getattr(u, "cache_creation_input_tokens", 0) or 0

    @property
    def calls(self) -> int:
        return sum(s.calls for s in self.stages.values())

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.stages.values())

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.stages.values())

    @property
    def cache_read_tokens(self) -> int:
        return sum(s.cache_read_tokens for s in self.stages.values())

    @property
    def cost_usd(self) -> float:
        return round(sum(s.cost_usd for s in self.stages.values()), 4)

    def summary(self) -> dict:
        return {
            "models": models.describe(),
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": sum(s.cache_write_tokens for s in self.stages.values()),
            "cache_hit": self.cache_read_tokens > 0,
            "cost_usd": self.cost_usd,
            "calls_by_stage": {k: v.calls for k, v in self.stages.items()},
            "cost_by_stage": {
                k: {"model": v.model, "calls": v.calls, "cost_usd": round(v.cost_usd, 5)}
                for k, v in self.stages.items()
            },
        }


_client: anthropic.Anthropic | None = None


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def client() -> anthropic.Anthropic:
    global _client
    if not available():
        raise LLMUnavailable(
            "ANTHROPIC_API_KEY is not set. The extraction and classification stages "
            "need it. Copy .env.example to .env and add a key from "
            "https://console.anthropic.com/settings/keys, or run the pipeline with "
            "--stub to exercise everything except the model calls."
        )
    if _client is None:
        # SDK retries 408/409/429/5xx with backoff. Vision extraction on a
        # multi-page PDF is slow, so the timeout is raised from the 10 min default
        # only where it needs to be, per call.
        _client = anthropic.Anthropic(max_retries=3)
    return _client


def structured(
    output_model: type[T],
    *,
    system: str,
    content: list[dict] | str,
    usage: Usage | None = None,
    stage: str = "unknown",
    max_tokens: int = 8000,
    effort: str = "high",
    cache_system: bool = True,
    timeout: float | None = None,
) -> T:
    """One constrained model call. Returns a validated instance of `output_model`.

    `system` is sent as a cacheable block so a run of 60 invoice lines pays for the
    policy prefix once rather than sixty times.
    """
    sys_blocks: list[dict] = [{"type": "text", "text": system}]
    if cache_system:
        sys_blocks[0]["cache_control"] = {"type": "ephemeral"}

    if isinstance(content, str):
        content = [{"type": "text", "text": content}]

    # The model, and the request shape it accepts, are resolved per stage. Sending
    # an Opus-shaped body to Haiku is a 400, not a degraded answer.
    spec = models.for_stage(stage) if stage in models.STAGES else models.for_stage("classify")

    kwargs: dict[str, Any] = {
        "system": sys_blocks,
        "messages": [{"role": "user", "content": content}],
        "output_format": output_model,
        **models.request_params(spec, effort, max_tokens),
    }
    if timeout is not None:
        kwargs["timeout"] = timeout

    try:
        resp = client().messages.parse(**kwargs)
    except anthropic.NotFoundError as e:
        raise LLMUnavailable(f"Model {spec.id} not available to this key: {e}") from e
    except anthropic.RateLimitError as e:
        raise RuntimeError(f"Rate limited after retries at stage {stage}: {e}") from e
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"API error at stage {stage} ({e.status_code}): {e}") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError(f"Could not reach the API at stage {stage}: {e}") from e

    if resp.stop_reason == "refusal":
        raise RuntimeError(f"Request declined at stage {stage}: {resp.stop_details}")

    if usage is not None:
        usage.add(stage, spec.id, resp.usage)

    parsed = resp.parsed_output
    if parsed is None:
        raise RuntimeError(f"Stage {stage} returned no parsed output (stop={resp.stop_reason})")
    return parsed
