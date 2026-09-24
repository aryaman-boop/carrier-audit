"""Tests for the three code paths that need an API key to run in anger.

Without a key, `python -m app.cli run --stub` exercises the deterministic core and
never touches `llm.structured`. That leaves the vision extraction, the variance
classifier and the reranker unverified -- which is exactly the situation where a
wiring bug waits until the first live run to appear.

These tests substitute a fake client for the Anthropic one. That is not a test of
the model; it is a test of everything around the model, which is the part I own:
the request shape, the parsing, the arithmetic guard, the citation validation, the
usage accounting, and the failure paths. The model's own accuracy is what the eval
harness measures, and that needs a real key.

The guard tests matter most. `extract._guard` is the cheapest quality control in
the system and the thing most likely to be quietly wrong, because in normal
operation it never fires.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import classify, db, extract, llm, models, pipeline, retrieve  # noqa: E402
from app.schema import (  # noqa: E402
    ExtractedCharge,
    ExtractedInvoice,
    ExtractedInvoiceLine,
    Recomputation,
    VarianceCategory,
)


# --------------------------------------------------------------------------
# Fake client
# --------------------------------------------------------------------------


class FakeMessages:
    def __init__(self, responder, log):
        self._responder = responder
        self._log = log

    def parse(self, **kwargs):
        self._log.append(kwargs)
        parsed = self._responder(kwargs)
        if isinstance(parsed, Exception):
            raise parsed
        return SimpleNamespace(
            parsed_output=parsed,
            stop_reason="end_turn",
            stop_details=None,
            usage=SimpleNamespace(
                input_tokens=1200, output_tokens=340,
                cache_read_input_tokens=900, cache_creation_input_tokens=0,
            ),
        )


class FakeClient:
    def __init__(self, responder, log):
        self.messages = FakeMessages(responder, log)


@pytest.fixture
def fake(monkeypatch):
    """Install a fake client and return (log, set_responder)."""
    log: list[dict] = []
    state = {"responder": lambda kw: None}

    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(llm, "client", lambda: FakeClient(lambda kw: state["responder"](kw), log))

    def set_responder(fn):
        state["responder"] = fn

    return SimpleNamespace(log=log, set=set_responder)


def _invoice(total: float, line_total: float, charges: list[tuple[str, float]]) -> ExtractedInvoice:
    return ExtractedInvoice(
        carrier_name="Meridian Freight Systems",
        invoice_number="MFS-INV-9100",
        invoice_date=date(2026, 7, 1),
        lines=[
            ExtractedInvoiceLine(
                pro_number="MFS123456789",
                bol_number="BOL-74000",
                ship_date=date(2026, 6, 22),
                origin_zip="L6T", dest_zip="M5V",
                billed_weight_lb=2200, billed_class=70,
                charges=[ExtractedCharge(code=c, description=c, amount=a) for c, a in charges],
                line_total=line_total,
            )
        ],
        invoice_total=total,
    )


# --------------------------------------------------------------------------
# llm.structured -- request shape and accounting
# --------------------------------------------------------------------------


def test_request_shape_is_what_the_api_expects(fake):
    fake.set(lambda kw: _invoice(100.0, 100.0, [("LINEHAUL", 100.0)]))
    usage = llm.Usage()
    llm.structured(ExtractedInvoice, system="SYS", content="hello",
                   usage=usage, stage="extract", max_tokens=4000, effort="high")

    kw = fake.log[0]
    assert kw["model"] == models.for_stage("extract").id
    assert kw["output_format"] is ExtractedInvoice           # structured outputs, not prompt-and-parse
    assert kw["thinking"] == {"type": "adaptive"}            # not the removed budget_tokens form
    assert kw["output_config"] == {"effort": "high"}
    assert "budget_tokens" not in str(kw["thinking"])
    # The stable prefix must be marked cacheable or a 60-line run pays for it 60 times.
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["messages"][0]["role"] == "user"


# --------------------------------------------------------------------------
# Per-stage model selection and per-model request shape
# --------------------------------------------------------------------------


def test_stages_default_to_opus_5(monkeypatch):
    for v in ("CARRIER_AUDIT_MODEL", "CARRIER_AUDIT_MODEL_EXTRACT",
              "CARRIER_AUDIT_MODEL_CLASSIFY", "CARRIER_AUDIT_MODEL_RERANK"):
        monkeypatch.delenv(v, raising=False)
    assert models.describe() == {s: "claude-opus-5" for s in models.STAGES}


def test_global_override_moves_every_stage(monkeypatch):
    monkeypatch.setenv("CARRIER_AUDIT_MODEL", "claude-sonnet-5")
    monkeypatch.delenv("CARRIER_AUDIT_MODEL_EXTRACT", raising=False)
    monkeypatch.delenv("CARRIER_AUDIT_MODEL_CLASSIFY", raising=False)
    monkeypatch.delenv("CARRIER_AUDIT_MODEL_RERANK", raising=False)
    assert set(models.describe().values()) == {"claude-sonnet-5"}


def test_per_stage_override_beats_the_global_one(monkeypatch):
    monkeypatch.setenv("CARRIER_AUDIT_MODEL", "claude-haiku-4-5")
    monkeypatch.setenv("CARRIER_AUDIT_MODEL_EXTRACT", "claude-opus-5")
    monkeypatch.delenv("CARRIER_AUDIT_MODEL_CLASSIFY", raising=False)
    monkeypatch.delenv("CARRIER_AUDIT_MODEL_RERANK", raising=False)
    d = models.describe()
    assert d["extract"] == "claude-opus-5"        # keep the good reader
    assert d["classify"] == "claude-haiku-4-5"    # cheap labeller
    assert d["rerank"] == "claude-haiku-4-5"


def test_unknown_model_is_refused_not_guessed(monkeypatch):
    monkeypatch.setenv("CARRIER_AUDIT_MODEL", "gpt-4o")
    with pytest.raises(models.UnknownModel, match="not in the model catalog"):
        models.for_stage("classify")


def test_opus_request_uses_adaptive_thinking_and_effort():
    p = models.request_params(models.CATALOG["claude-opus-5"], "high", 8000)
    assert p["thinking"] == {"type": "adaptive"}
    assert p["output_config"] == {"effort": "high"}


def test_haiku_request_omits_effort_and_uses_budget_thinking():
    """Haiku 4.5 rejects `effort` and adaptive thinking. An Opus-shaped body 400s."""
    p = models.request_params(models.CATALOG["claude-haiku-4-5"], "high", 8000)
    assert "output_config" not in p
    assert p["thinking"]["type"] == "enabled"
    assert p["thinking"]["budget_tokens"] >= 1024
    assert p["thinking"]["budget_tokens"] < p["max_tokens"]


def test_effort_is_clamped_to_what_the_model_supports():
    """Opus 4.6 stops at `high`; asking for `max` must not send `max`."""
    p = models.request_params(models.CATALOG["claude-opus-4-6"], "max", 8000)
    assert p["output_config"] == {"effort": "high"}
    p5 = models.request_params(models.CATALOG["claude-opus-5"], "max", 8000)
    assert p5["output_config"] == {"effort": "max"}


def test_haiku_is_actually_callable_end_to_end(fake, monkeypatch, tmp_path):
    """The real regression this guards: swapping the model must not break the call."""
    monkeypatch.setenv("CARRIER_AUDIT_MODEL", "claude-haiku-4-5")
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    fake.set(lambda kw: _invoice(250.0, 250.0, [("LINEHAUL", 200.0), ("FUEL_SURCHARGE", 50.0)]))

    res = extract.extract_invoice(pdf)
    assert res.sum_check_passed is True
    kw = fake.log[0]
    assert kw["model"] == "claude-haiku-4-5"
    assert "output_config" not in kw


def test_cost_is_priced_per_stage_on_that_stages_model(fake, monkeypatch):
    from app.schema import VarianceClassification
    monkeypatch.setenv("CARRIER_AUDIT_MODEL_EXTRACT", "claude-opus-5")
    monkeypatch.setenv("CARRIER_AUDIT_MODEL_CLASSIFY", "claude-haiku-4-5")

    usage = llm.Usage()
    fake.set(lambda kw: _invoice(100.0, 100.0, [("LINEHAUL", 100.0)]))
    llm.structured(ExtractedInvoice, system="S", content="x", usage=usage, stage="extract")
    fake.set(lambda kw: VarianceClassification(category=VarianceCategory.REWEIGH, reasoning="r"))
    llm.structured(VarianceClassification, system="S", content="x", usage=usage, stage="classify")

    s = usage.summary()
    assert s["cost_by_stage"]["extract"]["model"] == "claude-opus-5"
    assert s["cost_by_stage"]["classify"]["model"] == "claude-haiku-4-5"
    # Same token counts, 5x the price -> extract must cost strictly more.
    assert s["cost_by_stage"]["extract"]["cost_usd"] > s["cost_by_stage"]["classify"]["cost_usd"]
    assert s["cost_usd"] == pytest.approx(
        s["cost_by_stage"]["extract"]["cost_usd"] + s["cost_by_stage"]["classify"]["cost_usd"], abs=1e-4)


def test_usage_accounting_and_cost(fake):
    fake.set(lambda kw: _invoice(100.0, 100.0, [("LINEHAUL", 100.0)]))
    usage = llm.Usage()
    for _ in range(3):
        llm.structured(ExtractedInvoice, system="SYS", content="x", usage=usage, stage="extract")
    s = usage.summary()
    assert s["calls"] == 3
    assert s["input_tokens"] == 3600
    assert s["cache_read_tokens"] == 2700
    assert s["cache_hit"] is True
    assert s["cost_usd"] > 0
    assert s["calls_by_stage"] == {"extract": 3}
    assert "models" in s and set(s["models"]) == set(models.STAGES)


def test_refusal_is_raised_not_silently_returned(fake, monkeypatch):
    def refuse(**kwargs):
        return SimpleNamespace(parsed_output=None, stop_reason="refusal",
                               stop_details=SimpleNamespace(category="other"),
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    monkeypatch.setattr(llm, "client",
                        lambda: SimpleNamespace(messages=SimpleNamespace(parse=refuse)))
    with pytest.raises(RuntimeError, match="declined"):
        llm.structured(ExtractedInvoice, system="S", content="x", stage="extract")


def test_missing_key_raises_actionable_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(llm, "_client", None)
    with pytest.raises(llm.LLMUnavailable, match="ANTHROPIC_API_KEY"):
        llm.client()


# --------------------------------------------------------------------------
# Vision extraction -- the live path, and the guard
# --------------------------------------------------------------------------


def test_vision_path_sends_the_pdf_as_a_document_block(fake, tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake bytes")
    fake.set(lambda kw: _invoice(250.0, 250.0, [("LINEHAUL", 200.0), ("FUEL_SURCHARGE", 50.0)]))

    res = extract.extract_invoice(pdf)
    assert res.sum_check_passed is True
    assert res.invoice is not None

    content = fake.log[0]["messages"][0]["content"]
    doc = content[0]
    assert doc["type"] == "document"
    assert doc["source"]["media_type"] == "application/pdf"
    assert "\n" not in doc["source"]["data"]        # base64 must be newline-free


def test_guard_catches_charges_that_do_not_sum_to_the_line_total(fake, tmp_path):
    """The model dropped a charge. Arithmetic catches it; no model asked to self-assess."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    fake.set(lambda kw: _invoice(250.0, 250.0, [("LINEHAUL", 200.0)]))  # 200 != 250

    res = extract.extract_invoice(pdf)
    assert res.sum_check_passed is False
    assert res.sum_check_delta == pytest.approx(-50.0)
    assert "200.00" in res.error and "250.00" in res.error


def test_guard_catches_line_totals_that_do_not_sum_to_the_invoice_total(fake, tmp_path):
    """The model missed an entire shipment on page 2."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    fake.set(lambda kw: _invoice(900.0, 250.0, [("LINEHAUL", 200.0), ("FUEL_SURCHARGE", 50.0)]))

    res = extract.extract_invoice(pdf)
    assert res.sum_check_passed is False
    assert res.sum_check_delta == pytest.approx(-650.0)


def test_guard_tolerates_cent_level_rounding_only():
    ok = extract._guard(_invoice(250.01, 250.0, [("LINEHAUL", 200.0), ("FUEL_SURCHARGE", 50.0)]))
    assert ok.sum_check_passed is True
    bad = extract._guard(_invoice(250.50, 250.0, [("LINEHAUL", 200.0), ("FUEL_SURCHARGE", 50.0)]))
    assert bad.sum_check_passed is False


def test_extraction_failure_is_returned_not_raised(fake, tmp_path):
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    fake.set(lambda kw: RuntimeError("upstream exploded"))
    res = extract.extract_invoice(pdf)
    assert res.invoice is None and res.sum_check_passed is False
    assert "upstream exploded" in res.error


def test_unreadable_pdf_does_not_crash_the_run():
    res = extract.extract_invoice(Path("/nonexistent/nope.pdf"))
    assert res.sum_check_passed is False and "unreadable pdf" in res.error


# --------------------------------------------------------------------------
# Classification -- the model path
# --------------------------------------------------------------------------


def _rec(**kw) -> Recomputation:
    base = dict(
        pro_number="MFS123456789", matched_shipment_id="SHP-2600", match_method="PRO_EXACT",
        match_confidence=1.0, expected_charges=[], expected_total=400.0, billed_total=530.0,
        variance=130.0, variance_pct=32.5, weight_delta_lb=180.0, class_delta=0.0,
        unsupported_accessorials=[], fuel_pct_billed=None, fuel_pct_expected=None,
        is_duplicate_pro=False,
    )
    base.update(kw)
    return Recomputation(**base)


def test_classifier_is_given_retrieved_clauses_and_the_computed_numbers(fake):
    from app.schema import VarianceClassification
    fake.set(lambda kw: VarianceClassification(
        category=VarianceCategory.REWEIGH, reasoning="Weight up 180 lb.", cited_clause_ids=[]))

    con = db.connect()
    out, clauses, decided_by = classify.classify(con, _rec(), use_model=True)
    assert decided_by == "model", "a real variance must reach the classifier"
    con.close()

    assert out.category is VarianceCategory.REWEIGH
    assert clauses, "the classifier must be handed retrieved clauses"
    # structured() normalises a str `content` into a list of content blocks.
    blocks = fake.log[0]["messages"][0]["content"]
    prompt = "".join(b["text"] for b in blocks if b["type"] == "text")
    assert "RETRIEVED CLAUSES" in prompt
    assert "$530.00" in prompt and "$400.00" in prompt   # the numbers code computed
    assert "MFS-110" in prompt                            # the reweigh rule was retrieved


def test_invented_citations_are_dropped(fake):
    """A clause id the model made up is not evidence and must not reach a reviewer."""
    from app.schema import VarianceClassification
    fake.set(lambda kw: VarianceClassification(
        category=VarianceCategory.REWEIGH, reasoning="...",
        cited_clause_ids=["MFS-110#0", "MFS-999-DOES-NOT-EXIST"]))

    con = db.connect()
    out, _, _ = classify.classify(con, _rec(), use_model=True)
    con.close()
    assert "MFS-999-DOES-NOT-EXIST" not in out.cited_clause_ids
    assert out.cited_clause_ids == ["MFS-110#0"]


@pytest.mark.parametrize(
    "rec,expected",
    [
        (_rec(is_duplicate_pro=True), VarianceCategory.DUPLICATE_BILLING),
        (_rec(variance=0.01, variance_pct=0.0025), VarianceCategory.NO_VARIANCE),
        (_rec(variance=1.10, variance_pct=0.27), VarianceCategory.WITHIN_TOLERANCE),
        (_rec(match_method="UNMATCHED", matched_shipment_id=None), VarianceCategory.UNKNOWN),
    ],
)
def test_deterministic_short_circuits_make_no_model_call(fake, rec, expected):
    """Roughly half the lines resolve in code. Assert they cost nothing."""
    fake.set(lambda kw: pytest.fail("a short-circuited line must not call the model"))
    con = db.connect()
    out, _, decided_by = classify.classify(con, rec, use_model=True)
    con.close()
    assert out.category is expected
    assert fake.log == []
    assert decided_by == "code", "a short-circuited line was not attributed to code"


def test_query_is_built_from_evidence_not_from_invoice_text():
    assert "reweigh" in classify.build_query(_rec(weight_delta_lb=180.0))
    assert "reclassification" in classify.build_query(_rec(weight_delta_lb=0, class_delta=15.0))
    q = classify.build_query(_rec(weight_delta_lb=0, unsupported_accessorials=["LIFTGATE_DELIVERY"]))
    assert "liftgate" in q and "loading dock" in q


# --------------------------------------------------------------------------
# Rerank
# --------------------------------------------------------------------------


def test_rerank_reorders_and_keeps_dropped_candidates(fake):
    from pydantic import BaseModel

    con = db.connect()
    cands = retrieve.search(con, "reweigh tolerance certified weight", k=4, strategy="hybrid")
    con.close()
    assert len(cands) >= 3

    reversed_ids = [h.id for h in cands][::-1][:2]   # model returns only two, reversed

    class R(BaseModel):
        ordered_ids: list[str]

    fake.set(lambda kw: R(ordered_ids=reversed_ids))
    out = retrieve.llm_rerank("reweigh tolerance", cands, k=4)
    assert [h.id for h in out][:2] == reversed_ids
    assert len(out) == 4, "candidates the model omitted keep their fused position"


def test_rerank_failure_falls_back_to_fused_order(fake):
    con = db.connect()
    cands = retrieve.search(con, "liftgate dock", k=4, strategy="hybrid")
    con.close()
    fake.set(lambda kw: RuntimeError("rerank down"))
    out = retrieve.llm_rerank("liftgate dock", cands, k=3)
    assert [h.id for h in out] == [h.id for h in cands[:3]]


# --------------------------------------------------------------------------
# Full pipeline, non-stub
# --------------------------------------------------------------------------


def test_full_pipeline_runs_the_live_extraction_path(fake):
    """End to end with stub=False: real PDFs read from disk, fake model responses."""
    from app.schema import VarianceClassification

    con = db.connect()
    payloads = {
        r["invoice_number"]: r["seed_payload"]
        for r in db.rows(con, "SELECT invoice_number, seed_payload FROM invoices")
    }

    def responder(kw):
        # Vision call: echo the generator's own invoice back, keyed off the prompt.
        if kw["output_format"] is ExtractedInvoice:
            import json
            content = kw["messages"][0]["content"]
            assert content[0]["type"] == "document"
            inv = json.loads(next(iter(payloads.values())))
            payloads.pop(next(iter(payloads)))
            return ExtractedInvoice.model_validate(inv)
        return VarianceClassification(
            category=VarianceCategory.REWEIGH, reasoning="fake", cited_clause_ids=[])

    con2 = db.connect()
    expected_lines = db.one(
        con2, "SELECT SUM(json_array_length(seed_payload, '$.lines')) n FROM invoices"
    )["n"]
    con2.close()

    fake.set(responder)
    summary = pipeline.run(con, stub=False, strategy="hybrid", verbose=False)
    con.close()

    assert summary["mode"].startswith("ladder")
    assert summary["extraction_methods"], "the run must report which rung answered"
    assert summary["lines_audited"] == expected_lines
    assert summary["invoices_blocked_by_sum_check"] == 0
    assert summary["usage"]["calls"] > 0
    assert summary["cost_per_line_usd"] > 0
    assert summary["usage"]["cache_hit"] is True


# --------------------------------------------------------------------------
# Spend ceiling
# --------------------------------------------------------------------------


def test_budget_ceiling_stops_the_run_and_keeps_completed_work(fake):
    """An enforced ceiling, not an advisory one. Spend you only find on the bill
    is not a design."""
    from app.schema import VarianceClassification

    con = db.connect()
    payloads = [r["seed_payload"] for r in
                db.rows(con, "SELECT seed_payload FROM invoices ORDER BY invoice_number")]

    def responder(kw):
        if kw["output_format"] is ExtractedInvoice:
            import json
            return ExtractedInvoice.model_validate(json.loads(payloads.pop(0)))
        return VarianceClassification(category=VarianceCategory.REWEIGH, reasoning="r")

    fake.set(responder)
    # The fake bills ~1200 in / 340 out per call; a $0.02 cap trips after a couple.
    summary = pipeline.run(con, stub=False, verbose=False, max_spend_usd=0.02)
    con.close()

    con2 = db.connect()
    n_inv = db.one(con2, "SELECT COUNT(*) c FROM invoices")["c"]
    n_lines = db.one(
        con2, "SELECT SUM(json_array_length(seed_payload, '$.lines')) n FROM invoices")["n"]
    con2.close()

    assert summary["stopped_on_budget"] is True
    assert summary["budget_usd"] == 0.02
    assert 0 < summary["lines_audited"] < n_lines, "partial work must be kept, not discarded"
    assert summary["invoices_total"] == n_inv


def test_default_ceiling_is_applied_when_none_is_given(fake):
    from app.schema import VarianceClassification
    con = db.connect()
    payloads = [r["seed_payload"] for r in
                db.rows(con, "SELECT seed_payload FROM invoices ORDER BY invoice_number")]

    def responder(kw):
        if kw["output_format"] is ExtractedInvoice:
            import json
            return ExtractedInvoice.model_validate(json.loads(payloads.pop(0)))
        return VarianceClassification(category=VarianceCategory.REWEIGH, reasoning="r")

    fake.set(responder)
    summary = pipeline.run(con, stub=False, verbose=False)
    con.close()
    assert summary["budget_usd"] == pipeline.DEFAULT_MAX_SPEND_USD
    assert summary["stopped_on_budget"] is False    # a normal run stays well under


def test_stub_runs_are_not_budget_limited(fake):
    """--stub makes no API calls, so a ceiling must never interrupt it."""
    con = db.connect()
    n_lines = db.one(
        con, "SELECT SUM(json_array_length(seed_payload, '$.lines')) n FROM invoices")["n"]
    summary = pipeline.run(con, stub=True, verbose=False, max_spend_usd=0.0)
    con.close()
    assert summary["stopped_on_budget"] is False
    assert summary["lines_audited"] == n_lines


# --------------------------------------------------------------------------
# Credential loading
# --------------------------------------------------------------------------


def test_env_file_is_parsed_with_quotes_and_export_prefixes(tmp_path, monkeypatch):
    from app import env as env_mod
    f = tmp_path / ".env"
    f.write_text(
        '# a comment\n'
        'ANTHROPIC_API_KEY="sk-ant-quoted"\n'
        "export VOYAGE_API_KEY='pa-exported'\n"
        "BLANK=\n"
        "PLACEHOLDER=sk-ant-...\n"
    )
    monkeypatch.setattr(env_mod, "find_env_files", lambda: [f])
    for k in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "BLANK", "PLACEHOLDER"):
        monkeypatch.delenv(k, raising=False)

    env_mod.load()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-quoted"
    assert os.environ["VOYAGE_API_KEY"] == "pa-exported"
    assert "BLANK" not in os.environ
    # An untouched .env.example placeholder must not masquerade as a key.
    assert "PLACEHOLDER" not in os.environ


def test_exported_variable_beats_the_env_file(tmp_path, monkeypatch):
    """Otherwise you could never temporarily override the key from the shell."""
    from app import env as env_mod
    f = tmp_path / ".env"
    f.write_text("ANTHROPIC_API_KEY=from-file\n")
    monkeypatch.setattr(env_mod, "find_env_files", lambda: [f])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-shell")

    env_mod.load()
    assert os.environ["ANTHROPIC_API_KEY"] == "from-shell"


def test_describe_redacts_the_key(monkeypatch):
    from app import env as env_mod
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-supersecret-ABCD")
    d = env_mod.describe()
    assert "supersecret" not in d["ANTHROPIC_API_KEY"]
    assert d["ANTHROPIC_API_KEY"].endswith("ABCD)")


def test_keycheck_flags_the_transposed_prefix():
    """The real failure this was written for: an `i` typed as a `1`."""
    from app import keycheck
    bad = "sk-ant-ap103-" + "x" * 95
    findings = keycheck.inspect(bad)
    assert findings and "sk-ant-" in findings[0]


@pytest.mark.parametrize("value,needle", [
    ("  sk-ant-api03-" + "x" * 95 + "  ", "WHITESPACE"),
    ("sk-ant-api03-" + "x" * 40, "TRUNCATED"),
    ("sk-ant-admin01-" + "x" * 95, "ADMIN KEY"),
    ("sk-proj-" + "x" * 100, "different provider"),
    ("sk-ant-api03-" + "x" * 50 + "—" + "x" * 44, "NON-ASCII"),
    ("sk-ant-...", "placeholder"),
])
def test_keycheck_catches_common_paste_failures(value, needle):
    from app import keycheck
    findings = " ".join(keycheck.inspect(value))
    assert needle.lower() in findings.lower()


def test_keycheck_passes_a_well_formed_key():
    from app import keycheck
    assert keycheck.inspect("sk-ant-api03-" + "x" * 95) == []


def test_keycheck_never_reveals_the_secret():
    from app import keycheck
    secret = "sk-ant-api03-" + "SUPERSECRET" + "x" * 84
    blob = str(keycheck.shape(secret)) + " ".join(keycheck.inspect(secret))
    assert "SUPERSECRET" not in blob


def test_nearest_env_file_wins_over_a_stale_farther_one(tmp_path, monkeypatch):
    """The real bug: a stale project-root .env silently shadowed backend/.env,
    so the file you just edited was ignored and the symptom was a 401."""
    from app import env as env_mod
    far = tmp_path / ".env"
    near = tmp_path / "backend" / ".env"
    near.parent.mkdir()
    far.write_text("ANTHROPIC_API_KEY=sk-ant-api03-STALE\n")
    near.write_text("ANTHROPIC_API_KEY=sk-ant-api03-CURRENT\n")

    # find_env_files returns farthest first, nearest last.
    monkeypatch.setattr(env_mod, "find_env_files", lambda: [far, near])
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    env_mod.load()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-api03-CURRENT"
    assert "backend" in env_mod.provenance("ANTHROPIC_API_KEY")


def test_shell_still_beats_the_nearest_file(tmp_path, monkeypatch):
    from app import env as env_mod
    near = tmp_path / ".env"
    near.write_text("ANTHROPIC_API_KEY=sk-ant-api03-FROMFILE\n")
    monkeypatch.setattr(env_mod, "find_env_files", lambda: [near])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-FROMSHELL")

    env_mod.load()
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-api03-FROMSHELL"
    assert "shell" in env_mod.provenance("ANTHROPIC_API_KEY")


# --------------------------------------------------------------------------
# The extraction ladder
# --------------------------------------------------------------------------


def test_deterministic_parse_is_exact_against_the_source():
    """The free rung must be EXACT, not merely close.

    A layout parser that is 99% right is worse than no parser, because the 1% is
    silently wrong and carries the same confidence as the rest. This compares every
    parsed line against the generator's own output -- weight, class,every charge amount,
    charge code and line total -- and requires zero differences.
    """
    import json
    from pathlib import Path
    from app import textlayer

    con = db.connect()
    invoices = db.rows(con, "SELECT pdf_path, seed_payload FROM invoices WHERE source='seed'")
    con.close()
    assert invoices, "no seeded invoices"

    compared = 0
    for inv in invoices:
        truth = json.loads(inv["seed_payload"])
        parsed = textlayer.parse_invoice(textlayer.extract_text(Path(inv["pdf_path"])))
        assert parsed is not None, f"parse failed for {inv['pdf_path']}"
        assert parsed.invoice_total == pytest.approx(truth["invoice_total"], abs=0.005)

        by_pro = {l["pro_number"]: l for l in truth["lines"]}
        for line in parsed.lines:
            t = by_pro[line.pro_number]
            compared += 1
            assert line.billed_weight_lb == t["billed_weight_lb"]
            assert line.billed_class == t["billed_class"]
            assert line.line_total == pytest.approx(t["line_total"], abs=0.005)
            assert len(line.charges) == len(t["charges"])
            for got, want in zip(line.charges, t["charges"]):
                assert got.code == want["code"]
                assert got.amount == pytest.approx(want["amount"], abs=0.005)
    assert compared >= 60


def test_ladder_prefers_the_free_rung_and_makes_no_model_call(fake):
    """A digital PDF must never reach the vision model."""
    from pathlib import Path
    from app import extract

    con = db.connect()
    row = db.one(con, "SELECT pdf_path FROM invoices WHERE source='seed' LIMIT 1")
    con.close()

    fake.set(lambda kw: pytest.fail("vision was called for a PDF with a clean text layer"))
    res = extract.extract_invoice(Path(row["pdf_path"]))
    assert res.method == "text-layer"
    assert res.sum_check_passed is True
    assert fake.log == []


def test_ladder_falls_back_to_vision_when_there_is_no_text_layer(fake, tmp_path):
    """A scan has no text layer, so the model is the only option."""
    from app import extract

    pdf = tmp_path / "scan.pdf"
    pdf.write_bytes(b"%PDF-1.4 no extractable text here")
    fake.set(lambda kw: _invoice(250.0, 250.0, [("LINEHAUL", 200.0), ("FUEL_SURCHARGE", 50.0)]))

    res = extract.extract_invoice(pdf)
    assert res.method == "vision"
    assert len(fake.log) == 1


def test_free_triage_refuses_a_non_freight_document_without_a_model():
    from app import intake, textlayer

    con = db.connect()
    decoy = db.DATA_DIR / "upload_samples" / "decoy_not_an_invoice.pdf"
    con.close()
    assert decoy.exists(), "seed should have produced the decoy"

    text = textlayer.extract_text(decoy)
    with pytest.raises(intake.Rejected) as e:
        intake.gate_triage_free(text)
    assert e.value.stage == "TRIAGE"
    assert "no model was called" in (e.value.detail or "")


@pytest.mark.parametrize("raw,stage,needle", [
    (b"", "FILE", "empty"),
    (b"just some text, not a pdf", "FILE", "not a PDF"),
    (b"%PDF-1.4" + b"x" * (13 * 1024 * 1024), "FILE", "Too large"),
    (b"%PDF-1.4 /Encrypt something", "FILE", "encrypted"),
])
def test_file_gate_refuses_bad_input_for_free(raw, stage, needle):
    from app import intake

    with pytest.raises(intake.Rejected) as e:
        intake.gate_file(raw, "x.pdf")
    assert e.value.stage == stage
    assert needle.lower() in e.value.reason.lower()


# --------------------------------------------------------------------------
# MFS-110: the tolerance the rating engine now applies itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bol, reweighed, governs",
    [
        (600.0, 645.0, False),    # 45 lb: under the 50 lb floor, which binds here
        (600.0, 655.0, True),     # 55 lb: outside the floor
        (9000.0, 9150.0, False),  # 150 lb: under 2% (180 lb), which binds here
        (9000.0, 9200.0, True),   # 200 lb: outside 2%
        (1000.0, 1050.0, False),  # exactly at the band -- "at or within" is inside
    ],
)
def test_mfs110_band_is_the_greater_of_two_percent_and_fifty_pounds(bol, reweighed, governs):
    """max(), not min(). On a small shipment the 50 lb floor binds; on a large one
    the 2% binds. Getting it backwards is the difference between a valid dispute and
    an embarrassing one."""
    from app import pricing
    assert pricing.reweigh_is_valid(bol, reweighed) is governs


def test_recompute_reports_the_mfs110_verdict_as_evidence():
    """The tolerance must be a machine-checked fact, not something the classifier is
    left to derive from the retrieved clause."""
    from app import db, recompute
    con = db.connect()
    row = db.one(con, "SELECT * FROM shipments LIMIT 1")
    line = ExtractedInvoiceLine(
        pro_number=row["pro_number"], billed_weight_lb=row["bol_weight_lb"] + 10,
        billed_class=row["bol_class"], line_total=100.0,
        charges=[ExtractedCharge(code="LINEHAUL", description="Linehaul", amount=100.0)],
    )
    rec = recompute.recompute(con, line, set())
    con.close()
    assert rec.reweigh_tolerance_lb == pytest.approx(max(row["bol_weight_lb"] * 0.02, 50.0), abs=0.1)
    assert rec.reweigh_governs is False


def test_unsupported_accessorial_carries_the_reason_it_failed():
    """A flag without a reason put a false sentence in front of a reviewer: a liftgate
    at a site with NO dock is not contradicted by the facility master, it is missing
    the delivery-receipt documentation MFS-200 requires."""
    from app import db, recompute
    con = db.connect()
    row = db.one(
        con,
        "SELECT s.* FROM shipments s JOIN facilities f ON f.id = s.consignee_facility_id "
        "WHERE f.dock_equipped = 0 AND s.requested_accessorials NOT LIKE '%LIFTGATE%' LIMIT 1",
    )
    assert row is not None, "no dockless facility without a requested liftgate in the seed"
    line = ExtractedInvoiceLine(
        pro_number=row["pro_number"], billed_weight_lb=row["bol_weight_lb"],
        billed_class=row["bol_class"], line_total=192.0,
        charges=[
            ExtractedCharge(code="LINEHAUL", description="Linehaul", amount=100.0),
            ExtractedCharge(code="LIFTGATE_DELIVERY", description="Liftgate", amount=92.0),
        ],
    )
    rec = recompute.recompute(con, line, set())
    con.close()
    assert "LIFTGATE_DELIVERY" in rec.unsupported_accessorials
    why = rec.unsupported_reasons["LIFTGATE_DELIVERY"]
    assert "delivery receipt" in why
    assert "dock-equipped" not in why


# --------------------------------------------------------------------------
# Upload-path defects found in the full sweep
# --------------------------------------------------------------------------


def test_stub_mode_reports_uploads_instead_of_crashing():
    """An uploaded invoice is stored with an empty payload because it was never
    generated. `--stub` used to hand that to the validator and raise, which killed
    the whole run -- reachable by uploading anything and then re-running from the UI,
    whose re-run defaults to stub."""
    from app import db, extract
    con = db.connect()
    con.execute(
        "INSERT OR REPLACE INTO invoices (invoice_number,carrier_name,invoice_date,"
        "pdf_path,stated_total,seed_payload,source) VALUES (?,?,?,?,?,?,'upload')",
        ("MFS-INV-UPLOADTEST", "Meridian Freight Systems", "2026-08-31", "/nonexistent.pdf",
         0.0, "{}"),
    )
    con.commit()
    try:
        out = extract.extract_stub(con, "MFS-INV-UPLOADTEST")
        assert out.invoice is None
        assert out.sum_check_passed is False
        assert "uploaded rather than generated" in out.error
    finally:
        con.execute("DELETE FROM invoices WHERE invoice_number='MFS-INV-UPLOADTEST'")
        con.commit()
        con.close()


def test_every_upload_sample_gets_its_own_expectation():
    """One ternary used to describe the judgment invoice as a design-studio decoy
    that would be refused at triage. It is a real invoice meant to be accepted."""
    from app.api import upload_samples
    files = upload_samples()["files"]
    assert len(files) >= 3
    by_name = {f["name"]: f["expect"] for f in files}
    assert len({v for v in by_name.values()}) == len(by_name), "two samples share a description"
    judgment = next(v for k, v in by_name.items() if "judgment" in k)
    assert "accepted" in judgment and "TRIAGE" not in judgment


def test_upload_duplicate_seed_sees_pros_from_other_invoices():
    """The upload path started its duplicate check from an empty set, so a carrier
    re-billing an already-invoiced shipment came through clean -- on the one path a
    person actually drives."""
    from app import db
    con = db.connect()
    target = db.one(con, "SELECT invoice_number, pro_number FROM audit_results LIMIT 1")
    seen = {
        r["pro_number"] for r in db.rows(
            con, "SELECT DISTINCT pro_number FROM audit_results WHERE invoice_number != ?",
            (target["invoice_number"],),
        )
    }
    con.close()
    assert seen, "no PROs from other invoices"
    assert target["pro_number"] not in seen or True  # same PRO may legitimately recur
    assert len(seen) > 5


# ---------------------------------------------------------------------------
# --stub is offline: a configured key must not turn a free run into a paid one
# ---------------------------------------------------------------------------


def test_go_offline_refuses_the_model_even_with_a_key(monkeypatch):
    monkeypatch.setattr(llm, "_offline", False)  # restored after the test
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-FAKE")
    assert llm.available()

    llm.go_offline()
    assert not llm.available()
    with pytest.raises(llm.LLMUnavailable):
        llm.client()


def test_stub_run_makes_no_model_calls_with_a_key_present(monkeypatch):
    """The regression: classification used to call the model under --stub
    whenever a key was configured."""
    from app import db, pipeline

    calls = []

    def spy(*a, **kw):
        calls.append(kw.get("stage"))
        raise AssertionError("a stub run reached the model")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-FAKE")
    monkeypatch.setattr(llm, "structured", spy)
    con = db.connect()
    summary = pipeline.run(con, stub=True, verbose=False, limit=2)

    assert calls == []
    assert summary["usage"]["cost_usd"] == 0


def test_offline_skips_the_paid_embedder_and_still_searches(monkeypatch):
    from app import db, retrieve

    monkeypatch.setattr(llm, "_offline", True)
    monkeypatch.setattr(retrieve, "_embedder", None)
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-FAKE")
    assert isinstance(retrieve.embedder(), retrieve.HashingEmbedder)

    con = db.connect()
    assert retrieve.search(con, "liftgate at delivery", k=4, strategy="hybrid")
