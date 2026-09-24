"""Command line entry point.

    python -m app.cli seed                 build the synthetic book and the invoice PDFs
    python -m app.cli run [--stub]         audit every invoice, print the run summary
    python -m app.cli ask "<question>"     query the tariff corpus directly
    python -m app.cli prompt [--pro PRO]    print the exact classifier request, spend nothing
    python -m app.cli serve                start the API + review UI on :8000
"""

from __future__ import annotations

import argparse
import json
import sys

from . import db, llm, pipeline, retrieve, seed


def cmd_seed(args: argparse.Namespace) -> int:
    print("Generating the synthetic shipment book and carrier invoices...")
    stats = seed.seed_all(with_pdfs=not args.no_pdfs)
    print(json.dumps(stats, indent=2))
    print(f"\nInvoice PDFs: {db.INVOICE_DIR}")
    print(f"Database:     {db.DB_PATH}")
    return 0


def _resolve_reuse(con, value: str | None) -> str | None:
    if not value:
        return None
    if value == "last":
        row = db.one(con, "SELECT run_id FROM audit_results ORDER BY id DESC LIMIT 1")
        if not row:
            raise SystemExit("no previous run to reuse")
        return row["run_id"]
    return value


def cmd_run(args: argparse.Namespace) -> int:
    import os
    for flag, env in (("model", "CARRIER_AUDIT_MODEL"),
                      ("extract_model", "CARRIER_AUDIT_MODEL_EXTRACT"),
                      ("classify_model", "CARRIER_AUDIT_MODEL_CLASSIFY"),
                      ("rerank_model", "CARRIER_AUDIT_MODEL_RERANK")):
        if getattr(args, flag, None):
            os.environ[env] = getattr(args, flag)
    from . import models as _m
    try:
        print("models:", _m.describe())
    except _m.UnknownModel as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    con = db.connect()
    if not db.rows(con, "SELECT 1 FROM invoices LIMIT 1"):
        print("No data. Run:  python -m app.cli seed", file=sys.stderr)
        return 1
    if not args.stub and not llm.available():
        print("ANTHROPIC_API_KEY is not set, so vision extraction cannot run.\n"
              "Either export the key, or use --stub to exercise everything else.",
              file=sys.stderr)
        return 1
    cap = pipeline.DEFAULT_MAX_SPEND_USD if args.max_spend is None else args.max_spend
    print(f"Auditing ({'stub' if args.stub else 'vision'} extraction, {args.strategy} retrieval)"
          + ("" if args.stub else f"  [spend ceiling ${cap:.2f}]"))
    summary = pipeline.run(con, stub=args.stub, strategy=args.strategy, limit=args.limit,
                           max_spend_usd=args.max_spend,
                           reuse_run=_resolve_reuse(con, args.reuse_extraction))
    print("\n" + json.dumps(summary, indent=2))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """Query the tariff corpus. Retrieval only -- no generated answer.

    Deliberately: the reviewer wants the governing clause, quoted, not a paraphrase
    of it. A summary of a contract term is a liability in a dispute.
    """
    con = db.connect()
    hits = retrieve.search(con, args.question, k=args.k, strategy=args.strategy)
    print(f"\nstrategy={args.strategy}  embedder={retrieve.embedder().name}\n")
    for i, h in enumerate(hits, 1):
        print(f"{i}. [{h.id}] {h.title}   (score {h.score:.4f}, {h.source})")
        print(f"   {h.text[:400]}{'...' if len(h.text) > 400 else ''}\n")
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    """Print the exact request the classifier would send, and send nothing.

    Rebuilt from a stored audited line rather than typed out here, so what prints is
    what actually went to the model -- the same system block, the same evidence built
    by the same code, the same clauses the same retriever returned. A prompt
    transcribed into a doc drifts from the one the code sends within a week; this one
    cannot.

    It costs nothing because it stops at the boundary. That is the point: the request
    is inspectable without being billable, which is also what makes it reviewable in
    a pull request.
    """
    import json as _json

    from . import classify as classify_mod
    from . import models
    from .schema import AuditedLine, Recomputation, VarianceClassification

    con = db.connect()
    rows = db.rows(
        con,
        "SELECT payload FROM audit_results WHERE run_id = (SELECT run_id FROM audit_results "
        "ORDER BY id DESC LIMIT 1) ORDER BY id",
    )
    if not rows:
        print("No audited lines on file. Run `python -m app.cli run --stub` first.")
        return 1

    lines = [AuditedLine.model_validate_json(r["payload"]) for r in rows]
    if args.pro:
        picked = next((ln for ln in lines if ln.pro_number == args.pro), None)
        if picked is None:
            print(f"No line for PRO {args.pro} in the latest run.")
            return 1
    else:
        # A line the code short-circuited never reached the model, so showing its
        # "prompt" would be a fiction. Prefer one the classifier actually ran on.
        picked = next((ln for ln in lines if ln.decided_by == "model"), lines[0])

    rec: Recomputation = picked.recomputation
    billed = picked.extracted.charges
    query = classify_mod.build_query(rec, billed)
    hits = retrieve.search(con, query, k=4, strategy=args.strategy)
    retrieved = "\n\n".join(f"[{h.id}] {h.title}\n{h.text}" for h in hits)
    content = (f"EVIDENCE\n{classify_mod._evidence(rec, billed)}\n\n"
               f"RETRIEVED CLAUSES\n{retrieved}")

    spec = models.for_stage("classify")
    params = models.request_params(spec, "medium", 2000)

    if args.raw:
        print(_json.dumps({
            "system": [{"type": "text", "text": classify_mod.SYSTEM,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": content}]}],
            "output_format": VarianceClassification.model_json_schema(),
            **params,
        }, indent=2))
        return 0

    def rule(label: str) -> None:
        print(f"\n\033[1m{label}\033[0m\n" + "-" * 76)

    print(f"\nPRO {picked.pro_number}  invoice {picked.invoice_number}  "
          f"decided by {picked.decided_by}")
    print(f"model {spec.id}   params {params}")
    print("nothing is sent; this stops at the boundary")

    rule("SYSTEM  (one cacheable block -- 60 lines pay for this prefix once)")
    print(classify_mod.SYSTEM)

    rule("USER  (built by code from the facts, not from the PDF text)")
    print(content)

    rule("OUTPUT CONTRACT  (the enum is enforced here, not asked for in prose)")
    print(_json.dumps(VarianceClassification.model_json_schema(), indent=2))

    rule("RETRIEVAL QUERY  (constructed from the evidence -- there is no user)")
    print(query)
    print(f"\nstrategy={args.strategy}  embedder={retrieve.embedder().name}  "
          f"k={len(hits)}: {[h.id for h in hits]}")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check credentials and configuration without spending anything."""
    from . import env, models

    from . import keycheck

    print("Credentials")
    files = env.find_env_files()
    print(f"  .env files found: {[str(f) for f in files] or 'none'}")
    if len(files) > 1:
        print(f"  WARNING: {len(files)} .env files are in scope. The one NEAREST this")
        print(f"           package wins. Delete the ones you are not using -- a stale")
        print(f"           file shadowing the current one is hard to spot.")
    print(f"  key source:          {env.provenance('ANTHROPIC_API_KEY')}")
    for k, v in env.describe().items():
        print(f"  {k:<20} {v}")

    # Shape check before any network call. A 401 costs a round trip and tells you
    # nothing; almost every bad-paste failure is detectable locally and for free.
    ok, findings, shape = keycheck.report()
    if shape.get("present"):
        print(f"  key shape            len={shape['length']} prefix={shape['prefix']} "
              f"ascii={shape['ascii_only']}")
    if findings:
        print("\n  PROBLEMS WITH THE KEY ITSELF:")
        for f in findings:
            print(f"    * {f}")
        print("\n  Re-copy it from https://console.anthropic.com/settings/keys using the")
        print("  Copy button, then:  cd .. && echo 'ANTHROPIC_API_KEY=<paste>' > .env")
        return 1

    print("\nModels")
    try:
        for stage, mid in models.describe().items():
            spec = models.CATALOG[mid]
            print(f"  {stage:<10} {mid:<20} ${spec.price_in}/${spec.price_out} per MTok")
    except models.UnknownModel as e:
        print(f"  error: {e}")
        return 2

    print("\nData")
    con = db.connect()
    n_inv = db.one(con, "SELECT COUNT(*) c FROM invoices")["c"]
    n_cl = db.one(con, "SELECT COUNT(*) c FROM clauses")["c"]
    con.close()
    print(f"  invoices seeded: {n_inv}   tariff clauses indexed: {n_cl}")
    if n_inv == 0:
        print("  -> run: python -m app.cli seed")

    if not llm.available():
        print("\nThe model stages cannot run. Put a key in .env at the project root:")
        print("  echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env")
        return 1

    if args.ping:
        print("\nLive check (costs well under a cent)...")
        from pydantic import BaseModel

        class Ping(BaseModel):
            ok: bool

        try:
            out = llm.structured(Ping, system="Reply with ok=true.", content="ping",
                                 stage="classify", max_tokens=1024, effort="low")
            print(f"  API reachable, model responded: ok={out.ok}")
        except Exception as e:
            print(f"  FAILED: {e}")
            return 1
    else:
        print("\nKey is present. Add --ping to make one real call and confirm it works.")
    return 0


def cmd_reset_demo(args: argparse.Namespace) -> int:
    """Put the demo back to a clean, repeatable state.

    Exists because of a real trap: the upload gates dedupe on content hash, so
    rehearsing the upload demo makes the live one show DEDUPE instead of the
    six-gate accept path. Rehearsal silently breaking the performance is the worst
    class of demo bug, so it gets a command rather than a note in a README.

    Clears uploaded invoices, their audit runs and the upload history. Leaves the
    seeded book, the tariff index and the main audit run untouched.
    """
    con = db.connect()
    ups = db.rows(con, "SELECT invoice_number FROM invoices WHERE source='upload'")
    n_hist = db.one(con, "SELECT COUNT(*) c FROM uploads")["c"]
    for u in ups:
        con.execute("DELETE FROM audit_results WHERE invoice_number=?", (u["invoice_number"],))
        con.execute("DELETE FROM decisions WHERE invoice_number=?", (u["invoice_number"],))
    con.execute("DELETE FROM invoices WHERE source='upload'")
    con.execute("DELETE FROM uploads")
    con.commit()
    for f in db.INVOICE_DIR.glob("upload-*.pdf"):
        f.unlink()
    con.close()
    print(f"Cleared {len(ups)} uploaded invoice(s) and {n_hist} upload record(s).")
    print("The upload demo is repeatable again.")
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    """Everything that could embarrass you tomorrow, checked in one command."""
    from . import intake, models, textlayer

    ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label:<42}{detail}")

    print("Credentials and models")
    check("ANTHROPIC_API_KEY present", llm.available())
    try:
        models.describe()
        check("model catalog resolves", True, str(models.for_stage("classify").id))
    except Exception as e:
        check("model catalog resolves", False, str(e)[:50])

    print("\nData")
    con = db.connect()
    n_inv = db.one(con, "SELECT COUNT(*) c FROM invoices WHERE source='seed'")["c"]
    n_cl = db.one(con, "SELECT COUNT(*) c FROM clauses")["c"]
    n_vec = db.one(con, "SELECT COUNT(*) c FROM clause_vectors")["c"]
    n_up = db.one(con, "SELECT COUNT(*) c FROM uploads")["c"]
    n_upinv = db.one(con, "SELECT COUNT(*) c FROM invoices WHERE source='upload'")["c"]
    run = db.one(con, "SELECT run_id, COUNT(*) n FROM audit_results GROUP BY run_id "
                      "ORDER BY MIN(id) DESC LIMIT 1")
    n_judgment = db.one(con, "SELECT COUNT(*) c FROM ground_truth WHERE blind_spot=1")["c"]
    check("seeded invoices", n_inv >= 8, f"{n_inv} (incl. the judgment invoice)")
    check("judgment cases present", n_judgment >= 1,
          f"{n_judgment} known blind spot(s) tracked")
    check("tariff clauses indexed", n_cl > 0 and n_cl == n_vec, f"{n_cl} clauses / {n_vec} vectors")
    check("an audit run exists", run is not None,
          f"{run['run_id']} ({run['n']} lines)" if run else "run the eval")
    check("upload demo is repeatable", n_up == 0 and n_upinv == 0,
          "clean" if n_up == 0 else f"{n_up} prior upload(s) --- run: reset-demo")

    print("\nSample files")
    d = db.DATA_DIR / "upload_samples"
    holdout, decoy = d / "holdout_carrier_invoice.pdf", d / "decoy_not_an_invoice.pdf"
    judgment_pdf = d / "judgment_carrier_invoice.pdf"
    check("holdout invoice present", holdout.exists())
    check("decoy present", decoy.exists())
    check("judgment invoice present", judgment_pdf.exists(),
          "the upload that makes the model decide")
    if decoy.exists():
        v = textlayer.triage_text(textlayer.extract_text(decoy))
        check("decoy is refused for free", v.confident and not v.is_freight_invoice)
    if holdout.exists():
        parsed = textlayer.parse_invoice(textlayer.extract_text(holdout))
        check("holdout parses deterministically", parsed is not None,
              f"{len(parsed.lines)} lines" if parsed else "would fall back to vision")

    print("\nUI")
    check("frontend build present", (db.DATA_DIR.parent / "static" / "index.html").exists(),
          "run: cd frontend && npm run build")
    check("eval report present", (db.DATA_DIR / "eval_report.json").exists(),
          "run: python eval/run_eval.py")
    con.close()

    print("\n" + ("Ready." if ok else "NOT ready --- fix the FAIL lines above."))
    return 0 if ok else 1


def cmd_alerts(args: argparse.Namespace) -> int:
    """What needs the owner's attention, and what to do about it."""
    from . import alerts as alerts_mod

    con = db.connect()
    r = alerts_mod.summary(con)
    con.close()
    if not r["alerts"]:
        print("  Nothing to action."); return 0

    print(f"  {r['action']} to action · {r['watch']} to watch · {r['info']} for information\n")
    tone = {"ACTION": "!!", "WATCH": " ~", "INFO": "  "}
    for a in r["alerts"]:
        print(f"  {tone[a['severity']]} [{a['severity']}] {a['title']}")
        for line in _wrap(a["detail"], 74):
            print(f"        {line}")
        print(f"        -> {a['suggested_action'].split('.')[0]}.")
        print()
    return 0


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width)


def cmd_observe(args: argparse.Namespace) -> int:
    """Stage telemetry, as an on-call engineer would want it."""
    from . import observe

    con = db.connect()
    r = observe.report(con, limit_days=args.days)
    con.close()
    if r.get("traces", 0) == 0:
        print(r.get("note", "no telemetry")); return 0

    print(f"  window {r['window_days']}d   {r['traces']} traces, {r['events']} stage events   "
          f"${r['total_cost_usd']:.4f}\n")
    print(f"  {'stage':<12}{'calls':>7}{'err':>6}{'p50 ms':>10}{'p95 ms':>10}{'cost':>10}")
    print("  " + "-" * 55)
    for name, s_ in r["by_stage"].items():
        print(f"  {name:<12}{s_['calls']:>7}{s_['error']:>6}{s_['p50_ms']:>10.1f}"
              f"{s_['p95_ms']:>10.1f}{s_['cost_usd']:>10.4f}")

    if r["extraction_method_mix"]:
        print("\n  extraction ladder --- which rung answered")
        for m, v in sorted(r["extraction_method_mix"].items(), key=lambda kv: -kv[1]["n"]):
            print(f"    {m:<14}{v['n']:>4}   {v['share'] * 100:>5.1f}%")

    scr = r["schema_conformance_rate"]
    if scr is not None:
        print(f"\n  schema conformance   {scr * 100:.1f}%   <- the drift alarm; it degrades")
        print(f"                                   before accuracy does when a layout changes")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn
    uvicorn.run("app.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="carrier-audit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("seed", help="build synthetic data + invoice PDFs")
    p.add_argument("--no-pdfs", action="store_true")
    p.set_defaults(fn=cmd_seed)

    p = sub.add_parser("run", help="audit every invoice")
    p.add_argument("--stub", action="store_true", help="skip vision extraction")
    p.add_argument("--strategy", default="hybrid",
                   choices=["bm25", "dense", "hybrid", "hybrid_rerank"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--model", help="Set every model stage at once.")
    p.add_argument("--extract-model")
    p.add_argument("--classify-model")
    p.add_argument("--rerank-model")
    p.add_argument("--max-spend", type=float, default=None,
                   help=f"Hard USD ceiling for this run (default "
                        f"${pipeline.DEFAULT_MAX_SPEND_USD:.2f}). The run stops when hit.")
    p.add_argument("--reuse-extraction", metavar="RUN_ID", default=None,
                   help="Replay a previous run's extraction instead of re-reading the "
                        "PDFs. Costs only the classifier. Use 'last' for the latest run.")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("ask", help="query the tariff corpus")
    p.add_argument("question")
    p.add_argument("-k", type=int, default=4)
    p.add_argument("--strategy", default="hybrid",
                   choices=["bm25", "dense", "hybrid", "hybrid_rerank"])
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("prompt", help="Print the exact classifier request; sends nothing.")
    p.add_argument("--pro", default=None, help="Show the request for this PRO number.")
    p.add_argument("--strategy", default="hybrid",
                   choices=["hybrid", "bm25", "dense"])
    p.add_argument("--raw", action="store_true", help="Print the request body as JSON.")
    p.set_defaults(fn=cmd_prompt)

    p = sub.add_parser("doctor", help="check credentials and config; spends nothing")
    p.add_argument("--ping", action="store_true",
                   help="Make one tiny real API call to confirm the key works.")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("reset-demo", help="clear uploads so the upload demo is repeatable")
    p.set_defaults(fn=cmd_reset_demo)

    p = sub.add_parser("preflight", help="check everything before a demo; spends nothing")
    p.set_defaults(fn=cmd_preflight)

    p = sub.add_parser("alerts", help="recurring gaps that need the owner to change something")
    p.set_defaults(fn=cmd_alerts)

    p = sub.add_parser("observe", help="stage telemetry: latency, outcomes, cost, drift alarm")
    p.add_argument("--days", type=int, default=30)
    p.set_defaults(fn=cmd_observe)

    p = sub.add_parser("serve", help="run the API and review UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(fn=cmd_serve)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
