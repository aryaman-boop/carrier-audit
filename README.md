# carrier-audit

A/P reconciliation against carrier billing: read the carrier's invoice, price the
shipment ourselves from the contracted rate card, explain the difference, and put
the ones that need a person in front of a person with the evidence attached.

> **This is a demo build over synthetic data.** Synthetic carrier, synthetic rate
> card, synthetic invoices with a known set of injected billing defects. It is not
> deployed, has no users, and has never seen a real carrier invoice. It exists to
> show a way of building this kind of workflow, and the numbers below are measured
> on the synthetic set — they are not a claim about production accuracy.

---

## Ingestion: upload a PDF, six gates, cheapest first

`POST /api/upload` (and the panel in the UI) takes a PDF and runs it through a
chain ordered so that **every gate which can refuse a document for free runs before
any gate that costs money**:

| # | Gate | Checks | Cost |
|---|---|---|---|
| 1 | FILE | `%PDF-` magic bytes, size, page count, encryption | free |
| 2 | DEDUPE | content hash, invoice number already on file | free |
| 3 | TRIAGE | is this even a carrier freight invoice? | free, or ~$0.002 |
| 4 | EXTRACT | read it | free, or ~$0.07 |
| 5 | SUM CHECK | line items must reconcile to printed totals | free |
| 6 | OWNERSHIP | our carrier? our shipments? | free |

Measured, every refusal path costing nothing:

```
renamed text file    -> FILE     $0.00   This is not a PDF.
design studio bill   -> TRIAGE   $0.00   Not a carrier freight invoice.
same file again      -> DEDUPE   $0.00   This exact file has already been processed.
invoice already held -> DEDUPE   $0.00   Invoice MFS-INV-9100 is already on file.
holdout invoice      -> ACCEPTED $0.05   6 lines, 6/6 PRO numbers matched, $207.23 recoverable
```

**Gate 6 is the one people forget.** A document can be a valid freight invoice,
pass every technical check, and still not be *ours*. The test is whether the PRO
numbers resolve against our own shipment book; if they don't, auditing it would
compare a stranger's freight to our rate card and produce confident nonsense.

`app.cli seed` generates two samples under `data/upload_samples/` — a holdout
invoice that should be accepted and a decoy that should not.

## Extraction is a ladder, and the model is the top rung

The question worth asking before reaching for a model: *does one need to read this
at all?*

```
1  pdfplumber layout parse   free   deterministic
2  arithmetic validation     free   the parse must reconcile
3  vision model              ~$0.07 only when 1 or 2 fail
```

**Measured on these invoices, not assumed:**

- **MarkItDown** (Microsoft's `markitdown`, wrapping `pdfminer.six`) returns all
  the text *including every amount* — but in reading order, so right-aligned
  figures land in a separate run at the end, decoupled from the descriptions they
  belong to. Excellent for deciding *what a document is*. Unusable for deciding
  *what it says*.
- **pdfplumber with `layout=True`** preserves the spatial arrangement, so
  `Linehaul transportation charge ... $856.06` stays on one line. Parseable.

So all 8 invoices now extract **for free**, and the run cost fell from **$0.99 to
$0.43** — 31 model calls, all classification, zero extraction.

**Why the model still exists**, and these are the honest limits:

1. **Scanned invoices have no text layer.** The ladder detects that and falls to
   vision.
2. **Layout parsing is per-carrier.** This parser knows one template. The model
   generalises; the parser does not, and a parser per carrier is exactly the
   maintenance burden that made everyone reach for models in the first place.
3. **The arithmetic gate is what makes the cheap rung safe.** A regex that
   mis-associates an amount fails with total confidence, which is worse than a
   model misreading one. The parse is not trusted until it reconciles.

The same reasoning removes the triage model call: freight vocabulary, invoice
vocabulary and PRO-shaped tokens decide the obvious cases for nothing, and only a
genuinely ambiguous document costs $0.002.

## The argument

Most "AI for X" demos put a model at the centre and arrange the system around it.
This one does the opposite, and the reason is that in an audited finance workflow
almost nothing that matters is a language problem:

| Step | Who does it | Why |
|---|---|---|
| Read the invoice PDF | code, then **model** (vision) | Text-layer parse first; the model is the fallback for scans and layouts the parser does not know. |
| Check the line items sum to the total | code | Arithmetic. Catches most transcription error for free, and is what makes the free parse safe to trust. |
| Match invoice line → our shipment | code | A key lookup, with fuzzy entity resolution as the fallback. |
| Price the shipment | code | A join and multiplication. Non-determinism here is a defect. |
| Retrieve the governing clause | code | BM25 over the tariff, queried from the computed evidence rather than the invoice text. |
| Name the variance | code, then **model** (constrained enum) | Duplicates, unmatched lines and in-band variances are named by code. The model is reached only for a real variance code cannot explain. |
| Decide approve / dispute / escalate | code | Business policy. Finance changes it without touching a prompt. |

**The model names the variance. It does not find it, and it does not price it.**

Two model calls per invoice at most on the batch path, and on an upload a third --
a fifth of a cent of triage -- only when the free gate cannot decide. Roughly half
the lines never reach a model at all, because a duplicate PRO is a set membership
test, a two-cent variance is arithmetic, and a variance inside the auto-approve
band is a threshold.

**On the last run the extraction stage used no model whatsoever.** All nine invoices
came off the text layer (`mode: ladder -> text-layer x9`), and 35 of 66 lines were
named by code before any call was made -- 31 classify calls for the whole batch, and
not one extract call.

This is a **workflow, not an agent**. The steps are known in advance and run in a
fixed order; there is nothing for a planner to decide. An agent loop here would add
non-determinism, cost and failure modes to a process that has none of them.

### Where retrieval actually belongs

> The rate table is a join. The tariff rules are prose.
> Only the second one is a retrieval problem.

Rates, weight breaks, class factors and accessorial prices are keyed data — they
live in SQL and a model never sees them. What *is* retrieved is the rules text:
when a liftgate charge is valid, how a reweigh tolerance is computed, what
documentation the carrier owes on a dispute. That text is genuinely unstructured,
it is what a human auditor argues from, and the reviewer needs it **quoted in
full**, never paraphrased — a summary of a contract term is a liability in a
dispute.

Retrieval is hybrid: SQLite FTS5 BM25 + dense vectors, fused with reciprocal rank
fusion, with an optional LLM rerank. Measured per stage — see below.

---

## Running it

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
cd backend

python -m app.cli seed                 # synthetic book + invoice PDFs
python -m pytest tests/ -q             # 20 tests, no API key needed
python eval/run_eval.py --stub         # full eval, no API key needed
python -m app.cli serve                # review UI at http://127.0.0.1:8000
```

### The API key

Put it in a `.env` file at the **project root** — `~/Desktop/Projects/carrier-audit/.env`,
the same directory as `README.md`. It is loaded automatically by every entry point
(CLI, API server, eval harness, tests), from any working directory.

```bash
cp .env.example .env        # then edit, or:
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
python -m app.cli doctor    # confirms it loaded. Spends nothing.
python -m app.cli doctor --ping   # one tiny real call. Well under a cent.
```

A real exported environment variable always wins over the file, so
`ANTHROPIC_API_KEY=... python -m app.cli run` still works for a one-off override.
`.env` is in `.gitignore`, `doctor` prints only the last four characters, and the
key is never logged.

Then:

```bash
python eval/run_eval.py                # the real thing
```

### Cost control

Every run has an **enforced spend ceiling**, defaulting to `$2.50` — checked before
each invoice, so it is a ceiling rather than a number the run may overshoot once.
When it trips the run stops, keeps everything already audited, and marks the eval
report `PARTIAL` so a truncated run can never be mistaken for a complete one.

```bash
python -m app.cli run --limit 1 --max-spend 0.50    # smoke test, ~$0.20
python eval/run_eval.py --max-spend 2.00            # full run under a cap
```

`--stub` makes no API calls at all, so it is free and the ceiling never applies to
it. Start there, then spend one invoice, then run the whole thing.

Prompt caching makes repeat runs cheaper than the first: the system prefix is
marked cacheable and `cache_read_tokens` in the report tells you whether it is
actually working. If that number is zero across runs, something is invalidating
the prefix and the cost table above is wrong.

### Choosing models

Nothing here requires Opus 5. It is the default because it is the safe default,
not because the pipeline depends on it. **Each of the three model stages is
configured independently**, because they are not the same job and pretending they
are is how you end up paying frontier prices to pick a label from a list of nine.

```bash
python eval/run_eval.py --model claude-sonnet-5           # all three stages
python eval/run_eval.py --extract-model claude-opus-5 \
                        --classify-model claude-haiku-4-5 # mix
```

Or `CARRIER_AUDIT_MODEL`, `CARRIER_AUDIT_MODEL_EXTRACT`, `..._CLASSIFY`,
`..._RERANK`.

| extract | classify | run cost (8 invoices, 15 pages, 60 lines) | per line |
|---|---|---|---|
| opus-5 | opus-5 | $1.59 | $0.027 |
| opus-5 | haiku-4.5 | $1.05 | $0.018 |
| sonnet-5 | haiku-4.5 | $0.50 | $0.008 |
| haiku-4.5 | haiku-4.5 | $0.32 | $0.005 |

Projected from real page counts (15 pages across 8 PDFs) at a conservative 2,500
tokens per dense page, not from a run — the accuracy column is the
one that needs a key, and **cost without accuracy is not a decision**. The eval
harness exists precisely so that choice can be measured rather than argued:
run it per mix and compare macro F1 and dollar recall against the cost.

Where the money and the risk actually sit:

- **extract** is the stage to spend on. It is vision over a dense multi-page
  document, and everything downstream trusts its transcription. It is also the
  stage the sum-check guard protects, so a cheaper model here is *safe to try* —
  a bad read fails the arithmetic and escalates rather than silently corrupting an
  audit.
- **classify** is the stage to save on. It picks one of nine labels given numbers
  code already computed and clauses retrieval already found. Most of the thinking
  has been done for it.
- **rerank** is trivial ordering. Haiku, or turn it off.

`app/models.py` holds the catalog, and **an unknown model id is refused rather
than sent blind** — because the request shape is not uniform across the family and
a wrong body is a 400, not a worse answer:

- Haiku 4.5 accepts neither `output_config.effort` nor adaptive thinking; it takes
  the older `budget_tokens` form. An Opus-shaped request to Haiku fails outright.
- `xhigh` / `max` effort exist on Opus 5/4.8/4.7 and Sonnet 5 but not on Opus 4.6
  or Sonnet 4.6, so effort is clamped to what each model supports.
- Haiku 4.5 has a 200K context window against 1M for the rest.

Non-Anthropic providers are a bigger change than a config flag. `app/llm.py` is the
only file that talks to a model, so the swap is contained to one file — but
structured outputs, the document content-block shape and the thinking parameters
all differ, so it is a rewrite of that file rather than a setting.

Rebuild the UI after a frontend change:

```bash
cd frontend && npm install && npm run build   # outputs into backend/static/
```

Other commands:

```bash
python -m app.cli ask "they billed a liftgate at a dock-equipped warehouse"
python -m app.cli prompt                 # the exact classifier request, rebuilt from a real line
python -m app.cli prompt --pro <PRO> --raw
python -m app.cli run --stub --strategy bm25
python eval/run_eval.py --stub --strategy hybrid_rerank
```

`prompt` prints what the classifier would send and sends nothing: the cacheable system
block, the evidence the rating engine built, the clauses the retriever returned, and the
output schema that makes an invalid category unrepresentable. It is rebuilt from a stored
audited line rather than transcribed, so it cannot drift from the request the code
actually makes.

`--stub` swaps vision extraction for the generator's own output, so the rest of the
pipeline runs with no key and no cost. It makes extraction **100% accurate by
construction**, so any eval using it measures the classifier and policy layer only.
The report labels those runs, because that number must never be quoted as if the
model had read the page.

---

## Measured results

Latest run (`run-1753ed29`), 9 invoices / 16 pages / 66 lines, synthetic set:

```
extraction               ladder -> text-layer x9
                         9/9 invoices read WITHOUT a model call
lines audited            66        (incl. 5 judgment cases)
decided by code          35        before any model call
automation rate          56.1%     33 approved, 4 disputed
escalation rate          43.9%     29 to review

classification accuracy  93.8%     61 of 65 scored, macro F1 0.912
known blind spots        1         EXCLUDED from the score, counted separately
dollar recall            99.76%    $7,769.14 identified, $18.71 missed
false disputes           0
invoices blocked by guard 0

run cost                 $0.46     $0.0070/line, 31 calls, ALL classification
                                   (was $0.99 before the extraction ladder)
```

All four misses, by name:

| PRO | truth | variance | called | by |
|---|---|---|---|---|
| MFS470102900 | FUEL_SURCHARGE_ERROR | $1.15 | WITHIN_TOLERANCE | code |
| MFS396019009 | FUEL_SURCHARGE_ERROR | -$0.67 | WITHIN_TOLERANCE | code |
| MFS194720263 | FUEL_SURCHARGE_ERROR | -$1.49 | WITHIN_TOLERANCE | code |
| MFS421130405 | UNAUTHORIZED_ACCESSORIAL | $195.00 | UNKNOWN | **model** |

Three of the four never reached a model: they are fuel variances of a dollar or
less that the $2.00 / 1% band absorbed, and two of those three are the carrier
billing *under* the rate card, so there is nothing to recover on either. The
fourth is the detention line, left as a scored miss on purpose.

And the $18.71 that was missed was not missed by detection. $1.15 sat inside the
rounding band; $17.56 (`MFS726998810`) was **classified correctly** and
auto-approved under the $25 ceiling.

Accuracy fell from 98.4% when the judgment cases were added — 98.3% -> 95.3% ->
96.9% after the retrieval fix, and 93.8% on the current set. That is the point:
they are hard on purpose. A score that only ever goes up is measuring the wrong
thing.

### The judgment invoice

`MFS-INV-9300` exists because every other defect here is mechanically detectable,
which makes for a flattering demo and a dishonest one: the model only ever names
what arithmetic already found.

**Five lines, four of which code cannot resolve.** On every one,
`unsupported_accessorials` is empty and each machine-checked fact reconciles --
code can see money is missing and cannot say why. The answer is only in the clause.

| line | variance | the contractual point |
|---|---|---|
| Winter operations surcharge | $185.00 | no contracted rate exists for it |
| Liftgate billed twice | $92.00 | MFS-200 caps it at one per delivery |
| 11 hours of detention | $195.00 | MFS-220 caps it at 8 in a day |
| Fuel on the wrong base | $57.96 | MFS-130 applies it to linehaul only |
| Residential + limited access | $0.00 | MFS-210 forbids the pair --- **blind spot** |

**Line 1 --- a charge the contract has no rate for.** A "Winter operations
surcharge" of $185. `unsupported_accessorials` is empty, weight, class, fuel and
linehaul all reconcile, and code has nothing to say. The model read the tariff,
found no provision for it, and returned `UNKNOWN`:

> *"The entire $185.00 variance is an undescribed 'OTHER' line item with no
> contracted counterpart, while weight (+0 lb), class (+0), fuel (24.60% vs 24.60%
> published) and linehaul ($856.06) all reconcile."*

That is the system declining rather than inventing a category, and it is the one
line in the book where the model does work code could not.

**Line 2 --- a blind spot, kept deliberately.** Residential delivery *and* limited
access, both requested on the bill of lading, both priced correctly. MFS-210 says
they shall not both be assessed and the lower is waived --- so the carrier owes
$76.50. But the variance is **$0.00**, so nothing escalates and the system never
sees it.

**The pipeline escalates on dollar variance. A charge that is correctly priced but
contractually impermissible is invisible to it.** That line is flagged
`blind_spot=1` in ground truth, excluded from the score and counted separately:
scoring it as a model error would blame the classifier for a design limitation, and
hiding it would be worse. Catching it needs a permissibility pass over accessorial
*combinations* --- not a better classifier.

### What the judgment cases caught --- a second evidence bug, one layer up

Two of them initially went to `UNKNOWN` with the model diagnosing the arithmetic
perfectly and then declining. The citations gave it away: one cited MFS-400 and
MFS-130, the other cited **nothing at all**. Neither had seen its governing clause.

`build_query` builds the retrieval query from the *machine-checked evidence* --- so
when code could not characterise the problem, no condition fired, the query fell
through to a generic "application of rates" string, and the right rule was never
retrieved. **An unexplained variance is exactly when retrieval matters most, and it
was exactly when the query carried the least information.**

The fix: a residual variance now pulls the vocabulary of every accessorial actually
on the invoice. All four judgment cases retrieve their governing clause, and the
double-liftgate line went from `UNKNOWN` to `UNAUTHORIZED_ACCESSORIAL` citing
MFS-200. Accuracy 95.3% -> 96.9%.

**The detention line still answers `UNKNOWN` --- and now cites MFS-220.** It
retrieved the rule, read it, and declined anyway. That is a taxonomy gap rather
than a model failure: there is no category for *a legitimate accessorial billed in
excess of what the tariff permits*. It is left as a scored miss rather than
relabelled, because changing ground truth to match model output improves the number
and not the system.

Per category:

| category | precision | recall | n |
|---|---|---|---|
| NO_VARIANCE | 1.000 | 1.000 | 21 |
| REWEIGH | 1.000 | 1.000 | 7 |
| UNAUTHORIZED_ACCESSORIAL | 1.000 | 1.000 | 7 |
| WITHIN_TOLERANCE | 0.857 | 1.000 | 6 |
| FUEL_SURCHARGE_ERROR | 1.000 | 0.800 | 5 |
| RATE_CARD_MISMATCH | 1.000 | 1.000 | 5 |
| RECLASS | 1.000 | 1.000 | 5 |
| DUPLICATE_BILLING | 1.000 | 1.000 | 4 |

**The deterministic core alone scores 51.7% accuracy and 100% dollar recall with
zero model calls.** The rating engine finds and quantifies every variance on its
own; the model's contribution is naming it. That split is the design.

### What the eval actually caught

The first vision run scored 93.3%, with REWEIGH recall at 0.571 — much the worst
category. The model's own reasoning explained why:

> *"The +25 lb weight delta falls at or within the MFS-110 tolerance ... so the
> 1,950 lb BOL weight governs and would explain only about $6 of the $44.94
> variance"*

It was right. The other $38 was a `REWEIGH_FEE` line, and `_evidence()` showed the
classifier the *expected* charges but never the charges the carrier actually
**billed**. It was asked to account for $44.94 while shown facts covering $6 of it,
and it abstained to `UNKNOWN` — the correct response to incomplete evidence, which
the eval scored as a miss.

Adding a charge-by-charge reconciliation to the evidence took accuracy to **98.3%**
and REWEIGH to a perfect 7/7. Re-verifying cost **$0.38** because
`--reuse-extraction` replays the stored transcription rather than re-reading the
PDFs.

Two things worth taking from that:

- **When a classifier abstains, check the evidence before you touch the prompt.**
  Prompt-tuning that failure would have taught the model to guess.
- **The business number did not move.** Dollar recall stayed at 99.6% and the
  missed dollars stayed at $23.15, because those three lines were escalated to a
  human either way. The fix improved the *label a reviewer sees*, not the money
  recovered. Reporting it as a recovery improvement would be dishonest.

The one remaining error is a `$1.17` fuel variance classified `WITHIN_TOLERANCE` —
which it is, under the $2.00 / 1% band. That is the policy layer deciding a dollar
is not worth a reviewer's minute, and it is left visible rather than tuned away.

Retrieval, recall@4 of the governing tariff clause:

| strategy | recall@4 | MRR |
|---|---|---|
| bm25 | 100.0% | 1.000 |
| dense (local hashing embedder) | 100.0% | 0.917 |
| hybrid RRF | 100.0% | 1.000 |
| hybrid + LLM rerank | 100.0% | 1.000 |

**BM25 outranks dense on this corpus, and rerank buys nothing at all.** Both are the
expected results and both are worth having measured. The queries are exact tokens —
*reweigh*, *liftgate*, *NMFC*, rule numbers — and lexical retrieval is better than
dense at exact tokens. The rerank stage costs a model call per query and moved
neither recall nor MRR, so it stays off.

---

## What is deliberately not here

- **No agent loop.** See above. It would be the wrong shape.
- **No knowledge graph.** The entities here — shipment, invoice, carrier, lane,
  facility — already have foreign keys. Extracting an LLM-built graph on top of
  data that already has referential integrity pays extraction cost to lose it.
  GraphRAG earns its keep on thematic questions over an unstructured corpus; this
  is not that.
- **No vector database.** A few hundred clauses. SQLite FTS5 here, `tsvector` +
  `pgvector` on the Postgres you already run in production. A third system to
  operate would need to justify itself and cannot.
- **No LLM arithmetic.** Anywhere. See `pricing.py`.
- **No confidence score from the model.** A model's confidence in its own answer is
  the weakest signal available. The escalation triggers are numeric: dollar
  thresholds, match confidence, and an explicit `UNKNOWN` category.

## Layout

```
backend/app/
  schema.py      the output contract — every closed vocabulary in one file
  ratecard.py    the carrier contract, split into its structured and prose halves
  pricing.py     the rating engine. pure arithmetic, no model, ever
  seed.py        synthetic shipments + invoices with injected defects + ground truth
  pdfgen.py      renders the invoice PDFs the vision stage reads
  extract.py     stage 1 — vision transcription + the sum-check guard
  recompute.py   stage 2 — entity resolution + recompute from the rate card
  retrieve.py    hybrid BM25 + dense + RRF + optional rerank
  classify.py    stage 3 — constrained variance classification over retrieved rules
  policy.py      stage 4 — approve / dispute / escalate. business policy, in code
  pipeline.py    orchestration
  api.py         review queue API
backend/eval/
  run_eval.py    classification, retrieval and business-outcome metrics
  METRICS.md     what each metric is, why it and not the alternative, what it
                 costs to be wrong on it, and what none of them measure
backend/tests/
  test_model_paths.py  the three model-dependent paths, against a fake client
frontend/        React + TypeScript review UI
```

## Known gaps

- The dense retrieval backend defaults to a **local hashing embedder**, not a
  semantic model — it captures lexical and morphological overlap and nothing more.
  Set `VOYAGE_API_KEY` for real embeddings; the eval will show what the upgrade
  buys. Calling the default "semantic search" would be exactly the overstatement
  this project argues against.
- Everything above is measured on **synthetic invoices this repo generated**, whose
  defects it also injected. That is the right way to get a stable eval set and the
  wrong way to estimate production accuracy: real carrier invoices have layouts this
  generator never produced, and a real rate card has edge cases this one does not.
  Treat 98.3% as evidence the pipeline is sound, not as a forecast.
- The sum-check guard has **never fired on a real run** — 0 of 8 invoices blocked.
  It is covered by tests in both failure modes, but its production behaviour is
  unobserved.
- One defect line (`FUEL_SURCHARGE_ERROR`, $1.17) falls inside the rounding band
  and is auto-approved. That is arguably correct behaviour rather than a miss, but
  it is a real threshold interaction and it is left visible rather than tuned away.
- `POST /api/run` is synchronous. Fine for 8 invoices; a real vision run is minutes
  of work and belongs on a queue, not in a request handler.
