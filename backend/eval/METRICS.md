# What is measured, and why

Every metric in `run_eval.py` is here with four things: what it is, why *this* one
rather than an obvious alternative, what it costs to be wrong on it, and what it
fails to capture. A metric nobody can justify is a metric nobody should act on.

The organising idea: **this system has three failure surfaces that fail
independently**, so it needs three families of metric. Extraction can be perfect
while classification is useless. Classification can be excellent while the policy
layer loses money. Retrieval can be flawless and contribute nothing. A single
headline number would hide all three.

---

## 1. Classification quality

### Per-category precision and recall

**Definition.** For a category like `REWEIGH`: precision is *of the lines we
labelled REWEIGH, how many really were*; recall is *of the lines that really were
REWEIGH, how many did we find*.

**Why per-category rather than one overall number.** The categories have completely
different consequences, so a blended score is meaningless. Missing a
`DUPLICATE_BILLING` means paying an invoice twice. Missing a `WITHIN_TOLERANCE`
means a reviewer spends a minute on a line that was fine. Those are not
interchangeable errors and should not be averaged into one figure.

**What being wrong costs, per direction:**

| Error | Operational cost |
|---|---|
| Low recall on a defect category | A real overcharge is not named. It is still escalated (code found the variance), so a human still sees it — the cost is context, not money |
| Low precision on a defect category | We tell a reviewer "this is a reweigh" when it isn't. They waste time building the wrong dispute, or worse, send it to the carrier |
| Anything → `UNKNOWN` | The cheapest possible failure. Routes to a human by design |

**What good looks like here.** Precision matters more than recall for every defect
category, because code has already caught the variance — the classifier is naming
it, not detecting it. A wrong name is worse than no name.

### Macro F1, not micro or weighted

**Definition.** F1 is the harmonic mean of precision and recall. Macro means every
category counts equally regardless of how many examples it has.

**Why macro.** The classes are heavily imbalanced — 21 `NO_VARIANCE` against 4
`DUPLICATE_BILLING`. A micro or weighted average lets a system look excellent by
being right about the common, easy case and wrong about every rare, expensive one.
Macro-averaging refuses that trade: getting `DUPLICATE_BILLING` wrong costs exactly
as much in the score as getting `NO_VARIANCE` wrong, which is closer to how it
costs in reality.

**What it misses.** Macro over-weights tiny categories. With n=4, one error moves
`DUPLICATE_BILLING`'s F1 by 0.25 and macro F1 by 0.03. On a 60-line set that is
noise wearing a decimal point. **Do not read a 2-point macro F1 move as a real
change at this sample size.**

### Accuracy

Reported, but the weakest number in the report and deliberately not the headline.
On a set that is 35% `NO_VARIANCE`, a system that labels everything `NO_VARIANCE`
scores 35% while being completely useless. Accuracy is included because people
expect it and because a large gap between accuracy and macro F1 is itself
diagnostic — it means performance is concentrated in the easy classes.

### Confusion pairs, with direction

**Why direction is recorded, not just a count of errors.** `REWEIGH → UNKNOWN` and
`REWEIGH → RATE_CARD_MISMATCH` are the same "one error" to accuracy and completely
different events operationally. The first is the model abstaining — safe, routed to
a human, working as designed. The second is a confident wrong answer that could put
a bad dispute in front of a carrier.

**This is the metric that found the real bug.** Three reweighs went to `UNKNOWN`
rather than to a wrong category, which said the model was abstaining rather than
confused — which pointed at the evidence being incomplete rather than the prompt
being weak. An undirected error count would have said "3 errors" and taught nothing.

---

## 2. Business outcome

### Dollar recall, not line recall

**Definition.** Of the dollars genuinely recoverable, what share did the system
flag for recovery (auto-disputed or escalated)?

**Why dollar-weighted rather than counting lines.** The line counts and the money
do not correlate. A $1,900 duplicate and a $4 accessorial are one line each and
nothing like each other. Freight audit exists to recover money, so the metric is
denominated in money. A system that catches 90% of *lines* while missing the one
large one has failed at the job it was bought for.

**What being wrong costs.** Missed dollars are permanently lost — you paid the
invoice. This is the metric with no recovery path, which is why it is the one to
optimise if forced to choose.

**What it misses.** It says nothing about *effort*. A system that escalates every
line scores 100% dollar recall and saves nobody any work. It has to be read against
the escalation rate — see below. Neither number is meaningful alone.

### False disputes — tracked separately, on purpose

**Definition.** Clean lines the system auto-disputed to the carrier without a human.

**Why it is not folded into precision.** Because it is denominated in a different
currency. Every other error costs money or minutes; this one costs **carrier
relationship**, which no credit memo recovers and which does not convert to
dollars. Averaging it into a precision figure would let a good dollar number pay
for a bad relationship, which is not a trade anyone at the company would authorise.

**Target: zero, and it is a release gate rather than a score to improve.** This is
why the policy layer auto-disputes exactly one category — duplicate billing, where
the evidence is documentary rather than interpretive. A duplicate PRO is a fact. A
reweigh being invalid is an argument.

### Automation rate and escalation rate

**Definition.** The share of lines that reached a decision with no person, and its
complement.

**Why both, when one is `1 −` the other.** Because they are read in opposite
directions and by different people. Automation rate is what the business bought.
Escalation rate is the reviewer's workload, and it is the number that must not
*silently drop* — a sudden fall usually means something broke open, not that the
system got smarter. Reporting both makes the alarm obvious rather than requiring
arithmetic.

**What it costs to game.** Automation rate is trivially inflated by widening the
auto-approve band, which converts directly into missed dollars. **Never read it
without dollar recall next to it.** The pair is the metric; either alone is a
half-truth.

### Human override rate

**Definition.** Of the lines a reviewer decided, how often did they choose a
different category than the system?

**Why this is the best single signal once live.** Every other metric on this page
needs ground truth, and production has none — nobody labels invoices for you. The
override rate needs no labels: it is generated for free by people doing their job.
It is the only metric here that survives contact with production unchanged.

**And the override log is the next eval set.** Every override is a labelled example
of a case the system got wrong, produced by a domain expert, at zero marginal cost.

*Currently `null` in the report — no reviewer decisions have been logged.*

---

## 3. Retrieval quality

### Recall@k of the governing clause

**Definition.** For each defect category, is the tariff rule that actually governs
it in the top *k* retrieved chunks?

**Why recall and not precision.** The consumer of retrieval here is (a) a language
model that reads all *k* results, and (b) a human who is shown them. Both tolerate
an extra irrelevant clause; neither tolerates the governing rule being absent. A
missing clause means the reviewer disputes without the rule in front of them.

**Why k=4.** It is what the pipeline actually passes to the classifier. Measuring
recall@10 while shipping k=4 measures a system nobody runs. The eval mirrors the
production call, including using the same query builder, so the number transfers.

**Why MRR alongside it.** Recall@4 is binary and saturates — every strategy here
scores 100%, so it cannot rank them. Mean reciprocal rank asks *where* in the list
the right clause landed, which is what separates BM25 (MRR 1.000, always first)
from the dense retriever (0.917, sometimes second). Position matters because the
model attends more to earlier context and a reviewer reads top-down.

**What this measurement actually bought.** It killed a feature. The LLM rerank
stage scored identically to BM25 alone on both numbers while costing a model call
per query, so it is off by default. That decision is worth more than the metric.

**What it misses.** The corpus is 14 clauses from a single synthetic tariff. Recall
against 14 chunks says almost nothing about recall against 14,000. Treat 100% as
"retrieval is not currently the bottleneck", never as "retrieval is solved".

---

## 4. Cost

Cost per audited line, and cost split by stage and by model. Reported because it is
a real input to a build-versus-buy decision, and split by stage because
"which stage is expensive?" is the question that makes model selection tractable.
Prompt-cache hit rate is asserted in the same block — a cache nobody verifies is a
cache that silently stopped working, and a cost table computed over a broken cache
is wrong in the expensive direction.

---

## What none of this measures

Naming the blind spots is part of the job. These are known and unaddressed:

1. **Extraction accuracy is not measured field-by-field.** The sum-check guard is
   binary — an invoice passes or it doesn't. A transcription that is wrong but
   internally consistent (two amounts transposed between lines) passes the guard
   and is never scored. Fixing that needs a field-level golden set against the
   generator's own output.
2. **The eval set is synthetic and self-generated.** This repo made the invoices
   *and* injected the defects it grades itself on. That is the right way to get a
   stable, repeatable set and the wrong way to estimate production accuracy. Real
   carrier invoices have layouts this generator never produced.
3. **n=60.** Small enough that one line is 1.7 points of accuracy. No confidence
   intervals are reported, and differences under ~5 points should not be treated as
   real.
4. **No latency budget.** 149 seconds for nine invoices off the text layer (307 when
   the vision path runs) is fine for a nightly batch
   and useless if anyone ever wants this interactive. Wall time is reported but
   nothing is measured against a target.
5. **No adversarial cases.** Every invoice is well-formed. Nothing tests a scanned
   page, a rotated page, a handwritten annotation, a carrier who changes their
   layout, or a corrupt PDF.
6. **The policy thresholds are asserted, not derived.** The $25 auto-approve ceiling
   is a defensible starting guess, not an optimum. Deriving it needs the real
   distribution of variances and the real cost of a review-minute.
7. **Retrieval is scored on 6 hand-specified category→rule pairs**, not on a
   labelled query set. It measures whether the obvious clause is found for the
   obvious case.
8. **The category taxonomy is incomplete, and the eval can see it.** There is no
   label for *a legitimate accessorial billed in excess of what the tariff
   permits*. Detention past MFS-220's eight-hour cap is the case: the accessorial
   was authorised, only the excess hours were not. Ground truth says
   `UNAUTHORIZED_ACCESSORIAL`; the model retrieves MFS-220, reads it, and answers
   `UNKNOWN`. **The model is arguably right and the label is arguably wrong.** It
   is left as a scored miss rather than relabelled, because changing ground truth
   to match model output improves the number and not the system. The real fix is a
   category like `ACCESSORIAL_OVERBILLED`, which is a taxonomy change and needs the
   A/P desk's agreement, not an engineer's.
9. **Recall@4 is measured on the queries the PIPELINE constructs, not on free
   text a human types — and the gap is real.** Spot-checked 2026-09-10 against ten
   natural ops-language questions, the retriever found the governing clause in the
   top three **8 times out of 10**. The two misses are instructive: *"they charged
   us for a tailgate"* returns nothing relevant, because BM25 has no lexical
   overlap with "liftgate" and the default hashing embedder matches characters
   rather than meaning; and *"same shipment invoiced twice"* returns the
   residential/limited-access clause instead of the duplicate-billing one. **The
   100% figure is honest for what it measures and would be dishonest if quoted as
   free-text search quality.** The fix for the first miss is a real embedding
   backend (`VOYAGE_API_KEY`); the eval would then show what it bought.

---

## If you can only optimise one

**Dollar recall**, with **false disputes at zero** as a hard gate.

Everything else is instrumentation for understanding *why* those two moved. A
system with 80% classification accuracy, 100% dollar recall and no false disputes
is doing its job; a system with 99% accuracy that missed the one large duplicate
is not.
