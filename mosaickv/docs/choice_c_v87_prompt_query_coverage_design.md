# Choice C v87: prompt-query coverage for OCR/text evidence

## Status and scientific boundary

This is an implemented, preregistered calibration candidate. At the time of
preregistration it had not been run on a benchmark, had no measured quality
result, and was not a paper claim. It does not revise v86. No v86 answer,
reference, score, failed-sample identity, generated token sequence, or
per-sample telemetry was used to choose a block or threshold. Calibration must
use a new disjoint ledger, and any held-out ledger must remain unmaterialized
until a candidate is frozen.

The complete behavior is versioned in
`configs/experiments/choice_c_v87_prompt_query_coverage_development.yaml`.
The new path is disabled by default. Revision 87.1 preregisters 32 fresh
DocVQA and 32 fresh TextVQA calibration examples, paired FullKV and Prompt
Top-K controls, four label-blind folds, and the unchanged memory and fidelity
gates.

## Problem addressed

The stable-head selector previously reduced the last prompt window to one
number per source block. Conceptually, it averaged the attention from all
question/prompt tokens and then selected the blocks with the largest total
mass. This is efficient, but an average can hide rare evidence. For example,
one question token may strongly attend to a small printed number while the
other prompt tokens attend elsewhere. The number's block can have a modest
average even though losing it changes the answer.

LLaVA exposes image-grid positions but does not expose OCR boxes or OCR text.
The implementation therefore does not claim to use an OCR system. It protects
query-conditioned visual and text evidence using only the model's prompt-side
attention, without task labels or answer text.

## Algorithm

For every layer and KV head, eager prefill attention has shape
`[query head, prompt query, source position]`. Grouped query heads are averaged
into the corresponding KV head, but the final prompt-query axis is retained.
Token attention is then summed within each fixed-size KV block. Each block has
a vector whose entries say how much each recent prompt query attends to it.

The ordinary stable-head Prompt Top-K selection remains the immutable anchor.
For selected set `S`, the new coverage value is:

```text
C(S) = sum over prompt queries q of max over selected blocks b of A[q, b]
```

The reranker considers removing one selected optional block and adding one
unselected block. A replacement is accepted only when all conditions hold:

1. donor and recipient have the same token count;
2. donor and recipient have the same modality;
3. neither mandatory block membership nor route ownership changes;
4. prompt-query coverage increases by the configured normalized minimum; and
5. the retained scalar Prompt Top-K mass stays above the configured fraction
   of the anchor mass.

Candidate ordering and ties use node IDs, making selection deterministic.
The initial candidate permits one replacement, retains at least 99.75% of the
selected optional Prompt Top-K mass, and requires at least 0.1% normalized
query-coverage gain. These are preregistered development values, not tuned or
validated results.

## Why the constraints matter

- Equal-size replacement leaves the dense physical position capacity and KV
  byte count unchanged.
- Same-modality replacement cannot manufacture an apparent memory win by
  exchanging text for image tokens or vice versa.
- Mandatory blocks can never be donors.
- The Prompt Top-K mass floor prevents a rare-query gain from discarding a
  large amount of aggregate evidence.
- Keeping this path exclusive with adaptive visual swapping and future-query
  stable-head reranking prevents a later heuristic from silently undoing the
  preservation decision.
- Missing, incomplete, negative, nonfinite, or inconsistently shaped
  per-query attention fails closed.

## Audit telemetry

Each stable route records baseline and final query coverage, normalized
coverage gain, baseline and final Prompt Top-K mass, normalized prompt loss,
candidate and rejection counts, and exact donor/recipient node pairs. The
runtime trace aggregates these values and also retains route-level records for
routes with accepted or rejected proposals.

## Validation completed in this implementation step

CPU tests establish that the reranker is deterministic, preserves active
positions and exact bytes, keeps mandatory blocks, improves a known synthetic
rare-query evidence case, never replaces across modalities, and fails closed
when per-query evidence is incomplete. A Hugging Face unit test verifies that
retaining prompt queries and averaging them afterward reproduces the existing
scalar pooling, and that token-to-block pooling is consistent.

These tests prove implementation properties only. They do not prove DocVQA,
TextVQA, chart, or OCR quality.

## Required next empirical step

Do not run a broad sweep. Materialize the preregistered disjoint, label-blind
calibration ledgers from a clean commit, then compare FullKV, Prompt Top-K, and
this single candidate under identical examples and generation controls. Keep
the gates unchanged: at least 30% lower active and total cache memory than
Prompt Top-K, and paired FullKV-answer fidelity no more than five points below
Prompt Top-K at both task and fold levels. Only a clean calibration pass may
create a frozen candidate and authorize a new untouched held-out evaluation.
