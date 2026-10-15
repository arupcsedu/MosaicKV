# Output-preserving exact selection, development v1

Status: opt-in implementation, **not a measured improvement or a paper result**.
The frozen shared-byte candidate, historical predictions, and paper-native
PrefixKV/VL-Cache/LOOK-M lanes are not replaced. This is an explicitly labeled
MosaicKV variant: prompt-only evidence, no graph, no synthesized prototype K/V,
no residuals, no repair. It retains original keys/values and positions.

## Objective and interpretation

For one layer and KV head, each row of `A` is a distinct query head × prompt
position. Grouped-query attention (GQA) heads are **not averaged**. Only the last
configured prompt window is captured. Let `V` be the original source values:

\[
Y=AV,\quad m_q(S)=\sum_{i\in S}a_{qi},\qquad
L(S)=\sum_q\left\|Y_q-
\frac{\sum_{i\in S}a_{qi}v_i}{m_q(S)}\right\|_2^2.
\]

This is exactly the selected-key softmax output for the *fixed captured query
and mask*, apart from numerical rounding. It is not the output of a new complete
forward pass: upstream hidden states can change after compression. Prompt
reconstruction therefore does **not** guarantee answer preservation or future
query fidelity. Each prompt row is normalized in FP32 to remove capture rounding;
causal zeros remain zero. A selected subset with negligible/zero mass for any
row fails explicitly, rather than inventing probabilities. FP64 is retained in
numerical unit fixtures only.

Attention-output matching is related work, not a new general principle:
[Fast KV Compaction via Attention Matching](https://arxiv.org/abs/2602.16284).
Our proposed variant searches exact source subsets, rather than constructing
replacement keys/values. No claim of superiority to that work is supported here.

## Algorithm

1. Obtain the same dense per-head physical capacity and mandatory positions as
   the common runtime. Use its vectorized Prompt Top-K selector as the anchor.
   Only one-token blocks and HF eager attention are supported initially.
2. Measure `L` on every layer/KV-head route. Rank by decreasing error; break ties
   by layer then KV head. Optimize at most `max_routed_heads` routes above the
   configured error threshold. Other routes retain their exact Top-K selection.
3. For each optimized route, construct small donor/recipient shortlists using
   the membership sensitivity
   \[
   g_i=2\sum_q\frac{a_{qi}}{m_q(S)}
       (\widehat Y_q-Y_q)^\top(v_i-\widehat Y_q).
   \]
   Optional selected positions with large `g` are donor candidates; unselected
   positions with small `g` are recipient candidates. Stable sorting gives
   deterministic physical-position tie breaking. Mandatory positions cannot be
   donors. A shortlist is a heuristic, not an exhaustive/global optimizer.
4. Test at most `shortlist_size²` equal-byte pairs per round in bounded chunks.
   Recompute the retained numerator **and denominator** for each pair. Reject
   insufficient query mass or nonfinite output. Accept only a strict objective
   decrease exceeding `minimum_improvement`, verified again by reconstructing
   directly from the proposed subset. At most `max_swaps_per_head` changes occur.
5. Gather exact source blocks once using the same configured gather policy as
   the compared methods, then use the unchanged packer, validity masks,
   original logical sequence length, next decode position, and append-only decode.
   Initial/final/peak live-cache storage remains measured by the common tracker.

The selector never writes into original attention, K/V, or model weights. Swaps
remain inside one layer/head, preserving that route's capacity and byte cost.
No RoPE inversion or averaging occurs. The retained subset may cross modalities,
as Prompt Top-K does; this is eviction, not cross-modality merging.

The work is linear in positions for evidence and shortlisting, plus bounded pair
evaluation: per optimized head approximately `O(NQD + swaps × (NQD + k²QD))`.
Chunked pair temporaries are `O(pair_chunk_size × Q × D)`; there is no `N×N`
graph. Captured prompt attention still costs `O(layers × query_heads × window × N)`.
All-route error computation and per-head synchronization have real overhead:
latency improvement has **not** been established. Timings include selection,
exact construction, packing, and complete compression/end-to-end execution.

## Configuration and ablations

`configs/development/output_preserving_exact_v1.yaml` is a strict resolved-input
example for the existing COCO model/precision/control lane. Its near-one ratio
avoids an additional nominal-ratio sub-cap; the byte cap chooses capacity.
Exactly `retention_ratio: 1.0` is the untouched FullKV identity gate, not a way to
apply a smaller cap. All comparison methods must use the same capacity controls.
Unsupported backends, multi-token blocks, graph/prototype/repair settings, draft
forecasting, and legacy selector tuning fail configuration validation.

Separate ablations, applied through the recorded `output_preserving` section:

| Effective label suffix | Query evidence | Swaps/head | Purpose |
| --- | --- | --- | --- |
| `__anchor` | Individual | 0 | Verify exact agreement with common Prompt Top-K |
| `__pooled_swaps` | Mean query | 2 | Isolate the cost of discarding query identity |
| No suffix | Individual | 2 | New reconstruction-routed candidate |

All labels start with `mosaickv_exact__output_preserving_v1`. Traces report
anchor/final selected blocks (or catalogue hashes outside full diagnostics),
per-head error before/after, routed heads, candidate pairs, accepted swaps, and
zero-mass rejections. Existing canonical configurations omit the optional section
when absent, preserving old config hashes. The frozen shared-byte validator
explicitly refuses this unregistered variant. Frozen execution stays on its
original immutable worktree; its content seal is not regenerated for this work.

## Fair development test, not yet submitted

`configs/experiments/output_preserving_exact_v1.yaml` records an execution-disabled
16-example development plan. Before any run, bind a clean source SHA, a reviewed
disjoint sample ledger, and comparable evaluator/reference provenance. The plan
does not materialize data or submit jobs. If an untouched pool is unavailable,
stop or explicitly preregister a retrospective development cohort; never call
reused examples fresh.

The development matrix still needs its own runner/contract registration before
submission. The old frozen-lane launcher must reject the new variant and the
new 25.5/28 MiB caps. Do not weaken that guard: a separately bound development
runner must validate the original comparator source/policy hashes while using
this new protocol's budget grid. No historical campaign configuration is edited.

Run every compressed method at **every** shared cap: 20, 25.5, 28, and 40 MiB.
Include Prompt Top-K, `vl_cache_reimpl`, frozen MosaicKV, and all three candidate
ablations. Keep FullKV solely as a quality reference. Do not substitute an old
BF16 prediction for the FP16 reference. Preserve all COCO caption references and
compute corpus CIDEr over the cohort and within each paired-bootstrap replicate.
Use identical models/revisions, samples, media, prompts, generation, precision,
attention, block/mandatory policy, warmups, repetitions, and GPU controls.
For each cap, require the largest feasible physical cache and identical measured
initial, final, and peak bytes; unequal feasible sizes are not same-budget rows.

At a common cap this tests **quality and latency**, not a 30% memory reduction.
The memory claim is a separate curve comparison: identify the smallest measured
quality-eligible competitor point and test whether the candidate attains equal
quality with at least 30% less measured storage. Report paired uncertainty; a
development point estimate is not a held-out claim. Non-inferiority margin is
zero in the planned equal-quality test. A tiny cohort may be inconclusive.

Illustration only, not measurements: if decode adds 7.875 MiB to both caches,
40 → 28 MiB initial is a 30% initial reduction but only
`1 − 35.875/47.875 = 25.065%` final. At 25.5 MiB initial, final reduction is
`1 − 33.375/47.875 = 30.287%`, **provided** that capacity is physically attainable
and quality passes. Report actual growth and each memory boundary separately;
never assume these illustrative sizes or choose the 40 MiB baseline by default.

This development plan does not change the model, dataset, metric, retention
convention, or result labels of any original paper-native reproduction lane.

## Verification scope

CPU tests cover selected-softmax equivalence, causal masks, strict configuration,
GQA query preservation, deterministic bounded swaps, mandatory tokens, input
immutability, equal-capacity selection, exhaustive tiny one-swap comparisons,
Top-K anchor parity, retention-1 token/logit parity over 16 tokens, repeat-request
isolation, and actual initial/final/peak cache accounting in tiny random-weight
HF models. These are implementation checks, **not pretrained quality evidence**.
Real-checkpoint A100 parity, quality, memory curves, and latency remain unrun.

On 2026-10-14, 288 selected CPU regression tests passed, including the 30 new
candidate checks. One immutable-source fingerprint test intentionally rejects
the development checkout; it was run separately and passed in the unchanged
`mosaickv-coco-shared-retry1-20261014` frozen worktree. The content seal was not
updated or bypassed. Ruff passed. Legacy synthetic and paper-candidate canonical
JSON/hash identities were also compared against `git show HEAD` and are unchanged.
These dirty-development-worktree checks are not canonical experiment results.

To repeat CPU checks (all caches stay in scratch):

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/mosaickv/src"
/scratch/djy8hg/env/mosaickv/bin/pytest \
  mosaickv/tests/unit/test_output_preserving.py \
  mosaickv/tests/integration/test_output_preserving_runtime.py \
  -q -o cache_dir=/scratch/djy8hg/cache/mosaickv/pytest-output-preserving
```
