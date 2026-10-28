# Profiler-justified unit-reference budget reuse

This is one **candidate implementation optimization**, not a new selector.
Do not claim a speedup until its GPU correctness/equivalence and factorial
measurements complete. The original reference campaign remains unchanged.

## Measured motivation

The complete 56-request COCO profiler passed its execution gate. The corrected
CPU summary separates host ranges from identically named CUDA annotations:

`/scratch/djy8hg/runs/evicastkv/fox-review-20261028-v6-coco-scoped/writing/profiler-review-v2-nonduplicated/`

Physical-budget construction averaged 2,244.358 ms inclusive CPU time over its
48 observed compressed requests. This is **profiler-contaminated diagnostic
time**, not an unprofiled latency measurement or a promised saving. Exact-tier
bookkeeping was another costly operation, but a speculative rewrite could
change source-view lifetimes; it is deliberately not modified here.

## What changes, and what does not

The old budget path reconstructs a second complete `FullKVState` whenever a
physical reference block size or mandatory policy is explicit. In this frozen
lane, that reference uses the same one-token blocks and mandatory positions as
the already validated source. Reconstructing all descriptors is redundant.

The guarded path applies **only** when both block sizes are one and the
effective mandatory sets are identical. It obtains integer bytes per head-slot
from the actual K/V tensor geometry and dtype, then applies the unchanged
ceil/min/floor capacity arithmetic. A validated complete unit partition has
one physical position per block and the same mandatory-position count in every
head; its attainable lattice is exactly the integers. Canonical layer/head
dictionary ordering is preserved even for an unusual source descriptor order.

Different block sizes or protection policies retain the original factory path.
The slow function, with the added fast-path guard removed, must have the same
AST as the pinned reference function. Randomized CPU tests compare capacities,
byte targets, per-head limits, ordering, and rejection behavior.

No attention observation, normalization, score, routing, candidate universe,
mandatory token, head minimum, objective, tie, source K/V, logical position,
allocation policy, output schedule, precision or backend changes. Compatible
Top-K and local ports receive the same common optimization. No prefill or
graph is cached across end-to-end requests.

## Gated execution

From a clean source containing this candidate:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/evicastkv_optimization.py \
  --parent /scratch/djy8hg/runs/evicastkv/fox-review-20261028-v6-coco-scoped \
  --root NEW_EMPTY_OPTIMIZATION_DIRECTORY
```

Finite callbacks run smoke, a representation canary and all 128 COCO validation
checks (32 consumed inputs × two compressed methods × two caps). Equivalence
also checks the reference's selected IDs, routed/candidate domains, forecast
observations, quotas, slot metadata, all 64 output tokens, and actual
initial/final/peak bytes. Each independent representation oracle still requires
bitwise source entries and the original numerical thresholds. No old failure
or untested MileBench path is relabeled as passing.

Only then run the 16-input unprofiled implementation factorial: reference and
optimized Top-K, reference and optimized EviCastKV at both 20/40 MiB, plus
FullKV once per input as an uncompressed reference. Every arm has five complete
warmups and three fresh request repetitions. Pair orientation is alternated,
not merely cyclically shifted; actual order is recorded. The reference budget
function is compiled once from the hash-bound clean reference worktree using
the same model/classes. It caches code only, never requests or GPU tensors.

The separate four-contrast timing family uses 10,000 paired input-media cluster
draws on per-input repeat means. It does not change the original six quality
contrasts. Report regressions and inconclusive intervals honestly. Two workers
at most, one A100/12 CPUs/64 GiB each, two-hour limit. Existing GPU jobs are not
canceled. Confirmation remains blocked by the original freshness/review gates.

Outputs include `optimization_diff.patch`, per-attempt raw rows and manifests,
`equivalence_report.json`, `before_after_runtime.csv`, `paired_factorial.json`,
`optimization_report.md`, and `GATE_optimization.json`. Tensor equality evidence
is the independent per-run source oracle plus cross-version identical selected
positions; no unsaved cross-run GPU-tensor hash is claimed.
