# Lower memory at equal quality: paper experiment v1

This protocol answers a different question from an equal-byte comparison.
Equal-byte experiments ask which selector gives higher quality when both methods
receive the same physical cache.  This experiment asks whether MosaicKV can
reach a named baseline's quality while using at least 30% less measured active
KV storage.

The protocol is registered in
`configs/experiments/lower_memory_equal_quality_v1.yaml`. It is intentionally
execution-disabled until a clean immutable source SHA and immutable input
ledgers are bound. It does not change the running PrefixKV campaign.

## Claim boundary

For baseline point \(b\) and MosaicKV point \(m\), define the reduction at
cache boundary \(x\in\{\text{initial},\text{final},\text{peak}\}\) as

\[
R_x(b,m)=1-\frac{\operatorname{bytes}_x(m)}
                       {\operatorname{bytes}_x(b)}.
\]

The memory gate requires \(R_x\ge 0.30\) at every boundary, both in aggregate
and for every paired sample. Storage means unique live K/V backing storage;
padding and preallocation count. Model weights and allocator reservations are
reported separately and are not called KV compression.

Quality is tested with paired samples and the task's native reducer. For a
higher-is-better metric, the lower confidence bound of MosaicKV minus the
baseline must be at least zero. For lower-is-better PPL, the upper bound must
be at most zero. The zero margin is frozen before execution.

## Label-blind operating-point choice

All compressed methods run at every registered cap. For each baseline cap, the
analysis reads only initial/final/peak storage and selects the largest MosaicKV
cap that satisfies the 30% gate. It then opens quality. This gives MosaicKV the
largest permissible cache without choosing an operating point from observed
answers. Every registered comparison, including failures and physically
infeasible pairs, is reported. Family-wise inference is Bonferroni controlled
within each baseline/metric family.

No curve interpolation can pass a gate. Lines may be drawn only to visualize
the measured points. The headline must name the baseline, dataset, metric,
model, backend, and measured memory boundary.

## Lanes

### VL-Cache / COCO Caption

- LLaVA-v1.6-Mistral-7B, pinned revision, HF eager FP16.
- 512 paired COCO validation examples and 64 generated tokens.
- FullKV is the quality reference; compressed methods are Prompt Top-K,
  `vl_cache_reimpl`, and `mosaickv_exact`.
- Caps: 10, 20, 25, 40, and 80 MiB.
- Metric: complete-cohort CIDEr, recomputed inside all 10,000 paired bootstrap
  replicates.

The 25-versus-40 MiB comparison is important because the common 64-token
decode growth means 28 versus 40 MiB is not 30% smaller at the final boundary.
The gate uses measured values rather than this planning estimate.

### LOOK-M / representative MileBench

- LLaVA-1.5-7B, pinned revision, HF eager FP16.
- 192 balanced, previously unused examples over ActionPrediction,
  ActionSequence, CLEVR-Change, and Spot-the-Diff, plus a separate 16-example
  execution gate. The four registered subtasks contain 800 identities in
  total; 576 were already consumed by prior development/exclusion pools, so
  the protocol uses 48 paper examples and four gate examples per subtask and
  leaves four per subtask unused.
- Compressed methods: Prompt Top-K, `lookm_reimpl`, and `mosaickv_exact`.
- Caps: 16, 20, 28, 40, and 80 MiB.
- Accuracy and ROUGE-L are reduced separately using the registered MileBench
  hierarchy. Both must pass for a lane-wide claim.

LOOK-M remains a local reimplementation. These results must never be labeled
as official LOOK-M execution.

### PrefixKV

The active PrefixKV campaign is not modified. Once complete, a storage-only
step will choose registered pairs from 40/80/120/160/240 MiB without reading
answers, PPL, NLL, or ROUGE-L. Both corpus PPL and FullKV-reference ROUGE-L
must pass. Long append-only decoding may make a 30% final/peak reduction
physically impossible even when initial cache compression passes; that outcome
must be reported instead of weakening the boundary.

## Reuse and execution

Existing equal-byte results remain valid supporting evidence, but the primary
cross-budget claim uses one clean source snapshot across every curve point.
Historical nominal-retention rows are not reused. Existing FullKV predictions
may be reused only if source, model and processor revisions, prompts, media,
generation, precision, evaluator, and all control hashes match exactly.

Each worker requests one A100, 12 CPUs, and at most two hours. At most six
A100s may be occupied by MosaicKV jobs in aggregate. The running PrefixKV jobs
take precedence; this protocol does not authorize a submission while they are
active.

## Report wording

A positive comparison may be written as:

> On [dataset/model/backend], MosaicKV used X% less measured initial, Y% less
> final, and Z% less peak active KV storage than [named baseline], while the
> paired [metric] difference was D with a family-wise 95% confidence interval
> [L,U].

Do not shorten this to “30% better compression than all baselines” unless every
named lane and every required metric passes. Prototype merging and residual
repair remain outside this exact-selection experiment.
