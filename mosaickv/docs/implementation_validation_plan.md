# Implementation validation gate

This gate is a predeclared, small-sample implementation check. It is not a
paper-result experiment and must not be reported as a full benchmark score.
Its purpose is to determine whether `mosaickv_exact` shows a quality signal
relative to simple baselines under the same realized active-KV-byte budget.
Negative and inconclusive outcomes must be retained.

## Fixed design

- Back end: Hugging Face eager attention, BF16, deterministic greedy decode.
- Models: LLaVA-1.5-7B and Qwen2.5-VL-3B.
- Paired slices: LLaVA/MMStar with 10 examples; Qwen/MMStar and Qwen/DocVQA
  with 20 examples each.
- Timed output length: exactly 32 greedy tokens for comparable systems cost.
  The task-quality answer stops before the first model EOS/end-of-turn token;
  traces retain both token counts and the complete fixed-length trajectory.
  A sample with no terminal token that reaches 32 remains flagged as
  potentially truncated.
- Target retention ratios: `1.0`, `0.4`, and `0.2`.
- All methods, including RandomKV, use seed 17 so deterministic subset
  selection yields exactly the same paired sample IDs. The later full
  experiment must use an explicit pinned sample ledger before varying the
  RandomKV selection seed.
- Primary budget: realized active KV bytes, not nominal token retention.
- `mosaickv_exact` uses budget-filling lazy greedy selection. A run fails if
  useful logical KV bytes fall below the target by more than the unavoidable
  fixed-block rounding gap. Logical bytes, packed bytes, and padding are all
  retained in the debug trace.

The exact matrix is
`configs/experiments/implementation_validation.yaml`. It contains 33 jobs:
13 LLaVA/MMStar jobs and 20 Qwen image-task jobs. Each job processes one
immutable subset/configuration and remains independently resumable.

## Interpretation gate

Retention-one output must match FullKV under the repository's documented
tolerance before compressed outputs are compared. At `0.4` and `0.2`, compare
paired task scores only when realized active KV bytes satisfy the common
accounting rule. Report RandomKV as the single predeclared seed-17 smoke
baseline, not as a seed-averaged result. LOOK-M remains labeled
`lookm_reimpl`; if its
token-granularity budget does not byte-match the block-16 stratum, report it
separately.

The primary directional contrast is Qwen2.5-VL-3B on DocVQA at retention
`0.2`: compare `mosaickv_exact` with `prompt_attention_topk` and the fixed
seed-17 `random_kv` baseline at matched realized active bytes. Qwen/MMStar and
LLaVA/MMStar, plus retention `0.4`, are predeclared replication contrasts.
Report every contrast. A higher primary paired score is only a promising
implementation signal; with 20 examples it is not evidence of statistical
superiority.

This small gate may establish that the implementation is promising or expose
a regression. It cannot establish statistical superiority, dataset-wide
quality, backend generalization, prototype-tier benefit, repair benefit, or a
systems-speed advantage.

## Time budget

The 33 jobs form at most five eight-GPU waves. Based on the measured LLaVA
packing cost and existing Qwen canaries, expected aggregate demand is about
4--7 GPU-hours. Allowing cold model loads, Weka variability, and finalization,
the expected wall time at eight-way concurrency is 0.8--1.5 hours. The gate
has a 2.5-hour wall-clock envelope when launched shortly after the daily
reservation begins. No job should be launched close enough to the 08:00
reservation rollover to be interrupted.

No jobs are submitted by defining or validating this matrix.
