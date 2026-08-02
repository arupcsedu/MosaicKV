# Experiment 4: residual repair

> Classification: exploratory small-sample validation. The run used a dirty worktree with preserved config/trace provenance and is not a canonical paper result.

## Design and completeness

The study contains 105 paired sample-policy rows and 686 decode-step records over seven fixed examples (MMStar 2, DocVQA 2, ChartQA 2, Video-MME 1). Qwen2.5-VL-3B, BF16, HF eager, deterministic greedy decoding, eight output tokens, retention 0.2, block size 16, and the physical packed-prefill KV-byte budget are fixed.

Immutable matrix index SHA: `7ba195d515ad1651a053ee4aff63a708d715aac72984fa8642f7fe69a271be3b`. All compressed cells for a paired sample were required to have exactly the same active KV byte count. Requested residual dtype and pinned/pageable placement were checked against the runtime trace.

Thresholds are an exploratory log-scale sensitivity sweep, not fitted benchmark hyperparameters. A canonical study must choose thresholds on a disjoint calibration set.

All 60 run manifests agree on Git `9e8b77c0cfe9af62d6a1a6ce5840a01543f17b72` with dirty patch SHA `41fa7a6d2cfdf2b68ac621c60df04f9c2460858055ef204df7f13e5fffd36d4f` and model revision `66285546d2b821cf421d4f5eb2576359d3770cd3`. The exploratory classification is retained explicitly; these rows must not be presented as clean-tree paper evidence.

## Primary policy comparison

| Variant | Triggered samples | Precision | Recall | Repair ms avg / p95 | Restored bytes avg | Quality gain | E2E cost (s) | Recovered / improved / incorrect / no effect |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_repair | 0/7 | N/A | 0.0000 | N/A / N/A | N/A | 0.0000 | 0.000 | 0 / 0 / 0 / 7 |
| entropy_only | 7/7 | 0.1429 | 1.0000 | 505.533 / 895.650 | 16384.0 | 0.1429 | 10.782 | 0 / 1 / 0 / 6 |
| prototype_risk_only | 5/7 | 0.0000 | 0.0000 | 329.725 / 669.638 | 16384.0 | 0.0000 | 5.306 | 0 / 0 / 1 / 6 |
| combined_risk | 7/7 | 0.0000 | 0.0000 | 418.474 / 888.572 | 16384.0 | 0.0000 | 13.220 | 0 / 0 / 1 / 6 |
| oracle_repair | 3/7 | 0.3333 | 1.0000 | 381.367 / 675.858 | 16384.0 | 0.1429 | 0.569 | 0 / 1 / 0 / 6 |

Precision uses beneficial triggered sample-policy cells divided by triggered sample-policy cells. A benefit is either recovery of the paired FullKV answer or a strict benchmark-score improvement. Recall uses failures that the oracle repaired as the repairable denominator.

## Sensitivity sweep

| Variant | Triggered | Precision | Recall | Avg repair ms | Restored bytes | Residual MiB | Quality gain | E2E cost (s) | Improved / incorrect / no effect |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| combined_risk | 7 | 0.0000 | 0.0000 | 418.474 | 16384.0 | 53.56 | 0.0000 | 13.220 | 0 / 1 / 6 |
| combined_aggressive_threshold | 7 | 0.0000 | 0.0000 | 470.484 | 16384.0 | 53.54 | 0.0000 | 20.359 | 0 / 1 / 6 |
| combined_conservative_threshold | 6 | 0.1667 | 1.0000 | 341.598 | 16384.0 | 53.61 | 0.0000 | 4.049 | 1 / 1 / 5 |
| combined_max_blocks_1 | 7 | 0.0000 | 0.0000 | 425.082 | 8192.0 | 53.61 | 0.0000 | 15.199 | 0 / 1 / 6 |
| combined_max_blocks_4 | 7 | 0.0000 | 0.0000 | 424.775 | 32768.0 | 53.48 | 0.0000 | 13.877 | 0 / 1 / 6 |
| combined_residual_fp16 | 7 | 0.0000 | 0.0000 | 451.343 | 16384.0 | 53.56 | 0.0000 | 14.830 | 0 / 1 / 6 |
| combined_residual_bf16 | 7 | 0.0000 | 0.0000 | 450.750 | 16384.0 | 53.56 | 0.0000 | 14.113 | 0 / 1 / 6 |
| combined_residual_fp8 | 7 | 0.0000 | 0.0000 | 453.470 | 16384.0 | 26.78 | 0.0000 | 14.794 | 0 / 1 / 6 |
| combined_residual_int8 | 7 | 0.0000 | 0.0000 | 448.540 | 16384.0 | 26.78 | 0.0000 | 16.155 | 0 / 1 / 6 |
| combined_cpu_pageable | 7 | 0.0000 | 0.0000 | 420.625 | 16384.0 | 53.56 | 0.0000 | 13.703 | 0 / 1 / 6 |

## Interpretation boundaries

The no-repair cache differed from FullKV on 1/7 answer strings; 1 were repairable by the evaluation-only oracle under this one-redecode, fixed-budget mechanism. Results are reported whether favorable, neutral, or harmful. With seven examples, percentages are diagnostics rather than population estimates.

Repair latency is measured as synchronized residual transfer plus synchronized re-decode time. Net end-to-end cost is paired against the no-repair row and includes all online work; the oracle's independent FullKV diagnostic trajectory is excluded from online latency by construction.

Restored bytes are the promoted source-format KV bytes that re-enter the active GPU cache. Residual storage bytes, which vary for FP8/INT8 versus lossless storage, remain separately recorded in the sample table and runtime trace.
