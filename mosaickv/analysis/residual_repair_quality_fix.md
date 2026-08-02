# Experiment 4: residual repair

> Classification: exploratory small-sample validation. The run used a dirty worktree with preserved config/trace provenance and is not a canonical paper result.

## Design and completeness

The study contains 77 paired sample-policy rows and 490 decode-step records over seven fixed examples (MMStar 2, DocVQA 2, ChartQA 2, Video-MME 1). Qwen2.5-VL-3B, BF16, HF eager, deterministic greedy decoding, eight output tokens, retention 0.2, block size 16, and the physical packed-prefill KV-byte budget are fixed.

Immutable matrix index SHA: `d09dc4d3b8f49fb0e8472c0009ba30f91df980fa19b4bb1c6599d78d6083aa3b`. All compressed cells for a paired sample were required to have exactly the same active KV byte count. Requested residual dtype and pinned/pageable placement were checked against the runtime trace.

Thresholds are an exploratory log-scale sensitivity sweep, not fitted benchmark hyperparameters. A canonical study must choose thresholds on a disjoint calibration set.

All 44 run manifests agree on Git `9e8b77c0cfe9af62d6a1a6ce5840a01543f17b72` with dirty patch SHA `15c30f973f9196dac44b0bb9c7257e21b149b6799999490ef432d724710253c2` and model revision `66285546d2b821cf421d4f5eb2576359d3770cd3`. The exploratory classification is retained explicitly; these rows must not be presented as clean-tree paper evidence.

## Primary policy comparison

| Variant | Triggered samples | Precision | Recall | Repair ms avg / p95 | Restored bytes avg | Quality gain | E2E cost (s) | Recovered / improved / incorrect / no effect |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| no_repair | 0/7 | N/A | 0.0000 | N/A / N/A | N/A | 0.0000 | 0.000 | 0 / 0 / 0 / 7 |
| quality_entropy_only | 7/7 | 0.1429 | 1.0000 | 470.731 / 905.429 | 14165.3 | 0.1429 | 10.319 | 0 / 1 / 0 / 6 |
| quality_guarded_prototype_risk | 4/7 | 0.2500 | 1.0000 | 341.068 / 727.551 | 13312.0 | 0.1429 | 3.983 | 0 / 1 / 0 / 6 |
| quality_guarded_combined | 7/7 | 0.1429 | 1.0000 | 455.310 / 906.571 | 13762.6 | 0.1429 | 10.748 | 0 / 1 / 0 / 6 |
| quality_oracle_evaluation_only | 3/7 | 0.3333 | 1.0000 | 448.567 / 707.565 | 16384.0 | 0.1429 | 0.437 | 0 / 1 / 0 / 6 |

Precision uses beneficial triggered sample-policy cells divided by triggered sample-policy cells. A benefit is either recovery of the paired FullKV answer or a strict benchmark-score improvement. Recall uses failures that the oracle repaired as the repairable denominator.

## Sensitivity sweep

| Variant | Triggered | Precision | Recall | Avg repair ms | Restored bytes | Residual MiB | Quality gain | E2E cost (s) | Improved / incorrect / no effect |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_combined_control | 7 | 0.1429 | 1.0000 | 408.227 | 15252.2 | 53.57 | 0.1429 | 14.045 | 1 / 0 / 6 |
| quality_guarded_combined | 7 | 0.1429 | 1.0000 | 455.310 | 13762.6 | 53.60 | 0.1429 | 10.748 | 1 / 0 / 6 |
| quality_guarded_max_blocks_1 | 7 | 0.1429 | 1.0000 | 441.652 | 6912.0 | 53.62 | 0.1429 | 9.685 | 1 / 0 / 6 |
| quality_guarded_max_blocks_4 | 7 | 0.1429 | 1.0000 | 455.492 | 28180.5 | 53.55 | 0.1429 | 10.426 | 1 / 0 / 6 |
| quality_guarded_residual_fp8 | 7 | 0.1429 | 1.0000 | 501.375 | 13762.6 | 26.80 | 0.1429 | 11.448 | 1 / 0 / 6 |
| quality_guarded_residual_int8 | 7 | 0.1429 | 1.0000 | 495.694 | 13762.6 | 26.80 | 0.1429 | 13.380 | 1 / 0 / 6 |

## Interpretation boundaries

The no-repair cache differed from FullKV on 1/7 answer strings; 1 were repairable by the evaluation-only oracle under this one-redecode, fixed-budget mechanism. Results are reported whether favorable, neutral, or harmful. With seven examples, percentages are diagnostics rather than population estimates.

Repair latency is measured as synchronized residual transfer plus synchronized re-decode time. Net end-to-end cost is paired against the no-repair row and includes all online work; the oracle's independent FullKV diagnostic trajectory is excluded from online latency by construction.

Restored bytes are the promoted source-format KV bytes that re-enter the active GPU cache. Residual storage bytes, which vary for FP8/INT8 versus lossless storage, remain separately recorded in the sample table and runtime trace.

## What changed and what it fixes

The earlier repair path removed a whole prototype after promoting only part of
its membership. A two-block repair from a ten-block cluster therefore restored
two exact blocks while silently dropping the representation of the other eight.
The corrected path algebraically downdates that prototype and preserves the
unpromoted evidence. It also prioritizes members with high weighted K/V
reconstruction error, spreads the first `R` slots across risk-ranked clusters,
and prevents prototype risk from changing a token below the configured entropy
floor.

These changes improve the measured quality outcome. The no-repair rows score
4/7; guarded prototype risk and guarded combined repair score 5/7. Both recover
`chartqa:1447`, and neither changes an answer incorrectly. In particular, the
guarded combined row preserves the two FullKV DocVQA answers that conflicted in
the earlier sweep.

Guarded prototype risk is preferable to combined risk in this sample: it uses
16 rather than 25 events and reduces paired mean end-to-end overhead from
10.748 to 3.983 seconds with the same score. This is the recommended bounded-run
policy, subject to disjoint calibration on a larger study.

## Storage dtype and restoration-count conclusion

FP8 and INT8 reduce average residual storage from 53.60 MiB to 26.80 MiB and,
unlike the earlier sweep, retain the 5/7 quality result. They still do not lower
latency: mean restored payload is only 13.44 KiB per event, while dequantization
and a full one-step re-decode dominate. The correct claim is therefore reduced
CPU residual capacity, not faster repair.

`R=1`, `R=2`, and `R=4` all recover the only repairable failure. Mean restored
bytes rise from 6.75 KiB through 13.44 KiB to 27.52 KiB per event with no extra
score. `R=1` has the lowest paired end-to-end cost among these combined-policy
rows (9.685 seconds), so larger `R` should be reserved for a calibrated case
where it demonstrates incremental benefit.
