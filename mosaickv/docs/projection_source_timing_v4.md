# Exact source storage: fixed consumed-input systems measurement

V3 array 22185690 completed, and its exact raw-KV representation references
passed. Union restoration reduced average divergence and passed absolute
limits, but all four unchanged relative per-prefix guards failed. Preserve
this negative causal result; do not enlarge quotas, change thresholds or
promote it to fresh quality evaluation.

The remaining useful engineering measurement is the cost of the unchanged
original selections under raw K/V, encoded K/V and hybrid source storage.
Apply all three formats to BOTH EviCastKV and Top-K on the same two consumed
inputs. No selection, attention, model, precision, prompt or target changes.

Keep the original bitwise retention-one and 32-token controls. For timing,
use five warmups and twenty paired blocks, randomizing all three formats
inside each block with seed 20261030. Time construction separately. Record
CUDA events and synchronized wall time for exactly 31 own-greedy updates.
Compare every trial's 32 logits and tokens bitwise to the independent ordinary
selected-cache reference. Logit CPU copies, reference creation, prefill,
selection and trace I/O remain outside the decode timer.

Disable reconstruction workspace instrumentation ONLY within timing. Obtain
memory evidence from the separate, fully instrumented same-selection runtime
controls. Do not use unaudited timing peaks as memory results. Report mean,
p50/p95 and 10,000-replicate paired timing-ratio intervals against both raw
separate and encoded K/V. Include a slower result as measured; no artificial
speedup is required for successful experiment execution.

This measures a shared storage format, not selector superiority, end-to-end
request latency, benchmark quality, whole-request peak or broad GQA support.
No fresh cohort, calibration or larger campaign follows automatically.
Run two one-A100/12-CPU workers with twenty-minute hard limits and finite CPU
afterany analysis, scratch-only common environment, maximum six A100s total.
