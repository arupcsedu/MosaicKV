# Spill-directed source-projection launch study

The completed consumed row-geometry diagnostic found 64-row/four-head
projection about 5–6% faster at the component-fill boundary, but with 76
spills versus zero for the control. It was correctly rejected. Thirty-two-row
alternatives were slower. Do not promote that negative diagnostic as a win.

Only after residual decode attribution verifies at least 250 ms of inclusive
source-projection stream time on BOTH consumed inputs, test the fixed launch
grid: `(rows,heads,warps)` = `(16,4,default8)` control, `(32,4,16)`,
`(64,4,16)`, `(32,2,8)`. More warps reduce per-thread accumulator pressure;
this is a hypothesis, not an assumed speedup or claim that spills disappear.
The reduction order, dtypes, matrices, source codes, original K/V corrections,
mandatory positions and quantization remain unchanged. Default None preserves
all historical launches. Warp overrides are explicit in plans and trial rows.

Reuse the existing all-layer paired/sequential K/V-bit, 128-token vocabulary
logit, storage, spill and >=5% component-fill improvement gates on both inputs.
Unsafe, numerically different, spilling or slower launches cannot advance.
If one common geometry passes, run the same five local adaptations, five
warmups and twenty complete-request trials per consumed input. The CPU suite
must pass at the same clean SHA. Follow dependencies without log polling.

Two one-A100 workers, 12 CPUs/worker, at most four aggregate A100s; bounded
30-minute microdiagnostics and 75-minute complete-request workers. Stop after
this finite study, whether positive or negative. No new cohort, reference
inspection for tuning, paper promotion or threshold change is permitted.
All comparisons remain source-precision/storage variants, independent quality
N=0, not official baseline executions or byte-matched algorithm curves.
