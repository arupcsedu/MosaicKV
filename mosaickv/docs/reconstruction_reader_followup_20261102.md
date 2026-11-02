# Prompt-source, text-overlay and generated-history reconstruction

Use only the same two consumed LLaVA inputs and frozen method references.
This is a source-precision/storage engineering variant, not validation of all
six MosaicKV mechanisms, official baselines, or paper-quality superiority.

First attribute projection and root reconstruction costs using request-object
ownership: prompt representation, protected-text projection inside prompt
reconstruction, and sealed generated-history packets. Record history
construction separately. Do not infer roles from tensor dimensions, sum nested
timers, or present instrumented stream time as end-to-end latency. Validate
all original logits and initial bytes while profiling.

Two fixes are frozen before this new profile. Both retain all original packets,
checkpoints, quantization headers, weights, K/V values, logical positions and
one-buffer storage. Neither adds tensor storage or changes row/head/warp
geometry. `reuse_headers` loads a tile's identical 128-row quantization headers
once and broadcasts them. `carry_literals` reuses the prior 64-value literal
count rather than decoding those same codes again. Original reductions and
BF16 rounding remain. Defaults are `original`; explicit request-local contexts
reset on exceptions and cannot leak across requests.

Profile one warmup and two diagnostic repetitions per method (EviCastKV and
the local PrefixKV adaptation). If prompt projection is the largest of the
three source projection roles, test header reuse first; otherwise test literal
count reuse first. Select only from aggregate operation timings, not answers.

Each attempted fix must pass all-pattern reader controls, original packet
hashes, complete 128-token vocabulary logits, and exactly matching initial,
final and declared decode-owned peak bytes for ALL FIVE local adaptations.
Resources must expose verified non-increasing spills for every encountered
kernel signature. Observe candidate history construction without dropping any
of its three exact checks. Inactive controls and instrumentation prevent any
whole-request GPU-peak claim. Helper buffers are allocated only AFTER the
primary storage audit and are explicitly not cache-peak evidence.

Time prompt-plus-sealed-history reconstruction with five warmups and twenty
randomized paired helper repetitions. Require a >=1% helper improvement on
BOTH inputs using a Bonferroni-four-comparison upper ratio bound (10,000
bootstrap draws). This is only an engineering promotion rule, not a weaker
quality or full-request gate. Then run the unchanged five methods, five
warmups and twenty uninstrumented complete requests per input with the same
reader option applied to every method. Require the original <=1.05x
all-baseline request-time gate; report every outcome.

Allow at most TWO automatic reader attempts. If the first is safe but not
beneficial, test the other independently. If helpful but the full-request gate
still fails, test their combination. Stop on numerical, completeness or
storage failures, and stop after the second attempt. No fresh examples,
calibration, row/warp sweep, or large quality campaign is authorized. Each
attempt and analysis uses write-once roots and `afterany` dependencies, never
continuous scheduler-log polling. Workers use one A100 and 12 CPUs each,
at most two concurrent GPUs; no changes to running snapshots or the main
dirty worktree. Memory points remain different measured encoded sizes, not
byte-matched algorithm comparisons. Independent quality N remains zero.
