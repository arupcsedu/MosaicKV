# Lossless affine-minimum representation control

The earlier FP16 minimum-and-scale experiment changed answers; retain that
negative result. This separate engineering option stores only a bit-exact BF16
minimum. The scale remains the very same FP32 tensor; codes, exact rows,
positions, sparse corrections, source tiling and histories do not change.
Reject any minimum not exactly BF16-representable, including signed-bit changes.
Never round it or silently change the fallback. This exploits the fact that
these affine minima are extrema of BF16 source values, not a new quantizer.

Apply identically to the frozen source variant and all four compatible FP32-
header local baselines. Preserve their prior FP16-header results too; a stronger
previous variant must not be hidden. No new selection, label-dependent fitting,
forecast, prototype, repair, training or benchmark cohort is enabled.

These baseline controls use the registered `head3_boundary8` K/V precision
variant on their selected/merged caches. They are not unmodified published
baselines or official implementations. A win here cannot substitute for the
separate faithful paper-native lanes or establish superiority over the best
unquantized or differently configured baseline.

Use only the same two consumed inputs. Require complete prior native-target
results before preparing the write-once plan; bind their hashes. Compare every
constructed K/V scalar with the immediate parent. Require bitwise 33-prefix and
128-greedy logits, complete per-token NLL and conditional greedy tokens against
the immutable previous controls. Perform the unchanged FullKV retention-one,
32-token guard, original-cache retirement and replay checks. Record encoding
time separately from validation replay cost and initial/final/temporary-layer
inclusive peak storage in generation and teacher-forcing phases separately.

This is a common representation optimization, not a method-specific advantage.
Do not compare an optimized EviCastKV with deliberately unoptimized baselines.
Different measured memories are not byte-matched; two consumed inputs are not
independent paper evidence. No larger campaign, fresh identity or superiority
promotion is authorized. Whole-request construction/GPU peaks remain uncertified.

The finite `auto_minimum` CPU successor runs after the parent analysis. It may
prepare this engineering check only when both parent tasks complete and the
frozen source has no worse PPL and ROUGE-L than the old EviCastKV shadow. This
is not a nearest-baseline superiority gate. Otherwise it writes a blocked
decision and submits no GPU work. It submits at most one CPU suite, one
admission job, two one-A100 workers and an afterany analysis. No recursive
research, threshold change or new identity is authorized.
