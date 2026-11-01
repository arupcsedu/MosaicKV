# Bounded prompt-only affine refit feasibility

Preserve the complete PPL/ROUGE trade-off result. The prior source variant
improved generated-reference overlap but did not pass both original metrics;
the automatic lossless-header GPU successor correctly remained blocked.
No human annotations, saved answers, task scores, or reference logits may be
used to fit this successor. Do not weaken the model guard or claim superiority.

Test one fixed common quantizer option: two alternating affine least-squares
and nearest-code assignment steps per group, with rollback whenever actual
BF16 reconstructed error increases. This is statistics of the current prompt,
not weight updates or a learned adapter. Scale/intercept remain FP32, codes
retain three bits, group size remains 128. No extra persistent parameters,
positions, exact K/V protections or logical metadata are introduced. This is
ordinary quantization refinement, not a claim of novelty.

Use only the same two consumed normalized-input captures. Exclude positions
already exact in every head from the source fit. Compare original versus refit
through a CPU projection/attention oracle using the checkpoint's runtime BF16
weights cast to FP32 for CPU arithmetic. Preserve the original phases and
exact rows. This CPU oracle is not native CUDA parity or model-quality evidence.

Require, on both inputs: identical physical source bytes, identical exact
protections, at least 5% lower attention-output MSE and nonincreasing p95
per-query/head error. The five-percent threshold is frozen before execution.
Report every layer and rollback count. If the numerical gate fails, do not
launch a GPU check or relax the threshold. If it passes, baseline-compatible
runtime integration and a separate bounded same-input model protocol are still
required; this plan grants no GPU or new-cohort execution.

The accompanying direct constructor is CPU-tested scaffolding for eliminating
a discarded shadow cache, with identical online memberships. It is not GPU
validated and does not repair the PPL trade-off. Do not claim its latency or
whole-request memory until separately measured. Larger campaigns remain blocked.
