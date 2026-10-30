# Same-GPU paired storage diagnostic

Exactly the two consumed Description/MM-Vet inputs are reused. All six primary
methods execute with identical pinned model/processor/media, BF16 eager backend,
32-token schedule and original source-slot capacity. This is **not** exact
encoded-byte matching, a native paper reproduction, or independent quality
evidence. PrefixKV and VL-Cache are explicitly append-only local adaptations.
LOOK-M keeps its actual merged K/V; no original-source regeneration is offered
to transformed tensors. FullKV is a quality reference and is never timed.

Every eligible exact selector receives the same already validated lossless
formats. Each layer chooses its minimum actual component storage deterministically,
not its fastest or highest-scoring format. The original selected/merged K/V and
ordinary 32-token outputs must match bitwise. All compressed methods are then
measured on the same GPU in each of five warmup and twenty timed blocks, with a
seeded random method order. Tensor states are constructed afresh for each trial.

Reported latency is decode-only: prefill, selection, constructor, patch creation
and artifact I/O are excluded. Trial-bootstrap ratios are conditional timing
uncertainty, not sample-level quality confidence intervals. All method formats
remain resident for this diagnostic; whole-request memory/HBM peak is not
certified. Observed storage is charged separately to the currently active method.

The finite afterany analyzer validates nested artifact hashes and retains failed
workers. No fresh cohort, calibration, or paper campaign can advance automatically.
