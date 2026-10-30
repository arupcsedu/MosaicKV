# Exact-corrected dual projection: consumed-input diagnostic

This finite diagnostic does not revise EviCastKV's selected memberships, original
weights, RoPE positions, original K/V values, or the 32-token equality controls.
It is offered to both frozen selectors, not only EviCastKV. It creates no fresh
cohort, calibration, or paper campaign. The previous layer-addition quality
diagnostic remains negative.

A single Triton program computes K and V for one head from shared original
normalized source rows. It writes original-position RoPE keys and values into
the existing layer lease; low-sharing pairs retain their original losslessly
encoded K/V. Original weights are read in place, not concatenated or copied.
Accumulation order can change bits: every differing BF16 element receives an
original-value correction with an int32 logical index. Three independent exact
reconstructions, retention-one and 32-token own-output parity are mandatory.
All maps and corrections count toward physical storage. If a dual format costs
more than the already validated compact format, it is not selected.

The finite chain is two one-A100 correctness workers followed, only on complete
controls, by two workers with five warmups and twenty randomized paired trials.
Timing excludes original prefill, selection, constructor and patch creation;
it cannot support an end-to-end speedup or whole-request HBM claim. No inference
quality improvement is possible from a bit-exact storage change.
