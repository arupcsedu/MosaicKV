# Common exact range-workspace follow-up

The compact integer quantizer preserved all original payload bits but did not
make the source variant's observed construction peak lower than Prompt Top-K.
Preserve that result and its blocked timing chain. The observed source peak
now occurs in conditional range preparation, not integer serialization.

This follow-up changes temporary storage only, identically for every method.
For time-axis groups, reduce strided 128-token windows of the original BF16
input without transposing/padding a full source and without expanding a full
per-feature mask. Reduce the final partial window without replicating its last
element; repetition cannot change a nonzero extremum. Fill one private BF16
output through nonoverlapping views. For feature-axis groups, every group is
either completely protected or unchanged: replace protected rows by positive
zero and retain every other scalar. Signed-zero time minima and subnormals
still use the original FP32 reference. No quantizer, FP32 rounding, stored
scale/minimum, selection, attention, position or precision policy changes.

CPU fixtures compare every header/code bit across full and partial groups,
noncontiguous inputs, all bit widths, protected groups and numerical edge
cases, and assert original inputs remain unchanged and masks remain row-sized.
The GPU follow-up must verify all ten original method/input packets, physical
storage and prefill logits exactly. Only after all pass AND source construction
peak is below every baseline on both inputs may the conditional same-engine
five-warmup/twenty-trial complete-request timing run. Otherwise preserve the
negative gate and do not launch timing.

Scope remains the same two consumed inputs, at most six aggregate A100s, no
fresh examples/calibration/large campaign. Packet identity reuses prior
complete greedy/teacher-logit evidence; it does not cure existing mixed PPL
versus baseline results. Encoded points differ in measured bytes: not a
byte-matched algorithm comparison, whole-request peak proof or paper claim.
