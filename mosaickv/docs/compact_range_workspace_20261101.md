# Common exact range-workspace follow-up

The compact integer quantizer preserved all original payload bits but did not
make the source variant's observed construction peak lower than Prompt Top-K.
Preserve that result and its blocked timing chain. The observed source peak
occurs in conditional range preparation on Description and transport fitting
on MM-Vet, not integer serialization. The first bootstrap correctly stopped
because its guard required range preparation to dominate both inputs; preserve
that negative launch record (no GPU run occurred).

This follow-up changes temporary storage only, identically for every method.
For time-axis groups, reduce strided 128-token windows of the original BF16
input without transposing/padding a full source and without expanding a full
per-feature mask. Reduce the final partial window without replicating its last
element; repetition cannot change a nonzero extremum. Fill one private BF16
output through nonoverlapping views. For feature-axis groups, every group is
either completely protected or unchanged: replace protected rows by positive
zero and retain every other scalar. Signed-zero time minima and subnormals
still use the original FP32 reference. Transport correction extraction and
independent assertion check every original scalar in chunks of 524,288; use
the same FP32 isclose and original BF16 assert_close operations and unchanged
tolerances, and concatenate correction indices in original order. Only
temporary comparison arrays shrink. No quantizer, FP32 rounding, stored
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

## Reference-path correction

The BF16-only workspace change did not reduce Description's peak: its guarded
FP32 reference path still used expanded masks and out-of-place replacements.
MM-Vet's fitting peak was reduced by bounded checks. Preserve both outcomes.
A CPU-only replay of the saved Description layer-28 normalized input and
source memberships found one finite, unprotected zero minimum (24 text
positions): this triggers the signed-zero fallback for the whole source.

Optimize the FP32 fallback itself without changing its padded reduction layout,
FP32 dtype, signed-zero tie behavior or original finite-value checks. Convert
the transposed BF16 source directly into one contiguous FP32 workspace. Use a
broadcast row mask, run the original padded masked minimum, replace protected
entries in that PRIVATE workspace with torch.where(out=...), then convert
directly into a contiguous BF16 prepared tensor. Do not create an intermediate
contiguous FP32 output. Caller values/masks remain immutable. The full legacy
path remains available through compact_ranges=False for independent controls.

GPU controls explicitly compare original and new packets for zero/subnormal
fallbacks as well as random full/partial groups. All ten model packets, byte
counts and prefill logits must still match the frozen parents before timing.
