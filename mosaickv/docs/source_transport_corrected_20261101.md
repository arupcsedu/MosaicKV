# Same-input source transport correction and common lossless text storage

Preserve the incomplete shared-source GPU experiment and complete operator
audit. The audit localized native-BF16 envelope violations to one Description
layer-25 key scalar in each range choice (two cases), unchanged by disabling
BF16 reduced-precision reduction. FP32/FP64 references did not explain those
away and are NOT substituted as a more permissive reference.

Correct the constructor, not the tolerance. Decode the SAME encoded source
temporarily during construction, project with frozen native BF16 weights and
original phases, retain existing protected K/V overlays, and store only the
scalar differences that violate the frozen rtol=.01/atol=2^-6 construction
envelope. Count every int32 index and BF16 value. Reject attempts to hide a
protected-row mismatch. This target is native projection of the quantized
source, NOT original FullKV values at unprotected positions, task logits,
answers or labels. The temporary decoded source and native projection are
construction costs and are NOT a certified construction-memory saving.

Separately extend the already common bit-corrected selected-text source
optimization to this representation. Preserve EVERY decoded K/V bit,
including approximation and numerical correction. If exact residuals and
metadata are not smaller, keep the original packet. Never improve a source
variant by silently replacing unprotected approximations with FullKV text.
This option is already offered to the baseline formats. It is not a new
evidence-selection mechanism or a lossy prototype.

Gate dependency: completed positive common sequential-lease verification on
both consumed inputs (including raw retention-1 parity) and full CPU suite.
Only the same two inputs; no cohort/calibration/confirmation. Three arms:

1. Historical EviCastKV source shadow, unchanged, common one-buffer transport.
2. Full-prompt conditional-range 3-bit source with charged numerical patches.
3. Exactly arm 2's decoded K/V using common lossless selected-text storage.

Require bitwise 33-prefix and 128-greedy equivalence for the two-buffer vs
one-buffer transport for EVERY arm, and for arm 2 vs arm 3. Require original
FullKV retention-1 <=.001 logits and identical tokens; protected K/V bitwise;
original prefill cache retirement; original logical positions; no FullKV
fallback; immutable encoded state; A-B-A isolation. The model guard remains
32 FullKV-conditioned prefixes, KL<=.02, max-logit<=1, greedy agreement 1.
No threshold or original byte cap is raised. Over-cap formats are unsupported.

Report actual initial/final/decode-owned peak storage, CPU residuals (zero),
numerical patches, all construction and lossless-text overhead, decode
timings and torch allocated/reserved peaks. All other mechanisms remain
disabled as in the original source-representation variant. No general latency,
whole-request peak, official MM-Vet/PPL, independent-quality or superiority
claim from these two consumed diagnostics. Two one-A100/12-CPU workers,
30min limit, finite afterany analysis, no automatic larger campaign.
