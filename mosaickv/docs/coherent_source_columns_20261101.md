# Coherent source-column diagnostic and direct-constructor parity

The earlier source affine refits and independent output-scalar corrections
are completed negative diagnostics. Preserve them, their identities and all
baseline outcomes. Do not turn a lower local reconstruction objective into a
quality claim. The latest correction changed an exactly tied FullKV prediction;
no tie-handling, precision, KL, logit or greedy-token threshold is relaxed here.

## Hypothesis and scope

Independent changes to K/V coordinates can disturb their shared projection
structure. Test a separately versioned alternative: keep the original quantized
normalized-input codes unchanged and store a few entire ORIGINAL input channels
exactly. Both frozen K and V projections consume the SAME corrected input.
This coupling is a hypothesis for improved behavior, not an established cause
of previous failures. No weights, coefficients or quantizer codes are trained.

Dense/sparse exceptions and channel-aware quantization have prior art; see
[KVQuant](https://arxiv.org/abs/2401.18079) and
[KIVI](https://arxiv.org/abs/2402.02750). Do not claim the general idea of
precision exceptions or channel quantization is novel. This diagnostic uses
exceptions in their shared pre-projection INPUT, not independent post-RoPE K/V.
There is no RoPE averaging, prototype merging, offload or residual repair.
This replaces the six original mechanisms and remains explicitly labeled a
source-precision VARIANT, not `mosaickv_exact` performance evidence.

Only the same consumed Description and MM-Vet inputs are enabled. Independent
N=0; fresh identity acquisition, calibration, paper campaigns, dataset expansion
and model-family claims remain disabled. Full MHA LLaVA-1.5-7B, pinned BF16/eager
HF settings, original prompts/media and 128 timed greedy predictions are fixed.
This does not establish a VL-Cache Mistral/GQA paper-native reproduction.

## Frozen construction

Use the fixed equal-weight prompt/four-step FullKV-draft sensitivity families
already implemented, averaging each family by its own query count. The draft
reuses the original prefill, borrows a read-only prefix with private history,
is discarded before final generation, and is charged in full. No reference
logit, annotation, teacher target, or benchmark score enters construction.

For each input channel c, compute sum_i sensitivity_ic*(X_ic-Xhat_ic)^2 and
retain the largest positive channels, with stable channel-ID tie breaking.
Store a uint8 map of W entries (255 means ordinary quantized input) plus original
BF16 values for selected channels at EVERY original position. An aggregate
1-MiB allowance, equally divided across layers, includes the map. At the two
observed geometries this permits at most 23 channels per layer; the count is
derived from physical cost, never chosen from answer quality.

Fund the allowance through the same lossless minimum encoding used in the
previous common diagnostic: minimum expansion must reproduce original FP32
bits; retain original FP32 scales. Preserve exact text and protected K/V rows,
logical positions, append-only raw history, original own hard caps, and h4
projection. Refit ONLY numerical transport-envelope exceptions against native
projection of this SAME corrected input, not against output logits. Count all
metadata and temporary layer leases. Do not certify whole-request peak memory.

Fused source reads apply channel overrides in registers; no full corrected
input buffer or second K/V pair may be materialized during decoding. Validate
map contents once during construction. Runtime lookup may inspect tensor
metadata, but must perform no tensor scan, sort, reduction or host sync.
Native-BMM and other unverified transports reject the new format explicitly.

## Orthogonal runtime correction

Test the existing direct constructor with lossless minimum DISABLED for its
parity control. It must obtain memberships online and never consume saved
selection IDs or create a discarded shadow quantizer. Require identical
memberships and every encoded payload bit against the previous constructor,
then bitwise fixed/greedy logits and complete teacher-forced NLL/tokens. Record
full direct selection+construction cost, not merely an incremental conversion.
Single fixed-order timings remain exploratory; warm repeated timings follow
only if correctness and quality justify them.

## Execution and gates

Commit the implementation and CPU tests before the canonical CPU-suite/admission
chain. Run two one-A100/12-CPU workers, 30-minute bounds, with finite afterany CPU
analysis. Stay within the aggregate A100 maximum; no continuous queue/log polling.

Before model loading, require actual CUDA fixtures: zero-overlay input oracle
agreement, corrected input CPU/CUDA bit equality, native-projection tolerance,
h1/h4 K/V bit equality, strided-prefix/history sentinels, and lossless minimum
K/V bit equality. Run the frozen source, direct-constructor control, coherent
source, and four old FP32 baseline controls. Baseline controls must reproduce
their historical logits exactly. These baseline variants are local
head3/boundary8 adaptations, not unmodified official baselines.

This FIRST stage is a source-construction/own-parent diagnostic, not a baseline
superiority comparison: the new column option is not yet offered to baselines.
Set baseline_comparison_admissible=false. Only a meaningful positive model
intervention justifies implementing the corresponding common stored-column
option for every baseline and then performing comparative diagnostics.

Retain raw FullKV bypass, complete ordinary/reinjected 97/593-target parity,
original-source retirement, immutable-format and replay controls. The unchanged
32-token guard is KL<=0.02, maximum-logit<=1.0, exact greedy-token agreement.
Require nonempty corrections, no worse complete PPL AND FullKV-reference ROUGE-L
on BOTH inputs relative to the frozen source for a positive diagnostic. Keep
metrics separate, preserve failures, and report every memory boundary. Neither
zero corrections nor successful execution alone qualifies. No automatic cohort
expansion, threshold change, or paper-level superiority promotion is allowed.
