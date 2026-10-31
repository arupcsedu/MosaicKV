# Source-funded, no-donor generation-delimiter completion

This is a separately versioned **padded-layout development variant**, not the
frozen exact-selection paper method or a pass of its zero-padding gate. It
does not enable semantic prototype merging, residual repair, model training,
new quantization, oracle answers, or a fresh/paper campaign. Freeze before
its GPU outputs; use only the same two consumed native diagnostic inputs.

## Evidence and fixed intervention

The common text-source diagnostic completed twenty rows, preserving complete
128-token logits bitwise for all five methods. Its lossless source already
stores every original prompt-text projection input. A label-blind audit also
found missing pieces of the generation marker. Previous donor-removal tests
were harmful. Therefore restore **only the exact complete ASSISTANT: suffix**
without removing any existing selection or changing its decoded K/V bits.
No additional visual tokens, quantitative prior sweep, draft or fit is used.

Verify the pinned tokenizer's ASSISTANT: IDs exactly match the input suffix;
otherwise fail unsupported. For each head append its missing marker tokens,
using ORIGINAL BF16 source K/V. No new quantization. All other kept values,
including LOOK-M's merged ones, stay bitwise identical. Shared original text
inputs and exact scalar residuals reconstruct the added values. An in-place
uint16 cycle permutation restores chronological physical order without a
second K/V tensor. Every original logical position and next decode position
remains unchanged.

Heads may need different numbers of tokens. Allocate the largest needed
width within a layer, use implicit-zero padding, and exclude invalid rows
with compact per-head validity counts. Dummy sorting keys are not token
positions; traces store padding as null memberships. Count every physical
padded row in the materialized layer lease. Count cycle/slot/count metadata,
source/phases/residuals and the complete per-head additive mask workspace.
One fused CUDA kernel builds the mask without additional arange/boolean
HBM arrays. Generated history is always valid. Only isolated HF eager,
batch-one, query-one decoding on the pinned MHA LLaVA is supported.

## Hard budget and comparisons

One atomic GLOBAL transaction admits completion only if actual unique
persistent backing bytes do not exceed that method's measured original
anchor packet bytes. If it cannot fit, retain the lossless source-only
representation and explicitly record no top-up; do not evict donors, fall
back to FullKV, alter a threshold or conceal capacity failure. Initial
funding alone cannot certify final/peak memory: measure those boundaries.

Offer the identical anchor/source/completion options to EviCastKV, LOOK-M,
Prompt Top-K, VL-Cache and PrefixKV local adaptations. Completion rows are
post-baseline adaptations, not paper-faithful upstream algorithms. Source
and anchor rows preserve the corresponding original comparisons. Different
encoded costs and own-anchor caps mean these are **rate/distortion diagnostic
points, not shared-byte-matched rows**. The source-slot ratio is only the
anchor's reference ratio; completion explicitly adds original text slots.

## Frozen controls and stopping conditions

Same LLaVA-1.5 revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9, BF16,
eager, processor, prompts/media, anchor source ratio .5, block1, last-token
mandatory policy, source memberships, boundary-eight-bit/middle-three-bit
codec and raw append-only native history. Keep original FullKV ordinary,
prefill and clone/repack controls. Before proposals, an all-valid mask must
match FullKV token agreement1 and maximum logit difference<=.001. Integer
cycles must preserve all 65536 BF16 bit patterns and history sentinels;
mask kernels must match CPU reference for BF16 and FP32 additive masks.

Use the UNCHANGED 32-token FullKV fixed-prefix guard: max KL<=.02, max logit
difference<=1, token agreement1. Preserve all failures. Source-only outputs
must remain bitwise identical to each method's anchor, and the old EviCastKV
anchor must match its immutable reference. Completion must preserve every
previously retained cache value bitwise; its answers MAY change, so actual
128-token greedy outputs and FullKV-reference ROUGE-L are recorded. Neither
the two examples nor a numerical pass establish benchmark superiority.

Report native decode-owned initial/final/peak backing, including all padding
and masks. Whole-request KV-only peak remains uncertified; inactive reference
formats and construction scratch are outside this diagnostic boundary. Record
decode CUDA/wall and anchor+conversion construction times, but fixed-order
single-trial diagnostics are not a randomized no-slower gate or E2E win.
Capture full provenance, configs, telemetry and per-arm partial checkpoints.

Complete CPU suite, then two one-A100/12-CPU workers, <=30min each, aggregate
ceiling six A100s, followed by a finite dependent CPU analysis. No larger or
fresh cohort is authorized. Advance only after meaningful model improvement,
smaller measured storage and correct mask/state/position controls; otherwise
preserve the negative result without retuning thresholds or inspected labels.

## Driver-only retry

The first execution incorrectly applied whole-output identity to the completion
arm, after successfully computing it. That assertion belongs only to the
lossless source-only arm; completion intentionally adds evidence. Preserve
the original failed run. The corrective source package is byte-identical to
9988fd0. Acceptance/rejection unit fixtures cover this exact branch boundary.
Reuse the four successful EviCastKV anchor/source rows and logits with their
original source/config/manifest fields and immutable hashes. Rebuild packets
for the current proposal, but do not rerun those four generation trajectories.
Run only the 26 failed/unattempted inference rows. Save computed tensor pairs
before semantic postconditions, so later analysis bugs cannot discard expensive
inference again. No algorithm, threshold, selector or numerical tolerance changes.
