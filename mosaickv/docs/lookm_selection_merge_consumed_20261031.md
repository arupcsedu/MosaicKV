# LOOK-M selection versus merging: consumed-input causal diagnostic

This is a new bounded diagnostic, not a fresh cohort, baseline reproduction,
performance comparison, or permission to resume a paper campaign. The two
inputs are exactly the previously consumed Description 000000532947 and MM-Vet
v1_197 inputs from the hash-bound complete-request canary. Online construction
receives only prompt/media, never annotated answers or evaluation references.

## Frozen controls

- Same pinned LLaVA-1.5-7B revision, BF16, HF eager, seed, processor, media,
  prompts, block size one, mandatory last prompt token, and 0.5 source capacity
  as the parent. No model or baseline algorithm changes.
- One ordinary prefill with attention capture. Compute the existing local
  LOOK-M balanced-layer plan once, without modifying selection or merging.
- Hold every layer/head's selected original positions and physical capacity
  identical in all four arms: selection-only original K/V; merged K with
  original V; original K with merged V; actual merged K/V. Only the last is
  `lookm_reimpl`; the others are explicitly labeled causal ablations.
- Use private ordinary BF16 DynamicCache tensors for every replay. Do not
  quantize, reconstruct, prototype, repair, alter weights, or renumber RoPE.
  Maintain independent layer lengths and original next decode positions.
- Preserve the original FullKV cache for reuse and hash it before/after. Thus
  allocator peaks here are NOT online whole-request memory measurements.

## Registered measurements

1. Clone FullKV and generate 128 greedy predictions, ignoring EOS only for
   this fixed-length diagnostic. Require token agreement 1 and maximum logit
   difference <= 0.001 against the saved ordinary FullKV trajectory.
2. For each arm, replay the same first 32 FullKV greedy tokens and measure
   the resulting 32 cache-dependent predictions. Exclude the common prefill
   logit. Retain the existing guard: every greedy prediction identical,
   maximum FP32 KL <= 0.02, maximum logit difference <= 1.0. Save FP64 KL too,
   without substituting it for the registered guard.
3. Separately decode each arm's own 128-prediction greedy trajectory. Store
   answers only through first EOS. Autoregressive cross-trajectory logit errors
   are labeled as potentially differing conditioning contexts, not causal
   identical-prefix evidence. No task score or statistical quality claim.
4. Require the actual merged trajectory to reproduce the saved unmodified
   LOOK-M lossless trajectory within 0.001 and exact token agreement.
5. Replay merged K/V again after other arms: require bitwise identical logits
   (clone/repack and A-B-A isolation control). Verify original source hash,
   common position catalog hash, equal initial/final/completed-boundary peak
   cache bytes, zero padding, and unchanged logical positions.

The factorial attribution is descriptive on two consumed inputs. A lower
fixed-prefix KL does not establish higher benchmark quality. Neither a guard
pass nor failure is a measured task score. Negative findings remain preserved.

## Execution and stopping

Full CPU unit/integration suite from a clean immutable commit precedes two
one-A100 workers (12 CPUs, 48 GiB host RAM, 20-minute limits). At most two GPUs
are admitted, below the repository's four-GPU aggregate ceiling. Use the common
environment and scratch-only cache policy. A finite `afterany` CPU reducer
checks hashes and reports all failures; no recurring log polling or additional
GPU sweep. No independent cohort is authorized by this diagnostic.

Shared planning time is recorded only as a diagnostic. The selection-only arm
still pays the shared merge-plan computation, so it cannot support a latency
claim about an optimized merge-free implementation. KV-only peaks are measured
at completed decode boundaries, not transitory concatenation or whole-request
HBM peaks. No superiority claim follows from these controls alone.
