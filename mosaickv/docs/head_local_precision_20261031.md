# Head-local protection and asymmetric precision: bounded consumed diagnostic

EviCastKV and MosaicKV name the same system. This is a separately labeled
training-free precision variant, not the validated graph/exact-only paper path.
Original selected positions, original K/V or actual LOOK-M merged K/V, logical
positions, mandatory last prompt token, and append-only generation remain fixed.

## Motivation and attribution

The existing packet protects the union of selected text slot indices across
all heads. A text slot in one head can cause an unrelated visual slot in another
head to be stored losslessly. Test protection per head instead; retain the
original first four/last eight selected entries, text values, boundary layers,
and existing <=0.5% correction exceptions. This is lossy for formerly incidental
visual protection and therefore needs actual model-level checks.

Also test 2-bit quantization separately on keys/values. Quantization axes follow
the existing per-channel keys/per-token values design; that general principle
is established by KIVI (Liu et al., ICML 2024), not claimed novel here:
https://proceedings.mlr.press/v235/liu24bz.html . No official KIVI code is invoked.

## Frozen arms and fairness

Run every method (Prompt Top-K, local PrefixKV/VL-Cache/LOOK-M reimplementations,
and the EviCastKV profiled-prior selector) under all five policies: unchanged
union 3/3-bit reference, head-local 3/3, head-local 3/2, head-local 2/3, head-local
2/2 (key/value bits). Do not choose a weaker baseline or deny it the common
codec. Do not alter the .5 raw source-slot capacity or source profiles. Actual
encoded bytes differ: these are representation/Pareto diagnostics, NOT shared
physical-byte-matched quality comparisons.

Use the same two consumed Description/MM-Vet inputs from the immutable parent
canary; no new identities or annotated answers are loaded. One pinned BF16
LLaVA-1.5-7B HF eager prefill is reused; normalization/query captures and the
unchanged selectors construct all packets before scoring. Two boundary layers
at each end remain entirely lossless. Packet buffers, indices, minima/scales,
exact payloads, history, preallocation and active scratch leases all count.

## Correctness and measurements

CPU tests must pass before GPU admission. Each GPU worker first checks all
lower-bit decode and head-local scatter paths against the CPU oracle, including
all 65,536 BF16 bit patterns and untouched history sentinels. Retention 1 remains
the existing untransformed FullKV bypass; the new format is compressed-only.

For every method/policy:

1. Replay the first 32 saved FullKV-generated tokens to compare 32 affected
   logits at identical prefixes. No annotated benchmark answers are used.
2. Decode its own complete 128-prediction greedy trajectory, with common raw
   BF16 history, no history eviction/quantization, and no FullKV fallback.
3. Require unchanged-union control to reproduce saved lossless-transport
   3-bit logits within .001 and token agreement 1. Preserve all failures.
4. Preserve the registered same-prefix guard relative to the union anchor:
   all 32 greedy tokens identical, FP32 KL <= .02, maximum logit error <=1.0.
   Report FP64 KL as a supplementary diagnostic, never a replacement guard.
5. Record full greedy-token identity, exact prompt bytes and decode-owned
   initial/final/observed peak backing, and decode CUDA/wall timing. The full
   prefill is retained for diagnosis, so no whole-request memory or end-to-end
   latency claim is allowed. Changed answers cannot be called quality-preserved
   solely because a short-prefix gate passes.

A common policy is eligible for further consideration only if all ten
method/input combinations pass transport/isolation, retain complete own-greedy
token identity, pass the unchanged prefix guard, and reduce prompt backing.
That is a representation diagnostic, not a paper result or quality superiority
claim. No fresh cohort or large sweep is automatically authorized here.

Six one-A100 jobs, each 12 CPUs and at most 30 minutes, divide two consumed
inputs into three method groups. Six aggregate A100s is the ceiling; submit only
when no other owned GPU work exists. A finite afterany CPU reducer checks all
artifact hashes. All caches stay under scratch and dependencies stay in the
existing common environment. If a policy fails, do not relax thresholds or
increase cohort size to conceal the failure.
