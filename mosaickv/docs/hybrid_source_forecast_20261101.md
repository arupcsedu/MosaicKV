# Four-step hybrid source-precision diagnostic (consumed inputs only)

The completed prompt-query-weighted GPU experiment is negative: reconstruction
gains did not preserve free-generation ROUGE-L or complete-target PPL. Preserve
its complete raw results and the earlier negative unweighted/component studies.
This successor tests a different, online signal: four actual greedy FullKV draft
steps from the original prefill, with equal prompt/draft family weights fixed
at 0.5/0.5 after averaging within each query family. No answer/reference/NLL
target or saved reference logits enter construction. No gradient updates occur.

Use only the same consumed Description and MM-Vet identities. Fresh cohorts,
calibration, paper promotion and larger campaigns remain disabled. Independent
N is zero. This is a shared-source precision variant, not the original six-part
exact-selection algorithm, and no prototype/residual-repair claim is allowed.

The forecast borrows the original prefix read-only; its generated history and
one-layer lease are private. Verify the original cache hash and logical state
before/after, remove all hooks, and discard the temporary decode state. Capture
post-RoPE queries, actual eager prompt attention probabilities without
renormalizing away history mass, pre-output-projection attention outputs, and
the four private raw-history K/V positions. Do not perform a second prefill or
emit the draft in final generation. Validate the five logits (including the
prefill prediction) against ordinary FullKV only after collecting evidence.
References are validation-only and never constructor inputs.

Use the frozen diagonal source sensitivity formula with actual draft output
and attention mass. Offer every baseline the same draft queries and history;
compute its hypothetical selected/merged-context sensitivity including causal
history. Preserve actual LOOK-M merged values and protected rows/exceptions.
Fit only the original prompt payload. Keep two affine steps, group128, original
3/8-bit policies, FP32 headers, h4 bit-preserving projection, actual CUDA execution,
and the existing physical caps. CPU/CUDA fixture comparisons remain mandatory.

Evaluate ten arms: frozen source plus hybrid source, and FP32 original plus
hybrid for each of Prompt Top-K, PrefixKV, VL-Cache, LOOK-M local adaptations.
Bind the completed negative result and historical controls. Every hybrid arm's
construction cost includes its parent and the complete online forecast cost;
also disclose integrity hashing and auxiliary memory. These points are not
byte-matched. Construction/whole-request peaks remain uncertified.

Retain raw 128-token bypass parity, complete ordinary/reinjected 97/593-target
parity, immutable payloads, original-source retirement, replay, all cache-byte
boundaries, complete phase timings and failures. Keep the unchanged 32-token
KL<=0.02, max-logit<=1.0 and exact greedy-token guard. Free-generation ROUGE-L
and complete teacher-forced PPL must each be no worse than the frozen source
on both inputs for a positive model diagnostic. Do not relax a gate based on
these outcomes or treat a diagnostic pass as independent quality evidence.

One CPU suite/admission chain submits two one-A100, 12-CPU, 30-minute workers
and a finite afterany analysis, within the aggregate six-A100 maximum. No
continuous log polling, new dataset downloads, or automatic cohort expansion.
