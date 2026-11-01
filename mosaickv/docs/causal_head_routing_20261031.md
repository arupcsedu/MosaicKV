# Bounded causal head routing on consumed inputs

Preserve the negative nonlinear-tail result. Its zero-tail and actual replay
controls passed, but one last-layer head could not pass MM-Vet's existing guard.
Do not run a fresh cohort or increase its head quota.

This separately versioned diagnostic tests whether the missing intervention is
upstream. Freeze exactly ONE candidate per decoder layer (32 total): choose the
head with greatest projected prompt-output reconstruction error on the FIRST
FOUR queries from a private 32-token FullKV draft. Ties choose the lower head ID.
This surrogate creates the shortlist ONLY; it does not authorize promotion.

Actually replay EVERY candidate for all 32 draft prefixes through the complete
remaining transformer. Preserve original prompt K/V, mandatory tokens, logical
positions, private histories, and frozen model weights. No donor eviction and no
second prefill. One original full-context BF16 head plus its fully charged
128-token private history must fit the ORIGINAL encoded anchor's physical cap.
Use the existing native-MHA-shape head transport, not H=1 BF16 approximations.

Pick only candidates with all 32 greedy tokens identical to online FullKV,
maximum FP32 KL <= 0.02, maximum logit difference <= 1.0, and mean KL no worse
than the compressed anchor. Order by mismatch count, maximum KL, maximum logit
difference, mean KL, layer and head. If none passes, retain the source-only
anchor with an inactive counted sidecar; never use a FullKV fallback.

Numerical controls remain maximum logit difference <= 0.001 and identical tokens
for clone and full-cache head recomputation. Inactive shadow must preserve all
128 original source-only logits bitwise. Chosen isolated probe must reproduce
the final fixed-prefix replay bitwise. All variants count identical sidecar
capacity. Report every rejected route, source hashes, per-proposal timings,
selected route, initial/final/decode-owned peak bytes and 128-token quality.

Also capture a diagnostic all-head LAST-layer attention ceiling through the
actual nonlinear tail. It has no feasible-cache performance interpretation;
it only distinguishes final-layer attention error from upstream state error.
Save the small actual residual/head captures for CPU audits without rerunning
GPU inference. No benchmark answers, references, scored output or inspected
failure identities determine proposal construction or acceptance.

One A100 / 12 CPUs / 30 minutes per consumed input, two workers concurrently;
finite afterany CPU completeness analysis. Common environment and scratch-only
caches. Source is immutable and all prior negative artifacts hash-bound.

Stopping rule: if no meaningful single-head intervention passes MM-Vet, do not
increase the quota or launch a larger sweep. This tests one causal hypothesis,
not superiority. Even success permits only separately preregistered independent
quality validation. Independent quality N=0. Different historical measured
memory points are NOT byte-matched evidence. Whole-request peak uncertified;
count all draft, scout and proposal overhead separately from greedy decode.
Disabled architecture mechanisms remain labeled in the manifest. No official
PrefixKV/MM-Vet, prototype/residual-repair, or paper performance claim is made.
