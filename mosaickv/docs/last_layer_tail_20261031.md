# Last-layer nonlinear-tail diagnostic (consumed inputs only)

The norm-ranked exact-head diagnostic completed but failed MM-Vet. Preserve
its immutable v2 artifacts and do not increase that policy's head quota.

This separately labeled, training-free diagnostic restricts restoration to ONE
head of the LAST decoder layer. After its attention output there are no further
transformer layers to approximate. Evaluate the actual frozen output projection,
residual additions, post-attention normalization, MLP, final normalization, and
vocabulary projection. A last-layer output change cannot modify any cached K/V
at a fixed conditioning prefix. This is not a claim about earlier-layer effects.

Frozen before execution:

- Same two consumed LLaVA-1.5 BF16/eager inputs; no new cohort or benchmark labels.
- One original prefill, private deterministic 32-token FullKV draft; discard draft.
- One private compressed replay on those online tokens captures residuals,
  actual queries, original head outputs, and full-prompt head alternatives.
- Rank all last-layer single-head corrections with a batched nonlinear tail.
  Batch geometry is only a heuristic: verify precisely FOUR leading heads using
  native batch-one tail computations at all 32 prefixes.
- Zero-correction B=1 tail must be bitwise identical to compressed replay.
- Eligibility: all 32 greedy tokens agree with the online FullKV draft,
  maximum FP32 KL <= 0.02, maximum logit difference <= 1.0, and mean KL does not
  exceed the compressed anchor. Deterministic order: eligibility, mismatches,
  maximum KL, maximum logit difference, mean KL, head ID.
- Choose one eligible head or retain the source-only anchor; no FullKV fallback.
- Native replay must exactly reproduce the chosen B=1 counterfactual logits.
- Original FullKV clone and actual recomputed head controls: token agreement 1
  and maximum logit difference <= 0.001. Do not relax after a failure.
- Keep 128 greedy predictions, score through first EOS, retain all timings.
- No donor removals, original K/V and RoPE positions preserved, private history
  buffers preallocated for 128 tokens and fully charged from initial storage.
- Sidecar must fit the ORIGINAL source anchor's physical storage cap, not an
  increased cap. All source and sidecar owners counted. Temporary FullKV scout
  buffers and ranking time reported separately; whole-request peak uncertified.

Baseline/context controls and prior negative results are hash-bound. Two variants:
inactive single-head shadow and chosen-head restoration (inactive on rejection).
Shadow must preserve the original source-only 128 logits bitwise. Both variants
must match each other's actual initial/final/decode-owned peak storage. Different
historical methods' encoded sizes still differ: this is NOT a byte-matched paper
comparison. Independent quality N=0. No superiority or official-MM-Vet claim.

Numerical success alone does not authorize another cohort. Require meaningful
accepted correction, the unchanged guard and no diagnostic ROUGE-L regression
on either input before considering independent benchmark-quality validation.
All six architecture mechanisms not exercised by this diagnostic are explicitly
disabled in its manifest. Prototype merging and residual repair remain disabled.
