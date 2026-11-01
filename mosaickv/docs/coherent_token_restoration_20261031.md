# Source-funded coherent visual-token membership restoration

The selected-original-BF16 causal control still failed MM-Vet. Restoring original
precision at the same positions cannot remove the observed pruning deficit.
Preserve that negative result and all prior head-restoration results.

This separately versioned construction ablation changes the intervention unit:
ONE original visual-token position is made present across ALL decoder layers
and KV heads wherever currently absent. Individual KV blocks remain strictly
layer/head-local: the cross-layer group is an atomic transaction, never a block
that mixes layers, heads or incompatible RoPE phases. Original selected decoded
K/V stay bitwise unchanged (including their existing quantization). Added K/V
are ORIGINAL BF16 post-RoPE values, losslessly encoded. No averaging, donor
eviction, key renumbering, model updates, residual repair or history eviction.

Source-sharing savings fund the additions inside the ORIGINAL encoded anchor's
physical cap. Sparse extra bodies store ONLY previously absent head items.
Already retained heads receive a masked virtual padding slot, not duplicate
evidence. In-place audited cache cycles restore sorted original logical order.
All extra bodies, indices, cycles, validity counts, history, padding materialization
and masks are charged at the appropriate initial/final/decode-owned peak boundary.
Whole-request/construction peaks remain uncertified and no lower-peak claim is made.

Frozen proposal and acceptance rules:

- Pinned LLaVA-1.5 BF16/eager, the same two CONSUMED diagnostic inputs.
- Reuse original prefill, private deterministic 32-step FullKV draft, no second
  prefill; never return that draft as the compressed answer.
- FIRST FOUR draft query distributions rank every authoritative visual position
  by signed joint prompt-output reconstruction improvement from adding it to
  existing retained output. Calculate each query separately before reduction;
  do not first pool their attention distributions. This surrogate ONLY proposes.
- At most EIGHT positive-gain visual positions, deterministic gain/position ties.
- Actually replay every one-token trajectory for ALL 32 draft prefixes through
  the complete model. Accept only if all greedy tokens agree, max FP32 KL<=0.02,
  max logit difference<=1.0 and mean KL does not worsen the source-only anchor.
- Order eligible plans by maximum KL, maximum logit difference, mean KL, position.
  No passing plan means source-anchor fallback, NEVER FullKV fallback.
- Actual chosen replay must reproduce its private probe bitwise; inactive anchor
  must reproduce its previous 128 logits bitwise. GPU masks/cycles match CPU
  oracles; every added BF16 item and every retained item checked bitwise.
- Ordinary full-cache clone tolerance remains 0.001 and greedy agreement 1.
  A full-retention input already contains the position and is left untouched.
- Final decode remains 128 greedy predictions, scoring only through first EOS.
  Record all timings and failures. Probe cost is not hidden as decode-only time.

Both inputs must have a meaningful accepted addition, pass the unchanged guard,
and avoid diagnostic FullKV-reference ROUGE-L regression before any fresh study
is considered. Stop rather than enlarging this quota if it fails. These are
not official MM-Vet scores, PPL evidence or statistically independent results.
N=0 independent samples. Graph/submodular/prototype/repair mechanisms are disabled
in this explicitly labeled exact-evidence restoration variant. Forecasting here
is online model-only evidence, not saved benchmark answers or evaluation oracle.

These are intentionally DIFFERENT measured storage points inside the same own
cap, NOT byte-matched comparison rows. No superiority over historical LOOK-M,
nearest-baseline, latency, or paper claim follows from execution success. A later
comparison would require frozen equal-budget curves, common storage options for
baselines, and independent quality validation.

One A100/12 CPUs/30 minutes per consumed input, two concurrent workers, scratch-only
caches, common environment, clean immutable source, full CPU suite, and finite
afterany CPU analysis. Preserve every rejected proposal and every prior negative.
