# V8 draft-conditioned one-head diagnostic

## Scope and preserved negative evidence

`bounded_head_v7_negative_seal.json` binds the original V7/V7.1 plans,
diagnostics, and saved logits. The three previously unextended MM-Vet proposals
reproduced their original prefixes but failed at steps 5, 5, and 6; FP64 KL
also increased. Both null controls passed. No original artifact is changed,
and no V7/V7.1 GPU retry is needed.

V8 uses only the same two consumed inputs, original model/configuration,
precision, eager attention, prompts/media, mandatory policy, 40 MiB lifetime
cap, and generated-history reserve. It is a separately labeled mechanistic
diagnostic, not an online candidate, benchmark-quality result, or paper claim.
Fresh-cohort evaluation, calibration, and larger campaigns remain disabled.

## Actual draft evidence instead of prompt-only ranking

Reuse the original prefill and a private FullKV cache. During the clone null
control's 32-token greedy draft, use the adapter's existing
`decode_one_token(..., capture_attentions=True)` interface. The reference and
anchor are advanced once per prefix and shared with all candidate probes.
There is no additional prefill. The ordinary MosaicKV pipeline may already
perform its configured forecast before the diagnostic hook; that unchanged
work is outside the diagnostic probe count.

Capture actual model attention probabilities, not a product of pre-RoPE query
projections with post-RoPE keys. Keep every draft prefix and every query head
separate, including grouped-query heads sharing one KV head. Normalize captured
probability rounding only for this surrogate. Registered model logits and
FP32 KL are not changed.

For one route and draft query, let `p` be its FullKV probability vector and `V`
its FullKV values. Let `S` be its protected prompt positions and `G` the original
generated positions actually retained by the common sink/recent ring. Use:

```text
target        = sum(all prompt and draft positions, p[i] * V[i])
retained_mass = sum(S union G, p[i])
retained_sum  = sum(S union G, p[i] * V[i])
output(S)     = retained_sum / retained_mass
loss(S)       = mean over prefixes, query heads, value dimensions
                of (output(S) - target)^2
```

Each atomic proposal restores one omitted prompt token and removes one
nonmandatory retained prompt token **within the same layer and KV head**.
Evaluate the pair jointly, including its changed normalization denominator.
Select four possible restores and four donors per route using single-edit
loss, then score their 16 atomic pairs. Keep the best positive-gain pair per
route and the best 16 routes globally, with deterministic ordering. This
replaces prompt-mass proposal construction; it does not replace actual-logit
acceptance. Mandatory tokens, exact original K/V, logical positions, capacities,
dtype, physical bytes, and zero padding are unchanged.

Scoring scans positions in 64-position chunks, not an N-by-N pair matrix.
Memory is linear in prompt length for fixed draft horizon and shortlist sizes.
The generated-history terms use FullKV values as a surrogate: real compressed
queries and generated values can differ. Reconstruction gain therefore does
not prove logit or answer preservation.

## Unchanged execution gates

Before creating a pool:

- Both cloned and repacked anchors must preserve K/V tensors and all 32 output
  logits bitwise, and pass the registered guard.
- All 32 FullKV and anchor logits must also reproduce the hashed V7.1 logits.
  An attention-capture or reconstruction-path discrepancy blocks proposals.

Screen at most 16 candidates for up to four prefixes using actual logits;
extend only the best two through the complete 32-prefix gate. Reject at the
first violation; partial success is not acceptance. Keep `atol=1e-7`,
`rtol=1e-6`, greedy-token protection, every-prefix KL/max-logit non-regression,
and checkpoints 1/4/16/32 unchanged. FP64 KL remains shadow-only. A diagnostic
pass requires an accepted **and meaningful** selected modification, not a tie.
All probe states are isolated and discarded; the original anchor is untouched.

The diagnostic ceiling is 248 decode calls: 64 candidate-control calls,
120 proposal calls, 32 FullKV calls and 32 anchor calls. This is a bound, not
a measured latency; existing pipeline forecasting and parity checks are extra.
Record evidence capture, pool construction, packing, screening, extension,
state preparation, wall time, probe counts, and private KV storage separately.
Save `draft-evidence.pt`, `logits.pt`, controls, pool gains, guard decisions,
source-isolation checks, artifact hashes, traces, and full run manifests.

## Launch boundary

After an explicitly authorized scoped commit creates a clean immutable SHA:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/head_substitution_v7.py \
  submit --protocol mosaickv/configs/experiments/bounded_draft_head_v8.yaml \
  --root /scratch/djy8hg/runs/mosaickv/bounded-draft-head-v8-20261026-v1
```

Only two one-A100/12-CPU workers, with two-hour limits, use `bi_fox_dgx` and
the existing aggregate-six-A100 admission controller. A diagnostic pass still
does not authorize fresh inputs, calibration, or a larger campaign.
