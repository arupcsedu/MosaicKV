# V7.1: consumed-input controls and survivor extension

This is a retrospective mechanistic follow-up, not a new online algorithm,
quality evaluation, calibration, or paper result. Its versioned protocol is
`configs/experiments/bounded_head_substitution_v7_1.yaml`. It reuses exactly the
two redacted V7 inputs, original configurations, model revision, FP16 eager
attention, 40 MiB lifetime cap, and generated-history policy. It reads no
benchmark answers or teacher-forcing targets. No new identity is selected.

## Why this diagnostic is needed

V7 rejected every tested substitution but left three MM-Vet proposals unextended
after they passed four prefixes and missed the two-plan shortlist. That does
not establish that all 16 proposals fail the complete gate. V7 also lacked a
same-anchor numerical null control; small FP32 KL differences alone cannot
distinguish a meaningful change from an implementation/numerical problem.

## Registered checks, unchanged

The existing guard checks **every one of 32 FullKV-generated draft prefixes**:
candidate KL and maximum logit error may not worsen relative to the compressed
anchor beyond `atol=1e-7`, `rtol=1e-6`; an anchor-correct greedy token must not
become incorrect. The checkpoints remain 1/4/16/32. Early rejection is allowed;
partial passing evidence is never acceptance. Original FP32 KL remains the
registered statistic. No threshold is adapted to these diagnostic results.

## Null controls and shadow precision

Before extending proposals, run both controls for the entire horizon:

1. Clone the unchanged anchor into independent cache tensors.
2. Repack that anchor from original FullKV tensors in exactly its recorded
   per-head physical slot order, without changing membership or sorting it
   differently.

Both must preserve bitwise K/V values, logical positions, physical bytes, and
all 32 output-logit tensors relative to the ordinary anchor probe. Compare
the original guard metrics as well. A control failure blocks extension and
algorithmic conclusions; it is not permission to loosen the guard.

Additionally compute distribution KL using FP64 log-softmax and accumulation
from the **same FP16 model logits**, alongside FP32 KL. Record direct
anchor/candidate FP64 KL, maximum logit difference, greedy token IDs, and
bitwise logit identity. FP64 accumulation cannot recover precision already lost
in FP16 model execution. It is diagnostic only and cannot authorize a proposal.
Save full-vocabulary logits in a hashed `logits.pt`, not verbose JSON arrays.

## Extend only the three already frozen proposals

Preparation validates the V7 plan and both reviewed diagnostic hashes, then
extracts all and only the `not_in_actual_logit_shortlist` events: none for
LLaVA-Description and three for MM-Vet. It binds their descriptors, position
hashes, and original four-prefix measurements. It creates no replacement pool.

If both controls pass, probe each of those three proposals from an isolated
clone through the unchanged 32-prefix gate, stopping at its first violation.
The first four registered measurements must reproduce V7 exactly; otherwise
the replay is inconclusive. A proposal needs a complete pass for acceptance
and a measured improvement beyond tolerance to count as meaningful. Even such
a diagnostic pass establishes neither benchmark quality nor a smaller cache.
All measured decode states are discarded; original cache tensors and logical
counters must remain unchanged. There is no second prefill for draft evidence.

Only if all three reproduce their original prefix and fail the registered
extension, with both controls passing, is draft-conditioned donor/restore
construction indicated for a separately versioned design. No proposal is
silently revised or promoted during this follow-up. Fresh cohorts and larger
campaigns remain blocked regardless of outcome.

## Execution and evidence

After an explicitly authorized scoped commit makes the worktree immutable:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/head_substitution_v7.py \
  submit --protocol mosaickv/configs/experiments/bounded_head_substitution_v7_1.yaml \
  --root /scratch/djy8hg/runs/mosaickv/bounded-head-substitution-v7-1-20261025-v1
```

Only two one-A100/12-CPU workers are submitted, each with a two-hour limit and
the existing aggregate-six-A100 admission controller under `bi_fox_dgx`.
Workers retain FullKV/reinjection parity, config/input/source hashes, complete
guard rows, FP64 shadow statistics, raw logits, private-state accounting,
cache/source isolation checks, structured failures, traces, and manifests.
The original V7 artifacts remain immutable. CPU synthetic tests are not
substitutes for these real-checkpoint controls.
