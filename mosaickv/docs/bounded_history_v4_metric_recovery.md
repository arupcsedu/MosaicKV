# V4 metric/storage recovery and isolated logit audit

This repair does not define another compression algorithm or promote V4. V4
remains a negative development result. All 16 planned identities are consumed;
no sample answers or failure identities may guide another candidate.

## Defect and repair

A one-token teacher-forcing target requires zero cache updates. Previously,
when no generated-history ring was installed, preallocation incorrectly used
the generation limit (1,024 outputs) and reserved 1,023 unused positions.
Uniform caches silently exceeded the metric-phase byte cap. Ragged V4 caches
also lost independent-layer metadata during replacement, causing the occupied
length safety check to fail. Generation predictions themselves are reusable.

The runtime now plans updates from the actual target length, skips zero-update
preallocation, preserves independent-layer metadata, and validates a replacement
before publishing it. The occupied-length safety check remains enforced.
Generation and teacher-forced phases now receive separate initial/final/peak
backing-storage checks, including preallocated storage rather than tensor views.
The existing common short-target history-capacity rule is unchanged.

## Immutable recovery scope

Parent: `bounded-history-memory-v4-20261024-v1` under the scratch run directory.
Plan SHA-256: `caa4616930ff505958577224a5704f1f5aed68401609dc6068aac592a0acbd62`.

`scripts/recover_bounded_history_v4.py` binds the parent configuration, immutable
ledgers, references, predictions, traces, failures, and available manifests.
It refuses missing generation predictions, duplicate valid phase observations,
changed input hashes, or expansion beyond the registered small repair.

- Recover all 160 ROUGE-L scores from saved predictions on CPU.
- Reuse every complete, storage-valid NLL/token-count observation.
- Retry only missing/invalid PPL, without regenerating benchmark predictions.
  The audited parent requires 21 computations across nine configurations.
- Record new producer/config SHAs separately from reused observation SHAs.
- Disclose missing parent manifests; never fabricate original GPU provenance.
- Keep failed samples explicit and compute corpus PPL only with all eight
  planned samples present. Exponentiate summed NLL divided by summed tokens.
- Retrospective 10,000-draw paired bootstrap intervals remain development-only
  and are not multiplicity-adjusted confirmatory claims.

All recovery artifacts are written to a new directory. Original raw artifacts
and the original incomplete analysis are never overwritten.

## Label-blind diagnostic, not another candidate

Two diagnostics use the first planned identity in each task's consumed ledger
at 40 MiB, selected by protocol order before examining failure identities.
Each reuses one prefill and clones FullKV, the exact-selection anchor, and V4's
accepted allocation. Four FullKV greedy draft updates supply the same prefix
to all states. Record per-step KL, maximum logit difference, greedy agreement,
and physical active bytes. Check original K/V and logical counters remain
unchanged. No benchmark target or answer enters this diagnostic.

The evaluation-only subclass exits after prefill planning and these short
probes, without performing the 1,024-token benchmark decode. Its traces are
explicitly not quality predictions or performance rows. Probe outcomes do not
change online selection or authorize promotion. The purpose is to test whether
accepted prompt-reconstruction improvements preserve actual autoregressive
logits before proposing another algorithm.

## Execution

From a clean, committed worktree:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
run_root=/scratch/djy8hg/runs/mosaickv/NEW_IMMUTABLE_RECOVERY_ROOT
/scratch/djy8hg/env/mosaickv/bin/python mosaickv/scripts/recover_bounded_history_v4.py \
  prepare --root "$run_root"
```

Review `plan.json`, `rejected-ppl.json`, and `negative-v4-seal.json`, then pass
the SHA-256 of `plan.json` to the script's `submit --root ... --plan-sha ...`
action. The controller enforces at most six aggregate A100 workers. Each worker
requests one A100, 12 CPUs, 96 GiB RAM, and two hours using `bii-gpu`, account
`bii_dsc_community`, and reservation `bi_fox_dgx`. CPU analysis depends on all
workers terminating and retains any explicit failures. No larger campaign is
submitted. Cache paths come only from the shared scratch cache environment.
