# Conservative output-impact allocation: development protocol V3

## Scope and preservation

V1/V2 raw results and derived analyses remain immutable. All completed GPU
workers passed; no old inference is retried. V2 did not demonstrate consistent
smaller-cache quality preservation. In particular, more capacity in heads with
large anchor error did not prove that the **constructed** cache had lower error.

V3 changes only MosaicKV's optional allocation transaction. It is separately
labeled `mosaickv_output_impact_guarded_v3` in the development matrix and
`mosaickv_exact__guarded_output_impact_layers__bounded_generated_history_v2`
in runtime provenance. The latter suffix identifies the unchanged **common
generated-history policy**, not the allocator version. Historical V1/V2 config
SHAs and behavior remain unchanged. No prototype/residual claims are enabled.

## Guard, target and decision

For layer/head `h`, use only the already available prompt-forecast attention
`a_hi` and original source values `v_hi`. Compute the full output and retained,
softmax-renormalized output:

```text
full_h = sum_i a_hi v_hi / sum_i a_hi
retained_h(S) = sum_{i in S} a_hi v_hi / sum_{i in S} a_hi
MSE_h(S) = mean_channels (retained_h(S) - full_h)^2
```

After building V2's proposed selection, measure both anchor and proposal errors
against the **same** full output. Accept the whole transaction only if every
layer/head satisfies:

```text
proposal_MSE <= anchor_MSE + 1e-7 + 1e-6 * anchor_MSE
```

The fixed tolerances cover FP32 working-value arithmetic; they are not an answer
quality margin. Working products use FP32 and reductions use FP64. A mean gain
cannot conceal another head's regression. Invalid/non-finite or zero-retained-
mass inputs fail execution explicitly instead of fabricating a passing error.

If any head regresses, reuse the existing **same-byte compressed anchor**;
do not pack/inject the rejected proposal and do not use FullKV fallback. This
is an atomic whole-plan guard, **not** a claim that every possible individual
donor/receiver exchange has been searched or optimally accepted. Conservative
rejection can reduce the new variant to the existing anchor on many examples.

K/V values, mandatory positions and logical positions remain unchanged.
Layer capacities remain equal across heads within each layer. Both accepted
and rolled-back caches have the same physical prompt bytes and unchanged
generated-history reserve, with zero cross-layer padding.

## Trace and verification

Traces retain per-head anchor, rejected/proposed and accepted-final MSEs,
proposed capacities, anchor/proposed/final position hashes, rejected heads,
decision, fixed tolerances, and measurement wall time including device
transfers. The latter is included in the existing timed allocation/packing
phase, not added a second time to total compression time.

Both generation and teacher-forced traces must validate. When their prompt
budgets match, their selection hashes and guard decisions must match: attaching
teacher-forcing labels may not change selection at the same budget. Under the
unchanged common policy, a teacher-forced target shorter than the configured
history capacity reserves only its available decode updates, leaving more
prompt capacity. Such different-budget selections are recorded separately,
not falsely required to be identical. Every method uses this same rule; PPL
comparisons are paired within that metric's realized budget. Raw failures
remain explicit. Initial/final/peak live KV storage includes prompt payload
and the complete preallocated history ring.

## Small disjoint development validation

The versioned YAML registers eight examples per task from LLaVA-Description and
paper-style MM-Vet, excluding all six IDs per task in the explicit V1/V2
development ledgers. Exclusion files are hash-bound and checked before writing
the selected ledger. Selection uses deterministic parent order, not answers,
scores or failure identities. These parent cohorts have been evaluated before;
this is **retrospective development**, not globally untouched confirmation.

All four compressed arms run every shared cap (32/40/80 MiB): Prompt Top-K,
the existing MosaicKV exact anchor, unguarded V2 and guarded V3. FullKV runs
once per task as a quality reference, never a compressed latency comparator.
The model/revision, FP16 eager attention, media/prompts, one-token blocks,
mandatory-token policy, deterministic 1,024-token trajectories, common 20%
history rule, five warmups and metrics are unchanged. Corpus teacher-forced
PPL and FullKV-reference ROUGE-L remain separate; ROUGE-L is not official
MM-Vet answer accuracy. PPL targets must be complete and never truncated.

This is 26 one-A100 workers / 208 sample-arm rows, admitted by the aggregate
six-A100 controller, with 12 CPUs, 96 GiB RAM and a two-hour limit per worker.
There are no old-worker retries, large campaign submissions, or automatic
promotion. Once authorized and committed, execute from the clean worktree:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/submit_bounded_history_memory_v3.py \
  --root /scratch/djy8hg/runs/mosaickv/bounded-history-memory-v3-NEW-IMMUTABLE-ID
```

The small per-task, same-byte paired analysis and acceptance telemetry can
expose regressions; they cannot establish a powered paper claim. Equal-byte
quality wins are not smaller-memory wins. Only a consistent smaller-cache
quality benefit can motivate freezing an independent quality target and
smallest-quality-eligible competitor rule for a **separately authorized**
powered confirmation. No such target, rule, or confirmation is authorized here.
