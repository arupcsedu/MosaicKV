# Consumed-input routing and tier instrumentation

EviCastKV is the paper name of MosaicKV. This is an instrumentation diagnostic,
not a new method, benchmark cohort, calibration, or paper campaign.

## Frozen scope

Two workers use already consumed inputs only, selected by the first-entry rule
without reading their outcomes:

1. First LLaVA-Description entry from the completed October 29 PrefixKV plan:
   Prompt Top-K, `prefixkv_selection_only_reimpl`, and `mosaickv_exact`, each
   at the identical historical 120 MiB cap and 1,024-token generation policy.
2. First COCO entry from the completed October 29 installed-output-guard
   development ledger: frozen exact and guarded exact EviCastKV, each at its
   existing 20 and 40 MiB caps and 64-token generation policy.

Model/processor revisions, inputs, arithmetic, mandatory-token policy, routing
fraction, guard thresholds, and all algorithm configurations are unchanged.
The plan binds historical configurations, ledgers, and media by SHA-256. No
dataset access, new sample selection, or answer-based policy fitting occurs.
The maximum is two one-A100 workers, each with 12 CPUs, 64 GiB requested host
memory, and a 30-minute Slurm limit. The aggregate owned-A100 ceiling remains
six; the existing admission controller registers the diagnostic array.

## Instrumentation

`execution_audit.collect_execution_audit` activates a request-local ContextVar.
Without that context the original functions execute without collecting data.
The collector is not consumed by the selector and holds no model tensors,
references, or labels. It records:

- every available layer/head routing score, deterministic rank and tie order,
  selection, threshold and fraction, and the exact router call label;
- graph/selection validation and prototype-safety checking;
- full tier construction and its nested exact-only constructor;
- source-membership validation and descriptor gathering;
- exact-tier invariants;
- state creation, membership/storage invariants, and active-layout metadata.

These spans use synchronized wall timing and CUDA Events with synchronization
at both boundaries. CUDA Event elapsed time is not described as kernel-only
time. Inclusive parent and child timings must not be summed twice; exclusive
wall time is also recorded. Child synchronization and timing overhead can
appear in parent exclusive time, so these are profiling measurements, not
uncontaminated production latency. All output explicitly marks
`profiler_contaminated: true`.

For each arm, run instrumentation off and then on. Both requests independently
start from prefill and retain the historical output length. An LM-head hook
captures full next-token logit vectors in both runs for comparison; no weights
or forward results are modified. It does not read benchmark references. Logit
vectors remain bounded CPU temporaries and are discarded after comparison.

Acceptance is strictly diagnostic:

- initial FullKV ordinary-generation and cache-reinjection parity for 16 tokens;
- identical off/on generated token sequences, selected-block/packing catalogs,
  and initial, final, and peak physical cache bytes;
- finite logits and unchanged historical maximum logit tolerances: 1e-3 for
  the PrefixKV scope, 1e-4 for the COCO scope;
- nonempty EviCastKV routing and operation records;
- deterministic replay of selected routes from all saved scores and ranks;
- complete worker results, error records, source/config/environment provenance.

The callback runs afterany, including on producer failure, and writes a new
derived operation table and routing ledger. There is no automatic promotion,
fresh-cohort submission, parameter tuning, or continuation to a large run.
Benchmark-quality and superiority conclusions cannot follow from this gate.

## Execution

After CPU tests and a clean instrumentation commit:

```bash
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/mosaickv/src:$PWD/mosaickv/scripts"
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_consumed_instrumentation.py prepare \
  --root /scratch/djy8hg/runs/evicastkv/NEW_CONSUMED_DIAGNOSTIC
# Pass the printed plan SHA explicitly to submit:
/scratch/djy8hg/env/mosaickv/bin/python \
  mosaickv/scripts/evicastkv_consumed_instrumentation.py submit \
  --root /scratch/djy8hg/runs/evicastkv/NEW_CONSUMED_DIAGNOSTIC \
  --plan-sha PLAN_SHA
```

The finite dependent CPU callback generates `analysis/report.md`,
`analysis/GATE.json`, `analysis/routing_scores.json`, and
`analysis/tier_operations.parquet` without requiring continuous log polling.
Failed attempts are immutable and must not be blindly overwritten or retried.
