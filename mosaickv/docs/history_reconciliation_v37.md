# Occurrence-safe historical audit closeout

This metadata-only correction extends the V36 closeout. It does not change the
frozen EviCastKV candidate, numerical tolerances, byte budgets, scoring,
statistical tests, or freshness requirements. Historical errors and model
outputs remain immutable.

## Why another derived analysis is necessary

The V36 global scanner preserves every new error occurrence, with its original
owner, ordinal, and raw-error hash. Its recovery wave resolves errors from the
older, fixed V18 error universe. Reading that recovery's remaining count as the
global remaining count would miss later observations. The raw global gate
correctly stays blocked, but a unified occurrence-level report was missing.

The previous importer also retained proof hashes without merging every
reviewed identity payload into the exclusion inventory. The correction imports
the eight hash-bound YAML proof workers and the retained oversize-recovery
identity inventory. It unions their identities and the V36 recovery identities
with the global exclusions; it never replaces the global exclusions.

## Invariants

- Validate all eight original error prefixes against the preserved raw error
  list, in owner order.
- Identify an observation by owner and ordinal, with a checked raw-error hash.
- A proof covering an original observation never clears a new observation,
  even when its path and raw-error hash are identical.
- Preserve raw errors, previous exclusions, failed workers and partial waves.
- Import proof payloads only through checked file hashes and source bindings.
- Keep the actual model producer SHA separate from the derived analyzer SHA.
- Never turn a recovered identity inventory into a media/source-group proof.
- Keep materialization, fresh-confirmation and paper-claim flags false. This
  analyzer is not an admission authority.

## Finite execution

Run the new CPU-only Slurm closeout with an `afterany` dependency on the V36
closeout job. The script first executes the complete CPU suite from its clean
source SHA, then creates a new derived report. It does not poll logs, rescan
historical directories, download models or datasets, or submit GPU jobs.

```bash
sbatch --dependency=afterany:22851603 \
  mosaickv/slurm/fox_history_reconcile_v37.sbatch \
  /scratch/djy8hg/workdir/fox-review-v37-reconciliation-code \
  /scratch/djy8hg/workdir/fox-review-v36/run-20261111-1/history \
  /scratch/djy8hg/workdir/fox-review-v37/run-20261111-1
```

`analysis/reconciled_history.json` contains the preserved raw errors,
occurrence-qualified resolutions and unresolved observations, and combined
identity exclusions. `analysis/decision.json` is the small, hash-linked summary.
The complete CPU-suite logs and JUnit record are stored under `cpu/full-cpu`.

## Remaining scientific dependency

Finishing traversal and reconciling errors are necessary but insufficient for
fresh COCO confirmation. The retained media-gap report still lacks execution
and source-media evidence for six historical video pilots and a dirty-source
run. Cleanup intent, absent directories, or different dataset names do not
establish that those executions could not expose confirmation media. The
unchanged media/source-group gate must be satisfied by actual recovered
evidence; this implementation does not manufacture such evidence.
