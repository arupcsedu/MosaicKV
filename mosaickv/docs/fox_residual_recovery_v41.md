# V41: bounded recovery of the 1,434 unresolved audit occurrences

This is metadata/provenance recovery, not a new performance experiment. It
reuses V40's 39,326 resolved occurrences and all existing identity exclusions.
The immutable parent contains 40,760 raw occurrences; every negative result
and the distinction between original and newly observed errors is preserved.

## Recovery scope and limits

* 1,261 oversized JSON files: a new strict byte-streaming lexer validates UTF-8,
  JSON syntax and duplicate keys without decoding discarded answer/reference
  values. Unknown numeric tensors are pruned, not materialized as null arrays.
  Retained metadata is bounded to 8 MiB/100,000 live nodes; depth, scalar,
  chunk, and active-key limits are fixed in source. Each file needs a complete
  projection/hash pass and a separate full hash pass with matching signatures.
* 140 Parquet files: physically empty files can qualify under a closed
  Arrow/pandas footer grammar and exact no-data-page geometry. Nonempty files
  remain blocked without an exact historical producer/artifact proof. Column
  names, current code, or the absence of `sample_id` are not sufficient.
* 22 anchored YAML files: a bounded parser-event DAG handles backward aliases
  and declared merge precedence without object constructors. Tags, cycles,
  forward references, ambiguity, and resource-limit violations remain failures.
* Nine miscellaneous regular files: use exact registered fixture, empty-file,
  nonfinite-aggregate, or already completed V6 derived-recovery proofs where
  available. Nonempty credentials are never opened. A metadata-only recovery
  of a NaN-containing summary does not restore numerical validity.
* Two external directory aliases: only the exact registered links to the
  pinned PrefixKV configuration tree are considered. All 20 physical profile
  files must match their upstream blobs and contain only numeric layer budgets.
  This is explicitly a source-alias proof, not a hash of experimental results.

Worker limits remain **210 seconds and 2 GiB per pass**, with at most eight
CPU workers. The two files exceeding 1 GiB use separate bounded processes for
their two passes; the per-worker bound is not raised. CPU, canary, bulk and
analysis are finite `afterany` stages. A failed canary prevents bulk JSON work,
while small proofs can still be recorded. Accepted canary passes are reused.
Failed and resource-deferred occurrences stay in the denominator.

## Historical media and source recovery

Reuse the hash-bound V36 backup catalog rather than repeat its traversal.
Inspect only project-scoped Git metadata, deduplicate shared object databases,
and look for the exactly registered historical source commits. A separate new
bare repository may request those exact objects from the project's public
remote with credentials/prompts disabled and a fixed time bound. Existing
repositories, branches and results are not overwritten.

Source availability, cleanup intent and cached videos are not execution/media
exposure proofs. Original execution journals, media/frame hashes, sampling
recipes and the dirty-worktree snapshot must still be recovered and validated.
No helper may infer these from filenames or claim an untouched cohort.

## Admission

The full current-source CPU suite must pass with zero failures, errors or
skips before real proofs run. The smallest and largest registered JSON files
are canaries before bulk recovery. Final reconciliation binds each proof to
its exact owner, ordinal, raw error hash, source signature and proof artifact.
Every old record and exclusion is retained. No GPU, model, dataset download,
materialization, sample selection, or fresh confirmation is authorized by this
metadata-only chain. All existing scientific gates remain unchanged.
