# V40: occurrence-specific historical metadata recovery

This CPU-only continuation preserves V39's completed traversal, 6,318 imported
proof files, and all 2,848 successful Parquet recovery proofs. It does not repeat
them. V39 recorded 40,760 raw error occurrences, 39,052 resolved occurrences,
and 1,708 unresolved occurrences; raw evidence and original/new denominators
remain unchanged.

The only recovery scope is the 274 newly observed, explicitly registered
occurrences: 256 graph Parquets from producer
`7cf31bf285ea7b5f4ae7d7caa615921f2e89f417` and 18 ordinary YAML files. The
Parquet producer must satisfy the same eight pinned AST constraints, closed
writer schema, metadata-only footer proof, provenance chain, and two-pass file
hash transaction as V39. YAML uses the unchanged strict event projector;
anchors remain unsupported. Each receipt binds owner, ordinal, original error
hash, canonical path, signature, current file hash, and semantic proof.

The full CPU suite must pass with zero failures, errors, or skips before eight
bounded proof subprocesses may run. Each subprocess retains the existing
210-second and 2-GiB two-pass limits. An `afterany` Slurm continuation checks the
CPU gate even when its prerequisite fails, retaining failures rather than
silently proceeding. Neither stage requests GPUs. The common environment and
all caches remain under `/scratch/djy8hg`.

No inference, dataset scan, sample materialization, identity selection, or fresh
confirmation is authorized here. Large JSON, older anchored YAML, unsupported
aggregate Parquet, credential and external-alias records remain unresolved.
Missing historical media/source-group provenance is a separate blocker even
if every metadata parsing error were resolved. Producer attestation recovers
metadata exposure only; it does not qualify historical performance claims or
certify cohort freshness.
