# V44: reduce audit-reader memory without discarding identities

This is a CPU historical-metadata recovery tool, not a KV-cache algorithm,
model-quality experiment, or permission to run fresh COCO confirmation.

This implementation is distinct from the concurrent record-sharded diagnostic
at `0d45f555141bf6f2760cba73bef924c952a69f47`. That diagnostic passed its
grammar/identity canaries, but reset 210 seconds for every fragment rather than
the complete file. Its bulk worker 22900327 was stopped and its dependent
analyzer 22900328 cancelled after the resource-contract mismatch was identified.
All its completed artifacts remain preserved; they are not claimed as passes
of the unchanged whole-projection resource limit. No source or task artifacts
from the 32 previously qualified V41 pairs are regenerated.

The V43 CPU suite passed, but both consumed resource canaries failed their
unchanged live-node limit. A bounded charge observer on file 122 measured
100,001 live nodes and 1,800,003 projected bytes after reading 231,211,008 of
350,044,258 source bytes. Integer ID occurrences and their repeated record/array
scaffolding—not the 8-MiB byte ceiling—caused that failure. The diagnostic was
incomplete and is not accepted as a source projection proof.

## Representation and exactness

V44 retains the strict V43 JSON grammar, duplicate-key checks, UTF-8 checks,
redaction and semantic pruning. Completed array children can move to private,
typed JSON spools at a 1-MiB/12,500-node watermark. A tagged codec prevents a
source object from impersonating a spool reference. Full spool signatures,
hashes, lengths and replay counts are checked before acceptance. A bounded
flat-object fast path avoids repeated Python token-dispatch work for scalar
records while preserving the same validation and projection decisions. A
bounded scalar-member-run path also handles scalar fields surrounding nested
profiler `args` objects. Fast runs and nested/escaped/chunk-boundary fallback
fields share the same duplicate-key set; neither path bypasses validation.

Identity collection replays complete records only after EOF, when late parent
dataset fields are known. It uses the exact legacy namespace/configuration/split,
dataset-descriptor precedence, group and media-hash rules. A scratch SQLite
sink deduplicates identities without accumulating worker-wide Python sets.
Typed, independently hashed shards retain every exclusion. Unknown/forbidden
benchmark values are never decoded into the projection or sink.

The unchanged limits are 210 seconds and 2 GiB per independent pass, 8 MiB of
live projected metadata, 100,000 live metadata nodes, depth 64 and 1-MiB source
scalars. Source and auxiliary replay/export reads share the same 2-GiB budget;
spooling does not provide an unbounded second input. Buffers and shard references
are charged while live. SQLite's reservation and temporary files remain bounded
and scratch-only. A complete projection still requires a separate whole-source
closing hash pass; temporary-spool hashes cannot substitute for it.

## Selective continuation and admission

The completed V44 closeout contains exactly 1,203 unresolved JSON occurrences.
Only those indices can be invoked. Its 26 complete semantic proofs are reused
as historical identity evidence, not relabeled as unchanged whole-file resource
passes. The 32 completed V41 JSON pairs, V42 correction, non-JSON records and
every old exclusion are inherited, never replayed or relabeled. Canaries select
the largest still-unresolved consumed V41 failure in each resource category,
with source/registration bindings checked; already completed files are excluded.
Bulk starts only if both canaries produce complete, independently closed proofs.

Each eight-file wave closes and validates all successful parse hashes before
another wave. Missing/failed hashes or any other unqualified pass stop later
waves. Reconciliation rechecks the live canonical source signature and all
shard hashes/types/counts/order. The final receipt contains only new exclusions
relative to the immutable parent plus preceding qualified transactions; its
bound proof retains the complete shards. A failed shard never updates cumulative
exclusions. This avoids repeating identical profiler IDs throughout the ledger
without losing any of the six identity fields or their scopes.

The full current-source CPU suite must pass without failures, errors or skips.
All stages are finite Slurm `afterany` jobs with explicit acceptance checks.
Failed and deferred occurrences remain in raw records and denominators.
Freshness, source-group disjointness, materialization and GPU confirmation remain
disabled. Successful JSON recovery cannot replace missing historical media or
source provenance, and creates no quality, memory-superiority or latency claim.
