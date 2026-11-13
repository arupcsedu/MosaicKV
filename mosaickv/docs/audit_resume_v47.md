# V47: preserve V46 proofs and continue the bounded CPU audit

No fresh COCO confirmation, materialization, or GPU experiment is authorized.
Source parsing retains one 210-second deadline, 2 GiB of combined input reads,
8 MiB of live metadata and 100,000 live nodes. The checkpoint cap remains
512 MiB, each CPU allocation remains at or below 32 GiB, and bulk retains its
original bounded wall time and eight-subprocess maximum.
The inherited 64-GiB new-artifact ceiling is also enforced before receipt
exports and before bulk waves, with conservative per-wave headroom. Reaching
it defers work rather than increasing the allowance or dropping evidence.

## Actual V46 outcomes

Bulk job 22905721 ended after four waves because item 114 exceeded 210 seconds.
It had parsed 274,989,056 of 348,802,166 source bytes; replay had not started.
All 33 other attempted transactions completed their parse and independent
closing hash. Their original source bodies must not be parsed again.

Analyzer 22905722 failed while writing its checkpoint at the unchanged 512-MiB
cap. This was not an absent-proof or model-inference failure. Its 33 saved
per-file proofs, raw task results and identity shards remain authoritative
inputs, and the failed output directory is never overwritten.

## Reader correction

Optional scalar-layout learning now stops after its first actual byte-budget
refusal. Previously a nearly full cache could repeatedly compile the same
unadmittable layout. Existing layouts and strict fallback parsing remain
active. Compilation, refusal and closure telemetry makes the diagnosis
inspectable; a passing synthetic saturation test is not itself proof that the
real failed item has been repaired.

## Exact checkpoint representation

Duplicate inline receipt payloads move to complete, immutable, hash-bound
sidecars. Their order, counts, exact semantic digests, original proof bindings
and membership in global exclusions are validated. Historical source files
are not opened by this recovery. Original report files remain unchanged.

The dominant repeated scope can be stored once against the unchanged sample
ID table, with explicit omitted IDs and additional rows. Hashes and counts bind
the exact full scoped-ID set. This is lossless factoring, not removed evidence:
the explicit decoder must reconstruct the original canonical rows exactly.
Old consumers reject the encoded dictionary rather than interpreting it as an
empty list. CPU reducers decode before validation; only the compact checkpoint
is written, still using the original bounded writer. No over-cap raw legacy
export is published and no RAM limit is raised.

## Finite progression

1. Run all current-source CPU tests.
2. Reconcile all 33 completed V46 pairs in a new derived directory, preserving
   their producer SHA separately from the new analyzer SHA.
3. Require 1,303 unresolved occurrences and exclude all 93 completed JSON
   sources: 32 original, 26 historical semantic proofs and 35 strict pairs.
4. Retry only failed item 114 as a consumed-input execution canary.
5. Only after its complete pair passes, process remaining registered JSON
   items. A missing or failed phase stops later waves.
6. Run afterany CPU analysis, retaining all completed, failed and deferred
   occurrences and every original owner/ordinal/error hash.

The remaining non-JSON provenance gaps remain explicit. Even a successful
reader/audit continuation cannot create a freshness or superiority claim.
