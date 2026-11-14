# V53: original negative-result seals recover artifact hashes, not runtime writers

Earlier recovery inspected run-adjacent JSON metadata. A new source-first
check found that nineteen committed calibration seals separately record thirty-
three currently unbound Parquet paths and full SHA-256 digests. These digests
match the saved V49/V52 transactions. This is legitimate historical artifact
evidence, but the seal's Git commit identifies the **seal**, not the analyzer
that originally generated the Parquet. Worker/protocol/source-snapshot revisions
remain distinct from runtime-writer provenance.

The registry pins each original seal commit, repository path, Git blob ID,
SHA-256 and registered targets. Verification checks that the old commit is an
ancestor of the already reviewed V52 source, that its seal blob is unchanged
in the current clean source, and that its contents match the pinned hash. An
event-only YAML projector retains only the negative-seal contract, run root,
and explicit artifact path/hash map. No YAML constructors are executed; aliases,
tags, merge keys, duplicates, multiple documents and resource excess fail closed.

The existing transaction classifier checks every reused opaque proof. Thirty-
three records may advance only to `artifact_binding_only`, with the original
producer fields still null. All unaffected V52 records are preserved verbatim.
This cannot turn negative calibration into positive, identify benchmark/media
exposure, or resolve a historical semantic-audit occurrence.

All 134 Parquet transactions and completed JSON proofs are reused without source
body opens, rehashes, row decoding or historical code execution. The credential
fixture is unread. Limits remain 210 seconds, 2 GiB combined input, 8 MiB live
metadata, 100,000 nodes, depth 64, 1 MiB scalars, 512 MiB per checkpoint, 64 GiB
new artifacts, 32 GiB RAM and at most eight worker processes. This finite
CPU-suite → seal recovery → `afterany` analysis requests no GPUs. Fresh COCO,
materialization, GPU experiments and paper-campaign admission remain blocked.

The separate small external-tooling catalog recovered for the modest-memory
analysis is only a partial lead. Its wrapper hash matches an archived Slurm
script, but that does not bind the catalog itself to the original execution.
Equal-byte historical writing tables likewise do not transfer original producer
identity from a later regeneration. Neither fact is used for runtime promotion.
