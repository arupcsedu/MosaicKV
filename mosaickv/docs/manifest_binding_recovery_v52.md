# V52: source-derived manifest fields and scoped review readiness

This continuation fixes a concrete provenance-projector omission. The two
original consumed-instrumentation analyzers declare their output digests in
`artifact_sha256`, which the earlier flat-hash whitelist did not retain. The
field is admitted only as a flat hash map; answers, predictions, metrics, and
unknown scalar values remain lexically skipped. Malformed or duplicate JSON
still fails closed. Each original manifest is independently hash-closed.

Three original output links are nominated: the routing diagnostic's tier table
and the compact-view diagnostic's tier and representation-timing tables. Their
recorded output digests must match the already completed V49 transactions.
Exact original writer and clean-source helper Git blobs are independently
verified. Input model/source hashes are not promoted into runtime identities.
All 134 old two-pass Parquet proofs are reused, without opening target bodies.
All unaffected V51 records, including dirty-archive/ad-hoc/snapshot limitations,
are preserved verbatim. No previous identity proof is repeated.

## Independent review work

The readiness audit distinguishes passed engineering controls from unresolved
scientific gates. Later bounded-source and COCO numerical controls do not erase
earlier negative variants, validate all six mechanisms, or establish fresh
quality/latency superiority. Gate-specific model, dtype, producer, and memory
boundaries must remain explicit. Whole-request KV ownership certification is
not a whole-device/allocator-reservation certificate.

## Unchanged execution boundaries

Recovery keeps the 210-second, 2-GiB combined-read, 8-MiB live-metadata,
100,000-node, 64-level JSON, 512-MiB checkpoint, 64-GiB new-artifact, 32-GiB RAM,
and eight-worker ceilings. This finite CPU suite → recovery → afterany analysis
chain requests no GPUs. The credential fixture remains unread. Historical
semantic exposure and media/source-group completeness are separate open gates;
all fresh-COCO, materialization, GPU and paper-campaign permissions remain false.
