# V49: recover historical byte bindings without inventing producer provenance

V48's hash-bound inventory nominates exactly 134 still-unresolved Parquet
occurrences. The protected credential fixture is excluded from all I/O. All
successful JSON identity proofs, source hashes and exclusion checkpoints are
reused unchanged. Targeted provenance-field extraction from nominated metadata
is a distinct operation; it does not repeat the completed JSON identity audit.

## Narrow evidence scope

Only immediate named metadata siblings are examined: manifest, report,
GATE_analysis, provenance, metadata, and the two calibration-report names.
The explicitly recorded source attestation for the two archived source trees
is also examined. No directory traversal, raw predictions, model/dataset access,
historical code execution, or Parquet row decoding occurs.

The bounded JSON reader validates the entire document lexically, but constructs
only root source hashes, eligibility flags, and flat artifact-SHA maps. Question,
answer, score and other payload values never reach the JSON scalar decoder.
Descriptor checks, independent target-file hash passes and a closing metadata
check detect source changes. Current hashes alone never establish historical
artifact provenance.

Three evidence categories remain distinct:

1. **Original runtime writer plus artifact binding:** the original manifest
   records an output hash and writer SHA, the current artifact matches that
   recorded hash twice, and the exact reviewed Git source and source-cleanliness
   guard closure match. Only two explicitly reviewed writer paths can qualify.
2. **Committed snapshot artifact binding:** recorded source-attestation file
   hashes match exact committed Parquet blobs and the current archived files.
   This does not turn a dirty snapshot into a clean original execution. Missing
   original dirty-patch closures remain outstanding.
3. **Artifact binding only:** an original report records the exact matching
   output hash, but the actual analyzer source is not attested. In particular,
   a source SHA copied from an input plan is not the analyzer's execution SHA.

Missing or conflicting evidence stays explicitly unqualified. File names,
schemas, current source versions and enclosing run labels are not provenance.
These byte bindings are not yet closed-writer semantic identity proofs: this
continuation does not alter the 135 unresolved occurrences or grant freshness.

## Unchanged limits and finite execution

All recovery work shares a 210-second deadline and a 2-GiB combined-read budget;
projected metadata remains bounded by 8 MiB and 100,000 nodes. Each checkpoint
remains at most 512 MiB, aggregate new artifacts at most 64 GiB, and the CPU
allocation at most 32 GiB. The single recovery process is below the existing
eight-worker ceiling. No limit is increased to accommodate missing evidence.

Run full current-source CPU tests, then bounded recovery, then afterany CPU
analysis. Preserve all per-occurrence witnesses and failures under a new
immutable output root. No result authorizes fresh COCO confirmation,
materialization, GPU experiments or a paper campaign.
