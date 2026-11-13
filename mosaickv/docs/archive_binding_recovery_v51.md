# V51: originally recorded source archives, with original limitations preserved

This CPU-only continuation addresses the user's expanded backup and Slurm
archive search. It extends V50's immutable bindings and reuses its 134 completed
artifact hash transactions. It does not reopen any Parquet body or credential
fixture, rerun JSON identity proofs, execute historical code, or authorize GPU
work or fresh COCO confirmation.

## Recovered source content

The retrospective COCO v2 report originally recorded a hash for
`analysis_source.zip` and a seven-file `source_code_hashes` catalog. The ZIP
contains the original analyzer and the previously missing CIDEr helper, plus
the remaining registered scorer and test sources. Verification requires:

- the original report's complete-file hash;
- the original ZIP hash from that report;
- independent opening/closing ZIP byte hashes and stable descriptors;
- safe, regular, unencrypted Python-only members;
- all and only the declared source files, matched by unique basename and full
  SHA-256, with no missing, extra, aliased or modified members; and
- the original Parquet output hashes matching the already saved witnesses.

The report explicitly records `analysis_worktree_dirty: true`. Verification
preserves that fact and the recorded base SHA; it does not turn this into a
clean committed runtime. The status is therefore
`dirty_archive_source_content_and_artifact_binding_verified`, not a clean
runtime-writer status. All experiment and paper-admission flags stay false.

The static-history manifest separately records a generator SHA and output
hashes. Its saved adjacent `generate.py` can be verified by exact content hash.
The reported `producer_source_sha` belongs to its input, not the generator.
Its source-content binding is explicitly ad hoc, with no claimed producer
commit or canonical runtime qualification.

## Preserved boundaries

Neither source-content recovery resolves historical identity/media exposure.
No historical unresolved occurrence is removed. Completed proofs, original
dirty flags, partial snapshot bindings, unavailable source records and all
negative results remain inspectable.

No limit changes: 210 seconds per bounded recovery, 2 GiB combined input,
8 MiB live/expanded metadata, 100,000 nodes, 64-level JSON nesting, 512 MiB
per checkpoint, 64 GiB aggregate new artifacts, at most 32 GiB CPU RAM and
eight worker processes. ZIP members also respect the existing 1-MiB scalar
ceiling. Source is read in memory only; extraction to disk is forbidden.

The finite chain is current-source CPU tests → capsule recovery → afterany
analysis. A failed capsule cannot authorize a larger search or inference run.
