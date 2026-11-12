# Selective closed-producer recovery

V38 passed all 8,494 CPU tests and all eight verification owners. Its
occurrence reconciliation retained 40,760 raw errors, with 36,204 resolved and
4,556 unresolved. These are audit counts, not benchmark-quality measurements.

The 1,544 attempted Parquet proofs failed before body hashing because the
reviewed whole-worker AST belonged to a different historical revision. Another
1,304 candidates were resource-deferred. Review of the two exact recorded
producer revisions shows that the diagnostic writer/schema is unchanged; the
worker differences concern other experiment paths. The new registry pins both
actual whole-worker hashes, preserving all seven other AST constraints,
worker/record/audit bindings, exact footer schemas and before/after file hashes.
Unknown producers remain unsupported. Historical source is inspected, not run.

The finite retry requires the complete current-source CPU suite, then one real
proof canary for each reviewed producer. Only after both pass does it process
the 2,848 failed/deferred files in at most two waves. Each wave retains eight
210-second workers with a 2-GiB two-pass budget. Successful canaries are not
hashed again. Completed V38 YAML/JSON proofs are imported, not rerun. Blocked,
missing and unprocessed candidates remain unresolved in the full denominator.

No numerical/model/quality gate, worker I/O bound, historical raw artifact or
GPU execution is changed. The 899 newly observed occurrences are not waived
by proofs for older identical paths or hashes. Historical media/source-group
gaps remain a separate blocker; this recovery cannot authorize fresh COCO.
