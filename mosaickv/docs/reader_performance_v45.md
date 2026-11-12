# Strict bounded JSON-reader continuation V45

This is CPU-only historical identity recovery. It neither changes EviCastKV
inference nor authorizes fresh COCO confirmation or a GPU experiment.

## Measured V44 bottleneck

Clean source `f4a2362009aafeb40b24b97ad0e13d5bfa27a695` passed all 8,853 CPU
tests. Consumed file 115 completed its whole projection in 88.87 seconds and
closed its independent source hash. Its peak live projection used 590,849
bytes and 12,502 nodes. That complete proof is inherited, never rerun.

File 91 reached the unchanged 210-second deadline after reading all
1,018,490,881 source bytes. It processed 42,048,502 scalar fields but replayed
only 917,504 of 20,855,640 auxiliary bytes before failing. Bulk correctly
remained blocked. Reconciliation preserved the successful proof and every
failure: 1,337 historical occurrences remained unresolved, including 1,202
JSON occurrences. No whole-file deadline was reset per fragment.

V45 targets repeated scalar-field dispatch and repeated identity bookkeeping,
not the safety limits or identity semantics. Bounded learned layouts may
accelerate only already-validated plain-scalar grammar; escaped, nested and
unrecognized cases retain the exact fallback. Disk deduplication remains the
authority behind any bounded identity hot cache. All cache ownership belongs
to the existing live-memory budget, and source/auxiliary reads and all parse,
replay and export work share the same original input/deadline bounds.

The registration excludes the original 32 completed V41 pairs, 26 completed
record-sharded semantic proofs, and V44's completed strict pair. The 26 record
proofs retain their original classification; they are not falsely relabeled
as passes of the strict whole-file resource contract. New canaries must be
still-unresolved, consumed resource failures with original receipt bindings.

Only a passing complete-current-source CPU suite and both independently
closed resource canaries can authorize selective bulk work. Any failed or
missing phase stops later waves. Reconciliation is an `afterany` dependent
CPU job so negatives remain explicit. The 210-second, 2-GiB, 8-MiB and
100,000-node bounds are unchanged. No secret/credential body is read.
