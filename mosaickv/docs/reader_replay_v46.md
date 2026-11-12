# V46: exact private integer-ID replay

This CPU-only historical audit continuation does not authorize fresh COCO or
GPU experiments. It preserves the original 210-second whole-projection,
2-GiB combined-input, 8-MiB live-metadata and 100,000-node limits.

## Observed remaining bottleneck

Frozen V45 (`7d568b2b7d5215ece91c658ef4702b9420a293b7`) passed all 8,872 CPU
tests. Consumed canary 121 completed in 72.49 seconds, with 56.61 seconds of
source parsing and 15.87 seconds of identity replay/export. Its complete,
independently closed proof is inherited and must not be regenerated.

Canary 91 parsed its complete 1,018,490,881-byte source in 168.574 seconds.
At the unchanged 210-second deadline it had replayed 20,774,912 of 20,855,640
private bytes, with export still unfinished. It correctly remained unresolved;
near-completion is not a pass. Bulk did not run. The resulting ledger retains
1,336 unresolved occurrences, including 1,201 JSON occurrences.

## Narrow correction

Only complete, canonical private integer-ID dictionary packets receive a
fast decoder. It returns the same ordinary dictionary and exact logical
ownership cost as the existing tagged decoder. Other tags, strings, floats,
split packets and malformed packets retain the strict fallback. Raw-source
JSON grammar, projection, redaction and scalar limits are unchanged.

The enclosing iterator still verifies private-file ownership, signatures,
complete byte coverage, hashes, record counts and replay completion. Both
paths use the same source-plus-auxiliary byte budget and original deadline.
An ID-only plain dictionary also avoids redundant scope copying, but its
value still follows normal recursion with the unchanged context and media
flag. Dictionary subclasses retain the generic path.

All 60 completed JSON transactions remain excluded: 32 original V41 pairs,
26 record-sharded semantic proofs, and two strict bounded pairs. The semantic
proofs keep their historical resource classification rather than becoming
strict-deadline passes. New bulk work requires full-current-source CPU tests
and both unresolved consumed resource canaries to pass. Every failure or
missing closing hash blocks later waves. No safety threshold is weakened.
