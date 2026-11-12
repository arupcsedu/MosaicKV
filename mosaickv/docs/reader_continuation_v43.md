# V43 bounded reader correction

V41's smallest/largest canaries passed, but broader execution exposed slow
object-heavy traces and excessive retained metadata. The bulk stage was stopped
after confirmed resource failures. Completed parse receipts were retained;
job 22895417 supplied independent closing hashes without repeating parsing.
V41 analysis and V42's one-item alias correction preserved 70 new proofs.
The resulting V42 ledger has 1,364 unresolved occurrences: 1,229 JSON,
134 nonempty Parquet tables without qualified producer source, and one
protected nonempty credential fixture.

The twelve-second diagnostic on consumed file 58 recorded millions of
`fill`, whitespace, peek and punctuation calls, plus repeated key hashing and
JSON key decoding. V43 targets those operations, not the acceptance limits.
The legacy collector and its exclusion semantics remain unchanged. Fast paths
must validate the same syntax/UTF-8 and retain all identity/scope/media leaves.
Pruning or compacting unused projected leaves is permitted only when collector
identity sets are identical, including late dataset scope and media contexts.
Forbidden benchmark values must never be JSON-decoded.

The limits remain 210 seconds and 2 GiB per independent pass, depth 64,
1-MiB scalar size, 8-MiB retained metadata and 100,000 live metadata nodes.
There is no larger timeout or silent capacity fallback. The current full CPU
suite must pass without failures, errors or skips. Two already consumed
resource-failing files become correctness/performance canaries. Bulk requires
both complete projections and independent closing hashes.

Only the 1,229 unresolved JSON occurrences are eligible. No successful old
proof, nonempty Parquet, protected credential or source-recovery query is
repeated. Each eight-file wave closes its successful hashes immediately. One
resource failure or missing worker stops subsequent waves and preserves the
remaining denominator. Ordinary malformed input remains explicitly negative.

All input ledgers and parent decisions are hash-bound. New evidence is appended
to V42 rather than replacing V40/V41/V42 artifacts. Freshness and confirmation
permissions remain false. Parsing success cannot replace the still-missing
historical execution/media records or dirty-source snapshot. No GPU work,
dataset materialization, benchmark-quality run or superiority claim is part of
this audit continuation.
