# Exact structural-template key serialization

The device-record model profile identified 47.5 ms of instrumented graph time
inside metadata hashing; 45.4 ms was the generic dataclass conversion loop.
These cProfile values are not expected uninstrumented savings. The follow-up
eliminates deep copies of the frozen scalar/tuple-only `BlockEvidenceMetadata`
while retaining every field, ordered logical position, modality, layer/head,
eligible set, configuration and execution control in the SHA256 key.

The explicit `graph_direct_metadata_key` switch requires structural templates
and defaults off. Unknown metadata subclasses retain generic serialization.
Hash identity is mandatory, including non-ASCII OCR strings, all optional fields
and alignment IDs. Complete sparse COO edges/weights/diagnostics must be identical.
The bank retains no metadata owners, K/V, queries, media, predictions or labels.
No template invalidation rule, graph heuristic, threshold or cache budget changes.

First run the CPU-only 603-node hash/graph probe. Require hash and graph identity
and a median operator ratio at most 0.8 before considering a GPU timing test.
Any GPU comparison uses the same two consumed inputs, deterministic controls,
eight warmups and 24 balanced paired trials. Compare the best device-record
hybrid to the otherwise identical direct-key hybrid, with storage-only and
common-codec controls. Require all 128 logit vectors bitwise and identical
initial/final/decode-owned-peak cache storage. The strict adjusted upper ratio
must be at most 1.00 for a latency improvement; a neutral result is not a win.
This changes execution only and cannot establish improved benchmark quality,
additional KV savings, whole-request peak certification or six-stage superiority.
