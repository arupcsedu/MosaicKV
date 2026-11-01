# CPU feasibility only: exact inter-layer code-byte deltas

Use only the same two CONSUMED full-layer source captures. Do not read benchmark
answers, labels, references or failure identities. No new model execution. The
original normalized BF16 tensors, model revision and input provenance are bound
by SHA. Recompute unprotected three-bit code streams with the frozen quantizer.
These are NOT the current GPU constructor's globally protected/masked packets;
do not report their measured CPU serialization as actual GPU-KV storage.

Fixed codec before execution: 128-BYTE blocks (not a token/block-policy change),
nonzero XOR bitmaps of 16 bytes plus literals, or raw CURRENT blocks. Charge one
uint32 offset per block plus the endpoint; its high bit identifies raw blocks.
When offsets/literals do not save bytes, retain the complete original raw stream.
Different lengths and the first layer require raw fallback. Preserve every byte
and replay the entire chain. No ranges, scales, token memberships, exact overlays
or quantized scalar values change.

Engineering feasibility threshold: BOTH inputs must save at least 10% of their
original code-stream bytes AFTER counting offsets and one largest-code decoder
scratch buffer. This threshold is NOT a task-quality gate or GPU-memory claim.
The corpus-level quality, initial/final/peak cache accounting, baseline fairness,
latency and independent-data gates remain unchanged.

If feasible, implementation must be generic and offered to the frozen baseline
code streams too. Do not optimize EviCastKV while forcing baselines to retain
redundant storage. GPU encoding/decoding, a private per-request cursor, explicit
scratch ownership, mandatory-token/logical-position preservation, bitwise full
K/V/model controls, and bounded same-consumed-input latency validation are still
required BEFORE claiming an operational storage win. A CPU pass permits further
engineering, not automatic GPU submission, fresh calibration or paper promotion.
