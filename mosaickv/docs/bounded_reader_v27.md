# Bounded original-source reader: lifecycle verification

This is a storage/transport engineering variant, not the six-mechanism
EviCastKV algorithm and not an independent benchmark-quality confirmation.
The two inputs are already consumed development identities. No fresh cohort
or paper campaign is authorized by this protocol.

## Passed predecessor

V26, source `008f330edbf19421f658761efe6349d4198e2487`, measured all 128
predictions and all logits bitwise equal on both inputs. It constructed raw
comparators with exactly matching initial, final and decode-owned peak tensor
backing **and actual CUDA allocator blocks**. The replayable gate is at
`/scratch/djy8hg/workdir/fox-review-v26/run-20261111-3/analysis/GATE_bounded_reader_v26.json`.
This does not certify its prefill/construction peak. Its old 49 MiB lossless
source target remains unsupported; the measured original-source payload is
approximately 150 MiB, before history and workspace.

## V27 change

Run the original native eager prompt attention at every layer. Only after that
attention returns, construct its source or selected-raw packet and retire its
native K/V owners. Never retain an all-layer original cache or copied snapshot.
Weak references include tensor view bases: dropping a view alone is not a
retirement proof. The raw selection must hash-identically match V26.
CUDA retirement additionally requires the actual allocator allocation
generation's `FREE_REQUESTED` and `FREE_COMPLETED` events after capture.
An alias, a reused address, weak-reference death, or a free request alone
cannot satisfy this check. A bounded native memory-history trace fails closed
on truncation; sealed receipts are independently replayed on CPU.

Source construction reuses the same full-width component pool and one-head
scratch as decoding. It must reconstruct every original BF16 word with zero
patches, or fail closed. Raw head K/V uses exact-cover joint allocations and
the same physical-byte lattice. Both arms retain the original logical decode
positions and append-only lossless generated history. No numerical threshold
is relaxed. The first prefill logit must also equal the frozen native output.

## Accounting boundary

The request ledger begins before native prefill and covers published native
cache K/V, cloned source captures, phase metadata, retained raw K/V,
method-specific selection buffers, constructor/reader pools, logical-position
metadata, incoming decode K/V and overlapping history growth allocations.
Each actual ownership event resolves strong tensor owners against the native
CUDA allocator snapshot. Complete ownership deltas are independently replayed
on CPU; lifecycle validation additionally requires every layer's publication,
construction and retirement and every one of the 127 decode updates.
Every ownership observation is device-synchronized. A per-ATen constructor
audit also charges finite-check predicates, selection intermediates, index
buffers and any other live constructor outputs. These instrumented timings
are not a performance benchmark.

This is **published cache-owned KV storage**, not total GPU memory. Model
weights, ordinary projection/RoPE/attention activations,
opaque vendor workspace and allocator-reserved free memory are explicitly
outside this KV boundary. The allocator-wide GPU peak is a different metric;
the protocol does not certify it by relabeling the KV ledger.
The matched boundary is CUDA GPU storage. CPU position-map backing is reported
separately and is not silently represented as an equal total-system footprint.

Compare the post-prefill initial, final and entire registered request's peak
for tensor backing and allocator blocks separately. A decode-only peak is
never substituted for the request peak. All method-specific construction
buffers must be included even when their lifetime is short.

## Execution

`mosaickv/scripts/evicastkv_bounded_reader_v27.py launch --root NEW_SCRATCH_ROOT`
requires a clean immutable checkout and all six pinned submodules. A full CPU
suite and one small GPU arithmetic canary precede conditional release of four
one-A100, 12-CPU, 15-minute workers (two consumed inputs × two methods).
Admission allows at most six aggregate A100s. A finite `afterany` CPU analyzer
runs even after failure; write-once receipts prevent duplicate submission.
No continuous log polling is used. Instrumented worker time is not latency
superiority evidence. Failure records and partial ownership traces are kept.
