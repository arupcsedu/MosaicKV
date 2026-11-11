# V33: exact-cover raw ownership for bounded-reader verification

This is a storage/correctness diagnostic on the same two consumed inputs, not
fresh benchmark evidence, a latency comparison, or six-mechanism validation.
Keep the passed V26/V31 source evidence and every incomplete audit intact.

## Problem and narrow correction

The raw comparator retains 32 independent head owners per layer. Exhaustive
constructor observations repeatedly enumerate the immutable K/V aliases,
resolve and serialize their physical ownership, and replay the mapping. The
bounded source reader already passes 128 predictions with bitwise identical
logits; bookkeeping must not be confused with a numerical reader failure.

Use an opt-in, separately named packed raw packet with one exact-size owner
per layer. Its head-local K/V views are contiguous, disjoint, and cover the
entire owner exactly. Per-head capacities, stable rankings, selected words,
logical positions, mandatory tokens, and the 512-byte head-token lattice do
not change. Persistent padding is zero. No smaller cache is inferred from a
shorter view; the whole packed owner is charged as soon as it is allocated,
including regions that have not yet been filled.

The original `RaggedRawPacket` and its cross-head-alias rejection remain
unchanged. The new representation has its own exact-cover constructor checks.
It returns the complete packed root to ownership enumeration rather than
thousands of immutable aliases. It does not suppress ATen observations,
synchronization, native snapshots at lifecycle transitions, or free receipts.
All constructor intermediate results still enter the existing weak audit.

## Required checks

- CPU equivalence for random and tied scores, mixed per-head capacities,
  mandatory positions, original K/V words, and exact backing bytes.
- Reject gaps, overlapping intervals, tail capacity, incorrect shape/dtype,
  malformed position maps, and incompatible scratch storage.
- Native tiny-model prefill and cleanup parity for both raw layouts.
- Full CPU suite and native GPU operator controls before model admission.
- Both consumed inputs: source versus FullKV and raw versus the frozen
  same-selected native path each produce 128/128 predictions and finite logits.
- Remeasure—not infer—initial, final, and full registered lifecycle peak for
  both tensor backing and actual allocator blocks. The six quantities must
  match exactly across source/raw arms.
- Retain every event, ownership delta, manifest, raw prediction, and actual
  allocation-generation `FREE_COMPLETED` receipt; independently replay them.

## Scope

The accounting boundary is published GPU KV plus required cache construction
and reader storage across prefill, construction, and decoding. It is not total
GPU memory: weights, ordinary model activations, opaque vendor workspace, and
reserved allocator segments remain explicitly excluded. CPU metadata is
reported separately. The old 49 MiB lossless-source target remains unsupported.

Reuse `evicastkv_bounded_reader_v28.py` under its explicit V33 policy and a new
immutable run root. Four one-A100/12-CPU/15-minute workers and an `afterany`
CPU analyzer remain bounded to the two consumed inputs, with at most six A100s
in aggregate. No new research cohort or paper campaign is authorized by a
successful engineering gate.
