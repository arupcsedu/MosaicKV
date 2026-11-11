# V31: equivalent CPU ranking for bounded whole-request verification

This is a consumed-input storage/correctness diagnostic, not a latency result,
fresh quality confirmation, or activation of the six research mechanisms.

V28's source worker completed with 128/128 bitwise-logit parity. Its raw
comparator timed out after the registered 15 minutes before a complete row was
written. Preserve that negative run; never infer a paired peak certificate
from the successful source arm alone.

## Narrow correction

The globally balanced raw allocation gives each head either `lower` or
`lower + 1` positions. The remainder is a contiguous prefix in layer/head
order. Rank those two capacity groups in at most two independent, stable
per-row sorts instead of calling the same scorer once per head. Reorder each
head's selected set by original position exactly as before. Do not obtain a
smaller set by truncating a source-position-sorted larger set.

Validate contiguous CPU int64 position maps with exact Python integer bounds
and adjacent monotonicity checks. This preserves validation while avoiding
temporary CPU Boolean tensors. GPU indices, selection scores, mandatory
positions, original K/V, per-head allocation layout, direct gathers, and all
logical positions remain unchanged.

The constructor observer still runs for **every executed ATen operation**.
No owner observation or synchronization is suppressed. Fewer CPU operations
are actually executed; these operations previously triggered complete walks
of every retained GPU owner even though they created no GPU storage.

## Controls

- Compare selected positions, wordwise K/V, bytes, and position-map hashes
  against the original constructor on random and tied scores.
- Reject malformed maps, nonfinite scores, infeasible capacities, aliasing,
  non-exact owners, and mandatory-token violations as before.
- Retain actual allocator snapshots, continuous-live reuse proofs and native
  `FREE_COMPLETED` receipts. Preserve every prefill/construction/decode phase.
- Both paired arms use the same charged reader workspace and 128 predictions.
- Require exact tensor-backing and allocator-block initial, final, and peak
  bytes; no accounting tolerance or numerical threshold is relaxed.
- Run the complete CPU suite and native operator control before model workers.

The existing finite runner `evicastkv_bounded_reader_v28.py` now registers the
explicit V31 policy. Use a new output root. Workers remain one A100/12 CPUs,
15 minutes each, with at most six aggregate A100s and an `afterany` analyzer.
No paper campaign or fresh cohort is enabled by this correction.
