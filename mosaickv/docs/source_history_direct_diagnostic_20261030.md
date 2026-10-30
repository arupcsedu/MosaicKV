# Direct source-history constructor differential

The preceding 128-token source-history canary passed exact logits and reduced
final and observed decode-owned peak storage. It retained the source tail's
additional storage. This successor changes only construction work, not the
codec, selected positions, source tail, RoPE phases, original bits or budget.

The reference builder creates and validates a dual-projection intermediate,
then builds the encoded operator against that intermediate. Counting its
original-bit comparisons, this performs nine projection passes per sealed
layer. The direct builder creates the final encoded operator against the actual
cached K/V: one uncorrected projection builds exact sparse bit corrections,
followed by the same three independent final checks. This needs four passes.
No final checks are removed. Source maps, packets and correction values should
be identical; GPU verification must prove bytes and model outputs rather than
assume equality from the arithmetic argument.

A separately versioned consumed-input canary requires bitwise 128-token logits,
identical initial/final physical bytes, and no increase in observed decode-owned
peak. Only if it passes, apply the identical differential to all six paths and
run five warmups plus twenty randomized paired decode trials per method/input.
FullKV remains excluded from latency comparison. The two-stage chain uses
afterany analysis, at most six A100s total, twelve CPUs each and thirty-minute
worker limits. Existing active runs and source snapshots remain untouched.

Raw `ordinary_history` / `ordinary` fields are legacy comparator labels here:
the immutable plan explicitly identifies that comparator as the **reference
source-history constructor**, not ordinary K/V history. `source_history` / `source`
identify the direct constructor. Do not confuse this constructor differential
with the earlier K/V-versus-source-history study or add medians across jobs.

This does not establish independent quality or general superiority. Whole-request
peak remains uncertified; no fresh cohort or paper campaign is launched.
