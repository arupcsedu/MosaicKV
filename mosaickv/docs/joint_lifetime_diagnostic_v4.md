# V4: joint prompt/history layer allocation, consumed-input diagnostic

EviCastKV and MosaicKV are the same system. This is a separately labeled
adaptation, not official PrefixKV. Frozen campaigns and their results remain
unchanged. V2 and V3 completed without execution failures but did not pass
the combined PPL/ROUGE-L point gate. Do not promote either to calibration.

## Evidence and hypothesis

The V2 same-prompt controls generated identical first 128 tokens before history
eviction. The 97-token Description teacher-forcing target did not evict history;
the 593-token MM-Vet target did. Its PPL was 7.7404 append-only, 7.9297 FIFO,
and 7.8058 with importance retention. Thus prompt compression and generated
history compression make distinct contributions. These are inspected, consumed
inputs, not independent confirmation or permission to tune from answer labels.

V4 tests a geometry-only hypothesis: uniform history capacity may discard
important history in sensitive layers while retaining unnecessary history in
other layers. Use the same pinned numeric PrefixKV profile as V3, with weight
`w_l = 1 - forget_ratio_l`. Allocate a fixed total of `128 * number_of_layers`
generated positions using deterministic weighted water filling, with at least
69 positions per layer (four sink, 64 recent, one replaceable). No references,
answer scores, target lengths, draft probes or learned weights enter allocation.
The upstream profile does not certify its calibration identities; this remains
development-only evidence.

After allocating history `g_l`, allocate prompt lengths `p_l` under the same
208 MiB hard cap, enforcing

`sum_l c_l * (p_l + g_l) + max_l c_l * (p_l + g_l) <= B`.

Report the lattice shortfall, actual per-layer prompt/history lengths, growth
scratch and resident bytes. Equal geometry preserves the total history bytes;
unequal geometry uses measured costs in the cap. Prompt anchor resizing, exact
K/V preservation, layer-local physical packing, mandatory policy and original
logical positions are unchanged. There is no ragged-head padding, quantization,
prototype, residual store or repair. The V3 uniform mode is the default and its
config hashes remain unchanged.

The independent-layer decode validator checks each layer's update count and
actual length. The global maximum length may plateau before growing again;
it is not incorrectly assumed to increase once per token until saturation.
The complete fixed reservation is identical for generation and teacher forcing,
even when a short target only needs a smaller actual history allocation.

## Bounded experiment and controls

Use only the same two consumed Description/MM-Vet inputs, BF16 pinned LLaVA,
eager attention, forced 1,024-token generation and frozen FullKV outputs.
Compare both Top-K and EviCastKV under both frozen mandatory policies, all
with the identical V4 allocator and cap. Eight one-arm shards, one A100 and
12 CPUs each, maximum six concurrently; 30-minute limit and 16 GiB host RAM.

Reuse V3 native PrefixKV controls, including their common decode-transaction
memory replay, only after hash validation. This avoids remeasuring an unchanged
native path. Native and lifetime-controlled methods are different operating
points, never called byte-matched. Native timing is descriptive and unpaired.
Each new worker still runs ordinary generation/reinjection parity plus a shared
retention-one exact-transport control. The shared control bypasses the expensive
Top-K identity selector startup; it is not a method-specific Top-K selection
parity claim and its startup time is outside measured method latency.

Detailed ownership replay must reproduce all tokens. Initial/final physical KV,
active decode transactions including working copies, global registered owners,
and whole-request CUDA allocator peak are separately labeled. Generic internal
KV transient coverage and whole-request KV peak remain uncertified; an observed
decode-transaction bound is not silently renamed a whole-request bound.

The finite afterany CPU analyzer requires complete rows, exact matched-policy
initial/final/decode-transaction bytes, smaller three-boundary storage than
the native operating point, and non-worse PPL and ROUGE-L against native and
matched Top-K. N=1 per task gives point diagnostics, not statistical evidence.
No automatic calibration, fresh cohort or paper campaign follows this run.
