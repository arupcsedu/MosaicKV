# V3: pinned-profile lifetime diagnostic (not a paper-ready method)

EviCastKV and MosaicKV denote the same system. This is a separately labeled
adaptation, not official PrefixKV and not a change to any frozen campaign.

## Hypothesis and timing of the design

V2 tests more prompt capacity, explicit question preservation, fixed
generation/PPL prompt reservation and cheaper history bookkeeping. While its
GPU workers run, V3 is defined using only the pinned upstream layer-profile
shape and cache geometry, not their answers, references, scores or failures.
The profile assigns substantially different capacities to layers; uniform
allocation can discard evidence in sensitive layers while overspending on
less sensitive ones. This is a hypothesis, not proof that V3 improves quality.

The profile is the numeric list in pinned PrefixKV commit
`597f1ab032704951550f93bcc8a23f1454b80aa4`,
`confs/prefixkv_llava-v1.5-7b_0.5.json`. Interpret each entry as the same
forget ratio as the audited native loader, retaining weight `1 - forget`.
Its calibration identities are not included in that upstream list; therefore
independent-calibration provenance is **not certified**. This diagnostic must
not become a paper-faithful PrefixKV reproduction or fresh-confirmation claim.

## Feasible allocation and exact construction

Use deterministic weighted water filling of prompt positions with mandatory
minima and source-length maxima. Every layer uses one common length for all
its KV heads, avoiding padding. With prompt lengths `p_l`, history capacity
`G` and per-layer K+V position cost `c_l`, enforce

`sum_l c_l (p_l + G) + max_l c_l (p_l + G) <= B`.

The maximum-layer term reserves a conservative old/new growth-copy scratch
allocation. Report the remaining capacity explicitly. The heap algorithm is
proportionally fair and maximal under a single positive increment; it is not
claimed globally optimal for arbitrary heterogeneous-cost knapsacks.

Keep the frozen exact selector's selected blocks as an anchor. Where the
profile shrinks a route, keep mandatory positions then its highest prompt-mass
anchor positions. Where it expands a route, preserve the anchor and add omitted
positions by prompt mass. Deterministic ties use original positions. Original
K/V values and logical positions are gathered verbatim into layer-local storage.
No intermediate uniform packed cache, prototype, quantization, offload or
additional model probe is introduced. Both Top-K and EviCastKV receive the same
profile, mandatory policy, growth rule and realized byte boundary.

The uniform selector anchor may require rounding its **metadata** budget up by
one lattice step to cover the final profile. Only the final physically feasible
layer-local selection is packed. Traces label the anchor separately from the
final selected slots and record actual layer capacities and bytes.

## Conditional experiment

The CPU controller waits for V2's automatic analysis. It stops without GPU
submission if V2 has incomplete/failed execution, or if V2 already advanced a
qualifying candidate. Only a cleanly completed negative V2 permits V3.

Use the same two consumed inputs, pinned BF16 LLaVA-1.5, HF eager, seed and
forced 1,024-token output. At the 208 MiB hard transaction cap and 128-position
generated history, compare Top-K/EviCastKV under (a) the frozen mandatory
policy and (b) mapped-question/boundary protection. Eight method/input/policy
workers plus two remeasured native controls use one A100 and 12 CPUs each,
aggregate maximum six. The finite CPU
afterany analyzer validates every attempted sample, hashes, actual byte matching,
NLL/token records, PPL and reference ROUGE-L separately.

Reuse V2's immutable FullKV reference outputs only after checking checkpoint,
revision, generation parameters and sample identity exactly. New 16-token
generate, reinjection and retention-1 parity checks still run per worker. Do
not rerun those compatible FullKV predictions merely for scoring. The reused
native PrefixKV rows remain separate operating points. Native controls are
remeasured with the new ownership hooks and must reproduce their previous
answers, NLL and token counts exactly; their independent one-trial timing is
not a paired microbenchmark. No cross-budget comparison is called byte-matched.

Each method/input/policy is an independent shard, with a 30-minute limit and
16 GiB host RAM. The preceding 12-arm workers measured approximately 2.5 GiB
maximum RSS, leaving substantial headroom without reserving unused RAM.

The peak gate uses the new `audited_decode_transaction_peak_bytes` in both
methods, including preallocation, concatenation, eviction, growth copies and
incoming one-token K/V. Historical phase-sampled peaks remain separately
recorded and are not used for this peak gate. Completed transaction owners are
removed from the active ledger but remain in global weak ownership if unused
snapshots really stay alive. This prevents inactive storage from being hidden
or misclassified as active decode KV.

Detailed memory observation has a separate replay and must preserve all primary
tokens. Its runtime is not the method's latency. Explicit weak-owner hooks also
cover clone snapshot construction, dense/layer-local packing, append-only
preallocation, PrefixKV source/destination eviction buffers, and incremental
growth copies. Unit fixtures require simultaneous old/new storage accounting
without extending source lifetime. Report registered KV ownership
and whole-request PyTorch CUDA-allocator peaks as different quantities, without
certifying unobserved internal KV transients or whole-process memory. Record
the first EOS position and forced post-EOS token count: savings after an answer
ends cannot establish long-answer quality preservation. Keep the historical
forced-length workload unchanged rather than silently changing stopping rules.

## Gates and stopping

Every matched-policy pair must have equal measured initial/final/peak KV.
The candidate must reduce all three boundaries versus the native control and
be non-worse in both PPL and ROUGE-L against native PrefixKV and matched Top-K.
N=1 per task gives point diagnostics only, not statistical non-inferiority.
No fresh cohort, independent-calibration certificate, superiority claim or
paper campaign is authorized automatically. A negative result is preserved;
successful execution alone never opens larger evaluation.
