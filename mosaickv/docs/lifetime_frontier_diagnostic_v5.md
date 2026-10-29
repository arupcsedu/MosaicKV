# Prompt-protected history-capacity frontier: consumed diagnostic V5

This is a causal capacity study, not a new selection-policy superiority claim.
It may run only after V4 completes cleanly without a qualifying point candidate.
It reuses exactly the same two consumed inputs; independent confirmation and
paper-scale execution remain disabled. The original campaigns are untouched.

## Why another diagnostic is informative

The preceding controls separated prompt loss from generated-history loss. A
128-position history evicts substantial context during long PPL targets and
forced 1,024-token generation. Redistributing those same positions cannot
establish whether more history would fix the remaining PPL deficit.

V5 keeps prompt selection constant while increasing history capacity. This
does not conceal a budget increase: **both Top-K and EviCastKV run at every
registered capacity**, with identical generation, mandatory policy, pinned
profile, inputs, eager attention, BF16 weights and exact original K/V.

Register mean/uniform history lengths 128/256/384/512. On the pinned 32-layer
LLaVA geometry, the corresponding complete resident-plus-growth caps are
223/289/355/421 MiB. For per-layer position costs `c_l`, use

`B(G) = 223 MiB + (G - 128) * (sum_l c_l + max_l c_l)`.

Only equal layer costs are supported here. Subtracting history and growth
charges yields the same prompt allocation for every G. The analyzer requires
identical selected prompt-slot hashes across these four capacities within each
method, as well as identical measured bytes between methods at each capacity.
The geometry-only calculation predicts approximately 148.8 MiB initial prompt
storage; this is **planning, not a measured result**. The actual runtime repeats
allocation using mandatory minima and rejects disagreement with the plan.

This tests smaller, attainable lifetime savings rather than silently retaining
an unachievable 30% claim. Native PrefixKV is a separate operating-point control;
it is never called byte matched. The candidate must still be non-worse in PPL
and ROUGE-L separately, against both native PrefixKV and its same-budget Top-K
counterpart, while reducing initial, final and common decode-transaction peaks.
All capacity points, including negative ones, remain in the report.

## Execution and memory boundaries

Sixteen independent method/task/history shards; one A100 and 12 CPUs each;
maximum six concurrent; 30-minute limit, 16 GiB host RAM. Retention-one and
generation/reinjection controls run per worker. Hash-bound compatible FullKV
outputs and V3 native controls are reused, not regenerated. Timing is one-trial
descriptive and native-control timing is unpaired.

Detailed ownership uses a separate replay and must preserve all tokens. Live
initial/final bytes, active decode transactions including working buffers,
registered global KV owners, and whole-request CUDA allocator peak are separate
quantities. A generic whole-request KV/internal-transient certificate is still
not claimed.

V5 also clarifies a trace-labeling issue: `budget_reservation` now describes the
actual layer-profile prompt/history/growth allocation. The old uniform estimate
is retained under `uniform_anchor_metadata_reservation`, rather than presenting
the metadata-only anchor budget as actual physical allocation. Numerical
selection, construction and native PrefixKV behavior are unchanged by this fix.

N=1 per task is a diagnostic, not statistical non-inferiority. No result here
automatically authorizes calibration, fresh inputs or a paper campaign.
