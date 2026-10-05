# Byte-capped unified comparison lanes

`paper_native_byte_capped_lanes_v1.yaml` defines the replacement for every
cross-method comparison based on nominal retention.  It is execution-disabled
until its materialized ledgers and immutable source commit are reviewed.

Each non-FullKV method receives the same per-example physical active-K/V byte
cap.  It may retain the largest cache it can construct at or below that cap.
A result is rejected if either method exceeds the cap or their realized active
bytes differ.  The report records the configured cap, realized bytes, byte
underfill, and the method's feasibility status; it never interpolates scores.

The VL-Cache lane uses token-sized blocks and applies the common cap while
apportioning the paper's layer-adaptive allocation.  Its nominal retention is
only an upper bound and is not reported as the comparison budget.

The PrefixKV lane preserves a pinned offline profile's relative layer-allocation
shape while re-apportioning that shape to the common byte cap.  This is labeled
`prefixkv_reimpl` with `byte_budget_profile_rescaling=true`; it is not claimed
to be an official PrefixKV run or a published operating point.

FullKV is included only as a reference and is not expected to meet compressed
caps.  Quality uses the task-native metric separately for every task.  Latency
is reported only after dedicated matched-cap trials with the same warmups,
repetitions, GPU controls, and method order counterbalancing.
