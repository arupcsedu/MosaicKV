# Reusing the common last-layer suffix in the guarded hybrid

The fixed storage-only control has no counterfactual MLP. The hybrid must pay for
its guard; it cannot be compared only with an older, slower hybrid. Prior paired
trials attribute about 0.20 seconds to additional construction and about 0.29
seconds to decode-side work. These are consumed-input engineering observations,
not independent performance claims.

The default-off `OverlayPolicy(anchor_suffix="shared_graph")` reuses the existing
verified final-layer epilogue graph and its charged workspace. It saves the active
hidden vector before replaying the parent counterfactual. Graph replay does not
advance ordinary layer ordering, alter history, rewrite any K/V, change weights,
or allocate a new persistent graph/workspace. Its extra temporary saved hidden
activation is reported separately. Ordinary and guard graph replay counts are
both exported.

The graph is the same exact native arithmetic already used by every method. The
guard keeps its registered finite-value check, greedy-token requirement, KL limit
0.02, and maximum-logit-difference limit 1.0. Early/middle/late model controls
compare the borrowed parent hidden output against the eager native calculation
bitwise. Every 128-logit vector must also match the frozen hybrid reference.
Unsupported geometry and missing or misordered graphs fail closed.

Run `source_hybrid_latency.py prepare --bitmap-prefix --shared-suffix --root
NEW_PATH` from a clean source. Each of two one-A100/12-CPU workers first profiles
one consumed input, recording actual planner phase times and Python cumulative
time separately from benchmark trials. Only after bitwise checks pass does it
execute six warmups and twenty-one balanced timed repetitions for storage-only,
storage-only with the common optimized codec, and the new hybrid. An afterany CPU
analysis retains failures, measured storage and multiplicity-adjusted paired
latency intervals. Maximum aggregate admission remains six A100s. No labels, new
cohorts or larger campaign are authorized by this engineering protocol.

This guard preserves the storage-parent answer; it does not establish improved
task quality. The all-head lossless source codec is also supplied to the storage
control. Its memory reduction is not attributed to six adaptive components.
