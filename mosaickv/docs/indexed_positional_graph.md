# Exact indexed positional search

Measured construction traces attributed about 0.25--0.26 seconds to the
603-node routed graph. A separately labeled synthetic CPU shape profile found
positional fallback's full-target scan/sort and repeated Enum sorting in COO
finalization among the dominant operations. This is execution optimization,
not a new evidence policy, changed budget, graph thinning or quality result.

For unrestricted same-layer/head compatibility, centers are already sorted.
Use binary search to locate equal-center groups, then consume nearest groups
on either side until top-d is full or the configured span is exceeded. Consume
indices inside a group in ascending order: a naive left-pointer walk would
break the original tie rule for duplicate centers. Keep the reference path
available, and leave restricted compatibility behavior unchanged. Hoist the
immutable sorted EdgeType table out of the per-edge finalization loop.

Randomized CPU regressions compare every chosen neighbor against the original
all-pair enumeration, including duplicate centers, span boundaries and ties.
The full 603-node graph must retain every row, column, type, floating-point
weight and diagnostic exactly. The pretrained GPU gate then requires identical
actual COO hashes and bitwise full-128 logits for both graph-only and combined
inline-repair execution. Physical storage, selected rows, prototypes, repair
policy and numerical guards cannot change. Reference is the default until
explicitly selecting the indexed execution policy in a frozen experiment.

Only after that gate passes may a five-warmup/twenty-paired-trial benchmark
compare original execution with the combined indexed+inline optimization.
That benchmark establishes an engineering speed change, not superiority over
all baselines or independent quality evidence. Larger campaigns remain blocked.
