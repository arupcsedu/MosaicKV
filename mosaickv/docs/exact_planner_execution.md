# Exact planner execution: zero-overlap bound and local targets

The consumed-input profile attributes most spatial-graph Python work to
54,180 rectangle-distance calls. The zero-overlap implementation retains the
same spatial grid and compatibility rules. When at least k compatible rectangles
overlap or touch a source rectangle, those zero-distance edges have the exact
maximum score 1.0; the k smallest target IDs are therefore the exact winners.
Only these cases short-circuit. All other cases use the original scalar distance
and score calculations. No approximate distance, changed tie break, all-pairs
matrix, removed edge source, or altered graph objective is introduced.

Random mixed-layer/head/modality/container regressions compare complete spatial
COO tuples. A 603-node synthetic patch-grid regression compares all graph edge
types, scores, order, and diagnostics while requiring over ten times fewer
scalar distance evaluations. Synthetic timing is not paper evidence.

`reuse_precision_targets` computes the two unchanged original-attention targets
once inside a single precision transaction, reusing them for early/late/future
proxy metrics and prototype checks. It also reuses the identical baseline metric
dictionary rather than recomputing it. CPU tests compare every metric, decision,
weight, selection and replacement tensor bitwise. Nothing is cached between
requests, and no benchmark labels are accepted.

Historical timing mode also selected a matched-group anchor filter. New runs
make this choice explicit with `prototype_anchor_scope="selected"`, so profiling
cannot change their plan. `legacy` preserves old artifact interpretation and
`matched_groups` explicitly selects that historical component-study convention.
The new scope is identical to the frozen unprofiled full-hybrid convention.

The bounded worker profiles cold and warmed execution separately, outside timed
trials. It checks full model logits against immutable references before and on
every trial. A common `cpu_torch_threads` setting is registered for all arms,
distinct from the twelve CPUs allocated to each worker. This permits testing
one compute thread for small CPU attention matrices without granting a different
CPU policy to the hybrid. Native numerical checks are not relaxed if changing
the thread count changes any frozen output.

Both original and optimized-codec storage-only controls remain required. Shared
source coding is not credited as a benefit of the six adaptive components. These
two inputs remain consumed engineering diagnostics; no quality, freshness,
whole-request KV-peak or all-baseline superiority claim follows automatically.
