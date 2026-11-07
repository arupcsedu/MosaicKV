# Prompt 5 — Optimize only measured bottlenecks and preserve semantics

Apply 00_MASTER.md. Require Prompt 4's actual hotspot report. Preserve a runnable reference implementation and its configuration.

Choose at most two highest-cost, tractable implementation bottlenecks. Possible changes include tensorized descriptors, batched gathers, segmented operations, removing redundant transfers, caching request-invariant metadata, or reducing synchronization. Implement only changes justified by the profiler. Do not build a new serving engine or change attention backend, dtype, objective, routing, budgets, candidate domain, or stopping policy in this stage.

Maintain the existing physical allocation policy, mandatory entries, per-head minima, layout padding, rounding and filling rules. Never use unconstrained global top-K. Preserve deterministic tie-breaking; do not add epsilon perturbations to disguise changed rankings. If vectorized reduction order changes selections, either recover the reference selections or register the implementation as a distinct algorithmic variant requiring quality evaluation.

On identical saved DEVELOPMENT inputs/observations, compare reference and optimized selections, routed heads, candidate membership, allocation quotas, retained K/V, original positions, validity masks, target/actual bytes and output traces. Saved inputs may be reused for this diagnostic; full performance requests must be recomputed. Re-run structural and numerical gates after each change. Changes to common packing or score collection must benefit Top-K and other compatible methods too, without adding unnecessary graph work to them.

Measure the paired factorial: reference Top-K, optimized Top-K, reference EviCastKV at 1/256, optimized EviCastKV at 1/256. Include FullKV as an uncompressed timing reference. Use the same 16 timing inputs per workload, declared budgets, five warmups and three repeats, all timed unprofiled. Compare selected IDs and outputs before calling this semantics-preserving. Report regression or no gain as valid outcomes.

Release source-cache/query-hook/debug references at the same documented lifecycle boundary across comparable configurations. Do not make memory look lower by excluding live allocations or latency look lower by moving computation outside the timer.

Deliver optimization_diff.patch, equivalence_report.json, before_after_runtime.csv, raw repeated timing files, regression_tests/, optimization_report.md and GATE_optimization.json. Classify each change as semantics-preserving, corrected bug, or algorithm change. Mark historical results affected by a discovered bug; do not compare their invalid numbers with new measurements. A speedup in common infrastructure is not evidence of a graph-selection quality benefit.
