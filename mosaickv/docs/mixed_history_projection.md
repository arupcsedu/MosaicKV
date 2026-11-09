# Single-projection history execution: bounded correctness then timing

The completed five-warmup/twenty-repeat study
`header-history-timing-20261108-v1` measured the 64-exact-recent history variant
at 1.307894 / 1.309677 times the storage-only comparator's end-to-end latency
and 1.365248 / 1.367412 times its decode latency on the two consumed inputs.
Both received identical common header/predicate optimizations. Preserve this
negative timing result; no independent quality or complete memory claim follows.

Source inspection identifies repeated work: four recent exact fragments plus
one approximate older fragment each run a separate reconstruction/projection.
The execution-only successor retains those SAME source archives for aging but
reconstructs them into the existing request-private BF16 workspace and issues
one projection per component for the complete history. It allocates no second
persistent source copy. K/V are checked against the original unfused reader,
not merely against a relaxed model-logit tolerance.

Native GEMM row-count changes can change floating-point words. Build and charge
exact sparse corrections against the complete unfused K/V, including older
history, under the unchanged 32-KiB per-layer correction cap. Preserve all three
original verification passes, strided destinations, scratch sentinels, phases,
mandatory tokens, all logical positions, and request-owner checks. Reuse one
shared phase pair per epoch, retire obsolete owners only when no active packet
uses them, and include all old/new construction overlap in the allocator ledger.
Default runtime behavior remains unchanged outside this explicit scope.

CPU tests cover forced shape-dependent errors and repairs, one projection per
component, strided sentinels, source-owner identity, wrong-owner/phase rejection,
repeated aging with 64 exact source tokens, and actual backing accounting.
Checkpoint-free CUDA controls use known identity/diagonal projections and real
native kernels. Only after they pass may the same consumed pretrained inputs
run unfused/fused controls. Require bitwise equality of all 128 logit vectors
with the unfused policy, the unchanged FullKV-parent numerical guard, 192 actual
layer/epoch constructions, identical initial storage and no final expansion.

If that bounded gate passes, exactly one conditional randomized paired timing
study is allowed: five warmups and twenty repetitions per arm/input, no allocator
instrumentation in timed requests, complete raw rows and provenance, and strict
equal-or-faster ratio upper bound <=1.00. It cannot override the previous
initial/peak memory gate or launch a fresh cohort/paper campaign. Every producer
has an `afterany` reviewer; aggregate admission stays at six A100s.

## Separate LOOK-M accounting retry

The common-header checks passed all 128-logit comparisons for Prompt Top-K,
PrefixKV and VL-Cache on both consumed inputs. LOOK-M exceeded the original
500,000-entry allocator trace capacity, so those four rows are invalid, not
evidence of a numerical failure. Retry only LOOK-M with 8,000,000 trace entries;
keep the overwrite rejection and every numerical/accounting check enabled.
Preserve/reuse the six valid non-LOOK-M pairs. These remain local precision
variants and within-method transport tests, not official reproductions or
cross-method byte-matched paper comparisons.
