# Uniform split eager attention: consumed-input request screen

The universal operator failed on adaptive layouts. The separately registered
uniform-only operator passed its word/owner tests and cold replay screen. This
request experiment tests that narrower optimization; it does not relabel the
universal failure. The previous fused correction predicate remains disabled.

Run the same two consumed Description/MM-Vet inputs, original checkpoint,
BF16/eager, 128 predictions, prompt/source/history policy, mandatory positions,
constructor repairs, packet signatures and 128 parent vocabulary word checks.
Compare async Source with and without split replay, plus both optimized
PrefixKV controls. Offer the feature to all methods: nonuniform adaptive
layers retain the original eager path and allocate no replay workspace.

The replay runs only QK, scaling, original mask addition, native FP32 softmax,
BF16 conversion and PV. Between QK and PV the existing materializer overwrites
the same charged K/V lease. No second cache, changed reducer, SDPA, weight
change, query forecast, new selection or benchmark-label access is introduced.
The exact BF16-to-FP32 conversion has an explicit owned buffer to avoid a
hidden per-layout cast allocation. CUDA controls must prove identical words,
padding and owner retirement for actual A-B-A lengths and a shorter uniform
baseline BEFORE the first comparison request. Private graph pool reservation
must be zero. Do not reset the enclosing request's allocator peaks.

Each request owns at most 384 KiB explicit attention workspace. Charge it in
cache shared ownership and decode-owned peak, retire it before the existing
lease, retain initial/final bytes, and expose library workspace separately.
Audit every capture for unowned allocations; timing requests use the same
code without allocator-snapshot instrumentation. Count cold graph capture,
warming, descriptor creation, copies and retirement inside request latency.

Run the complete same-SHA CPU suite, then two one-A100/12-CPU workers with five
warmups and three paired randomized timed repetitions. Terminal afterany
analysis checks correctness even on producer failure. The 4% mean-ratio point
screen is not a statistically adjusted latency pass. No independent quality
examples are added; these tests cannot establish task superiority, whole-
request GPU peak, algorithmic novelty or paper readiness. Fresh evaluation
and larger campaigns remain blocked pending independent quality evidence.
