# Lossless construction bookkeeping diagnostic (November 4, 2026)

This is a consumed-input execution ablation, not new benchmark-quality evidence.
The deferred-validation run from `debbf82` passed all packet, parent-logit and
storage controls but failed the strict request-latency gate. Preserve its raw
records. No fresh cohort, calibration, or paper-scale submission is authorized.

Three measured construction operations are addressed without changing cache
selection, encoded words, repair tolerances, or reader arithmetic:

1. Deduplicate immutable, identity-shared head-position validation and lookup
   within one `text_slots` call. Distinct/mutable rows retain full validation.
   There is no cross-call cache or request-state sharing.
2. Construct the ordered retained overlay indices directly from the existing
   CPU filter. Preserve the first-row nonempty fallback, exact bodies and order;
   remove only the redundant GPU Boolean reduction and dynamic indexing.
3. Opt in to one correction-index extraction after identical chunked FP32
   `isclose` predicates. A temporary Boolean mask is bounded at 4 MiB. Above
   that bound, retain and explicitly count the original chunked path. No mask
   persists into the returned packet or decode. Report actual mask bytes,
   validation flags and torch allocated/reserved maxima separately. These
   figures do not certify whole-request GPU peak memory.

The common operations apply identically to all five local method adaptations.
All six main algorithm mechanisms remain disabled in this storage experiment.
The frozen source-precision parents, 128 predictions, logical positions,
mandatory membership and exact initial/final/decode-owned peak storage are
unchanged. Matching frozen packet hashes and all vocabulary words is required.

## Finite execution chain

Clean immutable source -> full CPU suite -> two one-A100 controls (same two
consumed inputs) -> paired request timing -> strict dependent CPU analysis.
The controls include real CUDA patch-word equivalence, negative predicates,
and randomized correction-operator trials (five warmups, twenty repetitions).
Operator speed is diagnostic only; it cannot substitute for request timing.

Request timing retains five warmups, twenty randomized paired repetitions and
the registered bootstrap family correction. Require the upper adjusted latency
ratio <=1.00 against **every** measured baseline control, including its fastest
reader. <=1.05 remains a reported trade-off, not a pass. Only a strict paired
pass may launch the already bounded all-five-method extension. Any failure
produces a preserved write-once negative decision. No automated fresh evaluation.

Independent benchmark-quality N remains zero. This experiment cannot establish
superiority of the full EviCastKV design, official baseline performance, or
smaller storage at independently validated equal answer quality.
