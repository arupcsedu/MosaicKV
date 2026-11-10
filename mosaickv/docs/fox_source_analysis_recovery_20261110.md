# V7 saved-logit analysis recovery

The V7 source workers completed all fourteen registered rows at source
`de339c2a66ed9804a1348ae0767f27588074983a`. The original analyzer stopped when
it compared a JSON digest of recomputed floating-point KL observations against
the producer's digest. That comparison incorrectly required CPU reductions to
be bitwise identical across different instruction sets and thread layouts.
Keep that failed analysis, its blocked bridge-admission record and all raw
predictions/logits unchanged.

The correction is **only analysis verification**, not a new model, codec,
allocation, scoring rule, numerical guard, or benchmark run. Use a new derived
directory and explicitly bind the clean producer and analyzer SHAs separately.
The saved producer artifacts remain authoritative and retain their hashes.

## Verification contract

Use AVX2 and twelve CPU threads for recomputation, matching the producer's
CPU-reduction path. Booleans, token IDs, tensor shapes, greedy agreement,
maximum logit differences and registered pass/fail decisions must match
exactly. Only recomputed KL scalars/arrays receive a machine-precision-based
roundoff check: eight FP32 epsilons or sixty-four FP64 epsilons, scaled by
`max(1, abs(recorded), abs(recomputed))`. Both observations must be finite.
Record observed drift and the exact verification policy.

This does **not** increase the model's `KL <= 0.02` or logit-difference `<= 1.0`
threshold, or permit token changes. Reject any discrepancy in the registered
decision even if its reported KL values happen to satisfy the arithmetic
verification tolerance. The separate Mistral FullKV tolerance stays `1e-4`.

## Finite execution

Run the complete CPU suite from the corrective clean SHA. Recover only the
saved source analysis; there is no source GPU retry. If execution is complete
and the native/clone FullKV controls pass, release the already prepared two
consumed common-model bridge workers using their original immutable source
and configurations. Analyze those workers after completion, including errors.
Publish a separate recovery packet; do not overwrite the original V7 packet.

Use one A100 and twelve CPUs per bridge worker, at most two workers, two-hour
limits, `bii-gpu`, account `bii_dsc_community`, and reservation `bi_fox_dgx`.
Retain the aggregate six-A100 ceiling. All cache/output paths remain on scratch.
No fresh cohort, candidate freeze, large sweep, superiority assertion, or
threshold relaxation is authorized by this recovery.

## Scientific outcome remains separate

The source-protection guards were mixed across the two inputs. Completing
analysis must not convert those negatives into passes. The local direct-3-bit
format that exceeds the shared ceiling remains an uncapped numerical
diagnostic, not an equal-byte baseline. Historical native and shape-matched
Mistral failures, incomplete freshness certification, and larger request-wide
planner peaks remain visible regardless of software recovery success.
