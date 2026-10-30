# Lossless row-predictor feasibility

This bounded follow-up preserves the two consumed identities and all frozen
selections. It captures only the actual encoded prompt bodies: normalized
source tensors, direct K/V, and LOOK-M's actual merged K/V. No weights,
annotations, future benchmark outputs, or optimizer state are exported. All
six primary methods receive exactly the same candidates.

The registered row predictors are no prediction, XOR against
the preceding physical tensor row with independent 16-row anchors, and a
reversible ordered-BF16 integer delta/zigzag transform with the same anchors.
Both transforms are bijections over all 65,536 BF16 bit patterns; no floating-point
subtraction, approximation, quantization or token eviction occurs. Every
CPU packet is round-tripped against its original integer bits. A four-byte
predictor/codec tag is charged. Ties keep the existing packet. Each predictor
is offered both the existing exponent codec and a lossless sparse-high-byte
packet (eight low bits, a high-byte presence bitmap, block offsets and literal
nonzero high bytes). This exposes narrow integer residuals instead of forcing
every value through a fixed eleven-bit minimum. These are six fixed candidates,
not an answer-guided hyperparameter search.

The CPU study chooses minimum serialized bytes per tensor, never answer
quality or encoding time. It verifies CPU/GPU baseline codec size equality.
Prospective installed storage substitutes measured CPU packet bytes into the
previously measured prompt ownership boundary. This is explicitly **not** an
installed GPU KV-memory or latency result: GPU decoding, workspace, and model
parity remain unverified for the new formats.

Two one-A100 captures feed two four-CPU, 30-minute codec workers, followed by
afterany completeness/hash aggregation. No GPU decoder follows automatically.
Only a >=3% prospective EviCastKV reduction on both consumed inputs justifies
separately versioned decoder work. A negative result stops this direction.
No fresh cohort, calibration, or paper campaign is authorized.

This continues the repository's lossless/source-storage study. Generic
reconstruction and lossless coding are established ideas; these simple
predictors are engineering candidates, not a novelty or superiority claim.

The first tensor captures are reusable. Their selection metadata contains the
repository's BudgetUnit enum. A hash-verified, weights-only load may narrowly
allowlist that class inside a temporary safe_globals context; unrestricted
pickle loading remains forbidden. Future exports normalize selection metadata
to JSON primitives. A new CPU-only retry plan binds both the old capture SHA
and the repaired analysis SHA, never copies or regenerates the GPU captures,
and explicitly rejects GPU submission under the retry root.
