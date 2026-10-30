# Compact original projection rows: fail-closed numerical experiment

V5 profile array 22186937 and analysis 22186938 completed. For EviCastKV's
two consumed inputs, original-shape K/V GEMMs accounted for about 54% of
device time nested inside the CPU source-reconstruction range. Do not sum
CPU-correlated ATen events, GPU kernel events and inclusive annotations:
their activity overlaps. V4 remains the actual repeated latency evidence.

Candidate: decode only the archived normalized rows, run the original frozen
K/V projections at that compact row count, apply the original RoPE at those
rows' original positions, and remap per-head gathers through the union map.
This removes the full zero canvas/scatter and avoids projecting unused rows.
It changes no selected position, source value, mandatory token, logical
position, precision, attention math, model weight or generated-history rule.

Changing GEMM shape can change BF16 rounding. Therefore first test every
selected K/V bit against original post-RoPE prefill K/V at all 32 layers, for
BOTH selectors and BOTH consumed inputs. Save partial proofs on failure.
Any mismatch stops the compact experiment; do not relax the tolerance or
continue into timing. The full-shape path remains the default.

If all static proofs pass, require the existing retention-one and own-greedy
32-token/logit controls. Count source/metadata/history/reconstruction backing
with the same workspace audit. Include new compact RoPE and remapping
temporaries. No unaudited memory or whole-request peak claim is permitted.

Only a completed V6 proof authorizes V7 repeated paired timing, from the same
immutable source, with five warmups/twenty blocks and all three formats for
both selectors. V7 keeps the same inputs, seed, boundaries and reference as
V4. A speedup is measured, not promised. Neither step changes the negative
union result or authorizes fresh quality data, calibration or paper campaigns.

Two one-A100/12-CPU workers, ten-minute correctness limits (twenty for timing),
finite CPU afterany analysis, maximum six A100s total, scratch-only caches.
