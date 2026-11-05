# Corrected 192-row Source execution, same consumed input screen

This successor explicitly differs from the sealed negative zero-correction
operator. Retain the same selected/encoded evidence, original weights,
positions, original K/V words and all 128 vocabulary-logit vectors. Refit and
store actual original-word corrections for the new GEMM geometry; enforce the
unchanged 32 KiB per-layer and 1 MiB per-request caps. This is lossless numerical
transport, not uncertainty-driven residual repair or prototype merging.

The original 128-row path remains default. The new 192-row engine allocates
only its two 192x4096 BF16 buffers plus the original incoming vector, never an
old pair followed by a larger pair. Copy original full/tail/joint algorithm
descriptors without a timing search. Use four chunks rather than five at the
two registered source lengths, 600 and 604. Fitting and exhaustive stride
validation use that same engine in the existing bounded background pipeline.
No new prefill, quantization, graph, selection or generated-history change.

The new workspace costs exactly 1 MiB more. Native numerical patch bytes may
change and are counted, not hidden: initial/final deltas must equal the actual
patch delta, and the decode-owned peak delta must equal that plus 1 MiB. Gate
smaller initial/final/observed decode-owned peak against every measured Prefix
reader using actual storage. Do not claim whole-request or vendor workspace
peak certification. The larger temporary fit workspace remains audited.

Only the corrected exact operator result for both geometries can release the
runtime screen. Preserve the failed zero-correction operator and all negative
history results. Native history remains OFF. Freeze the common input-reuse
choice only if every own mean improved on both inputs, otherwise retain the
previous common norm policy. Apply every eligible optimization to baselines
too; report ineligible native Source formats as no-ops.

Four arms: old/new Source, optimized/original Prefix readers. Five warmups and
three randomized paired complete-request trials, two consumed inputs, one
A100/12 CPUs per worker; finite afterany analysis even on failure. Require all
word/stride/retirement/position/packet-signature controls before timing. The
4% point screen is not formal latency evidence and authorizes no fresh or
larger quality campaign. This retains the explicitly labeled shared-source
precision execution variant with six primary mechanisms disabled; independent
benchmark-quality N=0. It does not establish complete EviCastKV superiority.
