# Bounded common epilogue-to-normalization input reuse

The common epilogue already returns the same fixed BF16 output buffer to each
next decoder layer. The input norm unnecessarily copies this exact buffer to
another BF16 buffer before its original float32 conversion. Read the existing
epilogue output directly for layers 1--31. Only layer 0 must copy the new token
embedding to that same buffer. Retain the original operators and all rounding.

Use the original norms for the first token, then require exactly 3,906
later-layer copy bypasses and 126 embedding copies. Fail closed if a later
hidden pointer is not exactly its preceding epilogue output, if an embedding
unexpectedly aliases that output, or if the live same-request owner is missing.
Destroy norm graphs before their epilogue source owner. Preserve all 128 logit
vectors, packet signatures, original K/V words, histories and positions.

Retain and charge the original now-unused 8 KiB norm input buffer as well: this
is a controlled execution-only ablation, not an invented memory saving. Apply
identically to Source and both Prefix readers. Compare against the predecessor
norm policy only if it improved every own mean on both inputs; otherwise the
old arm uses the original norm, while the new arm tests norm+input reuse.
No changes to fresh identities, quality scope, repair thresholds, outputs,
metadata checks, whole-request peak certification or statistical acceptance.
Five warmups/three paired timings are an engineering screen, not formal
latency evidence. Independent quality N=0; no larger campaign authorized.
