# Bounded CPU prototype/residual source feasibility

This is a separately versioned CPU screen after the negative pure rank-96/128
screen (`source-rank-feasibility-20261107-v1`). It does not activate a new cache
backend, update the validated storage variant, access labels, or authorize a
fresh cohort. The same two consumed, hash-bound BF16 source captures are used.

The earlier pure-factor payload lost high-rank source information and failed in
all layers after basis quantization. This screen retains residual information.
Freeze three families before execution:

1. `lut2`: two-bit source codes with four BF16 centroids per feature and
   128-position original-time group.
2. `spatial4_binary_residual`: a four-bit source prototype for each 2x2 image
   patch group plus a one-bit residual with two BF16 centroids per time group.
3. `spatial8_binary_residual`: the same with 2x4 image groups.

The pinned LLaVA checkpoint has a 336/14 = 24x24 CLIP patch raster. Installed
`transformers/models/clip/modeling_clip.py`, `CLIPVisionEmbeddings.forward`,
flattens the patch embedding's last two dimensions before transposing to the
sequence. Require exactly 576 non-text original positions and retain the
original mappings. Pool **pre-projection normalized source inputs**, not
post-RoPE keys. No averaging of incompatible RoPE phases is proposed.

Use four fixed Lloyd updates for CPU scalar centroids, no gradients or model
weight changes. Exclude text, globally exact rows and padding from centroid
fitting. Assign using the **stored BF16** centroids and rounded source addition,
not an optimistic FP32 reconstruction. Every non-exact image row must have
source MSE no greater than the existing three-bit quantizer (+1e-12). Restore
largest scalar errors using BF16 literals plus int16 columns and int32 CSR row
pointers. Use a complete BF16 row plus an int16 map when cheaper. Charge all
actual patch arrays, maps, prototype headers and packed codes. Original exact
K/V, protected text, logical positions and model weights are unchanged.

The first CPU execution showed that choosing the smallest raw format separately
could increase net storage by disrupting the next layer's delta codec. Preserve
that negative attempt. The corrective chooser uses an exact finite dynamic
program over numerically feasible formats, predecessor-dependent codec costs,
and the maximum charged replay workspace. Include whole-stream raw fallback and
prefer the parent on equal cost. The parent at every layer is always available,
so the net CPU payload cannot exceed the baseline. Apply the **same** codec and
workspace rule to the original and adaptive streams. The old baseline
totals must match the prior screen exactly. Require at least 10% net CPU source
payload reduction on both consumed inputs before considering native reader
implementation. Preserve all rejected formats, layers and errors.

This measures source-array payload only, not active GPU KV or whole-request
peak. Fit/decode temporaries and model-time reconstruction costs are not measured
by this CPU screen. The arrays constitute a format feasibility experiment;
neither prototype/residual source construction nor sparse literal reconstruction
is yet supported by the serving kernel. A positive screen would require actual
native reader integration, tensor lifetime accounting, retention-1 controls,
unchanged model-level token/KL/logit gates and independent quality validation.
Source MSE preservation alone does **not** prove answer quality preservation.
No larger campaign is submitted automatically.

Scalar quantization, source-space factorization, residual coding and sparse
exceptions are established ideas; this study makes no novelty or reproduction
claim for those principles. Related source-factor reconstruction includes
[Palu](https://arxiv.org/abs/2407.21118). No external implementation is imported.
