# Common exact scale transport for compatible local baseline variants

The original source-only binding depended on a three-bit source-code stream.
The frozen local baseline precision variants instead contain mixed 3/8-bit K/V
bodies, and PrefixKV can have different layer lengths. Feeding those bytes to
the source predictor would be incorrect. The new header-only binding retains
the original code tensors unchanged and serializes only their FP32 affine
scale bits. Charge all dictionaries, selectors and per-role maximum FP32 replay
buffers. A complete transaction that does not save storage retains the original
format. Raw BF16 archives without affine headers remain unchanged. Existing
encoded source packets are not converted twice.

CPU tests cover mixed bits, original payload identity, exact reconstructed K/V,
charged cache backing storage, raw no-op and common-scope restoration. All four
frozen local variants—Prompt Top-K, `prefixkv_reimpl`, `vl_cache_reimpl`,
`lookm_reimpl`—receive a consumed-input before/after GPU check. Both arms use
identical selection, merge behavior, model, media, prompts, dtype, eager backend,
generation, common softmax/predicate policy and native-history setting. The
historical per-input cost-only text policies are preserved verbatim, hashed, and
identical within each pair; they are not silently forced to one input's policy.

Measure all 128 vocabulary logits against that baseline's own control, not
EviCastKV outputs. Independently validate ownership certificates and initial/
final storage. Keep failed/unsupported rows. These are within-method transport
controls on consumed inputs—not official reproduction, fresh benchmark quality,
clean latency evidence, or cross-method byte-matched operating points. The
existing baseline identities and fidelity caveats remain unchanged.

The finite CPU → real-header operators → two one-A100/12-CPU model workers →
`afterany` review stays under the aggregate six-A100 limit. No fresh cohort or
paper campaign follows automatically. The separate clean timing job is unaffected.
