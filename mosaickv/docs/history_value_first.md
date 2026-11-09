# History V before prompt V: bounded schedule-only diagnostic

The current one-buffer eager reader reconstructs prompt K, history K, prompt V,
then history V. The shared source workspace is overwritten by prompt V, so the
history source must be unpacked again. After QK/softmax finishes, the old K
lease is no longer needed. Reconstruct **history V before prompt V**, reuse the
source left by history K exactly once, then skip its duplicate value fill.
Attention summation order does not change; all original native GEMMs, RoPE,
patches and constructor checks remain. No additional GPU owner is created.

Require one consolidated NativeHistoryFormat or MixedHistory packet, the same
request owner, nongrad eager execution, original logical membership, no source-
destination alias, and a complete K-to-V transaction. Tickets retain only weak
references. Between K and V the audited eager scores use separate buffers; no
unreviewed source-writing callback is supported. Reject changed source identity
and checked version metadata, cross-layer consumption and incomplete commits.
Inference tensors have no version counter: correctness also relies on the
private, source-pinned read-only score boundary, not a claim of detecting every
possible external in-place mutation.

CPU regressions -> actual native/mixed CUDA operators with an explicit prompt-
overwrite hazard -> same two consumed pretrained inputs -> all compatible local
baseline controls -> randomized direct three-arm timing. Operator promotion
requires unchanged words and adjusted equal-or-faster host intervals in both
registered source formats. Model promotion requires all 128 vocabulary-logit
words unchanged, identical initial/final storage, complete requested KV/workspace
ownership, and actual reuse. Strict latency uses adjusted upper ratio <=1.00.

Apply the schedule to the storage-only comparator and compatible local baseline
precision variants too. This is not an official-baseline reproduction, fresh
quality evaluation, full six-mechanism result or whole-process HBM certificate.
History compression starts after answer EOS on these consumed inputs; do not
present its 128-step tail savings as demonstrated answer-phase savings. No new
cohort or larger campaign is authorized by this diagnostic.
