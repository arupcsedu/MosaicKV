# Complete native-target scoring on consumed inputs

This is evaluation-only diagnostic extension of the FP32 bit-preserving source
tiling study, not a fresh benchmark or official PrefixKV reproduction. Both
Description and MM-Vet identities were already consumed; independent N is zero.
No calibration, new cohort or paper campaign is authorized. Preserve negatives.

Freeze the chosen global head tiling from the operator test. Require the model
parity result before preparing this run. Construct formats from prompt-derived
captures before accessing any annotation. Do not pass targets to construction,
numerical fitting, text-source optimization, or selection. Keep all existing
three-bit source codes, FP32 headers and frozen numerical guards unchanged.

Compare the source variant, its untiled controls, old EviCastKV shadow, and all
four local baselines. Baselines use the same common lossless text compression
and one-buffer eager transport. Include both FP32 and earlier FP16 baseline
header variants; do not hide a stronger baseline. This is an unequal measured
memory diagnostic, not an equal-byte superiority experiment.

After construction, access the immutable same-prompt/media native annotations.
Tokenize without added special tokens; score every target token with no EOS
stopping or truncation. The first loss uses the existing prefill logits; later
losses condition on preceding annotation tokens. The explicit maximum is 1024
target tokens. Reject longer targets rather than truncate. One target requires
zero cache updates and no cache construction/preallocation.

Run ordinary cloned FullKV and raw reinjected FullKV over the complete targets.
Require identical conditional greedy tokens and <=0.001 maximum logit difference.
Retain additive per-token NLL, token count and exact annotation/ledger hashes;
calculate PPL from summed NLL divided by summed tokens. Keep PPL and generated
FullKV-reference ROUGE-L separate. Report 97/593-token actual counts only when
confirmed by execution, not as synthetic measured observations.

Each metric phase records initial/final/temporary-layer-inclusive peak cache
storage independently. Teacher forcing has longer history than 128-token
generation; do not reuse its memory figures across phases. These are decode-owned
storage boundaries, not whole-request GPU peaks. FullKV remains a correctness
and quality reference, not a compressed-method latency comparator. Complete
rows, manifests, failures, original-cache retirement and replay checks remain.
