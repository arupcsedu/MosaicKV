# Common native history reader: unchanged words, explicitly counted owners

Prerequisite: the Llama-compatible operator study passes every 16..112-row
epoch's raw words, tail sentinels and unchanged 32-KiB numerical-repair cap.
Its native/reference GPU-event ratios were 0.19..0.28. These are synthetic
operator intervals, not request latency or independent quality evidence.

Offer the native history execution option identically to every method. Preserve
the original lossless normalized source, phases, source packing, 16-token raw
tail and original positions. Construct actual sparse native-versus-original
K/V repairs while raw history is available. Keep three K and three V original
word checks before committing each layer. At consolidation, recompute repairs
at the changed GEMM shape; do not join corrections calibrated at another shape.
Keep all original repeated verification and anti-alias scratch checks.

Use one request-private zero-workspace FP32 native descriptor set. Reuse the
already charged Source prompt input/output buffers where available; otherwise
allocate and charge exactly two 128-row BF16 buffers (2 MiB). Attach before the
history audit's exclusion snapshot to prevent double-counting the same owners
as both shared storage and scratch. Retire before the parent prompt pool and
before final independent enumeration. Failed requests also retire. No owner,
handle, activation, source or model reference persists across requests.

Initial prompt packet bytes must remain identical to each own control. Final
history bytes may change only by the measured difference in numerical repairs;
source packets and all positions remain identical. Report actual initial,
final and observed decode-owned peak storage, not nominal sizes or uncharged
workspaces. This is a common lossless execution variant, not an algorithmic
selection improvement. The Source precision variant still disables/replaces
the six mechanisms; independent benchmark-quality N=0.

Use the same two consumed inputs, 128 predictions, randomized method order,
five warmups and three timed repetitions. Compare original and optimized
Source and both PrefixKV graph readers; repeat all 128 parent vocabulary-word
checks, original packet digest and no-fallback controls. Whole-request GPU
peak remains uncertified. No fresh or larger quality submission is authorized.
