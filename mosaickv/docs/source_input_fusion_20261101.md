# Same-input compact-source, grouped projection and lossless input-fusion gate

Prerequisites: completed positive corrected-source 32-prefix model guards and
three-boundary reductions; completed compact-header CPU gates; complete
operator microbenchmark choosing fused_h2 with ALL K/V bits unchanged.
Native BMM was faster but not bitwise identical; it has NOT passed a model
gate. Do not reuse the microbenchmark as model quality or end-to-end latency.

Only the same two consumed Description/MM-Vet inputs. Preserve negative
history, phase, quota and reconstruction results. No fresh cohort/promotion.

Register separate ablations, not one unexplained change:

- old FP32-header source shadow, unchanged control;
- prior conditional source and its lossless text-source form, unchanged;
- same shadow with compact affine FP16 headers;
- full-prompt compact source with bitwise-validated two-head fused tiling;
- full-prompt compact source with native BF16 BMM/256-row workspace;
- exactly the native source/text K/V above, using integrated exact inputs.

Offer compact headers and common sequential attention to all five existing
selectors: EviCastKV shadow, Prompt Top-K, LOOK-M, VL-Cache and PrefixKV local
reimplementations. Retain BOTH header variants for every baseline, and verify
the FP32-header controls against historical 128-greedy/33-prefix logits.
Never omit a stronger unmodified point if compaction worsens its quality.
These remain precision/representation variants, NOT official reproductions.

Integrated exact inputs decode lossless original text-normalization inputs
directly into the existing bounded native source chunk. Project once through
the main native K/V matrices instead of projecting the text a second time.
Fit exact scalar corrections to the IMMEDIATELY PRECEDING native source/text
representation, not to task logits/answers. Verify non-overlay scalars are
unchanged except the prior numerical-transport corrections. Require three
bitwise K/V materializations; keep the old representation when exact fusion
does not reduce payload. Count all input buffers, indices and corrections.
Native source workspaces count throughout their actual lifetime, including
the ordinary two-buffer control; no hidden full normalized-input tensor.

Retain original byte caps, original positions, untouched mandatory tokens,
raw append-only history, no FullKV fallback, protected K/V bitwise, original
cache retirement, immutable encoded state and A-B-A isolation. Every arm
requires two-buffer/one-buffer logit equality. Native-vs-fused is a separate
model-level experiment; do not require or falsely claim their equivalence.
Integrated-input vs its native parent MUST have identical 33-prefix and
128-greedy logits. FullKV retention-1: <=.001 logits/identical tokens. Model
guard unchanged: 32 cache-dependent FullKV prefixes, KL<=.02, logits<=1,
greedy agreement 1. Score reference ROUGE-L through EOS, time all128 tokens.
PPL/official MM-Vet judge scores are NOT measured in this stage.

Report each measured initial/final/decode-owned peak separately, per-arm cap,
the extra native workspace and format choices, component construction times,
decode CUDA/wall time and GPU allocator peaks. Source component timings do
not include the historical protected-position planner/setup and therefore
are NOT end-to-end latency. Record total setup separately. Whole-request and
construction peak remain uncertified. No same-byte or superiority claim
from these unequal measured development points; independent quality N=0.

Full CPU suite first, then two one-A100/12CPU workers <=30min and afterany
analysis. No larger campaign and no automatic frozen-candidate artifact.
