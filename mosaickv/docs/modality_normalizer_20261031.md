# Draft-conditioned modality normalization — consumed diagnostic

The preceding source-funded delimiter variant improved COCO corpus CIDEr on
sixteen consumed images but did not establish superiority over the strongest
LOOK-M point. It worsened the consumed MM-Vet example. These negative evidence
and confidence intervals remain unchanged; no fresh or paper cohort is released.

## Hypothesis and fixed intervention

Retaining extra text while discarding visual tokens can alter the conditional
visual/text attention odds even though no retained K/V bits change. Diagnose
that mechanism, not another byte-cap or prior-mixing sweep. Use exactly four
deterministic FullKV draft steps from the original prefill and an isolated cache.
No second prefill, benchmark annotation, or reference logit enters the estimator.
For each layer, head, and draft query, compute original and retained visual/text
masses. The correction is the mean of individual query log-odds differences:

`gamma = mean_q(log(M_full_visual/M_full_text)
                    - log(M_kept_visual/M_kept_text))`.

Clamp gamma to [-log(2), log(2)]. Add `alpha * gamma` ONLY to retained visual
logits; text, mandatory delimiter, generated history, and padding receive no
additional bias. Preregister alpha=0, 0.5, 1.0 as zero/half/full ablations. This
preserves each query/head before reduction rather than pooling their masses.
If either retained/original modality has zero mass at any draft query, bypass
that head with zero bias and record unidentified odds as null, never infinity.
This is a forecast approximation: merged or quantized keys may not match the
original attention, and later queries may differ. No fidelity guarantee follows.

## Scope and execution

Use only the same two consumed Description/MM-Vet inputs. The unchanged
`head3_boundary8` balanced selection and no-donor delimiter constructor remain
fixed. Keep K/V, physical/logical memberships, mandatory tokens, output policy,
precision, and original decode positions unchanged relative to that completion.
No prototype or residual transformation is enabled. This is an explicit variant
with graph/submodular planning disabled, not the frozen full paper architecture.
The implementation is method-agnostic, but this first diagnostic runs EviCastKV
only: it cannot establish superiority over a baseline.

Charge FP32 bias and validity metadata, all persistent encoded payload, shared
rotary owners, padding, raw append-only history, and the fused mask workspace.
The complete initial allocation must fit the original actual anchor bytes.
Zero/half/full controls use identical metadata shapes and must match measured
initial, final and decode-owned peak storage. This is a causal transport ablation,
not proof of largest-feasible byte-matched competing method allocations.
Inactive FullKV/reference/candidate packets and construction workspaces remain
outside decode-owned peak, so whole-request KV superiority is not certified.

Require full CPU-suite success first, then two one-A100 workers (maximum six
aggregate A100s), 12 CPUs and 30 minutes each. Use the unchanged common environment
and scratch-only caches. A finite afterany CPU analyzer preserves errors.

## Controls and stopping rule

The FullKV clone replay must match the pinned reference within 0.001 logits and
identical tokens. Zero bias must reproduce ALL saved 128 completion logits
bit-for-bit. Fused CUDA masks must equal CPU fixtures. Four-step forecast logits
must match the existing ordinary FullKV path and leave its cache unchanged.
Estimator parameters are frozen BEFORE the 32-prefix evaluation-only replay.
Save per-prefix actual visual/text/history mass alongside logits, tokens, traces,
exact bytes, hashes and manifests. Score greedy output only through first EOS.
ROUGE-L here compares to FullKV output; it is not official MM-Vet judging or PPL.

The model-level guard remains 32 prefixes, KL<=0.02, maximum logit difference<=1,
and greedy-token agreement=1. A useful intervention must improve MM-Vet without
losing Description quality, and pass unchanged guards; successful execution alone
is insufficient. If it fails, preserve the actual layer/prefix mass telemetry and
diagnose the failure; do not launch a fresh or larger cohort or loosen thresholds.
Even a diagnostic pass would require independent quality validation and a
separately implemented shared-physical-byte competitor lane.
