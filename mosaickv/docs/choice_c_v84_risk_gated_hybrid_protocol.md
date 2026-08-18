# Choice C v84: label-blind risk-gated hybrid reranking

## Scientific boundary

V84 is a new calibration protocol, not a continuation of v83 tuning. The v83 seal is used only to locate samples that must be excluded. V84 candidate definitions do not use v83 answers, references, benchmark scores, failed-sample identities, or per-sample telemetry.

The only empirical design input is label-blind v82 route telemetry: normalized prompt-mass loss, normalized hybrid-forecast gain, prompt/forecast total variation, and replacement counts. These quantities do not contain benchmark references or task scores. Candidate thresholds are frozen before v84 sample materialization.

V84 materializes 96 new DocVQA and 96 new TextVQA calibration examples with seed 20280107. The materializer excludes repository ledgers, every earlier Choice-C input directory, the complete v83 held-out input directory, and prior observed-result rows. Any overlap is a hard error.

## Why a per-swap risk gate is needed

Hybrid future-query evidence can favor a visual block that Prompt Top-K did not retain. That replacement can help future decoding, but it can also remove text evidence needed by the next answer. A route-level prompt-mass floor is too coarse: several individually risky swaps can still fit under one aggregate limit.

For a proposed equal-size replacement of source block (s) by target block (t), V84 computes:

\[
L_{\mathrm{prompt}}=\frac{p_s-p_t}{\sum_i p_i},
\qquad
G_{\mathrm{future}}=\frac{q_t-q_s}{\sum_i q_i},
\]

where (p_i) is prompt-attention evidence and (q_i) is the isolated hybrid-forecast evidence for the same layer and KV head. The swap is accepted only if it satisfies the existing prompt-mass, gain-ratio, and divergence checks plus:

\[
L_{\mathrm{prompt}}\le L_{\max},
\qquad
G_{\mathrm{future}}\ge G_{\min}.
\]

All comparisons retain equal-size blocks, so accepted swaps cannot change the byte budget. Mandatory blocks are never replaced. Ties are deterministic.

## Frozen arms

The common model, processor, prompt, cache block size, 27.5 MB active-byte cap, deterministic decoding settings, and four-step isolated hybrid forecast are identical.

- `hybrid4_exact_no_rerank_control_b4_27_5` measures the exact-only cache without stable-head reranking.
- `hybrid4_exact_risk_conservative_b4_27_5` permits at most two replacements, requires 99.5% aggregate prompt-mass preservation, and uses (L_max=0.0015), (G_min=0.0025).
- `hybrid4_exact_risk_moderate_b4_27_5` permits at most four replacements, requires 99.0% aggregate prompt-mass preservation, and uses (L_max=0.0030), (G_min=0.0010).

The thresholds bracket central and upper portions of the pre-v83 v82 label-blind route distributions. No v83 outcome determined either threshold.

All v84 candidates are exact-only. Prototype merging, residual storage, and repair are disabled. Therefore a successful v84 result can support only an exact-selection claim, not a complete MosaicKV prototype/repair claim.

## Fail-closed calibration gates

Every candidate must satisfy on both tasks:

- at least 30% lower active KV bytes than Prompt Top-K;
- at least 30% lower active-plus-residual bytes than Prompt Top-K;
- zero prototype logical bytes and zero residual bytes;
- FullKV exact-answer agreement no more than five percentage points below Prompt Top-K;
- complete rows, deterministic execution, and a single clean source snapshot.

Each risk-gated arm must additionally record at least one accepted rerank and at least one risk-rejected rerank on each task. Its FullKV exact-answer agreement may not fall below the no-rerank control on either task. If a telemetry field is missing, a divergence check fails, or either activity condition is absent, the arm is ineligible.

The submission chain contains only ledger materialization, two one-A100 calibration workers, and CPU analysis/freezing. It contains no held-out job. A held-out protocol may be designed later only if all gates pass and the analyzer creates `analysis/frozen_candidate.yaml`.
