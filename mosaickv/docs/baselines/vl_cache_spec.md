# `vl_cache_reimpl`: ICLR VL-Cache specification

Status: implemented local reimplementation; **not official author code**. Every
configuration, trace, manifest, and result row uses the exact method label
`vl_cache_reimpl`.

This document covers *VL-Cache: Sparsity and Modality-Aware KV Cache
Compression for Vision-Language Model Inference Acceleration* by Tu, Vashchilenko,
Lu, and Xu, published at ICLR 2025. It does **not** cover the later similarly
named VLCache system for reuse across recurring images.

## Authoritative sources and implementation availability

- Full paper and embedded supplementary appendix:
  [arXiv:2410.23317v1](https://arxiv.org/abs/2410.23317), submitted 2024-10-29.
- Conference version: [ICLR 2025 OpenReview](https://openreview.net/forum?id=HMrcv7Q4Ub).
- The arXiv source contains Appendix A.1--A.5, including prompt construction,
  the three-kernel implementation sketch, speed methodology, and modality
  contribution/coverage definitions. There is no separate algorithm appendix
  beyond that material in the inspected artifact.
- Neither primary page provides an official software repository. No official
  implementation is assumed or represented here.

The arXiv paper source declares CC BY-NC-SA 4.0. That is a content license, not
a software license. This implementation derives behavior from the equations,
uses no copied author software, and retains paper attribution.

## Notation and online inputs

For layer `l`, query head `h`, prompt length `m`, and head dimension `d`, let

\[
Q^{(l,h)},K^{(l,h)}\in\mathbb{R}^{m\times d},\qquad
A^{(l,h)}=\operatorname{softmax}(QK^T/\sqrt d).
\]

The attention is causal. Let `v` be the first language position following the
last visual token and `tau = m - v`. The post-vision query set is
`P = {v, ..., m-1}`. The online method needs eager prefill attention
probabilities, the exact modality spans produced by the model processor, and
the unmodified full KV cache. It performs one selection immediately after
prefill. It does not use prototype merging, quantization, offloading, or repair.

The paper method is prompt-adaptive. It does **not** require an offline
calibration profile: sparsity and layer budgets are calculated on the current
prompt. The randomly sampled DocVQA, MathVista, and COCO-Caption examples in
Section 3 establish the observations; they are not described as an inference
profile.

## Paper-defined behavior

### Relative threshold and attention sparsity

Paper Equation (1) defines the row-relative filter

\[
\operatorname{ThresholdFilter}(A,p)_{qj}=
\begin{cases}
A_{qj}, & A_{qj}\ge p\max_{j'} A_{qj'},\\
0, & \text{otherwise}.
\end{cases}
\]

The paper uses `p = 0.01`. MosaicKV takes the maximum over causally visible
keys and counts only causal entries. For the post-vision region, the per-head
sparsity corresponding to Equation (2) and Algorithm 1 is

\[
\gamma'_{l,h}=
\frac{
\sum_{q\in P}\sum_{j\le q}
\mathbf{1}[\operatorname{ThresholdFilter}(A^{(l,h)},p)_{qj}=0]
}{
\sum_{q\in P}|\{j:j\le q\}|
}.
\]

Algorithm 1 averages over heads:

\[
\gamma'_l=\frac{1}{H}\sum_h\gamma'_{l,h},\qquad d_l=1-\gamma'_l.
\]

Thus a dense layer has larger `d_l` and should receive more cache.

### Layer-adaptive cache allocation

Given whole-model retention ratio `alpha` and `L` decoder layers, Algorithm 1
sets

\[
Z=\sum_l d_l,
\qquad
\widetilde\beta_l=\frac{d_l}{Z}\alpha L,
\qquad
\beta_l=\operatorname{clip}(\widetilde\beta_l,0.01,1).
\]

`beta_l` is the fractional cache budget for layer `l`. The allocation is
computed once after prefill, separately for every prompt; it is not a static
pyramid or an offline layer profile.

### Modality-aware token scoring

The modality awareness is in the **query boundary**, not a hand-written image
versus text weight. Visual-query rows are excluded, while post-vision language
queries score every causally visible prompt key. Section 3.2 defines

\[
s^{(l,h)}_j=\sum_{q\in P} A^{(l,h)}_{qj}.
\]

The highest-scoring positions are retained within the realized layer budget.
Summing rather than averaging post-vision query rows does not change Top-K,
but the implementation uses the stated sum and records its digest and total.
Both image/video and text key positions compete in this one ranking.

The paper's speed experiment uses the last 50 prompt tokens as query rows. The
accuracy algorithm instead defines the prompt-dependent post-vision region.
The default therefore uses all post-vision rows; `max_post_vision_queries: 50`
reproduces the speed-study interpretation and is recorded in the trace.

### Recent-token protection

Section 5 states that cache sparsification retains the most recent tokens and
uses a recent-token window equal to 10% of the cache budget. The text does not
unambiguously state whether that sentence applies only to comparison baselines
or to every method. The paper-experiment configuration implemented here is

\[
w_l=\lfloor 0.1 k_l\rfloor,
\]

with the latest `w_l` positions protected before attention Top-K. MosaicKV's
universal mandatory-token policy additionally protects every descriptor marked
non-compressible, normally the terminal prompt token. The fraction is an
explicit sensitivity parameter and can be set to zero to isolate Algorithm 1
and post-vision Top-K.

## Explicit implementation decisions

The paper leaves several details unspecified. They are isolated from the
paper equations in `build_vl_cache_reimpl_plan` and recorded under
`implementation_decisions` in every trace.

1. **All prompt keys, not only post-vision keys.** The prose and Algorithm 1 use
   `Q_post K_all^T` and say that critical visual and language tokens are both
   selected. One preliminary formula prints `K_post`, which would make visual
   selection impossible. The implementation follows Algorithm 1 and the stated
   purpose: post-vision queries attend to all causal prompt keys.
2. **GQA mapping.** Algorithm 1 averages per-query-head sparsity to get a layer
   budget, but does not specify how GQA query heads choose KV positions.
   Contiguous query-head groups belonging to one KV head are mean-reduced; Top-K
   is then independent per KV head. MHA is the group-size-one case.
3. **Integer budgets.** Clipping and integer rounding can change the global
   total. The implementation first preserves the paper's raw and clipped
   `beta_l`, then uses bounded largest-remainder apportionment to obtain one
   deterministic hard total. Ties favor lower layer indices.
4. **Top-K ties.** Equal scores favor the lower original physical position.
5. **Clipping bounds.** The paper defaults are `[0.01, 1]`. Changing either is
   explicitly an implementation sensitivity setting, not paper-default behavior.
6. **Short prompts.** Every layer/head must retain at least one position. A
   requested budget below clipping or mandatory bounds is rejected rather than
   silently exceeded.
7. **Uniform decoder geometry.** The paper's allocation assumes one prompt
   length and head geometry across decoder layers. The reimplementation rejects
   heterogeneous decoder KV-head counts or cache lengths instead of inventing
   a weighted generalization.
8. **Retention 1.0.** Selection is bypassed: every source position is copied
   exactly, and no other tier or transformation is created. The explicit hard
   budget must cover the full cache.
9. **Reference implementation versus optimized kernel.** This implementation
   consumes captured eager attention and materializes statistics for
   auditability. It does not claim the Appendix A.3 Triton kernel, whose three
   stages avoid materializing `QK^T` and recompute softmax for column reduction.

## Integer realization and exact budget

For the uniform geometry supported by the paper models, the hard source budget
is measured in token-sized `(layer, KV head, position)` blocks. The target is
the smaller of `floor(alpha * source_slots)` and `cache.budget_value`, rounded
down to a complete KV-head group. Fractional desired counts are
`beta_l * m`. Lower bounds are `max(1, floor(min_beta * m))`; upper bounds are
`floor(max_beta * m)`. Bounded apportionment realizes the exact feasible target.

For each layer/head, mandatory and recent positions are selected first. The
remaining positions are ordered by `(-s_j, physical_position)`. Selected K and
V positions are passed unchanged through the common `pack_runtime_payloads`
interface. Original logical sequence length and next decode position remain
separate from active packed-cache length.

## Calibration and evaluation split enforcement

Paper-default online inference needs no calibration. Calibration becomes
relevant only when choosing an ambiguity setting—threshold, clipping bounds,
recent fraction, or the 50-query cap. Such choices must be made using IDs listed
in `vl_cache.calibration_sample_ids`, with dataset ID, immutable revision, and
split recorded in configuration.

`assert_vl_cache_calibration_disjoint` rejects duplicate IDs and any overlap.
The HF runtime checks every evaluation `sample_id` against the configured
calibration IDs before prefill. `analyze_vl_cache_sensitivity` additionally
requires that its input sample is registered as calibration data and accepts an
explicit evaluation-ID set. It emits structural diagnostics only; task scores
must be produced later by the shared evaluation harness on the disjoint set.

## Sensitivity analysis

The supported ambiguity grid is:

- `sparsity_threshold`: recommended diagnostic values `0.005, 0.01, 0.02`;
- `recent_window_fraction`: `0.0, 0.1, 0.2`;
- `max_post_vision_queries`: full post-vision region or `50`;
- optional non-paper clipping bounds through configuration.

Each `VLCacheSensitivityPoint` records layer sparsity, realized per-layer
positions, active slots, exact retained bytes, and a digest of selected
positions. It is labeled
`synthetic_or_calibration_structural_diagnostic`, never as measured benchmark
quality. Selection is deterministic.

## Paper-to-code map

| Paper behavior | Code component | Classification |
|---|---|---|
| Equation (1), relative row threshold | `baselines.vl_cache.threshold_filter` | Paper equation |
| Equation (2), causal zero fraction | `post_vision_attention_statistics` | Paper equation restricted to Algorithm 1 post-vision rows |
| Algorithm 1 head mean and density | `post_vision_attention_statistics` | Paper algorithm |
| Algorithm 1 raw/clipped `beta_l` | `paper_layer_retention_ratios` | Paper algorithm |
| Section 3.2 accumulated post-vision score | `post_vision_attention_statistics` | Paper formula |
| Per-layer Top-K | `_select_positions` | Paper behavior plus deterministic tie rule |
| 10% recent window in Section 5 | `_select_positions` | Chosen paper-experiment interpretation |
| Integer global budget | `_bounded_apportion` | Implementation decision |
| GQA query-to-KV-head mapping | `post_vision_attention_statistics` | Implementation decision |
| Exact K/V extraction | `vl_cache_runtime_payloads` | Common-runtime implementation |
| Selection/byte/provenance trace | `VLCacheCompressionPlan.trace` | Reproducibility implementation |
| Calibration leakage guard | `assert_vl_cache_calibration_disjoint` and HF runtime | Project scientific-integrity rule |
| Ambiguity grid | `analyze_vl_cache_sensitivity` | Project diagnostic |

The implementation file is
[`src/mosaickv/baselines/vl_cache.py`](../../src/mosaickv/baselines/vl_cache.py).
The development configuration is
[`configs/hf_vl_cache_reimpl.yaml`](../../configs/hf_vl_cache_reimpl.yaml).

## Formula and trend validation status

CPU tests construct a known dense layer and a known sparse layer. They verify
Equation (1), the exact causal sparsity fractions, Algorithm 1 ratios, larger
realized budget for the denser layer, exact hard-budget accounting, deterministic
selection, common-packer payloads, full-cache retention parity at the tensor
level, sensitivity labeling, and calibration leakage rejection. A tiny eager
LLaVA architecture test with a synthetic expanded-image token boundary also
executes the shared packed-cache decode path and verifies retention-one greedy
token/logit parity. It loads random tiny weights and is a correctness test, not
a model-quality measurement.

This reproduces the paper's **structural trend** that denser layers receive more
cache. It is not a task-level reproduction and is not placed in a measured
results table. Exact published accuracy/latency reproduction has not been
claimed because the current development config is LLaVA-1.5 rather than the
paper's LLaVA-v1.6-Mistral-7B or LLaVA-v1.6-34B, and the paper does not specify
the complete package revisions or release its Triton implementation. A future
measured trend run must use identical prompts, media, tokenization, generation,
output length, cache budget, precision, and backend across methods and record
the full MosaicKV run manifest.

## Invocation

```bash
mosaickv evaluate \
  --config mosaickv/configs/hf_vl_cache_reimpl.yaml \
  --task synthetic_smoke \
  --output-dir /scratch/djy8hg/runs/vl_cache_reimpl
```

Direct flags require `--method vl_cache_reimpl --block-size 1
--budget-unit blocks`. Eager attention is mandatory until an audited fused
statistics kernel and its correctness parity tests exist.
