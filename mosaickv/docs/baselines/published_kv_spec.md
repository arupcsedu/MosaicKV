# Audited published KV-policy reimplementations

Status: implemented as `elastic_cache_reimpl`, `h2o_reimpl`,
`streamingllm_reimpl`, `pyramidkv_reimpl`, `snapkv_reimpl`, and
`roco_reimpl`. These names always
mean MosaicKV local reimplementations, never official code.

## Shared execution contract

All six methods use the same explicit Hugging Face prefill/decode loop, eager
attention capture, logical-position map, exact K/V gather, cache packer,
physical-byte accounting, prompts, media, generation settings, and timing
boundaries as MosaicKV. The common hard budget is applied to actual K plus V
bytes. Token block size must be one because each paper policy operates on
individual tokens, not MosaicKV multi-token blocks. At retention 1.0 every
method preserves every source position without pooling, eviction, merging,
quantization, or repair.

The implementation entry point is
`mosaickv.baselines.published_kv.build_published_kv_reimpl_plan`. Every plan
records selected and protected positions per layer/KV head, a SHA-256 of its
score vector, source/active slots, source/active bytes, and the explicit
`local_reimplementation_not_official` label.

## Elastic Cache

Official audit source:
`liuzuyan/ElasticCache@3138e03fc8894231d98a2e5353f27a59af854cc3`
(MIT), especially `kv_cache.py::ElasticCache` and equations 1--3 of the ECCV
2024 paper.

For every layer, the implementation averages attention importance across all
heads and sums it over every causal prompt query. The highest-scoring tokens,
together with the configured initial and terminal protected tokens, become
anchors shared by all KV heads in the layer. Every source position is assigned
to its nearest anchor with a deterministic left-anchor tie-break. K and V are
averaged independently in FP32 within each bucket and cast back to the original
cache dtype. Each merged slot retains its anchor's original logical position.

The generation stage implements the paper's second policy separately: after a
decode append would exceed the retained-ratio trajectory, one token is removed
at the fixed truncation rank `prompt_anchor_capacity + elastic_decode_offset`.
The default offset is -25, matching the audited author code. Every removal is
recorded in the runtime trace. Because the author implementation averages
cached post-RoPE LLaVA keys, `elastic_cache_reimpl` records the cached-key state
but does not reinterpret this paper-defined operation as a MosaicKV prototype.

## H2O

Source basis: H2O, arXiv:2306.14048, and the audited public tree at
`FMInference/H2O@ac75c2a8a9e76832b2a4139b9363373b56336bfb`.

For KV head (h), key position (i), and the last (w) observed prompt
queries, the local prefill-boundary score is

\[
s_{h,i}=\sum_{q=N-w}^{N-1} A_{h,q,i}.
\]

The final (w) positions are the recent set. Remaining capacity is filled by
descending accumulated attention, with physical position as the deterministic
tie-breaker. This is the prompt-compression H2O variant used by PyramidKV's
published comparison implementation. It is not the original H2O repository's
unbounded online decode experiment. The distinction is recorded in score
provenance rather than hidden.

The audited H2O tree has no top-level license file. MosaicKV therefore does
not redistribute it or expose an `h2o_official` label.

## StreamingLLM

Official audit source:
`mit-han-lab/streaming-llm@2e5042606d69933d88fbf909bd77907456b9b4dd`
(MIT), especially `streaming_llm/kv_cache.py::StartRecentKVCache`.

For capacity (C) and configured sink count (S<C), selection is exactly

\[
\{0,\ldots,S-1\}\;\cup\;\{N-(C-S),\ldots,N-1\}.
\]

No attention score is used. Sink and recent positions are protected. Logical
positions are preserved; packing does not renumber RoPE positions.

## SnapKV

Official audit source:
`FasterDecoding/SnapKV@e216ddc84c5bd210378cbdbbba12ba02102aa640`
(Apache-2.0), especially
`snapkv/monkeypatch/snapkv_utils.py::SnapKVCluster.update_kv`.

SnapKV sums attention from the final observation window, then applies the
configured one-dimensional average or maximum pool of odd kernel width (k):

\[
u_{h,i}=\operatorname{Pool}_{k}\!\left(
\sum_{q=N-w}^{N-1}A_{h,q,i}\right).
\]

The window itself is retained, and the highest pooled scores fill the older
capacity. GQA query heads are averaged only inside their corresponding KV-head
group. The default controls match the audited source: window 32, average pool,
kernel 5.

## PyramidKV

Official audit source:
`IsaacRe/PyramidKV@1ca16d6ea0bc3e48fc4cc8623bc4901bc131d511`
(Apache-2.0), especially
`pyramidkv/pyramidkv_utils.py::PyramidKVCluster.update_kv`.

Token scoring is the SnapKV observation-window pooled score. Capacity is not
uniform by layer. Let (C) be the average route capacity, (w) the observation
window, (b) the paper beta control, and (L) the layer count. The audited
source constructs an early-layer maximum and late-layer minimum:

\[
m=\max(1,(C-w)/b),\qquad M=\max(m,2(C-w)-m),
\]

then linearly decreases from (w+M) to (w+m). MosaicKV rescales these real
capacities to the exact shared byte target and uses bounded largest-remainder
rounding. This preserves the pyramidal order while preventing nominal ratios
from exceeding the physical byte budget.

## RoCo

Source basis: arXiv:2402.06262 and the audited EasyKV tree at
`DRSY/EasyKV@a1d71cae3b562d9a709dda3741bd63e46a09ad31`, especially the `roco`
branch in `easykv/easykv.py`.

For each token, the policy computes mean attention and standard deviation
across observed queries:

\[
\mu_i=\frac{1}{Q}\sum_q A_{q,i},\qquad
\sigma_i=\sqrt{\frac{1}{Q}\sum_q A_{q,i}^2-\mu_i^2}.
\]

The configured recent suffix and highest-variance fraction are protected from
eviction. The remaining capacity keeps the largest mean-attention tokens. This is the
deterministic batch equivalent of repeatedly applying the paper's
mean-attention eviction over a fixed prefill observation. It is explicitly
recorded as a prefill-boundary interpretation; it must not be presented as an
official EasyKV online-decode result.

The audited EasyKV tree has no top-level license file, so no source is
redistributed and no `roco_official` label exists.

## Fail-closed behavior

The runtime rejects non-eager attention for score-dependent methods, cache
block sizes other than one, missing or malformed attention tensors,
non-uniform decoder layer geometry, a protected window that cannot fit, a
retention-one budget smaller than FullKV, and byte budgets with non-uniform
per-token storage. Elastic Cache additionally requires a common layer-wise
anchor count for all KV heads. It never substitutes Prompt Top-K for a rejected
method.

## Official PrefixKV and LOOK-M

Their pinned MIT submodules remain unmodified. Official execution uses
`scripts/run_official_baseline.py`, which verifies the Git SHA, clean tree,
license hash, exact legacy package versions, scratch-only cache/output paths,
and writes `measurement_type=baseline_official_measured`. PrefixKV's audited
environment uses Torch 2.1.2 and Transformers 4.31.0. LOOK-M requires Torch
2.1.1 and Transformers 4.37.0. Those incompatible author stacks are isolated;
the launcher is available from the common project control plane, but the
official model process does not pretend to use the common Torch 2.8 runtime.

Official and reimplementation artifacts are comparable only after the parity
validator confirms identical model weights, processor, prompts, media,
generation, precision, attention backend, output length, and actual active K/V
bytes.
