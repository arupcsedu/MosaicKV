# Consumed-input streaming prefill transaction gate

The common exact-history stage passed all ten local method/input workers. It
reduces final/decode-owned peak storage with identical logits but does not remove
the earlier full native prefill allocation. This separately scoped engineering
gate changes WHEN a layer is serialized, not its selection or stored values.

Use the same two consumed inputs, checkpoint, BF16 eager backend, media,
processor and frozen source3-direct policy. No fresh data, calibration,
benchmark-label fitting, baseline superiority or paper promotion is authorized.
Only one A100/12 CPUs per worker, two workers, 45-minute hard bounds. Wait for
the preceding tiling array/analysis before admitting GPU work; do not exceed six
aggregate A100s. Full CPU unit/integration tests precede admission.

## Public cache boundary

Installed Transformers 4.55.2 `cache_utils.Cache.update` delegates to its
`DynamicLayer.update`. Provide a request-private subclass through
`past_key_values` in `PreparedInputs.model_inputs`. The audited
`models.llama.modeling_llama.LlamaAttention` consumes ORIGINAL native K/V for
prefill. Its post-forward hook then constructs and owns that layer's packet,
checks disjoint backing, and retires its original K/V. Preserve the original
logical length even though `keys`/`values` become absent. Reject second updates,
decoding through this temporary cache, crop, beams, non-eager, GQA and gradients.
Export immutable formats to the existing isolated decode cache afterward.

The real HF CPU fixture must establish unchanged prefill logits, length/mask
compatibility, complete retirement and hook cleanup. Failure must retain native
state and prevent partial export. No weights or installed source are modified.

## Two arms, no invented comparison

1. Full-prompt lossless BF16 K/V serialization: original bits checked three
   times per layer, then 128-token and complete teacher-forcing identity against
   the saved ordinary FullKV reference. This is an identity/storage control,
   NOT a compressed FullKV systems row.
2. Streaming execution of `source3_direct_fp32`: precompute the SAME global
   allocator/maximum-layer-scratch counts from the pinned profile, then perform
   the SAME balanced local selection, source coding, exact text, numerical
   exception fitting and four-head prompt tiling after each attention layer.
   Do not independently allocate one layer: that changes scratch rounding and
   capacities. Preserve BF16 head-mean then FP32 query-mean reduction order.

Before either arm, build the validated nonstreaming source constructor once as
an evaluation control. Retain only CPU hashes, membership IDs and prefill logits;
release all its original native cache, captures and encoded GPU payload before
the streaming memory measurement. Require exact streamed payload hashes,
shared-backing byte count and selected positions. Require bitwise prefill and
128-greedy logits and complete 97/593-target vocabulary logits/NLL/counts.
References/targets remain evaluation-only and enter no encoder API.

## Accounting and limits

Charge every live native cache layer, encoded backing, shared map/phase, and
observed per-ATen encoder workspace/old-new overlap. Record partial state on
failure and require zero original K/V tensor objects after retirement. Preserve
all decode/history audit controls. Report initial/final decoded-cache storage,
observed prefill transaction peak, and maximum of observed prefill/decode-owned
peak. Compare the latter only against the known original native prefill K/V
lower bound; do not invent a measured whole-request reference peak.

Model weights, vision/attention activations outside the encoder, and opaque GPU
workspace are excluded from this declared tensor boundary. Record actual
PyTorch allocation/reservation peaks separately; whole-request GPU peak remains
uncertified. Instrumented single-run prefill timings are diagnostic, not a
production latency result. No byte-matched baseline or universal memory claim
follows from this own-parent gate. Baselines must receive compatible streaming
construction before any comparative claim. Do not alter their algorithms when
global layer statistics make immediate serialization unsafe.

Finite afterany analysis must preserve failures. Passing justifies further
common-backend engineering, not independent quality or paper-scale execution.
