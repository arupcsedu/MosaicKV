# Three-tier cache construction

`mosaickv.prototypes.construct_three_tier_cache` converts a `FullKVState`, a
sparse evidence graph, and a completed selection result into an auditable
exact/prototype/residual state. It does not mutate the source cache.

## Exact tier

Selected graph nodes are gathered unchanged with
`FullKVState.gather_exact_blocks`. Dtype, device, source positions, and byte
counts must match their `KVBlockDescriptor` metadata. Mandatory blocks remain
exact through the existing `MosaicKVState` invariant.

When `retention_ratio=1.0`, construction requires every source block to be
selected exact and takes a dedicated `retention_one` branch. It
gathers every block as exact, creates no prototype or residual payload, and
immediately performs bitwise full-cache reinjection. Prototype and residual
configuration is not consulted on this path.

## Prototype safety gate

The constructor compares the source `cached_key_state` with the selected
adapter's `AdapterCapabilities.cached_key_state`. Prototype merging is allowed
only when all of the following hold:

1. the adapter explicitly sets `supports_prototype_merge=true`;
2. source and adapter RoPE-state declarations agree; and
3. keys are `pre_rope` or RoPE is `not_applicable`.

Unknown or post-RoPE keys are never averaged. LLaVA-1.5, Qwen2.5-VL,
LLaVA-OneVision, and InternVL2.5 currently declare post-RoPE caches and
prototype support false. They therefore return
`exact_only_prototype_merge_unsafe`, preserving the selected exact blocks and
creating no transformed tier. This is a supported selection-only MosaicKV
mode, not prototype support.

## Prototype assignment and values

Every unselected block must have a graph edge to a compatible selected anchor.
Compatibility requires the same layer and KV head, an allowed directed
`source:anchor` modality pair, and the configured maximum logical-position
span. The highest edge weight wins; equal weights choose the lower anchor node
ID. Typed or reverse-direction parallel evidence is reduced by maximum weight
without materializing a dense matrix.

The anchor remains exact and is not averaged into the prototype. Its role is
to define the cluster and its active logical reference position. Assigned
unselected blocks supply the prototype members and graph-derived weights. Each
source block is first pooled over its cache sequence axis. K and V weighted
means are computed in FP32, kept as one active cache slot, and cast back to the
respective source cache dtype and device. Layers, KV heads, or incompatible
modalities are never mixed.

Construction fails closed to exact-only selection if any unselected node lacks
a compatible anchor, a cluster exceeds `group_size` or the full position-span
limit, or exact-plus-prototype storage exceeds the selector's block,
retained-slot, or byte budget (or a stricter explicit active byte budget).
Partial prototype coverage is never reported as a complete three-tier state.

Every `PrototypeRecord` stores its exact anchor, assigned source node IDs, raw
and normalized weights, and these diagnostics:

- assigned cluster size;
- weighted K and V mean-squared dispersion;
- modality composition;
- minimum, maximum, and span of source logical positions; and
- member bytes, prototype bytes, and active bytes saved.

## Residual tier

When residual storage is enabled, every original unselected block represented
by a prototype is copied to CPU and indexed by layer, KV head, prototype ID,
original logical position, physical source position, payload, and offset.
`lossless` is the default and initial research representation. FP16, BF16,
FP8 (`float8_e4m3fn`), and symmetric per-payload INT8 encodings are available
only when the runtime tensor library can represent them; INT8 records separate
K and V scales.

Production construction requires pinned CPU allocations by default. If the
PyTorch/CUDA runtime cannot provide pinned memory, construction raises
`PinnedMemoryUnavailableError` instead of claiming a repair-capable residual
tier. `require_pinned_memory=false` exists for CPU/mock validation only.
`restore_residual_payload` converts a stored payload back to its source dtype
and device.

Residual membership intentionally overlaps prototype source membership: the
prototype is active device KV, while the residual is the inactive original
source data used by the decode-time repair controller. It may never overlap exact
membership. `active_kv_bytes` counts exact plus prototype payloads;
`residual_kv_bytes` reports CPU storage; `total_stored_bytes` reports both.

## Position and length state

`ThreeTierCacheConstruction` retains the source original logical sequence
length and next decode position separately from each layer/head's physical
`active_cache_length`. `ActiveHeadLayout` records exact logical positions,
prototype IDs, and their selected-anchor logical positions. Current Hugging
Face decoding still rejects active/logical length divergence, so this module
does not claim compressed end-to-end adapter execution yet. See
[decode-time residual repair](decode_time_repair.md) for the backend-independent
promotion controller and its current integration boundary.

## Validation

The CPU suite checks manually computed weighted prototypes, deterministic
anchor selection, incompatibility fallback, byte accounting, lossless and
quantized residual round trips, original-position indices, current-adapter
exact-only behavior, and transformation-free retention one. A separate Torch
smoke validates BF16 and FP8 host encodings where the installed build exposes
them. These are implementation correctness tests, not measured research
results or model checkpoint acceptance.

Run the no-download GPU smoke, which additionally requires real CUDA pinned
memory, with:

```bash
sbatch mosaickv/slurm/three_tier_smoke.sbatch
```

On 2026-07-19, Slurm job `17110572` completed this no-download validation on
one NVIDIA A100-SXM4-80GB using Torch 2.11.0+cu130 and CUDA runtime 13.0. It
verified an FP16 prototype on `cuda:0`, two lossless pinned CPU residual
payloads, pinned lossless/FP16/BF16/INT8/FP8 storage and restoration, exact-only
fallback for all four current post-RoPE adapter classes, and a
transformation-free retention-one reconstruction. This synthetic-tensor
`validation_smoke` is hardware-path evidence only; it is not checkpoint
acceptance, a quality measurement, or a paper result.
