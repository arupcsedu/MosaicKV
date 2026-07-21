# Cache-state model

`mosaickv.cache_state` is the backend-independent state and accounting layer for
MosaicKV. Forecasting, graph construction, selection, prototype construction,
residual encoding, and repair are separate modules; the state layer validates
their payload and membership boundaries.

## State objects

- `ModalitySpan` maps a half-open logical sequence range to text, image, or
  video provenance, with optional image, frame, page, and region identifiers.
- `LogicalPositionMap` separates active physical cache slots from original
  logical positions and retains both the original logical sequence length and
  the next decode position.
- `KVBlockDescriptor` identifies one layer and one KV head, its physical and
  logical positions, optional token IDs, media provenance, K/V dtype and
  device, exact byte size, and mandatory/non-compressible status.
- `FullKVState` owns the unmodified source K/V tensors and their complete block
  partition.
- `ExactTier`, `PrototypeTier`, and `ResidualTier` own tier payloads and the
  source memberships represented by those payloads.
- `MosaicKVState` combines the tiers with logical-position state and checked
  `CompressionStatistics`. Exact and prototype payloads are active KV;
  residual payloads are separately accounted CPU storage and may reference the
  same source memberships as prototypes.

Blocks contain at most `block_size` consecutive source positions. A block ends
early at the sequence tail or when modality/media provenance changes. Blocks
never cross a layer or KV-head boundary. Source membership is represented by
`(layer, kv_head, physical_position)` so an invariant check can prove that
every source slot occurs exactly once.

## Gathering and reinjection

`FullKVState.gather_exact_blocks()` copies whole source blocks into an
`ExactTier`. `FullKVState.gather_selected_positions()` copies arbitrary
physical positions per `(layer, kv_head)` while preserving their original
logical positions. Both operations retain tensor dtype and device.

`MosaicKVState.retention_one(full)` gathers every source block into the exact
tier and immediately reconstructs the source with
`reconstruct_full_state(full)`. Reconstruction allocates new K/V tensors,
reinjects only the gathered payloads, and requires bitwise tensor identity. It
also verifies that original logical sequence length and next decode position
are unchanged. `to_cache_snapshot()` converts a reconstructed state back to
the typed Hugging Face cache snapshot used by the adapters.

## Enforced invariants

Construction fails with a useful exception when:

1. source positions are missing from or duplicated across source blocks;
2. exact source memberships overlap prototype or residual memberships;
3. descriptor or tier byte counts differ from `numel * element_size` storage;
4. a 100%-retention reconstruction differs from its source;
5. any mandatory source block is absent from the exact tier; or
6. physical or logical block positions are not strictly monotonic.

Prototype and residual memberships may overlap intentionally: the residual is
an inactive copy or encoding of a source block represented by a prototype.
`active_kv_bytes` counts only exact and prototype device payloads;
`residual_kv_bytes` and `total_stored_bytes` expose CPU and total storage.
The constructor and residual representation are documented in
[three-tier cache construction](three_tier_cache.md).

The CPU suite performs a deterministic randomized property sweep over layer
counts, KV-head counts, tensor widths, sequence lengths, modality layouts, and
block sizes. The Hugging Face GPU smoke gate runs 16-token greedy parity after
round-tripping the full cache through these core structures. A passing tiny
architecture test validates runtime mechanics only; checkpoint support still
requires the pinned-checkpoint gate documented in the adapter guide.

On 2026-07-19, Slurm job `17104011` completed this core 100%-retention gate on
one NVIDIA A100-SXM4-80GB with Torch 2.11.0+cu130 and Transformers 4.57.6.
LLaVA-1.5, Qwen2.5-VL, and LLaVA-OneVision tiny random architecture instances
each recorded 16/16 token agreement and maximum absolute logit difference
`0.0`. This is no-download runtime-mechanics evidence from random weights, not
checkpoint acceptance or an experimental result. A separate pinned 0.5B
LLaVA-OneVision attempt (`17103946`) stopped before inference because the
available borrowed environment lacks its required `torchvision` dependency;
no checkpoint-support claim is made from that failed run.
