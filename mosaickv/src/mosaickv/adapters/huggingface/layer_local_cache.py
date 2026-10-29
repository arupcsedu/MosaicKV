"""Exact, unpadded per-layer storage for the audited bounded-history transport.

Heads in a layer have equal physical lengths, but can retain different source
positions. Different layers need not have the same length. Original post-RoPE
keys and values are gathered verbatim; no model weights or positions change.
"""

from __future__ import annotations

from typing import Any

from mosaickv.adapters.huggingface.types import CacheLayerSnapshot, CacheSnapshot


def gather_layer_local_snapshot(
    adapter: Any,
    original_cache: Any,
    positions: tuple[tuple[tuple[int, ...], ...], ...],
) -> CacheSnapshot:
    """Own exact-size BHSD tensors and reject duplicate/out-of-range positions."""
    import torch
    from transformers import DynamicCache

    from mosaickv.adapters.huggingface.bounded_history_cache import assert_bounded_history_support

    assert_bounded_history_support(adapter)
    if type(original_cache) is not DynamicCache:
        raise RuntimeError("layer-local packing requires ordinary DynamicCache prefill")
    legacy, _ = adapter._legacy_layers(original_cache)
    if len(legacy) != len(positions):
        raise ValueError("layer-local selection/cache layer counts differ")
    layers = []
    from mosaickv.measurements.kv_transactions import observe_kv_owners

    observe_kv_owners("layer_local_pack_source", legacy)
    for (key, value), heads in zip(legacy, positions, strict=True):
        if key.ndim != 4 or key.shape[0] != 1 or key.shape != value.shape:
            raise ValueError("layer-local packing requires matching batch-one BHSD tensors")
        if len(heads) != key.shape[1] or not heads or not heads[0]:
            raise ValueError("layer-local packing requires a nonempty selection for every head")
        if len({len(row) for row in heads}) != 1:
            raise ValueError("layer-local packing rejects ragged heads within a layer")
        if any(
            tuple(sorted(set(row))) != row or row[0] < 0 or row[-1] >= key.shape[2] for row in heads
        ):
            raise ValueError("layer-local positions must be sorted, unique and inside source")
        indices = torch.tensor(heads, dtype=torch.long, device=key.device).unsqueeze(0)
        indices = indices.unsqueeze(-1).expand(1, key.shape[1], len(heads[0]), key.shape[-1])
        layers.append(
            CacheLayerSnapshot(key.gather(2, indices), value.gather(2, indices.to(value.device)), 2)
        )
        observe_kv_owners("layer_local_pack_destination", ((x.key, x.value) for x in layers))
    return CacheSnapshot(
        tuple(layers),
        DynamicCache,
        "cache_object",
        max(len(heads[0]) for heads in positions),
        adapter.capabilities.cached_key_state,
        independent_layer_lengths=True,
    )
