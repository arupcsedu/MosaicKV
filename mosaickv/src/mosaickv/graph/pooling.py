"""Block-level descriptor pooling without assuming one KV tensor layout."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from mosaickv.cache_state import FullKVState
from mosaickv.graph.types import BlockEvidenceMetadata, PooledBlockDescriptor


def _as_numpy_float(value: Any) -> np.ndarray[Any, np.dtype[np.float32]]:
    if type(value).__module__.startswith("torch"):
        value = value.detach().float().cpu().numpy()
    result = np.asarray(value, dtype=np.float32)
    if not bool(np.all(np.isfinite(result))):
        raise ValueError("graph descriptor source tensors must be finite")
    return result


def _normalize_axis(axis: int, rank: int, name: str) -> int:
    normalized = axis if axis >= 0 else rank + axis
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"{name} axis {axis} is invalid for rank {rank}")
    return normalized


def _mean_feature_vector(
    value: Any,
    *,
    positions: tuple[int, ...],
    sequence_axis: int,
    head_axis: int | None,
    head: int | None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    is_torch = type(value).__module__.startswith("torch")
    # ``pool_block_descriptors`` validates and stages NumPy inputs once per
    # layer.  Revalidating the entire layer for every block would turn sparse
    # pooling into quadratic host work.
    array: Any = value.detach().float() if is_torch else np.asarray(value, dtype=np.float32)
    rank = int(array.ndim)
    sequence = _normalize_axis(sequence_axis, rank, "sequence")
    if max(positions) >= array.shape[sequence]:
        raise ValueError("graph block position lies outside descriptor source tensor")
    if is_torch:
        import torch

        position_index = torch.tensor(positions, dtype=torch.long, device=array.device)
        selected: Any = torch.index_select(array, sequence, position_index)
    else:
        selected = np.take(array, positions, axis=sequence)
    reductions = {sequence}
    if head_axis is not None:
        normalized_head = _normalize_axis(head_axis, rank, "head")
        if normalized_head == sequence:
            raise ValueError("head and sequence axes must differ")
        if head is None or head < 0 or head >= array.shape[normalized_head]:
            raise ValueError("graph block KV head lies outside descriptor source tensor")
        if is_torch:
            import torch

            head_index = torch.tensor((head,), dtype=torch.long, device=array.device)
            selected = torch.index_select(selected, normalized_head, head_index)
        else:
            selected = np.take(selected, (head,), axis=normalized_head)
        reductions.add(normalized_head)
    # Cache and hidden-state tensors conventionally use axis zero for batch.  It
    # is reduced when it is not already the head or sequence axis.
    if rank >= 3 and 0 not in reductions:
        reductions.add(0)
    feature_axes = tuple(axis for axis in range(rank) if axis not in reductions)
    reduction_axes = tuple(sorted(reductions))
    axes = (*reduction_axes, *feature_axes)
    transposed: Any = selected.permute(axes) if is_torch else np.transpose(selected, axes)
    reduction_size = int(np.prod([selected.shape[axis] for axis in reduction_axes]))
    feature_size = int(np.prod([selected.shape[axis] for axis in feature_axes])) or 1
    if is_torch:
        pooled = transposed.reshape(reduction_size, feature_size).mean(dim=0).cpu().numpy()
    else:
        pooled = transposed.reshape(reduction_size, feature_size).mean(axis=0, dtype=np.float32)
    result = np.asarray(pooled, dtype=np.float32)
    result.setflags(write=False)
    return result


def _unit(value: np.ndarray[Any, np.dtype[np.float32]]) -> np.ndarray[Any, np.dtype[np.float32]]:
    norm = float(np.linalg.norm(value))
    result = value.copy() if norm == 0 else value / norm
    result = np.asarray(result, dtype=np.float32)
    result.setflags(write=False)
    return result


def pool_block_descriptors(
    full_state: FullKVState,
    *,
    hidden_states: Sequence[Any] | None = None,
    hidden_sequence_dimension: int = -2,
    metadata_by_node: Mapping[int, BlockEvidenceMetadata] | None = None,
) -> tuple[PooledBlockDescriptor, ...]:
    """Pool K, V, and optional hidden states for every cache block.

    Pooling averages batch, the selected KV head, and source positions while
    flattening any remaining feature axes.  This supports both ``[B,H,S,D]``
    and ``[B,H,D,S]`` cache layouts because the cache state records its axes.
    Hidden states must contain one tensor per layer and normally use
    ``[B,S,D]``.  The resulting descriptors live on CPU in float32 so graph
    construction does not retain accelerator tensors or autograd state.
    """

    hidden = tuple(hidden_states) if hidden_states is not None else None
    if hidden is not None and len(hidden) != len(full_state.layers):
        raise ValueError("hidden_states must contain exactly one tensor per cache layer")
    overrides = dict(metadata_by_node or {})
    valid_node_ids = set(range(len(full_state.blocks)))
    unknown = sorted(set(overrides) - valid_node_ids)
    if unknown:
        raise ValueError(f"metadata overrides reference unknown graph node(s): {unknown}")

    blocks_by_layer: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for node_id, block in enumerate(full_state.blocks):
        blocks_by_layer[block.layer].append((node_id, block))

    # Stage each layer on CPU once.  The previous block-at-a-time path issued
    # one synchronizing device transfer for every pooled K and V block, which
    # is prohibitive for multi-head VLMs while producing identical values.
    indexed_result: list[tuple[int, PooledBlockDescriptor]] = []
    for layer_index, layer in enumerate(full_state.layers):
        key_source = _as_numpy_float(layer.key)
        value_source = _as_numpy_float(layer.value)
        hidden_source = None if hidden is None else _as_numpy_float(hidden[layer_index])
        for node_id, block in blocks_by_layer[layer_index]:
            key = _mean_feature_vector(
                key_source,
                positions=block.physical_cache_indices,
                sequence_axis=layer.key_sequence_dimension,
                head_axis=layer.key_head_dimension,
                head=block.kv_head,
            )
            value = _mean_feature_vector(
                value_source,
                positions=block.physical_cache_indices,
                sequence_axis=layer.value_sequence_dimension,
                head_axis=layer.value_head_dimension,
                head=block.kv_head,
            )
            pooled_hidden = (
                None
                if hidden_source is None
                else _mean_feature_vector(
                    hidden_source,
                    positions=block.physical_cache_indices,
                    sequence_axis=hidden_sequence_dimension,
                    head_axis=None,
                    head=None,
                )
            )
            semantic_parts = [_unit(key), _unit(value)]
            if pooled_hidden is not None:
                semantic_parts.append(_unit(pooled_hidden))
            semantic = _unit(np.concatenate(semantic_parts).astype(np.float32, copy=False))
            evidence = overrides.get(node_id, BlockEvidenceMetadata()).with_block_defaults(block)
            indexed_result.append(
                (
                    node_id,
                    PooledBlockDescriptor(
                        node_id=node_id,
                        block=block,
                        pooled_key=key,
                        pooled_value=value,
                        pooled_hidden_state=pooled_hidden,
                        semantic_embedding=semantic,
                        evidence=evidence,
                    ),
                )
            )
    indexed_result.sort(key=lambda item: item[0])
    if tuple(item[0] for item in indexed_result) != tuple(range(len(full_state.blocks))):
        raise RuntimeError("pooled graph descriptors do not cover every cache block")
    return tuple(item[1] for item in indexed_result)


def pool_prompt_attention_coactivation(
    full_state: FullKVState,
    prompt_attentions: Sequence[Any],
) -> dict[int, np.ndarray[Any, np.dtype[np.float32]]]:
    """Pool prompt attention maps into one co-activation signature per block.

    Each layer must use ``[batch, query_heads, query_positions, key_positions]``
    or ``[query_heads, query_positions, key_positions]``.  Grouped-query heads
    are deterministically assigned to their corresponding KV head, then batch,
    query-head, and block-key axes are averaged.  The retained query-position
    vector is the block's prompt-side co-activation signature.
    """

    attention_layers = tuple(prompt_attentions)
    if len(attention_layers) != len(full_state.layers):
        raise ValueError("prompt_attentions must contain exactly one tensor per cache layer")
    result: dict[int, np.ndarray[Any, np.dtype[np.float32]]] = {}
    for layer_index, (attention, layer) in enumerate(
        zip(attention_layers, full_state.layers, strict=True)
    ):
        array = _as_numpy_float(attention)
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4:
            raise ValueError("prompt attention must have shape [batch, query_heads, queries, keys]")
        query_heads = int(array.shape[1])
        if query_heads % layer.kv_heads:
            raise ValueError("prompt query-head count must be divisible by KV-head count")
        if array.shape[-1] < layer.sequence_length:
            raise ValueError("prompt attention key axis is shorter than the active KV cache")
        if bool(np.any(array < 0)):
            raise ValueError(
                "prompt attention co-activation requires nonnegative attention weights"
            )
        query_heads_per_kv = query_heads // layer.kv_heads
        for node_id, block in enumerate(full_state.blocks):
            if block.layer != layer_index:
                continue
            query_start = block.kv_head * query_heads_per_kv
            query_end = query_start + query_heads_per_kv
            selected = np.take(
                array[:, query_start:query_end, :, :],
                block.physical_cache_indices,
                axis=-1,
            )
            pooled = selected.mean(axis=(0, 1, 3), dtype=np.float32)
            signature = np.asarray(pooled, dtype=np.float32)
            signature.setflags(write=False)
            result[node_id] = signature
    return result


__all__ = ["pool_block_descriptors", "pool_prompt_attention_coactivation"]
