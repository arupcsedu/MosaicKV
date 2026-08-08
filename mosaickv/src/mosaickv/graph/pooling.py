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


def _token_feature_matrix(
    value: np.ndarray[Any, np.dtype[np.float32]],
    *,
    sequence_axis: int,
    head_axis: int | None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Canonicalize one staged layer without copying once per cache block.

    The returned shape is ``[heads, sequence, reductions, features]`` when a
    head axis exists and ``[sequence, reductions, features]`` otherwise.
    Batch is retained in ``reductions`` so block pooling performs the same
    joint batch/position mean as the scalar reference implementation.
    """

    array = np.asarray(value, dtype=np.float32)
    rank = int(array.ndim)
    sequence = _normalize_axis(sequence_axis, rank, "sequence")
    head = None if head_axis is None else _normalize_axis(head_axis, rank, "head")
    if head == sequence:
        raise ValueError("head and sequence axes must differ")
    reductions = () if rank < 3 or 0 in {sequence, head} else (0,)
    excluded = {sequence, *reductions}
    if head is not None:
        excluded.add(head)
    features = tuple(axis for axis in range(rank) if axis not in excluded)
    leading = (sequence,) if head is None else (head, sequence)
    transposed = np.transpose(array, (*leading, *reductions, *features))
    reduction_size = int(np.prod([array.shape[axis] for axis in reductions])) or 1
    feature_size = int(np.prod([array.shape[axis] for axis in features])) or 1
    if head is None:
        return np.asarray(
            transposed.reshape(array.shape[sequence], reduction_size, feature_size),
            dtype=np.float32,
        )
    return np.asarray(
        transposed.reshape(
            array.shape[head],
            array.shape[sequence],
            reduction_size,
            feature_size,
        ),
        dtype=np.float32,
    )


def _torch_token_feature_matrix(
    value: Any,
    *,
    sequence_axis: int,
    head_axis: int | None,
) -> Any:
    """Canonicalize a torch layer without staging the full KV tensor on CPU."""

    array = value.detach().float()
    rank = int(array.ndim)
    sequence = _normalize_axis(sequence_axis, rank, "sequence")
    head = None if head_axis is None else _normalize_axis(head_axis, rank, "head")
    if head == sequence:
        raise ValueError("head and sequence axes must differ")
    reductions = () if rank < 3 or 0 in {sequence, head} else (0,)
    excluded = {sequence, *reductions}
    if head is not None:
        excluded.add(head)
    features = tuple(axis for axis in range(rank) if axis not in excluded)
    leading = (sequence,) if head is None else (head, sequence)
    transposed = array.permute((*leading, *reductions, *features))
    reduction_size = int(np.prod([array.shape[axis] for axis in reductions])) or 1
    feature_size = int(np.prod([array.shape[axis] for axis in features])) or 1
    if head is None:
        return transposed.reshape(array.shape[sequence], reduction_size, feature_size)
    return transposed.reshape(
        array.shape[head],
        array.shape[sequence],
        reduction_size,
        feature_size,
    )


def _pool_torch_token_feature_batch(
    matrix: Any,
    *,
    positions: tuple[tuple[int, ...], ...],
    heads: tuple[int, ...] | None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Pool one equal-width block batch on its source device, then copy only descriptors."""

    import torch

    if (
        not positions
        or not positions[0]
        or any(len(item) != len(positions[0]) for item in positions)
    ):
        raise ValueError("graph pooling batches require one nonzero block width")
    position_index = torch.as_tensor(positions, dtype=torch.long, device=matrix.device)
    sequence_length = int(matrix.shape[0] if heads is None else matrix.shape[1])
    if int(position_index.min()) < 0 or int(position_index.max()) >= sequence_length:
        raise ValueError("graph block position lies outside descriptor source tensor")
    if heads is None:
        selected = matrix[position_index]
    else:
        if len(heads) != len(positions):
            raise ValueError("graph pooling heads and position rows must align")
        head_index = torch.as_tensor(heads, dtype=torch.long, device=matrix.device)
        if int(head_index.min()) < 0 or int(head_index.max()) >= int(matrix.shape[0]):
            raise ValueError("graph block KV head lies outside descriptor source tensor")
        selected = matrix[head_index[:, None], position_index]
    pooled = selected.reshape(len(positions), -1, selected.shape[-1]).mean(dim=1)
    result: np.ndarray[Any, np.dtype[np.float32]] = (
        pooled.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    if not bool(np.all(np.isfinite(result))):
        raise ValueError("graph descriptor source tensors must be finite")
    return result


def _pool_token_features(
    matrix: np.ndarray[Any, np.dtype[np.float32]],
    *,
    positions: tuple[int, ...],
    head: int | None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    sequence_length = int(matrix.shape[0] if head is None else matrix.shape[1])
    if max(positions) >= sequence_length:
        raise ValueError("graph block position lies outside descriptor source tensor")
    if head is not None and (head < 0 or head >= matrix.shape[0]):
        raise ValueError("graph block KV head lies outside descriptor source tensor")
    source = matrix if head is None else matrix[head]
    selected = source[np.asarray(positions, dtype=np.intp)]
    pooled = selected.reshape(-1, selected.shape[-1]).mean(axis=0, dtype=np.float32)
    result = np.asarray(pooled, dtype=np.float32)
    result.setflags(write=False)
    return result


def _pool_token_feature_batch(
    matrix: np.ndarray[Any, np.dtype[np.float32]],
    *,
    positions: tuple[tuple[int, ...], ...],
    heads: tuple[int, ...] | None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Pool an equal-width block batch with the scalar reduction order.

    Advanced indexing materializes ``[blocks, positions, reductions, features]``.
    Reducing the flattened middle dimensions therefore performs the same
    per-block float32 mean as :func:`_pool_token_features`, while avoiding one
    Python/NumPy dispatch per cache block.  Callers group variable-width tail
    blocks separately.
    """

    if not positions:
        raise ValueError("graph pooling batch cannot be empty")
    width = len(positions[0])
    if width == 0 or any(len(item) != width for item in positions):
        raise ValueError("graph pooling batches require one nonzero block width")
    position_index = np.asarray(positions, dtype=np.intp)
    if bool(np.any(position_index < 0)):
        raise ValueError("graph block positions must be nonnegative")
    sequence_length = int(matrix.shape[0] if heads is None else matrix.shape[1])
    if int(position_index.max()) >= sequence_length:
        raise ValueError("graph block position lies outside descriptor source tensor")
    if heads is None:
        selected = matrix[position_index]
    else:
        if len(heads) != len(positions):
            raise ValueError("graph pooling heads and position rows must align")
        head_index = np.asarray(heads, dtype=np.intp)
        if bool(np.any(head_index < 0)) or int(head_index.max()) >= matrix.shape[0]:
            raise ValueError("graph block KV head lies outside descriptor source tensor")
        selected = matrix[head_index[:, None], position_index]
    pooled = selected.reshape(len(positions), -1, selected.shape[-1]).mean(
        axis=1,
        dtype=np.float32,
    )
    return np.asarray(pooled, dtype=np.float32)


def _readonly_row(
    matrix: np.ndarray[Any, np.dtype[np.float32]], index: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Return one immutable, independently owned descriptor vector."""

    result = np.asarray(matrix[index], dtype=np.float32).copy()
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
        torch_sources = layer.key.__class__.__module__.startswith("torch") and (
            layer.value.__class__.__module__.startswith("torch")
        )
        if torch_sources:
            key_tokens = _torch_token_feature_matrix(
                layer.key,
                sequence_axis=layer.key_sequence_dimension,
                head_axis=layer.key_head_dimension,
            )
            value_tokens = _torch_token_feature_matrix(
                layer.value,
                sequence_axis=layer.value_sequence_dimension,
                head_axis=layer.value_head_dimension,
            )
        else:
            key_tokens = _token_feature_matrix(
                _as_numpy_float(layer.key),
                sequence_axis=layer.key_sequence_dimension,
                head_axis=layer.key_head_dimension,
            )
            value_tokens = _token_feature_matrix(
                _as_numpy_float(layer.value),
                sequence_axis=layer.value_sequence_dimension,
                head_axis=layer.value_head_dimension,
            )
        hidden_source = None if hidden is None else hidden[layer_index]
        hidden_tokens = (
            None
            if hidden_source is None
            else (
                _torch_token_feature_matrix(
                    hidden_source,
                    sequence_axis=hidden_sequence_dimension,
                    head_axis=None,
                )
                if hidden_source.__class__.__module__.startswith("torch")
                else _token_feature_matrix(
                    _as_numpy_float(hidden_source),
                    sequence_axis=hidden_sequence_dimension,
                    head_axis=None,
                )
            )
        )
        layer_blocks = blocks_by_layer[layer_index]
        width_groups: dict[int, list[tuple[int, Any]]] = defaultdict(list)
        for node_id, block in layer_blocks:
            width_groups[len(block.physical_cache_indices)].append((node_id, block))
        pooled_keys: dict[int, np.ndarray[Any, np.dtype[np.float32]]] = {}
        pooled_values: dict[int, np.ndarray[Any, np.dtype[np.float32]]] = {}
        hidden_by_positions: dict[tuple[int, ...], np.ndarray[Any, np.dtype[np.float32]]] = {}
        for group in width_groups.values():
            positions = tuple(block.physical_cache_indices for _node_id, block in group)
            heads = tuple(block.kv_head for _node_id, block in group)
            batch_pool = (
                _pool_torch_token_feature_batch if torch_sources else _pool_token_feature_batch
            )
            key_batch = batch_pool(key_tokens, positions=positions, heads=heads)
            value_batch = batch_pool(value_tokens, positions=positions, heads=heads)
            for batch_index, (node_id, _block) in enumerate(group):
                pooled_keys[node_id] = _readonly_row(key_batch, batch_index)
                pooled_values[node_id] = _readonly_row(value_batch, batch_index)
            if hidden_tokens is not None:
                unique_positions = tuple(dict.fromkeys(positions))
                hidden_batch = (
                    _pool_torch_token_feature_batch
                    if hidden_tokens.__class__.__module__.startswith("torch")
                    else _pool_token_feature_batch
                )(hidden_tokens, positions=unique_positions, heads=None)
                for batch_index, item_positions in enumerate(unique_positions):
                    hidden_by_positions[item_positions] = _readonly_row(
                        hidden_batch,
                        batch_index,
                    )
        for node_id, block in layer_blocks:
            key = pooled_keys[node_id]
            value = pooled_values[node_id]
            pooled_hidden = None
            if hidden_tokens is not None:
                pooled_hidden = hidden_by_positions[block.physical_cache_indices]
            semantic_parts = [_unit(key), _unit(value)]
            if pooled_hidden is not None:
                semantic_parts.append(_unit(pooled_hidden))
            semantic = _unit(np.concatenate(semantic_parts).astype(np.float32, copy=False))
            override = overrides.get(node_id)
            evidence = (
                BlockEvidenceMetadata.from_block(block)
                if override is None
                else override.with_block_defaults(block)
            )
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
        if attention.__class__.__module__.startswith("torch"):
            import torch

            tensor = attention.detach().float()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4:
                raise ValueError(
                    "prompt attention must have shape [batch, query_heads, queries, keys]"
                )
            query_heads = int(tensor.shape[1])
            if query_heads % layer.kv_heads:
                raise ValueError("prompt query-head count must be divisible by KV-head count")
            if int(tensor.shape[-1]) < layer.sequence_length:
                raise ValueError("prompt attention key axis is shorter than the active KV cache")
            if bool(torch.any(tensor < 0)):
                raise ValueError(
                    "prompt attention co-activation requires nonnegative attention weights"
                )
            query_heads_per_kv = query_heads // layer.kv_heads
            by_head_width: dict[tuple[int, int], list[tuple[int, Any]]] = defaultdict(list)
            for node_id, block in enumerate(full_state.blocks):
                if block.layer == layer_index:
                    by_head_width[(block.kv_head, block.position_count)].append((node_id, block))
            for (kv_head, _width), indexed_blocks in by_head_width.items():
                positions = torch.as_tensor(
                    tuple(block.physical_cache_indices for _node_id, block in indexed_blocks),
                    dtype=torch.long,
                    device=tensor.device,
                )
                query_start = kv_head * query_heads_per_kv
                query_end = query_start + query_heads_per_kv
                source = tensor[:, query_start:query_end, :, :]
                selected = source[..., positions]
                pooled = selected.mean(dim=(0, 1, 4)).transpose(0, 1)
                host = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
                for row, (node_id, _block) in enumerate(indexed_blocks):
                    result[node_id] = _readonly_row(host, row)
            continue
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
