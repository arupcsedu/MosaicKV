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


def _pool_torch_token_feature_batch_device(
    matrix: Any,
    *,
    positions: tuple[tuple[int, ...], ...],
    heads: tuple[int, ...] | None,
) -> Any:
    """Pool one equal-width block batch without synchronizing it to the host."""

    import torch

    if (
        not positions
        or not positions[0]
        or any(len(item) != len(positions[0]) for item in positions)
    ):
        raise ValueError("graph pooling batches require one nonzero block width")
    sequence_length = int(matrix.shape[0] if heads is None else matrix.shape[1])
    minimum_position = min(min(row) for row in positions)
    maximum_position = max(max(row) for row in positions)
    if minimum_position < 0 or maximum_position >= sequence_length:
        raise ValueError("graph block position lies outside descriptor source tensor")
    position_index = torch.as_tensor(positions, dtype=torch.long, device=matrix.device)
    if heads is None:
        selected = matrix[position_index]
    else:
        if len(heads) != len(positions):
            raise ValueError("graph pooling heads and position rows must align")
        if min(heads) < 0 or max(heads) >= int(matrix.shape[0]):
            raise ValueError("graph block KV head lies outside descriptor source tensor")
        head_index = torch.as_tensor(heads, dtype=torch.long, device=matrix.device)
        selected = matrix[head_index[:, None], position_index]
    return selected.reshape(len(positions), -1, selected.shape[-1]).mean(dim=1)


def _pool_torch_token_feature_batch(
    matrix: Any,
    *,
    positions: tuple[tuple[int, ...], ...],
    heads: tuple[int, ...] | None,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Pool one equal-width block batch on its device, then copy its descriptors."""

    pooled = _pool_torch_token_feature_batch_device(
        matrix,
        positions=positions,
        heads=heads,
    )
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


def _readonly_view(
    matrix: np.ndarray[Any, np.dtype[np.float32]], index: int
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Return an immutable row view while retaining the owning matrix as its base."""

    result = np.asarray(matrix[index], dtype=np.float32)
    result.setflags(write=False)
    return result


def _unit_rows(
    value: np.ndarray[Any, np.dtype[np.float32]],
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Normalize a descriptor matrix in the scalar reference reduction order.

    ``np.linalg.norm(matrix, axis=1)`` is mathematically equivalent to calling
    :func:`_unit` for every row, but NumPy may use a different reduction order.
    The resulting few-ULP drift is enough to violate MosaicKV's deliberately
    strict decision-parity gate.  Descriptor transfer remains layer-batched;
    only this inexpensive host-side reduction follows the reference order.
    """

    source = np.asarray(value, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError("descriptor normalization requires a rank-two matrix")
    result = np.empty_like(source)
    for row_index, row in enumerate(source):
        result[row_index] = _unit(row)
    return np.asarray(result, dtype=np.float32)


def _unit(value: np.ndarray[Any, np.dtype[np.float32]]) -> np.ndarray[Any, np.dtype[np.float32]]:
    norm = float(np.linalg.norm(value))
    result = value.copy() if norm == 0 else value / norm
    result = np.asarray(result, dtype=np.float32)
    result.setflags(write=False)
    return result


def _pool_torch_descriptors_one_transfer(
    full_state: FullKVState,
    *,
    metadata_by_node: Mapping[int, BlockEvidenceMetadata],
    eligible_node_ids: frozenset[int] | None,
    node_table_ids: tuple[int, ...],
) -> tuple[PooledBlockDescriptor, ...]:
    """Pool all torch cache layers, then synchronize descriptor storage once."""

    import torch

    device_batches: dict[tuple[int, int], list[Any]] = defaultdict(list)
    batch_node_ids: dict[tuple[int, int], list[int]] = defaultdict(list)
    feature_width_by_layer: dict[int, tuple[int, int]] = {}
    blocks_by_layer: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for source_node_id in node_table_ids:
        block = full_state.blocks[source_node_id]
        blocks_by_layer[block.layer].append((source_node_id, block))
    for layer_index, layer in enumerate(full_state.layers):
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
        key_width = int(key_tokens.shape[-1])
        value_width = int(value_tokens.shape[-1])
        feature_width_by_layer[layer_index] = (key_width, value_width)
        width_groups: dict[int, list[tuple[int, Any]]] = defaultdict(list)
        for node_id, block in blocks_by_layer[layer_index]:
            if eligible_node_ids is None or node_id in eligible_node_ids:
                width_groups[block.position_count].append((node_id, block))
        signature = (key_width, value_width)
        for group in width_groups.values():
            positions = tuple(block.physical_cache_indices for _node_id, block in group)
            heads = tuple(block.kv_head for _node_id, block in group)
            keys = _pool_torch_token_feature_batch_device(
                key_tokens,
                positions=positions,
                heads=heads,
            )
            values = _pool_torch_token_feature_batch_device(
                value_tokens,
                positions=positions,
                heads=heads,
            )
            device_batches[signature].append(torch.cat((keys, values), dim=-1))
            batch_node_ids[signature].extend(node_id for node_id, _block in group)
    pooled_keys: dict[int, np.ndarray[Any, np.dtype[np.float32]]] = {}
    pooled_values: dict[int, np.ndarray[Any, np.dtype[np.float32]]] = {}
    semantic_by_node: dict[int, np.ndarray[Any, np.dtype[np.float32]]] = {}
    packed_groups: list[Any] = []
    packed_signatures: list[tuple[int, int]] = []
    packed_counts: list[int] = []
    for signature in sorted(device_batches):
        packed = torch.cat(tuple(device_batches[signature]), dim=0)
        packed_groups.append(packed.reshape(-1))
        packed_signatures.append(signature)
        packed_counts.append(int(packed.shape[0]))
    combined = torch.cat(tuple(packed_groups), dim=0)
    host = combined.detach().cpu().numpy().astype(np.float32, copy=False)
    offset = 0
    for signature, count in zip(packed_signatures, packed_counts, strict=True):
        key_width, value_width = signature
        width = key_width + value_width
        scalar_count = count * width
        matrix = host[offset : offset + scalar_count].reshape(count, width)
        offset += scalar_count
        if not bool(np.all(np.isfinite(matrix))):
            raise ValueError("graph descriptor source tensors must be finite")
        key_matrix = matrix[:, :key_width]
        value_matrix = matrix[:, key_width:]
        semantic_matrix = _unit_rows(
            np.concatenate((_unit_rows(key_matrix), _unit_rows(value_matrix)), axis=1)
        )
        identifiers = batch_node_ids[signature]
        if len(identifiers) != count:
            raise RuntimeError("deferred K/V descriptors lost block alignment")
        for row, node_id in enumerate(identifiers):
            pooled_keys[node_id] = _readonly_view(key_matrix, row)
            pooled_values[node_id] = _readonly_view(value_matrix, row)
            semantic_by_node[node_id] = _readonly_view(semantic_matrix, row)
    if offset != len(host):
        raise RuntimeError("deferred descriptor transfer has trailing storage")

    zero_by_layer: dict[
        int,
        tuple[
            np.ndarray[Any, np.dtype[np.float32]],
            np.ndarray[Any, np.dtype[np.float32]],
            np.ndarray[Any, np.dtype[np.float32]],
        ],
    ] = {}
    for layer_index, (key_width, value_width) in feature_width_by_layer.items():
        zero_key = np.zeros(key_width, dtype=np.float32)
        zero_value = np.zeros(value_width, dtype=np.float32)
        zero_semantic = np.zeros(key_width + value_width, dtype=np.float32)
        for value in (zero_key, zero_value, zero_semantic):
            value.setflags(write=False)
        zero_by_layer[layer_index] = zero_key, zero_value, zero_semantic
    derived_evidence_cache: dict[
        tuple[
            object,
            int | None,
            int | None,
            int | None,
            tuple[float, float, float, float] | None,
        ],
        BlockEvidenceMetadata,
    ] = {}
    result: list[PooledBlockDescriptor] = []
    for local_node_id, source_node_id in enumerate(node_table_ids):
        block = full_state.blocks[source_node_id]
        pooled = eligible_node_ids is None or source_node_id in eligible_node_ids
        zero_key, zero_value, zero_semantic = zero_by_layer[block.layer]
        key = pooled_keys[source_node_id] if pooled else zero_key
        value = pooled_values[source_node_id] if pooled else zero_value
        semantic = semantic_by_node[source_node_id] if pooled else zero_semantic
        override = metadata_by_node.get(source_node_id)
        if override is None:
            evidence_identity = (
                block.modality,
                block.image_index,
                block.frame_index,
                block.page_index,
                block.region,
            )
            evidence = derived_evidence_cache.get(evidence_identity)
            if evidence is None:
                evidence = BlockEvidenceMetadata.from_block(block)
                derived_evidence_cache[evidence_identity] = evidence
        else:
            evidence = override.with_block_defaults(block)
        result.append(
            PooledBlockDescriptor._from_validated(
                node_id=local_node_id,
                block=block,
                pooled_key=key,
                pooled_value=value,
                pooled_hidden_state=None,
                semantic_embedding=semantic,
                evidence=evidence,
                source_node_id=source_node_id,
            )
        )
    return tuple(result)


def pool_block_descriptors(
    full_state: FullKVState,
    *,
    hidden_states: Sequence[Any] | None = None,
    hidden_sequence_dimension: int = -2,
    metadata_by_node: Mapping[int, BlockEvidenceMetadata] | None = None,
    host_transfer_mode: str = "deferred_layer",
    eligible_node_ids: Sequence[int] | None = None,
    node_table_ids: Sequence[int] | None = None,
) -> tuple[PooledBlockDescriptor, ...]:
    """Pool K, V, and optional hidden states for every cache block.

    Pooling averages batch, the selected KV head, and source positions while
    flattening any remaining feature axes.  This supports both ``[B,H,S,D]``
    and ``[B,H,D,S]`` cache layouts because the cache state records its axes.
    Hidden states must contain one tensor per layer and normally use
    ``[B,S,D]``.  The resulting descriptors live on CPU in float32 so graph
    construction does not retain accelerator tensors or autograd state.
    """

    if host_transfer_mode not in {
        "deferred_layer",
        "deferred_layer_vectorized",
        "per_group_reference",
    }:
        raise ValueError(
            "host_transfer_mode must be 'deferred_layer', "
            "'deferred_layer_vectorized', or 'per_group_reference'"
        )
    hidden = tuple(hidden_states) if hidden_states is not None else None
    if hidden is not None and len(hidden) != len(full_state.layers):
        raise ValueError("hidden_states must contain exactly one tensor per cache layer")
    overrides = dict(metadata_by_node or {})
    valid_node_ids = set(range(len(full_state.blocks)))
    table_ids = (
        tuple(range(len(full_state.blocks)))
        if node_table_ids is None
        else tuple(int(node_id) for node_id in node_table_ids)
    )
    if len(table_ids) != len(set(table_ids)) or tuple(sorted(table_ids)) != table_ids:
        raise ValueError("graph node table IDs must be unique and source-ordered")
    unknown_table_ids = set(table_ids) - valid_node_ids
    if unknown_table_ids:
        raise ValueError(
            f"graph node table references unknown source node(s): {sorted(unknown_table_ids)}"
        )
    if not table_ids:
        raise ValueError("graph node table cannot be empty")
    unknown = sorted(set(overrides) - valid_node_ids)
    if unknown:
        raise ValueError(f"metadata overrides reference unknown graph node(s): {unknown}")
    eligible = (
        None
        if eligible_node_ids is None
        else frozenset(int(node_id) for node_id in eligible_node_ids)
    )
    if eligible is not None:
        unknown_eligible = eligible - valid_node_ids
        if unknown_eligible:
            raise ValueError(
                f"descriptor shortlist references unknown graph node(s): {sorted(unknown_eligible)}"
            )
        if not eligible:
            raise ValueError("descriptor shortlist cannot be empty")
    if (
        hidden is None
        and host_transfer_mode == "deferred_layer_vectorized"
        and all(
            layer.key.__class__.__module__.startswith("torch")
            and layer.value.__class__.__module__.startswith("torch")
            for layer in full_state.layers
        )
    ):
        return _pool_torch_descriptors_one_transfer(
            full_state,
            metadata_by_node=overrides,
            eligible_node_ids=eligible,
            node_table_ids=table_ids,
        )

    blocks_by_layer: dict[int, list[tuple[int, Any]]] = defaultdict(list)
    for source_node_id in table_ids:
        block = full_state.blocks[source_node_id]
        blocks_by_layer[block.layer].append((source_node_id, block))
    local_by_source = {
        source_node_id: local_node_id for local_node_id, source_node_id in enumerate(table_ids)
    }

    # Stage each layer on CPU once.  The previous block-at-a-time path issued
    # one synchronizing device transfer for every pooled K and V block, which
    # is prohibitive for multi-head VLMs while producing identical values.
    indexed_result: list[tuple[int, PooledBlockDescriptor]] = []
    derived_evidence_cache: dict[
        tuple[
            object,
            int | None,
            int | None,
            int | None,
            tuple[float, float, float, float] | None,
        ],
        BlockEvidenceMetadata,
    ] = {}
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
            if eligible is not None and node_id not in eligible:
                continue
            width_groups[len(block.physical_cache_indices)].append((node_id, block))
        pooled_keys: dict[int, np.ndarray[Any, np.dtype[np.float32]]] = {}
        pooled_values: dict[int, np.ndarray[Any, np.dtype[np.float32]]] = {}
        semantic_by_node: dict[int, np.ndarray[Any, np.dtype[np.float32]]] = {}
        hidden_by_positions: dict[tuple[int, ...], np.ndarray[Any, np.dtype[np.float32]]] = {}
        deferred_kv_batches: list[Any] = []
        deferred_kv_groups: list[list[tuple[int, Any]]] = []
        deferred_key_width: int | None = None
        for group in width_groups.values():
            positions = tuple(block.physical_cache_indices for _node_id, block in group)
            heads = tuple(block.kv_head for _node_id, block in group)
            if torch_sources and host_transfer_mode.startswith("deferred_layer"):
                key_device = _pool_torch_token_feature_batch_device(
                    key_tokens,
                    positions=positions,
                    heads=heads,
                )
                value_device = _pool_torch_token_feature_batch_device(
                    value_tokens,
                    positions=positions,
                    heads=heads,
                )
                key_width = int(key_device.shape[-1])
                if deferred_key_width is None:
                    deferred_key_width = key_width
                elif deferred_key_width != key_width:
                    raise RuntimeError("one cache layer produced inconsistent key descriptors")
                deferred_kv_batches.append((key_device, value_device))
                deferred_kv_groups.append(group)
            else:
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
        if deferred_kv_batches:
            import torch

            key_device = torch.cat(tuple(item[0] for item in deferred_kv_batches), dim=0)
            value_device = torch.cat(tuple(item[1] for item in deferred_kv_batches), dim=0)
            combined = torch.cat((key_device, value_device), dim=-1)
            host = combined.detach().cpu().numpy().astype(np.float32, copy=False)
            key_width = int(deferred_key_width or 0)
            key_matrix = host[:, :key_width]
            value_matrix = host[:, key_width:]
            semantic_matrix = None
            if host_transfer_mode == "deferred_layer_vectorized" and hidden_tokens is None:
                semantic_matrix = _unit_rows(
                    np.concatenate(
                        (_unit_rows(key_matrix), _unit_rows(value_matrix)),
                        axis=1,
                    )
                )
            offset = 0
            for group in deferred_kv_groups:
                for node_id, _block in group:
                    if semantic_matrix is None:
                        pooled_keys[node_id] = _readonly_row(key_matrix, offset)
                        pooled_values[node_id] = _readonly_row(value_matrix, offset)
                    else:
                        pooled_keys[node_id] = _readonly_view(key_matrix, offset)
                        pooled_values[node_id] = _readonly_view(value_matrix, offset)
                        semantic_by_node[node_id] = _readonly_view(semantic_matrix, offset)
                    offset += 1
            if offset != len(host):
                raise RuntimeError("deferred K/V descriptors lost block alignment")
        zero_key = np.zeros(int(key_tokens.shape[-1]), dtype=np.float32)
        zero_value = np.zeros(int(value_tokens.shape[-1]), dtype=np.float32)
        zero_key.setflags(write=False)
        zero_value.setflags(write=False)
        zero_semantic = np.zeros(zero_key.size + zero_value.size, dtype=np.float32)
        zero_semantic.setflags(write=False)
        for node_id, block in layer_blocks:
            pooled = eligible is None or node_id in eligible
            key = pooled_keys[node_id] if pooled else zero_key
            value = pooled_values[node_id] if pooled else zero_value
            pooled_hidden = None
            if hidden_tokens is not None and pooled:
                pooled_hidden = hidden_by_positions[block.physical_cache_indices]
            semantic = semantic_by_node.get(node_id)
            if semantic is None:
                if not pooled:
                    semantic = zero_semantic
                else:
                    semantic_parts = [_unit(key), _unit(value)]
                    if pooled_hidden is not None:
                        semantic_parts.append(_unit(pooled_hidden))
                    semantic = _unit(np.concatenate(semantic_parts).astype(np.float32, copy=False))
            override = overrides.get(node_id)
            if override is None:
                evidence_identity = (
                    block.modality,
                    block.image_index,
                    block.frame_index,
                    block.page_index,
                    block.region,
                )
                evidence = derived_evidence_cache.get(evidence_identity)
                if evidence is None:
                    evidence = BlockEvidenceMetadata.from_block(block)
                    derived_evidence_cache[evidence_identity] = evidence
            else:
                evidence = override.with_block_defaults(block)
            indexed_result.append(
                (
                    node_id,
                    PooledBlockDescriptor._from_validated(
                        node_id=local_by_source[node_id],
                        block=block,
                        pooled_key=key,
                        pooled_value=value,
                        pooled_hidden_state=pooled_hidden,
                        semantic_embedding=semantic,
                        evidence=evidence,
                        source_node_id=node_id,
                    ),
                )
            )
    indexed_result.sort(key=lambda item: item[0])
    if tuple(item[0] for item in indexed_result) != table_ids:
        raise RuntimeError("pooled graph descriptors do not cover every cache block")
    return tuple(item[1] for item in indexed_result)


def _pool_attention_block_batch(grouped, *, positions, heads):
    """Gather only requested (head,query,key) coordinates, not B*Q*all_keys.

    A block-size-one graph must not replicate its entire key axis for every
    block before gathering a single position. Preserve the original contiguous
    selected tensor and mean reduction, including repeated-head batches.
    """
    import torch

    if (
        grouped.ndim != 3
        or not positions
        or not positions[0]
        or len(positions) != len(heads)
        or len({len(p) for p in positions}) != 1
        or min(min(p) for p in positions) < 0
        or max(max(p) for p in positions) >= grouped.shape[-1]
        or min(heads) < 0
        or max(heads) >= grouped.shape[0]
    ):
        raise ValueError("valid equal-width prompt-attention block coordinates required")
    p = torch.as_tensor(positions, dtype=torch.long, device=grouped.device)
    h = torch.as_tensor(heads, dtype=torch.long, device=grouped.device)
    q = torch.arange(grouped.shape[1], device=grouped.device)
    selected = grouped[h[:, None, None], q[None, :, None], p[:, None, :]]
    return selected.mean(dim=2)


def pool_prompt_attention_coactivation(
    full_state: FullKVState,
    prompt_attentions: Sequence[Any],
    *,
    host_transfer_mode: str = "deferred_layer",
    eligible_node_ids: Sequence[int] | None = None,
    query_window: int | None = None,
) -> dict[int, np.ndarray[Any, np.dtype[np.float32]]]:
    """Pool prompt attention maps into one co-activation signature per block.

    Each layer must use ``[batch, query_heads, query_positions, key_positions]``
    or ``[query_heads, query_positions, key_positions]``.  Grouped-query heads
    are deterministically assigned to their corresponding KV head, then batch,
    query-head, and block-key axes are averaged.  The retained query-position
    vector is the block's prompt-side co-activation signature.
    """

    if host_transfer_mode not in {
        "deferred_layer",
        "deferred_layer_vectorized",
        "per_group_reference",
    }:
        raise ValueError(
            "host_transfer_mode must be 'deferred_layer', "
            "'deferred_layer_vectorized', or 'per_group_reference'"
        )
    if query_window is not None and query_window < 1:
        raise ValueError("prompt attention co-activation query_window must be positive")
    attention_layers = tuple(prompt_attentions)
    if len(attention_layers) != len(full_state.layers):
        raise ValueError("prompt_attentions must contain exactly one tensor per cache layer")
    eligible = (
        None
        if eligible_node_ids is None
        else frozenset(int(node_id) for node_id in eligible_node_ids)
    )
    if eligible is not None:
        unknown = eligible - set(range(len(full_state.blocks)))
        if unknown:
            raise ValueError(
                f"attention co-activation shortlist contains unknown nodes: {sorted(unknown)}"
            )
    result: dict[int, np.ndarray[Any, np.dtype[np.float32]]] = {}
    global_deferred_batches: list[Any] = []
    global_deferred_node_ids: list[int] = []
    global_deferred_minima: list[Any] = []
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
            if query_window is not None:
                tensor = tensor[..., -min(query_window, int(tensor.shape[-2])) :, :]
            query_heads = int(tensor.shape[1])
            if query_heads % layer.kv_heads:
                raise ValueError("prompt query-head count must be divisible by KV-head count")
            if int(tensor.shape[-1]) < layer.sequence_length:
                raise ValueError("prompt attention key axis is shorter than the active KV cache")
            query_heads_per_kv = query_heads // layer.kv_heads
            grouped = tensor.reshape(
                int(tensor.shape[0]),
                layer.kv_heads,
                query_heads_per_kv,
                int(tensor.shape[-2]),
                int(tensor.shape[-1]),
            ).mean(dim=(0, 2))
            by_width: dict[int, list[tuple[int, Any]]] = defaultdict(list)
            for node_id, block in enumerate(full_state.blocks):
                if block.layer == layer_index and (eligible is None or node_id in eligible):
                    by_width[block.position_count].append((node_id, block))
            deferred_batches: list[Any] = []
            deferred_node_ids: list[int] = []
            for _width, indexed_blocks in by_width.items():
                pooled = _pool_attention_block_batch(
                    grouped,
                    positions=tuple(
                        block.physical_cache_indices for _node_id, block in indexed_blocks
                    ),
                    heads=tuple(block.kv_head for _node_id, block in indexed_blocks),
                )
                if host_transfer_mode.startswith("deferred_layer"):
                    # Preserve the original independent per-head reduction,
                    # but defer its synchronizing device-to-host copy. Long
                    # VLM prompts otherwise issue one CPU synchronization per
                    # KV-head/block-width group (often thousands per sample).
                    deferred_batches.append(pooled)
                    deferred_node_ids.extend(node_id for node_id, _block in indexed_blocks)
                else:
                    host = pooled.detach().cpu().numpy().astype(np.float32, copy=False)
                    for row, (node_id, _block) in enumerate(indexed_blocks):
                        result[node_id] = _readonly_row(host, row)
            if deferred_batches:
                global_deferred_batches.append(torch.cat(tuple(deferred_batches), dim=0))
                global_deferred_node_ids.extend(deferred_node_ids)
                global_deferred_minima.append(tensor.amin())
            continue
        array = _as_numpy_float(attention)
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4:
            raise ValueError("prompt attention must have shape [batch, query_heads, queries, keys]")
        if query_window is not None:
            array = array[..., -min(query_window, int(array.shape[-2])) :, :]
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
            if block.layer != layer_index or (eligible is not None and node_id not in eligible):
                continue
            query_start = block.kv_head * query_heads_per_kv
            query_end = query_start + query_heads_per_kv
            selected_numpy = np.take(
                array[:, query_start:query_end, :, :],
                block.physical_cache_indices,
                axis=-1,
            )
            pooled = selected_numpy.mean(axis=(0, 1, 3), dtype=np.float32)
            signature = np.asarray(pooled, dtype=np.float32)
            signature.setflags(write=False)
            result[node_id] = signature
    if global_deferred_batches:
        import torch

        deferred = torch.cat(tuple(global_deferred_batches), dim=0)
        scalar_count = int(deferred.numel())
        combined = torch.cat(
            (deferred.reshape(-1), torch.stack(global_deferred_minima).reshape(-1))
        )
        host = combined.detach().cpu().numpy().astype(np.float32, copy=False)
        descriptor_width = int(deferred.shape[-1])
        descriptor_host = host[:scalar_count].reshape(-1, descriptor_width)
        if len(descriptor_host) != len(global_deferred_node_ids):
            raise RuntimeError("deferred attention descriptors lost block alignment")
        if bool(np.any(host[scalar_count:] < 0)):
            raise ValueError(
                "prompt attention co-activation requires nonnegative attention weights"
            )
        for row, node_id in enumerate(global_deferred_node_ids):
            result[node_id] = _readonly_row(descriptor_host, row)
    return result


__all__ = ["pool_block_descriptors", "pool_prompt_attention_coactivation"]
