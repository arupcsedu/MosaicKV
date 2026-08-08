"""Lossless algebraic downdates for partially repaired prototypes."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mosaickv.cache_state import FullKVState, tensor_storage_bytes
from mosaickv.prototypes.rope import RotaryPrototypeKeyTransform
from mosaickv.prototypes.types import (
    PrototypeDiagnostics,
    PrototypeMember,
    PrototypeRecord,
)


class PrototypeRefreshError(RuntimeError):
    """Raised when a partial repair cannot preserve the remaining prototype."""


@dataclass(frozen=True, slots=True)
class RefreshedPrototype:
    """Updated record and cache-ready payload after removing promoted members."""

    record: PrototypeRecord
    key: Any
    value: Any


def _fp32(value: Any) -> Any:
    if value.__class__.__module__.startswith("torch") and hasattr(value, "float"):
        return value.detach().float()
    return np.asarray(value, dtype=np.float32)


def _pool(value: Any, axis: int, position_weights: tuple[float, ...]) -> Any:
    fp32 = _fp32(value)
    normalized_axis = axis % int(fp32.ndim)
    if len(position_weights) != int(fp32.shape[normalized_axis]):
        raise PrototypeRefreshError(
            "prototype refresh position weights do not match the source block"
        )
    if fp32.__class__.__module__.startswith("torch"):
        import torch

        shape = [1] * int(fp32.ndim)
        shape[normalized_axis] = len(position_weights)
        torch_weights = torch.as_tensor(
            position_weights,
            dtype=fp32.dtype,
            device=fp32.device,
        ).reshape(shape)
        return torch.sum(fp32 * torch_weights, dim=normalized_axis, keepdim=True)
    shape = [1] * int(fp32.ndim)
    shape[normalized_axis] = len(position_weights)
    numpy_weights = np.asarray(position_weights, dtype=np.float32).reshape(shape)
    return np.sum(
        fp32 * numpy_weights,
        axis=normalized_axis,
        keepdims=True,
        dtype=np.float32,
    )


def _cast_like(value: Any, reference: Any) -> Any:
    if value.__class__.__module__.startswith("torch") and hasattr(value, "to"):
        return value.to(dtype=reference.dtype, device=reference.device)
    return np.asarray(value).astype(np.asarray(reference).dtype, copy=True)


def _mse(first: Any, second: Any) -> float:
    squared = (first - second) * (first - second)
    mean = squared.mean()
    return max(0.0, float(mean.item()) if hasattr(mean, "item") else float(mean))


def _subtract_members(old_mean: Any, members: tuple[tuple[float, Any], ...]) -> Any:
    result = old_mean
    for weight, value in members:
        result = result - value * weight
    return result


def _remaining_dispersion(
    old_dispersion: float,
    old_mean: Any,
    new_mean: Any,
    removed: tuple[tuple[float, Any], ...],
    remaining_weight: float,
) -> float:
    removed_about_old = sum(weight * _mse(value, old_mean) for weight, value in removed)
    remaining_about_old = max(0.0, old_dispersion - removed_about_old)
    result = remaining_about_old / remaining_weight - _mse(new_mean, old_mean)
    return max(0.0, result)


def refresh_prototype_after_promotions(
    full_state: FullKVState,
    record: PrototypeRecord,
    prototype_key: Any,
    prototype_value: Any,
    promoted_payloads: Mapping[int, tuple[Any, Any]],
    *,
    key_transform: RotaryPrototypeKeyTransform | None = None,
) -> RefreshedPrototype:
    """Remove promoted members while retaining an exact prototype of the remainder.

    The old weighted mean and promoted source payloads are sufficient for an
    algebraic downdate; unpromoted residual blocks never return to the GPU. For
    post-RoPE keys, the audited construction transform is required so
    subtraction happens in phase-neutral space.
    """

    promoted_ids = frozenset(int(node_id) for node_id in promoted_payloads)
    member_by_id = {member.node_id: member for member in record.members}
    if not promoted_ids or not promoted_ids < set(member_by_id):
        raise PrototypeRefreshError(
            "prototype refresh requires a non-empty strict subset of active members"
        )
    remaining_members = tuple(
        member for member in record.members if member.node_id not in promoted_ids
    )
    remaining_weight = sum(member.normalized_weight for member in remaining_members)
    if not math.isfinite(remaining_weight) or remaining_weight <= 0:
        raise PrototypeRefreshError("partial prototype has no positive remaining weight")

    layer = full_state.layers[record.layer]
    key_state = str(getattr(full_state.cached_key_state, "value", full_state.cached_key_state))
    old_key_mean = _fp32(prototype_key)
    if key_state == "post_rope":
        if key_transform is None:
            raise PrototypeRefreshError(
                "post-RoPE partial prototype refresh requires the audited key transform"
            )
        old_key_mean = key_transform.remove_anchor_phase(
            prototype_key,
            record.anchor_logical_position,
        )
    elif key_transform is not None:
        raise PrototypeRefreshError(
            "a rotary key transform cannot refresh a phase-neutral prototype"
        )

    removed_keys: list[tuple[float, Any]] = []
    removed_values: list[tuple[float, Any]] = []
    for node_id in sorted(promoted_ids):
        member = member_by_id[node_id]
        key, value = promoted_payloads[node_id]
        block = full_state.blocks[node_id]
        pooled_key = (
            key_transform.phase_neutral_pool(
                key,
                block,
                sequence_dimension=layer.key_sequence_dimension,
                position_weights=member.position_weights,
            )
            if key_transform is not None
            else _pool(key, layer.key_sequence_dimension, member.position_weights)
        )
        removed_keys.append((member.normalized_weight, pooled_key))
        removed_values.append(
            (
                member.normalized_weight,
                _pool(value, layer.value_sequence_dimension, member.position_weights),
            )
        )

    old_value_mean = _fp32(prototype_value)
    key_members = tuple(removed_keys)
    value_members = tuple(removed_values)
    new_key_mean = _subtract_members(old_key_mean, key_members) / remaining_weight
    new_value_mean = _subtract_members(old_value_mean, value_members) / remaining_weight
    normalized_members = tuple(
        PrototypeMember(
            node_id=member.node_id,
            raw_weight=member.raw_weight,
            normalized_weight=member.normalized_weight / remaining_weight,
            reconstruction_error=member.reconstruction_error,
            value_output_norm=member.value_output_norm,
            position_weights=member.position_weights,
        )
        for member in remaining_members
    )
    blocks = tuple(full_state.blocks[member.node_id] for member in normalized_members)
    # The cache slot and its logical/RoPE position are an immutable identity.
    # A weighted-medoid prototype chooses that identity once during prefill.
    # Removing members may change the statistical medoid, but moving an active
    # cache entry would invalidate the packed layout and the next decode mask.
    refreshed_anchor_position = record.anchor_logical_position
    refreshed_key = (
        key_transform.apply_anchor_phase(
            new_key_mean,
            refreshed_anchor_position,
            prototype_key,
        )
        if key_transform is not None
        else _cast_like(new_key_mean, prototype_key)
    )
    refreshed_value = _cast_like(new_value_mean, prototype_value)

    positions = tuple(position for block in blocks for position in block.original_logical_positions)
    modalities = Counter(block.modality for block in blocks)
    source_bytes = sum(block.byte_size for block in blocks)
    prototype_bytes = tensor_storage_bytes(refreshed_key) + tensor_storage_bytes(refreshed_value)
    diagnostics = PrototypeDiagnostics(
        cluster_size=len(normalized_members),
        key_dispersion=_remaining_dispersion(
            record.diagnostics.key_dispersion,
            old_key_mean,
            new_key_mean,
            key_members,
            remaining_weight,
        ),
        value_dispersion=_remaining_dispersion(
            record.diagnostics.value_dispersion,
            old_value_mean,
            new_value_mean,
            value_members,
            remaining_weight,
        ),
        modality_composition=tuple(sorted(modalities.items(), key=lambda item: item[0].value)),
        minimum_logical_position=min(positions),
        maximum_logical_position=max(positions),
        position_span=max(positions) - min(positions),
        source_member_bytes=source_bytes,
        prototype_bytes=prototype_bytes,
        active_bytes_saved=source_bytes - prototype_bytes,
    )
    return RefreshedPrototype(
        PrototypeRecord(
            prototype_id=record.prototype_id,
            layer=record.layer,
            kv_head=record.kv_head,
            anchor_node_id=record.anchor_node_id,
            anchor_logical_position=refreshed_anchor_position,
            members=normalized_members,
            assigned_node_ids=tuple(sorted(member.node_id for member in normalized_members)),
            diagnostics=diagnostics,
            represented_source_positions=sum(block.position_count for block in blocks),
            attention_log_bias_scale=record.attention_log_bias_scale,
            attention_bias_basis=record.attention_bias_basis,
            pooling_mode=record.pooling_mode,
            anchor_position_strategy=record.anchor_position_strategy,
            eviction_utility=record.eviction_utility,
            utility_provenance=f"{record.utility_provenance};partial_residual_downdate_v1",
        ),
        refreshed_key,
        refreshed_value,
    )


__all__ = [
    "PrototypeRefreshError",
    "RefreshedPrototype",
    "refresh_prototype_after_promotions",
]
