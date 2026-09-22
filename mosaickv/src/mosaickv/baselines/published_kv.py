"""Audited reimplementations of published exact-token KV policies.

The implementations in this module use MosaicKV's common cache packer and
physical-byte accounting.  They are local reimplementations, never official
author code.  Formula provenance and the exact pinned sources are documented
in ``docs/baselines/published_kv_spec.md``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from mosaickv.cache_state import FullKVState
from mosaickv.config import CacheConfig, PublishedKVConfig
from mosaickv.types import BudgetUnit, JsonObject, MosaicKVMethod


class PublishedKVReimplementationError(RuntimeError):
    """Raised when a paper policy cannot be represented without guessing."""


_SUPPORTED = frozenset(
    {
        MosaicKVMethod.ELASTIC_CACHE_REIMPL,
        MosaicKVMethod.H2O_REIMPL,
        MosaicKVMethod.STREAMINGLLM_REIMPL,
        MosaicKVMethod.PYRAMIDKV_REIMPL,
        MosaicKVMethod.SNAPKV_REIMPL,
        MosaicKVMethod.ROCO_REIMPL,
    }
)


@dataclass(frozen=True, slots=True)
class PublishedKVHeadState:
    """One layer/head token selection with auditable score provenance."""

    layer: int
    kv_head: int
    source_positions: int
    capacity: int
    selected_physical_positions: tuple[int, ...]
    selected_logical_positions: tuple[int, ...]
    protected_positions: tuple[int, ...]
    score_sha256: str
    retained_bytes: int
    merge_memberships: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if not 0 < self.capacity <= self.source_positions:
            raise PublishedKVReimplementationError("invalid published-KV head capacity")
        if len(self.selected_physical_positions) != self.capacity:
            raise PublishedKVReimplementationError("selected positions do not fill head capacity")
        if tuple(sorted(set(self.selected_physical_positions))) != self.selected_physical_positions:
            raise PublishedKVReimplementationError("selected positions must be unique and sorted")
        if any(
            position < 0 or position >= self.source_positions
            for position in self.selected_physical_positions
        ):
            raise PublishedKVReimplementationError("selected position is outside the source cache")
        if not set(self.protected_positions).issubset(self.selected_physical_positions):
            raise PublishedKVReimplementationError("protected positions must be retained")
        if len(self.score_sha256) != 64:
            raise PublishedKVReimplementationError("score provenance must be a SHA-256 digest")
        if self.retained_bytes < 1:
            raise PublishedKVReimplementationError("retained bytes must be positive")
        if self.merge_memberships:
            if len(self.merge_memberships) != self.capacity:
                raise PublishedKVReimplementationError(
                    "merge memberships must align one-to-one with selected anchors"
                )
            flattened = tuple(
                position for membership in self.merge_memberships for position in membership
            )
            if tuple(sorted(flattened)) != tuple(range(self.source_positions)):
                raise PublishedKVReimplementationError(
                    "merge memberships must partition every source position exactly once"
                )
            for anchor, membership in zip(
                self.selected_physical_positions,
                self.merge_memberships,
                strict=True,
            ):
                if anchor not in membership:
                    raise PublishedKVReimplementationError(
                        "every Elastic Cache bucket must contain its anchor"
                    )


@dataclass(frozen=True, slots=True)
class PublishedKVCompressionPlan:
    """Cache plan shared by the six audited paper baselines."""

    method: MosaicKVMethod
    full_state: FullKVState
    config: PublishedKVConfig
    heads: tuple[PublishedKVHeadState, ...]
    source_slots: int
    active_slots: int
    source_bytes: int
    retained_bytes: int
    score_provenance: str

    def __post_init__(self) -> None:
        if self.method not in _SUPPORTED:
            raise PublishedKVReimplementationError("unsupported published-KV method")
        if self.active_slots != sum(head.capacity for head in self.heads):
            raise PublishedKVReimplementationError("active slot accounting is inconsistent")
        if self.retained_bytes != sum(head.retained_bytes for head in self.heads):
            raise PublishedKVReimplementationError("retained byte accounting is inconsistent")
        if self.active_slots > self.source_slots or self.retained_bytes > self.source_bytes:
            raise PublishedKVReimplementationError("compressed cache exceeds its source")

    def to_trace(self) -> JsonObject:
        return {
            "method": self.method.value,
            "implementation_status": "local_reimplementation_not_official",
            "source_slots": self.source_slots,
            "active_slots": self.active_slots,
            "source_bytes": self.source_bytes,
            "retained_bytes": self.retained_bytes,
            "score_provenance": self.score_provenance,
            "heads": [
                {
                    "layer": head.layer,
                    "kv_head": head.kv_head,
                    "capacity": head.capacity,
                    "selected_physical_positions": list(head.selected_physical_positions),
                    "selected_logical_positions": list(head.selected_logical_positions),
                    "protected_positions": list(head.protected_positions),
                    "score_sha256": head.score_sha256,
                    "retained_bytes": head.retained_bytes,
                    "merge_memberships": [
                        list(membership) for membership in head.merge_memberships
                    ],
                }
                for head in self.heads
            ],
        }


def _numpy_attention(value: Any, layer: int) -> np.ndarray[Any, np.dtype[np.float32]]:
    if hasattr(value, "detach"):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] != 1:
        raise PublishedKVReimplementationError(f"attention layer {layer} must have shape [1,H,Q,K]")
    if not bool(np.all(np.isfinite(array))) or bool(np.any(array < 0)):
        raise PublishedKVReimplementationError(
            f"attention layer {layer} contains invalid probabilities"
        )
    return array


def _group_attention(
    raw: Any,
    *,
    layer: int,
    kv_heads: int,
    sequence_length: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    attention = _numpy_attention(raw, layer)[0, ..., :sequence_length]
    query_heads, query_length, key_length = attention.shape
    if key_length != sequence_length or query_heads % kv_heads:
        raise PublishedKVReimplementationError(
            f"layer {layer} query/KV head or sequence geometry is incompatible"
        )
    return cast(
        "np.ndarray[Any, np.dtype[np.float32]]",
        attention.reshape(
            kv_heads,
            query_heads // kv_heads,
            query_length,
            sequence_length,
        ).mean(axis=1, dtype=np.float32),
    )


def _pool_scores(
    values: np.ndarray[Any, np.dtype[np.float32]], kernel: int, mode: str
) -> np.ndarray[Any, np.dtype[np.float32]]:
    if kernel == 1 or values.size == 0:
        return values.astype(np.float32, copy=True)
    radius = kernel // 2
    padded = np.pad(values, (radius, radius), mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)
    if mode == "avgpool":
        return cast(
            "np.ndarray[Any, np.dtype[np.float32]]",
            windows.mean(axis=-1, dtype=np.float32),
        )
    return cast(
        "np.ndarray[Any, np.dtype[np.float32]]",
        windows.max(axis=-1).astype(np.float32, copy=False),
    )


def _sha_scores(values: np.ndarray[Any, np.dtype[np.float32]]) -> str:
    canonical = np.asarray(values, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _target_slots(full_state: FullKVState, cache: CacheConfig) -> int:
    source_slots = len(full_state.blocks)
    ratio_slots = math.floor(source_slots * cache.retention_ratio + 1e-12)
    if cache.retention_ratio == 1.0:
        if cache.budget_unit is BudgetUnit.BLOCKS and cache.budget_value < source_slots:
            raise PublishedKVReimplementationError(
                "retention 1.0 requires a block budget covering the full cache"
            )
        if cache.budget_unit is BudgetUnit.BYTES and cache.budget_value < full_state.active_bytes:
            raise PublishedKVReimplementationError(
                "retention 1.0 requires a byte budget covering the full cache"
            )
        return source_slots
    if cache.budget_unit is BudgetUnit.BLOCKS:
        return min(ratio_slots, cache.budget_value)
    if cache.budget_unit is not BudgetUnit.BYTES:
        raise PublishedKVReimplementationError(
            "published KV policies support block or physical-byte budgets"
        )
    costs = {block.byte_size for block in full_state.blocks}
    if len(costs) != 1:
        raise PublishedKVReimplementationError(
            "byte-budget paper baselines require uniform per-token K/V storage"
        )
    (slot_bytes,) = tuple(costs)
    ratio_bytes = math.floor(full_state.active_bytes * cache.retention_ratio + 1e-12)
    return min(ratio_slots, min(ratio_bytes, cache.budget_value) // slot_bytes)


def _bounded_capacities(
    desired: tuple[float, ...],
    *,
    target: int,
    lower: tuple[int, ...],
    upper: tuple[int, ...],
) -> tuple[int, ...]:
    if not (len(desired) == len(lower) == len(upper)):
        raise PublishedKVReimplementationError("capacity vectors must align")
    if target < sum(lower) or target > sum(upper):
        raise PublishedKVReimplementationError(
            f"cache target {target} is outside feasible range [{sum(lower)}, {sum(upper)}]"
        )
    values = [
        min(high, max(low, math.floor(want)))
        for want, low, high in zip(desired, lower, upper, strict=True)
    ]
    while sum(values) < target:
        choices = [
            (desired[index] - values[index], -index, index)
            for index in range(len(values))
            if values[index] < upper[index]
        ]
        if not choices:
            raise PublishedKVReimplementationError("cannot fill the cache target")
        values[max(choices)[2]] += 1
    while sum(values) > target:
        removal_choices = [
            (values[index] - desired[index], index)
            for index in range(len(values))
            if values[index] > lower[index]
        ]
        if not removal_choices:
            raise PublishedKVReimplementationError("cannot reduce to the cache target")
        values[max(removal_choices)[1]] -= 1
    return tuple(values)


def _route_capacities(
    method: MosaicKVMethod,
    *,
    layer_count: int,
    heads_per_layer: int,
    sequence_length: int,
    target_slots: int,
    config: PublishedKVConfig,
    physical_prompt_capacity: int | None,
) -> tuple[int, ...]:
    route_count = layer_count * heads_per_layer
    upper_value = sequence_length
    if physical_prompt_capacity is not None:
        if physical_prompt_capacity < 1:
            raise PublishedKVReimplementationError("physical prompt capacity must be positive")
        upper_value = min(upper_value, physical_prompt_capacity)
    lower_value = 1
    if method is MosaicKVMethod.STREAMINGLLM_REIMPL:
        lower_value = min(sequence_length, config.sink_tokens + 1)
    elif method is MosaicKVMethod.ELASTIC_CACHE_REIMPL:
        protected = set(range(min(sequence_length, config.elastic_start_tokens)))
        protected.update(
            range(max(0, sequence_length - config.elastic_protect_tokens), sequence_length)
        )
        lower_value = len(protected)
    elif method in {
        MosaicKVMethod.H2O_REIMPL,
        MosaicKVMethod.SNAPKV_REIMPL,
        MosaicKVMethod.PYRAMIDKV_REIMPL,
    }:
        lower_value = min(sequence_length, config.observation_window + 1)
    lower = (lower_value,) * route_count
    upper = (upper_value,) * route_count
    if any(low > high for low, high in zip(lower, upper, strict=True)):
        raise PublishedKVReimplementationError(
            "the configured protected window exceeds the physical cache capacity"
        )
    if method is MosaicKVMethod.PYRAMIDKV_REIMPL and target_slots < route_count * sequence_length:
        average = target_slots / route_count
        non_window = max(1.0, average - config.observation_window)
        minimum = max(1.0, non_window / config.pyramid_beta)
        maximum = max(minimum, 2.0 * non_window - minimum)
        step = 0.0 if layer_count == 1 else (maximum - minimum) / (layer_count - 1)
        layer_weights = tuple(
            config.observation_window + maximum - layer * step for layer in range(layer_count)
        )
        scale = target_slots / (sum(layer_weights) * heads_per_layer)
        desired = tuple(
            layer_weights[layer] * scale
            for layer in range(layer_count)
            for _head in range(heads_per_layer)
        )
    else:
        desired = (target_slots / route_count,) * route_count
    return _bounded_capacities(desired, target=target_slots, lower=lower, upper=upper)


def _ranked_fill(
    scores: np.ndarray[Any, np.dtype[np.float32]],
    capacity: int,
    protected: tuple[int, ...],
) -> tuple[int, ...]:
    keep = set(protected)
    if len(keep) > capacity:
        raise PublishedKVReimplementationError("protected positions exceed route capacity")
    ranked = sorted(
        (position for position in range(scores.size) if position not in keep),
        key=lambda position: (-float(scores[position]), position),
    )
    keep.update(ranked[: capacity - len(keep)])
    return tuple(sorted(keep))


def _select_route(
    method: MosaicKVMethod,
    grouped_attention: np.ndarray[Any, np.dtype[np.float32]] | None,
    *,
    kv_head: int,
    sequence_length: int,
    capacity: int,
    config: PublishedKVConfig,
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray[Any, np.dtype[np.float32]], str]:
    if capacity == sequence_length:
        scores = np.ones(sequence_length, dtype=np.float32)
        selected = tuple(range(sequence_length))
        return selected, selected, scores, "retention_one_identity"
    if method is MosaicKVMethod.STREAMINGLLM_REIMPL:
        sink = min(config.sink_tokens, capacity - 1)
        recent = capacity - sink
        protected = tuple(range(sink)) + tuple(range(sequence_length - recent, sequence_length))
        scores = np.zeros(sequence_length, dtype=np.float32)
        scores[list(protected)] = 1.0
        return (
            tuple(sorted(set(protected))),
            tuple(sorted(set(protected))),
            scores,
            "attention_sinks_plus_recent",
        )
    if grouped_attention is None:
        raise PublishedKVReimplementationError(f"{method.value} requires eager prefill attention")
    attention = grouped_attention[kv_head]
    if method in {
        MosaicKVMethod.H2O_REIMPL,
        MosaicKVMethod.SNAPKV_REIMPL,
        MosaicKVMethod.PYRAMIDKV_REIMPL,
    }:
        window = min(config.observation_window, capacity - 1, attention.shape[0])
        if window < 1:
            raise PublishedKVReimplementationError("observation window leaves no scored capacity")
        prefix_end = sequence_length - window
        protected = tuple(range(prefix_end, sequence_length))
        scores = attention[-window:, :].sum(axis=0, dtype=np.float32)
        if method in {MosaicKVMethod.SNAPKV_REIMPL, MosaicKVMethod.PYRAMIDKV_REIMPL}:
            pooled = _pool_scores(
                scores[:prefix_end],
                config.attention_pool_kernel,
                config.attention_pooling,
            )
            scores = scores.copy()
            scores[:prefix_end] = pooled
            provenance = "observation_window_attention_with_pooling"
        else:
            provenance = "observation_window_accumulated_attention"
        scores[prefix_end:] = np.inf
        return _ranked_fill(scores, capacity, protected), protected, scores, provenance
    if method is MosaicKVMethod.ROCO_REIMPL:
        # RoCo protects high-variance tokens from eviction, then retains the
        # largest mean-attention tokens in the remaining eviction scope.  This
        # is the deterministic batch equivalent of the paper's repeated
        # mean/variance policy at the prefill boundary.
        mean = attention.mean(axis=0, dtype=np.float32)
        deviation = attention.std(axis=0, dtype=np.float32)
        recent_count = min(
            capacity - 1,
            math.floor(capacity * config.recent_fraction + 1e-12),
        )
        recent_positions = set(range(sequence_length - recent_count, sequence_length))
        variance_count = min(
            capacity - len(recent_positions),
            max(1, math.floor(capacity * config.roco_protected_fraction + 1e-12)),
        )
        variance_protected = sorted(
            (
                position
                for position in range(sequence_length)
                if position not in recent_positions
            ),
            key=lambda position: (-float(deviation[position]), position),
        )[:variance_count]
        protected = tuple(sorted(recent_positions.union(variance_protected)))
        return (
            _ranked_fill(mean, capacity, protected),
            protected,
            mean,
            "mean_attention_with_variance_protection",
        )
    raise PublishedKVReimplementationError(f"unsupported method: {method.value}")


def _nearest_anchor_memberships(
    anchors: tuple[int, ...], sequence_length: int
) -> tuple[tuple[int, ...], ...]:
    """Partition source positions by nearest anchor with a stable left tie-break."""

    if not anchors:
        raise PublishedKVReimplementationError("Elastic Cache requires at least one anchor")
    groups: dict[int, list[int]] = {anchor: [] for anchor in anchors}
    for position in range(sequence_length):
        anchor = min(anchors, key=lambda candidate: (abs(candidate - position), candidate))
        groups[anchor].append(position)
    return tuple(tuple(groups[anchor]) for anchor in anchors)


def _elastic_layer_selection(
    grouped_attention: np.ndarray[Any, np.dtype[np.float32]],
    *,
    sequence_length: int,
    capacity: int,
    config: PublishedKVConfig,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    np.ndarray[Any, np.dtype[np.float32]],
    tuple[tuple[int, ...], ...],
]:
    """Implement Elastic Cache equations 1--3 at the prefill boundary."""

    if capacity == sequence_length:
        anchors = tuple(range(sequence_length))
        return (
            anchors,
            anchors,
            np.ones(sequence_length, dtype=np.float32),
            tuple((position,) for position in anchors),
        )
    # The paper averages token importance across all attention heads in a
    # layer, then sums every causal query contribution for each key position.
    scores = grouped_attention.mean(axis=0, dtype=np.float32).sum(axis=0, dtype=np.float32)
    protected_set = set(range(min(sequence_length, config.elastic_start_tokens)))
    protected_set.update(
        range(max(0, sequence_length - config.elastic_protect_tokens), sequence_length)
    )
    protected = tuple(sorted(protected_set))
    anchors = _ranked_fill(scores, capacity, protected)
    return anchors, protected, scores, _nearest_anchor_memberships(anchors, sequence_length)


def build_published_kv_reimpl_plan(
    full_state: FullKVState,
    attention_weights: tuple[Any, ...],
    method: MosaicKVMethod,
    config: PublishedKVConfig,
    cache: CacheConfig,
    *,
    physical_prompt_capacity: int | None = None,
) -> PublishedKVCompressionPlan:
    """Build a deterministic token-level plan under the common hard budget."""

    if method not in _SUPPORTED or not config.enabled:
        raise PublishedKVReimplementationError("published KV reimplementation is not enabled")
    if full_state.block_size != 1 or cache.block_size != 1:
        raise PublishedKVReimplementationError(
            "published KV reimplementations require block_size=1"
        )
    lengths = tuple(layer.sequence_length for layer in full_state.layers)
    head_counts = tuple(layer.kv_heads for layer in full_state.layers)
    if len(set(lengths)) != 1 or len(set(head_counts)) != 1:
        raise PublishedKVReimplementationError(
            "published KV policies require uniform decoder sequence lengths and KV-head counts"
        )
    layer_count = len(lengths)
    sequence_length = lengths[0]
    heads_per_layer = head_counts[0]
    if method is not MosaicKVMethod.STREAMINGLLM_REIMPL and len(attention_weights) != layer_count:
        raise PublishedKVReimplementationError(
            f"{method.value} requires one eager attention tensor per decoder layer"
        )
    target_slots = _target_slots(full_state, cache)
    capacities = _route_capacities(
        method,
        layer_count=layer_count,
        heads_per_layer=heads_per_layer,
        sequence_length=sequence_length,
        target_slots=target_slots,
        config=config,
        physical_prompt_capacity=physical_prompt_capacity,
    )
    blocks = {
        (block.layer, block.kv_head, block.physical_cache_indices[0]): block
        for block in full_state.blocks
    }
    heads: list[PublishedKVHeadState] = []
    provenances: set[str] = set()
    for layer_index in range(layer_count):
        grouped = (
            None
            if method is MosaicKVMethod.STREAMINGLLM_REIMPL
            else _group_attention(
                attention_weights[layer_index],
                layer=layer_index,
                kv_heads=heads_per_layer,
                sequence_length=sequence_length,
            )
        )
        elastic_selection = None
        if method is MosaicKVMethod.ELASTIC_CACHE_REIMPL:
            if grouped is None:  # pragma: no cover - attention count is validated above
                raise PublishedKVReimplementationError(
                    "Elastic Cache requires eager prefill attention"
                )
            layer_capacities = capacities[
                layer_index * heads_per_layer : (layer_index + 1) * heads_per_layer
            ]
            if len(set(layer_capacities)) != 1:
                raise PublishedKVReimplementationError(
                    "Elastic Cache requires a byte target yielding one shared capacity "
                    "for every KV head in a layer"
                )
            elastic_selection = _elastic_layer_selection(
                grouped,
                sequence_length=sequence_length,
                capacity=layer_capacities[0],
                config=config,
            )
        for kv_head in range(heads_per_layer):
            route_index = layer_index * heads_per_layer + kv_head
            memberships: tuple[tuple[int, ...], ...] = ()
            if elastic_selection is not None:
                selected, protected, scores, memberships = elastic_selection
                provenance = "layer_attention_sum_nearest_anchor_fp32_mean_merge"
            else:
                selected, protected, scores, provenance = _select_route(
                    method,
                    grouped,
                    kv_head=kv_head,
                    sequence_length=sequence_length,
                    capacity=capacities[route_index],
                    config=config,
                )
            retained_bytes = sum(
                blocks[(layer_index, kv_head, position)].byte_size for position in selected
            )
            heads.append(
                PublishedKVHeadState(
                    layer_index,
                    kv_head,
                    sequence_length,
                    capacities[route_index],
                    selected,
                    full_state.logical_positions.gather(selected),
                    protected,
                    _sha_scores(scores),
                    retained_bytes,
                    memberships,
                )
            )
            provenances.add(provenance)
    retained_bytes = sum(head.retained_bytes for head in heads)
    if cache.budget_unit is BudgetUnit.BYTES and retained_bytes > cache.budget_value:
        raise PublishedKVReimplementationError("retained K/V bytes exceed the hard byte budget")
    return PublishedKVCompressionPlan(
        method,
        full_state,
        config,
        tuple(heads),
        len(full_state.blocks),
        sum(head.capacity for head in heads),
        full_state.active_bytes,
        retained_bytes,
        "+".join(sorted(provenances)),
    )


def _position_payload(
    tensor: Any,
    *,
    head_axis: int,
    head: int,
    sequence_axis: int,
    position: int,
) -> Any:
    index: list[Any] = [slice(None)] * int(tensor.ndim)
    index[head_axis] = slice(head, head + 1)
    index[sequence_axis] = slice(position, position + 1)
    result = tensor[tuple(index)]
    clone = getattr(result, "clone", None)
    return clone() if clone is not None else result.copy()


def _merged_position_payload(
    tensor: Any,
    *,
    head_axis: int,
    head: int,
    sequence_axis: int,
    positions: tuple[int, ...],
) -> Any:
    """Average one Elastic Cache bucket in FP32 and restore cache dtype."""

    if len(positions) == 1:
        return _position_payload(
            tensor,
            head_axis=head_axis,
            head=head,
            sequence_axis=sequence_axis,
            position=positions[0],
        )
    import torch

    head_index: list[Any] = [slice(None)] * int(tensor.ndim)
    head_index[head_axis] = slice(head, head + 1)
    head_tensor = tensor[tuple(head_index)]
    index = torch.tensor(positions, dtype=torch.long, device=tensor.device)
    selected = torch.index_select(head_tensor, sequence_axis, index)
    return selected.float().mean(dim=sequence_axis, keepdim=True).to(dtype=tensor.dtype)


def published_kv_runtime_payloads(
    plan: PublishedKVCompressionPlan,
) -> tuple[
    dict[tuple[int, int], list[tuple[int, str, int, Any, Any]]],
    tuple[JsonObject, ...],
]:
    """Expose selected positions through the shared logical-position packer."""

    payloads: dict[tuple[int, int], list[tuple[int, str, int, Any, Any]]] = {}
    records: list[JsonObject] = []
    for head in plan.heads:
        storage = plan.full_state.layers[head.layer]
        protected = set(head.protected_positions)
        entries: list[tuple[int, str, int, Any, Any]] = []
        memberships = head.merge_memberships or tuple(
            (position,) for position in head.selected_physical_positions
        )
        for slot, (physical, membership) in enumerate(
            zip(head.selected_physical_positions, memberships, strict=True)
        ):
            logical = plan.full_state.logical_positions.logical_for_physical(physical)
            merged = plan.method is MosaicKVMethod.ELASTIC_CACHE_REIMPL and len(membership) > 1
            entries.append(
                (
                    logical,
                    plan.method.value,
                    physical,
                    _merged_position_payload(
                        storage.key,
                        head_axis=storage.key_head_dimension,
                        head=head.kv_head,
                        sequence_axis=storage.key_sequence_dimension,
                        positions=membership,
                    ),
                    _merged_position_payload(
                        storage.value,
                        head_axis=storage.value_head_dimension,
                        head=head.kv_head,
                        sequence_axis=storage.value_sequence_dimension,
                        positions=membership,
                    ),
                )
            )
            records.append(
                {
                    "layer": head.layer,
                    "kv_head": head.kv_head,
                    "slot": slot,
                    "physical_position": physical,
                    "logical_position": logical,
                    "tier": f"{plan.method.value}_{'merged' if merged else 'exact'}",
                    "selection_reason": (
                        "protected_anchor"
                        if physical in protected
                        else ("paper_anchor" if head.merge_memberships else "paper_score")
                    ),
                    "source_membership": list(membership),
                    "cached_key_state": plan.full_state.cached_key_state,
                    "implementation_status": "local_reimplementation_not_official",
                }
            )
        payloads[(head.layer, head.kv_head)] = entries
    return payloads, tuple(records)


__all__ = [
    "PublishedKVCompressionPlan",
    "PublishedKVHeadState",
    "PublishedKVReimplementationError",
    "build_published_kv_reimpl_plan",
    "published_kv_runtime_payloads",
]
