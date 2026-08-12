"""Conservative exact/prototype/residual cache construction."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, cast

import numpy as np

from mosaickv.adapters.huggingface.types import AdapterCapabilities, CachedKeyState
from mosaickv.cache_state import (
    ExactTier,
    FullKVState,
    KVBlockDescriptor,
    MosaicKVState,
    PrototypeTier,
    tensor_storage_bytes,
)
from mosaickv.config import PrototypeConfig, ResidualConfig
from mosaickv.graph.types import SparseEvidenceGraph
from mosaickv.prototypes.rope import RotaryPrototypeKeyTransform
from mosaickv.prototypes.types import (
    ActiveHeadLayout,
    PrototypeDiagnostics,
    PrototypeMember,
    PrototypeRecord,
    PrototypeSafetyAssessment,
    ThreeTierCacheConstruction,
    TierConstructionMode,
)
from mosaickv.residual.storage import build_residual_storage, empty_residual_storage
from mosaickv.selection.types import SelectionResult
from mosaickv.types import BudgetUnit


class PrototypeConstructionError(RuntimeError):
    """Raised for malformed graph/selection inputs or invalid prototype tensors."""


def _coerce_key_state(value: Any) -> CachedKeyState:
    if isinstance(value, CachedKeyState):
        return value
    raw = getattr(value, "value", value)
    try:
        return CachedKeyState(str(raw))
    except ValueError:
        return CachedKeyState.UNKNOWN


def assess_prototype_safety(
    full_state: FullKVState,
    capabilities: AdapterCapabilities,
    key_transform: RotaryPrototypeKeyTransform | None = None,
) -> PrototypeSafetyAssessment:
    """Reject post-RoPE merging unless an exact audited phase transform exists."""

    source_state = _coerce_key_state(full_state.cached_key_state)
    adapter_state = _coerce_key_state(capabilities.cached_key_state)
    declares_support = capabilities.supports_prototype_merge
    transform_provenance = None if key_transform is None else key_transform.provenance
    transform_matches = bool(
        key_transform is not None and key_transform.source_key_state is source_state
    )
    phase_neutral_source = source_state in {
        CachedKeyState.PRE_ROPE,
        CachedKeyState.NOT_APPLICABLE,
    }
    safe = (
        declares_support
        and source_state == adapter_state
        and (
            (phase_neutral_source and key_transform is None)
            or (source_state is CachedKeyState.POST_ROPE and transform_matches)
        )
    )
    if source_state is CachedKeyState.UNKNOWN:
        reason = "source cache RoPE state is unknown; prototype merging is disabled"
    elif adapter_state is CachedKeyState.UNKNOWN:
        reason = "adapter cache RoPE state is unknown; prototype merging is disabled"
    elif source_state != adapter_state:
        reason = (
            "source and adapter cached-key RoPE metadata disagree "
            f"({source_state.value} != {adapter_state.value})"
        )
    elif key_transform is not None and not transform_matches:
        reason = "prototype phase transform does not match the source cached-key representation"
    elif source_state is CachedKeyState.POST_ROPE and not transform_matches:
        reason = (
            "cached keys are post-RoPE and no matching audited phase transform was supplied; "
            "the implementation never averages different RoPE phases"
        )
    elif not declares_support:
        reason = "adapter does not declare prototype merging safe"
    else:
        reason = (
            f"adapter explicitly supports prototype merging for {source_state.value} cached keys"
            + (
                f" through phase transform {transform_provenance}"
                if transform_provenance is not None
                else ""
            )
        )
    return PrototypeSafetyAssessment(
        model_family=capabilities.model_family,
        adapter_cached_key_state=adapter_state,
        source_cached_key_state=source_state,
        adapter_declares_support=declares_support,
        phase_transform_provenance=transform_provenance,
        safe=safe,
        reason=reason,
    )


def _active_layouts(
    full_state: FullKVState,
    exact: ExactTier,
    records: Sequence[PrototypeRecord],
) -> tuple[ActiveHeadLayout, ...]:
    identities = {
        (layer_id, head)
        for layer_id, layer in enumerate(full_state.layers)
        for head in range(layer.kv_heads)
    }
    result: list[ActiveHeadLayout] = []
    for layer, head in sorted(identities):
        matching = tuple(
            record for record in records if record.layer == layer and record.kv_head == head
        )
        result.append(
            ActiveHeadLayout(
                layer=layer,
                kv_head=head,
                exact_logical_positions=tuple(
                    sorted(exact.selected_logical_positions(layer, head))
                ),
                prototype_ids=tuple(record.prototype_id for record in matching),
                prototype_anchor_positions=tuple(
                    record.anchor_logical_position for record in matching
                ),
            )
        )
    return tuple(result)


def _exact_only(
    full_state: FullKVState,
    selection: SelectionResult,
    safety: PrototypeSafetyAssessment,
    adapter_declares_residual_repair: bool,
    *,
    mode: TierConstructionMode,
    reason: str,
    exact_gather_mode: str,
) -> ThreeTierCacheConstruction:
    exact = selection.to_exact_tier(full_state, gather_mode=exact_gather_mode)
    residual_report = empty_residual_storage()
    state = MosaicKVState.create(full_state, exact=exact, residuals=residual_report.tier)
    return ThreeTierCacheConstruction(
        state=state,
        mode=mode,
        reason=reason,
        safety=safety,
        adapter_declares_residual_repair=adapter_declares_residual_repair,
        active_budget=selection.budget,
        exact_node_ids=selection.selected_node_ids,
        prototypes=(),
        residual_storage=residual_report,
        active_layouts=_active_layouts(full_state, exact, ()),
        original_logical_sequence_length=full_state.original_logical_sequence_length,
        next_decode_position=full_state.next_decode_position,
    )


def _retention_one(
    full_state: FullKVState,
    safety: PrototypeSafetyAssessment,
    selection: SelectionResult,
    adapter_declares_residual_repair: bool,
    *,
    exact_gather_mode: str,
) -> ThreeTierCacheConstruction:
    exact = full_state.gather_exact_blocks(
        full_state.blocks,
        gather_mode=exact_gather_mode,
    )
    residual_report = empty_residual_storage()
    state = MosaicKVState.create(full_state, exact=exact, residuals=residual_report.tier)
    state.reconstruct_full_state(full_state)
    exact_ids = tuple(range(len(full_state.blocks)))
    return ThreeTierCacheConstruction(
        state=state,
        mode=TierConstructionMode.RETENTION_ONE,
        reason="all source blocks are exact; no prototype or residual transformation was run",
        safety=safety,
        adapter_declares_residual_repair=adapter_declares_residual_repair,
        active_budget=selection.budget,
        exact_node_ids=exact_ids,
        prototypes=(),
        residual_storage=residual_report,
        active_layouts=_active_layouts(full_state, state.exact, ()),
        original_logical_sequence_length=full_state.original_logical_sequence_length,
        next_decode_position=full_state.next_decode_position,
    )


def _validate_inputs(
    full_state: FullKVState,
    graph: SparseEvidenceGraph,
    selection: SelectionResult,
) -> None:
    graph_blocks = tuple(node.block for node in graph.nodes)
    if graph_blocks != full_state.blocks:
        raise PrototypeConstructionError(
            "evidence graph nodes must align exactly with FullKV source blocks"
        )
    if len(selection.decisions) != len(graph.nodes):
        raise PrototypeConstructionError(
            "selection decisions must cover the same node table as the evidence graph"
        )
    expected = tuple(full_state.blocks[node_id] for node_id in selection.selected_node_ids)
    if selection.selected_blocks != expected:
        raise PrototypeConstructionError(
            "selected block descriptors do not align with their graph node IDs"
        )


def _undirected_anchor_weights(graph: SparseEvidenceGraph) -> Mapping[tuple[int, int], float]:
    weights: dict[tuple[int, int], float] = {}
    for source, target, weight in zip(
        graph.row_indices, graph.column_indices, graph.weights, strict=True
    ):
        identity = (min(source, target), max(source, target))
        weights[identity] = max(weights.get(identity, 0.0), float(weight))
    return weights


def _pair_span(first: KVBlockDescriptor, second: KVBlockDescriptor) -> int:
    positions = (*first.original_logical_positions, *second.original_logical_positions)
    return max(positions) - min(positions)


def _compatible(
    source: KVBlockDescriptor,
    anchor: KVBlockDescriptor,
    config: PrototypeConfig,
) -> bool:
    if source.layer != anchor.layer or source.kv_head != anchor.kv_head:
        return False
    modality_pair = f"{source.modality.value}:{anchor.modality.value}"
    if modality_pair not in config.allowed_modality_pairs:
        return False
    maximum_span = config.position_span_for(source.layer, source.kv_head)
    return maximum_span is None or _pair_span(source, anchor) <= maximum_span


def _descriptor_mse(first: Any, second: Any) -> float:
    left = np.asarray(first, dtype=np.float32)
    right = np.asarray(second, dtype=np.float32)
    if left.shape != right.shape:
        return math.inf
    delta = left - right
    return float(np.mean(delta * delta, dtype=np.float64))


def _descriptor_compatible(
    graph: SparseEvidenceGraph,
    source_id: int,
    anchor_id: int,
    config: PrototypeConfig,
) -> bool:
    """Apply optional pre-materialization member guards to adaptive clusters."""

    if not config.adaptive_grouping:
        return True
    source = graph.nodes[source_id]
    anchor = graph.nodes[anchor_id]
    if (
        config.max_member_key_error is not None
        and _descriptor_mse(source.pooled_key, anchor.pooled_key) > config.max_member_key_error
    ):
        return False
    return not (
        config.max_member_value_error is not None
        and _descriptor_mse(source.pooled_value, anchor.pooled_value)
        > config.max_member_value_error
    )


def _assign_to_anchors(
    graph: SparseEvidenceGraph,
    selected: frozenset[int],
    config: PrototypeConfig,
    *,
    prototype_anchor_node_ids: frozenset[int] | None = None,
) -> tuple[dict[int, tuple[int, float]], str | None]:
    anchors = selected if prototype_anchor_node_ids is None else prototype_anchor_node_ids
    if not anchors <= selected:
        return {}, "prototype anchor allowlist contains a non-selected source node"
    missing_exact_routes = tuple(
        node.node_id
        for node in graph.nodes
        if config.requires_exact_head(node.block.layer, node.block.kv_head)
        and node.node_id not in selected
    )
    if missing_exact_routes:
        return {}, (
            "per-head exact prototype policy requires source nodes to be selected exact: "
            f"{missing_exact_routes[:8]}"
        )
    weights = _undirected_anchor_weights(graph)
    neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (first, second), weight in weights.items():
        neighbors[first].append((second, weight))
        if first != second:
            neighbors[second].append((first, weight))
    selected_by_identity: dict[tuple[int, int, object], list[tuple[float, int]]] = defaultdict(list)
    for anchor_id in anchors:
        anchor = graph.nodes[anchor_id]
        selected_by_identity[
            (anchor.block.layer, anchor.block.kv_head, anchor.block.modality)
        ].append((anchor.logical_center, anchor_id))
    for identity_anchors in selected_by_identity.values():
        identity_anchors.sort()
    allowed_anchor_modalities: dict[str, set[str]] = defaultdict(set)
    for pair in config.allowed_modality_pairs:
        source_modality, anchor_modality = pair.split(":", maxsplit=1)
        allowed_anchor_modalities[source_modality].add(anchor_modality)

    def positional_candidates(node_id: int) -> tuple[tuple[int, float], ...]:
        """Return a bounded compatibility scaffold independent of evidence scores."""

        source = graph.nodes[node_id]
        candidates: list[tuple[float, int]] = []
        local_group_size = config.group_size_for(source.block.layer, source.block.kv_head)
        window = max(1, 2 * local_group_size)
        for anchor_modality in sorted(allowed_anchor_modalities[source.block.modality.value]):
            modality = type(source.block.modality)(anchor_modality)
            anchors = selected_by_identity.get(
                (source.block.layer, source.block.kv_head, modality),
                (),
            )
            if not anchors:
                continue
            centers = [center for center, _anchor_id in anchors]
            insertion = bisect_left(centers, source.logical_center)
            first = max(0, insertion - window)
            last = min(len(anchors), insertion + window)
            for center, anchor_id in anchors[first:last]:
                anchor = graph.nodes[anchor_id]
                if not _compatible(
                    source.block, anchor.block, config
                ) or not _descriptor_compatible(graph, node_id, anchor_id, config):
                    continue
                distance = abs(source.logical_center - center)
                span = config.position_span_for(source.block.layer, source.block.kv_head)
                normalized = 1.0 / (1.0 + distance) if span is None else 1.0 - distance / (span + 1)
                # Compatibility edges are deliberately weaker than evidence
                # edges. They make representation feasible without claiming
                # semantic evidence or affecting which real graph edge wins.
                weight = max(config.min_anchor_weight, 0.01 * max(normalized, 1e-12))
                candidates.append((weight, anchor_id))
        return tuple((anchor_id, weight) for weight, anchor_id in candidates)

    preferences: dict[int, tuple[tuple[float, int], ...]] = {}
    for node_id, node in enumerate(graph.nodes):
        if node_id in selected:
            continue
        weights_by_anchor: dict[int, float] = {}
        for anchor_id, weight in neighbors.get(node_id, ()):
            if anchor_id not in anchors:
                continue
            anchor = graph.nodes[anchor_id]
            if (
                weight > 0
                and weight >= config.min_anchor_weight
                and _compatible(node.block, anchor.block, config)
                and _descriptor_compatible(graph, node_id, anchor_id, config)
            ):
                weights_by_anchor[anchor_id] = max(weight, weights_by_anchor.get(anchor_id, 0.0))
        for anchor_id, weight in positional_candidates(node_id):
            weights_by_anchor[anchor_id] = max(weight, weights_by_anchor.get(anchor_id, 0.0))
        candidates = [(weight, anchor_id) for anchor_id, weight in weights_by_anchor.items()]
        if not candidates:
            return {}, f"unselected node {node_id} has no compatible selected graph anchor"
        # Higher graph weight wins; lower node ID is the deterministic tie-break.
        preferences[node_id] = tuple(sorted(candidates, key=lambda item: (-item[0], item[1])))

    def segmented_assignments() -> tuple[dict[int, tuple[int, float]], str | None]:
        """Return the deterministic anchor-cover assignment used as a safe fallback."""

        strata: dict[tuple[int, int, object], list[int]] = defaultdict(list)
        for node in graph.nodes:
            block = node.block
            strata[(block.layer, block.kv_head, block.modality)].append(node.node_id)
        result: dict[int, tuple[int, float]] = {}

        def assign_segment(member_ids: Sequence[int]) -> str | None:
            segment_anchors = frozenset(node_id for node_id in member_ids if node_id in anchors)
            if not segment_anchors:
                return f"prototype anchor-cover segment {tuple(member_ids)} has no selected anchor"
            for node_id in member_ids:
                if node_id in selected:
                    continue
                candidates = tuple(
                    item for item in preferences[node_id] if item[1] in segment_anchors
                )
                if not candidates:
                    return (
                        f"unselected node {node_id} has no compatible selected anchor "
                        "inside its anchor-cover segment"
                    )
                result[node_id] = candidates[0][1], candidates[0][0]
            return None

        for identity in sorted(strata, key=lambda item: (item[0], item[1], str(item[2]))):
            ordered = sorted(
                strata[identity],
                key=lambda node_id: (
                    graph.nodes[node_id].block.original_logical_positions[0],
                    node_id,
                ),
            )
            segment: list[int] = []
            for node_id in ordered:
                trial = [*segment, node_id]
                positions = [
                    position
                    for member_id in trial
                    for position in graph.nodes[member_id].block.original_logical_positions
                ]
                first = graph.nodes[trial[0]].block
                local_group_size = config.group_size_for(first.layer, first.kv_head)
                local_position_span = config.position_span_for(first.layer, first.kv_head)
                exceeds_capacity = len(trial) > local_group_size + 1
                exceeds_span = bool(
                    local_position_span is not None
                    and max(positions) - min(positions) > local_position_span
                )
                if segment and (exceeds_capacity or exceeds_span):
                    reason = assign_segment(segment)
                    if reason is not None:
                        return {}, reason
                    segment = [node_id]
                else:
                    segment = trial
            reason = assign_segment(segment)
            if reason is not None:
                return {}, reason
        return result, None

    # Scarce nodes choose first. When an anchor is full, a deterministic
    # augmenting-path relocation finds another compatible slot for an existing
    # member before rejecting a feasible capacitated assignment.
    assignments: dict[int, tuple[int, float]] = {}
    groups: dict[int, list[int]] = defaultdict(list)

    def cluster_fits(anchor_id: int, member_ids: Sequence[int]) -> bool:
        anchor_block = graph.nodes[anchor_id].block
        maximum_span = config.position_span_for(anchor_block.layer, anchor_block.kv_head)
        if maximum_span is None:
            return all(
                _descriptor_compatible(graph, node_id, anchor_id, config) for node_id in member_ids
            )
        positions = [
            position
            for candidate_id in (anchor_id, *member_ids)
            for position in graph.nodes[candidate_id].block.original_logical_positions
        ]
        return max(positions) - min(positions) <= maximum_span and all(
            _descriptor_compatible(graph, node_id, anchor_id, config) for node_id in member_ids
        )

    def assign(node_id: int, visited_nodes: set[int], visited_anchors: set[int]) -> bool:
        if node_id in visited_nodes:
            return False
        visited_nodes.add(node_id)
        for weight, anchor_id in preferences[node_id]:
            if anchor_id in visited_anchors:
                continue
            visited_anchors.add(anchor_id)
            members = groups[anchor_id]
            anchor = graph.nodes[anchor_id].block
            maximum_members = config.group_size_for(anchor.layer, anchor.kv_head)
            if len(members) < maximum_members and cluster_fits(anchor_id, (*members, node_id)):
                members.append(node_id)
                assignments[node_id] = (anchor_id, weight)
                return True
            # Relocate the weakest current member first. This retains the
            # highest possible local graph weights without nondeterminism.
            displaced_order = sorted(
                members,
                key=lambda member_id: (
                    assignments[member_id][1],
                    -member_id,
                ),
            )
            for displaced in displaced_order:
                old_assignment = assignments[displaced]
                members.remove(displaced)
                del assignments[displaced]
                if cluster_fits(anchor_id, (*members, node_id)) and assign(
                    displaced, visited_nodes, visited_anchors
                ):
                    members.append(node_id)
                    assignments[node_id] = (anchor_id, weight)
                    return True
                assignments[displaced] = old_assignment
                members.append(displaced)
        return False

    source_order = sorted(
        preferences,
        key=lambda node_id: (
            len(preferences[node_id]),
            -preferences[node_id][0][0],
            node_id,
        ),
    )
    for node_id in source_order:
        if not assign(node_id, set(), set()):
            fallback, fallback_reason = segmented_assignments()
            if fallback_reason is None:
                return fallback, None
            return {}, (
                f"unselected node {node_id} cannot be assigned within "
                "the configured per-head prototype capacity; "
                f"{fallback_reason}"
            )

    for anchor_id, member_ids in groups.items():
        final_anchor_block = graph.nodes[anchor_id].block
        maximum_span = config.position_span_for(
            final_anchor_block.layer, final_anchor_block.kv_head
        )
        if maximum_span is not None:
            blocks = [graph.nodes[anchor_id].block]
            blocks.extend(graph.nodes[node_id].block for node_id in member_ids)
            positions = [
                position for block in blocks for position in block.original_logical_positions
            ]
            if max(positions) - min(positions) > maximum_span:
                fallback, fallback_reason = segmented_assignments()
                if fallback_reason is None:
                    return fallback, None
                return {}, (
                    f"anchor {anchor_id} cluster exceeds its position span="
                    f"{maximum_span}; {fallback_reason}"
                )
    return assignments, None


def _pool_fp32(
    tensor: Any,
    axis: int,
    position_weights: Sequence[float] | None = None,
) -> Any:
    """Pool one cache block uniformly or with normalized per-position weights."""

    normalized_axis = axis % int(tensor.ndim)
    size = int(tensor.shape[normalized_axis])
    if position_weights is None:
        weights = tuple(1.0 / size for _index in range(size))
    else:
        weights = tuple(float(value) for value in position_weights)
        if len(weights) != size:
            raise PrototypeConstructionError(
                "prototype position weights do not match the cache block width"
            )
        if any(not math.isfinite(value) or value <= 0 for value in weights):
            raise PrototypeConstructionError(
                "prototype position weights must be finite and positive"
            )
        total = sum(weights)
        weights = tuple(value / total for value in weights)
    if tensor.__class__.__module__.startswith("torch") and hasattr(tensor, "float"):
        import torch

        source = tensor.detach().float()
        shape = [1] * int(source.ndim)
        shape[normalized_axis] = size
        weight_tensor = torch.as_tensor(
            weights,
            dtype=source.dtype,
            device=source.device,
        ).reshape(shape)
        return torch.sum(source * weight_tensor, dim=normalized_axis, keepdim=True)
    source = np.asarray(tensor, dtype=np.float32)
    shape = [1] * int(source.ndim)
    shape[normalized_axis] = size
    weight_array = np.asarray(weights, dtype=np.float32).reshape(shape)
    return np.sum(source * weight_array, axis=normalized_axis, keepdims=True, dtype=np.float32)


def _normalized_position_weights(
    block: KVBlockDescriptor,
    token_attention_by_head: Mapping[tuple[int, int], Any] | None,
) -> tuple[tuple[float, ...], float]:
    """Return one positive within-block forecast distribution and its mass."""

    positions = block.physical_cache_indices
    if token_attention_by_head is None:
        return tuple(1.0 / len(positions) for _position in positions), 1.0
    try:
        source = np.asarray(
            token_attention_by_head[(block.layer, block.kv_head)],
            dtype=np.float64,
        ).reshape(-1)
    except KeyError as error:
        raise PrototypeConstructionError(
            f"prototype token attention is missing layer/head {(block.layer, block.kv_head)}"
        ) from error
    if max(positions) >= len(source):
        raise PrototypeConstructionError(
            "prototype token attention is shorter than a source cache block"
        )
    values = np.asarray(source[list(positions)], dtype=np.float64)
    if not bool(np.all(np.isfinite(values))) or bool(np.any(values < 0)):
        raise PrototypeConstructionError("prototype token attention must be finite and nonnegative")
    tiny = float(np.finfo(np.float64).tiny)
    values = np.maximum(values, tiny)
    mass = float(values.sum())
    return tuple(float(value / mass) for value in values), mass


def _prototype_logical_position(
    anchor: KVBlockDescriptor,
    member_blocks: Sequence[KVBlockDescriptor],
    block_weights: Sequence[float],
    position_weights: Sequence[Sequence[float]],
    strategy: str,
) -> int:
    """Choose a deterministic original position at which to phase one prototype."""

    if strategy == "selected_anchor_start":
        return anchor.original_logical_positions[0]
    if strategy != "forecast_weighted_member_medoid":  # pragma: no cover - config guards this
        raise PrototypeConstructionError(f"unsupported prototype anchor strategy: {strategy}")
    total = float(sum(block_weights))
    if total <= 0:
        raise PrototypeConstructionError("prototype block weights must be positive")
    weighted_positions: list[tuple[int, float]] = []
    for block, block_weight, within in zip(
        member_blocks,
        block_weights,
        position_weights,
        strict=True,
    ):
        if len(within) != block.position_count:
            raise PrototypeConstructionError(
                "prototype position weights do not align with member positions"
            )
        for position, local_weight in zip(
            block.original_logical_positions,
            within,
            strict=True,
        ):
            weighted_positions.append((position, block_weight * local_weight / total))
    center = sum(position * weight for position, weight in weighted_positions)
    return min(
        (position for position, _weight in weighted_positions),
        key=lambda position: (abs(position - center), position),
    )


def _cast_like(tensor: Any, reference: Any) -> Any:
    if tensor.__class__.__module__.startswith("torch") and hasattr(tensor, "to"):
        return tensor.to(dtype=reference.dtype, device=reference.device)
    return np.asarray(tensor).astype(np.asarray(reference).dtype, copy=True)


def _weighted_average(
    pooled: Sequence[Any], weights: Sequence[float], reference: Any
) -> tuple[Any, Any]:
    if not pooled or len(pooled) != len(weights):
        raise PrototypeConstructionError(
            "prototype tensors and weights must be non-empty and aligned"
        )
    total = float(sum(weights))
    if total <= 0:
        raise PrototypeConstructionError("prototype weights must sum to a positive value")
    normalized = tuple(weight / total for weight in weights)
    if pooled[0].__class__.__module__.startswith("torch"):
        import torch

        torch_stacked = torch.stack(tuple(pooled), dim=0)
        weight_tensor = torch.as_tensor(
            normalized,
            dtype=torch_stacked.dtype,
            device=torch_stacked.device,
        ).reshape((-1,) + (1,) * (torch_stacked.ndim - 1))
        average = torch.sum(torch_stacked * weight_tensor, dim=0)
    else:
        numpy_stacked = np.stack(tuple(np.asarray(tensor) for tensor in pooled), axis=0)
        weight_array = np.asarray(normalized, dtype=numpy_stacked.dtype).reshape(
            (-1,) + (1,) * (numpy_stacked.ndim - 1)
        )
        average = np.sum(numpy_stacked * weight_array, axis=0)
    return average, _cast_like(average, reference)


@dataclass(frozen=True, slots=True)
class _DeviceMemberStatistics:
    """Deferred per-cluster reductions kept on device until one bulk transfer."""

    packed_members: Any
    packed_cluster: Any


_ResolvedMemberStatistics = tuple[
    tuple[float, ...],
    tuple[float, ...],
    float,
    float,
]


@dataclass(frozen=True, slots=True)
class _PendingPrototypeRecord:
    """Prototype metadata awaiting bulk-resolved dispersion statistics."""

    prototype_id: int
    anchor: KVBlockDescriptor
    anchor_id: int
    prototype_logical_position: int
    member_ids: tuple[int, ...]
    member_blocks: tuple[KVBlockDescriptor, ...]
    raw_weights: tuple[float, ...]
    position_weights: tuple[tuple[float, ...], ...]
    prototype_key: Any
    prototype_value: Any
    statistics: _DeviceMemberStatistics | _ResolvedMemberStatistics


@dataclass(frozen=True, slots=True)
class _PrototypeRecipe:
    """Tensor-independent prototype metadata used by the batched reducer."""

    prototype_id: int
    anchor: KVBlockDescriptor
    anchor_id: int
    prototype_logical_position: int
    member_ids: tuple[int, ...]
    member_blocks: tuple[KVBlockDescriptor, ...]
    raw_weights: tuple[float, ...]
    position_weights: tuple[tuple[float, ...], ...]


def _batched_pool_source_members(
    full_state: FullKVState,
    gathered_by_node: Mapping[int, tuple[Any, Any]],
    blocks_by_node: Mapping[int, KVBlockDescriptor],
    position_weights_by_node: Mapping[int, tuple[float, ...]],
    key_transform: RotaryPrototypeKeyTransform | None,
) -> tuple[dict[int, Any], dict[int, Any]]:
    """Pool all source blocks in shape-compatible batches.

    Long visual/document prompts can contain tens of thousands of block/head
    pairs.  Pooling each independently is launch-bound on a GPU.  Shape
    grouping preserves every source block and its local attention weights
    while reducing the launch count to the number of distinct block shapes.
    """

    node_ids = tuple(sorted(gathered_by_node))
    if not node_ids:
        return {}, {}
    first_key = gathered_by_node[node_ids[0]][0]
    if not first_key.__class__.__module__.startswith("torch"):
        numpy_pooled_keys: dict[int, Any] = {}
        numpy_pooled_values: dict[int, Any] = {}
        for node_id in node_ids:
            block = blocks_by_node[node_id]
            key, value = gathered_by_node[node_id]
            layer = full_state.layers[block.layer]
            local_weights = position_weights_by_node[node_id]
            if key_transform is None:
                numpy_pooled_keys[node_id] = _pool_fp32(
                    key,
                    layer.key_sequence_dimension,
                    local_weights,
                )
            else:
                numpy_pooled_keys[node_id] = key_transform.phase_neutral_pool(
                    key,
                    block,
                    sequence_dimension=layer.key_sequence_dimension,
                    position_weights=local_weights,
                )
            numpy_pooled_values[node_id] = _pool_fp32(
                value,
                layer.value_sequence_dimension,
                local_weights,
            )
        return numpy_pooled_keys, numpy_pooled_values

    import torch

    pooled_keys: dict[int, Any] = {}
    pooled_values: dict[int, Any] = {}
    shape_groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for node_id in node_ids:
        block = blocks_by_node[node_id]
        key, value = gathered_by_node[node_id]
        layer = full_state.layers[block.layer]
        shape_groups[
            (
                tuple(int(size) for size in key.shape),
                tuple(int(size) for size in value.shape),
                layer.key_sequence_dimension % int(key.ndim),
                layer.value_sequence_dimension % int(value.ndim),
                str(key.device),
                str(value.device),
            )
        ].append(node_id)
    for signature in sorted(shape_groups, key=str):
        members = tuple(shape_groups[signature])
        keys = torch.stack(
            tuple(gathered_by_node[node_id][0].detach().float() for node_id in members),
            dim=0,
        )
        values = torch.stack(
            tuple(gathered_by_node[node_id][1].detach().float() for node_id in members),
            dim=0,
        )
        key_axis = int(signature[2]) + 1
        value_axis = int(signature[3]) + 1
        local_position_weights_batch = tuple(
            position_weights_by_node[node_id] for node_id in members
        )

        if key_transform is None:
            key_width = int(keys.shape[key_axis])
            key_weight = torch.as_tensor(
                local_position_weights_batch,
                dtype=keys.dtype,
                device=keys.device,
            ).reshape(
                (len(members),)
                + (1,) * (key_axis - 1)
                + (key_width,)
                + (1,) * (keys.ndim - key_axis - 1)
            )
            pooled_key_batch = torch.sum(keys * key_weight, dim=key_axis, keepdim=True)
        else:
            if key_axis != keys.ndim - 2:
                raise PrototypeConstructionError(
                    "batched RoPE prototype pooling requires the cache sequence axis at -2"
                )
            positions = torch.as_tensor(
                tuple(blocks_by_node[node_id].original_logical_positions for node_id in members),
                dtype=torch.long,
                device=key_transform.cos.device,
            )
            if int(positions.min()) < 0 or int(positions.max()) >= int(key_transform.cos.shape[-2]):
                raise PrototypeConstructionError(
                    "prototype logical position lies outside the captured prefill RoPE phase"
                )
            cos = key_transform.cos[0, 0][positions].to(device=keys.device, dtype=torch.float32)
            sin = key_transform.sin[0, 0][positions].to(device=keys.device, dtype=torch.float32)
            phase_shape = (len(members),) + (1,) * (keys.ndim - 3) + tuple(cos.shape[-2:])
            cos = cos.reshape(phase_shape)
            sin = sin.reshape(phase_shape)
            denominator = (cos * cos + sin * sin).clamp_min(1e-12)
            neutral = (keys * cos - key_transform._rotate_half(keys) * sin) / denominator
            key_width = int(keys.shape[-2])
            key_weight = torch.as_tensor(
                local_position_weights_batch,
                dtype=neutral.dtype,
                device=neutral.device,
            ).reshape((len(members),) + (1,) * (neutral.ndim - 3) + (key_width, 1))
            pooled_key_batch = torch.sum(neutral * key_weight, dim=-2, keepdim=True)

        value_width = int(values.shape[value_axis])
        value_weight = torch.as_tensor(
            local_position_weights_batch,
            dtype=values.dtype,
            device=values.device,
        ).reshape(
            (len(members),)
            + (1,) * (value_axis - 1)
            + (value_width,)
            + (1,) * (values.ndim - value_axis - 1)
        )
        pooled_value_batch = torch.sum(values * value_weight, dim=value_axis, keepdim=True)
        for index, node_id in enumerate(members):
            pooled_keys[node_id] = pooled_key_batch[index]
            pooled_values[node_id] = pooled_value_batch[index]
    return pooled_keys, pooled_values


def _batched_torch_prototypes(
    recipes: Sequence[_PrototypeRecipe],
    pooled_keys_by_node: Mapping[int, Any],
    pooled_values_by_node: Mapping[int, Any],
    gathered_by_node: Mapping[int, tuple[Any, Any]],
    key_transform: RotaryPrototypeKeyTransform | None,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[_DeviceMemberStatistics, ...]]:
    """Compute ordered per-cluster reductions in count-compatible batches."""

    import torch

    prototype_keys: list[Any | None] = [None] * len(recipes)
    prototype_values: list[Any | None] = [None] * len(recipes)
    statistics: list[_DeviceMemberStatistics | None] = [None] * len(recipes)
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for recipe_index, recipe in enumerate(recipes):
        first_id = recipe.member_ids[0]
        key = pooled_keys_by_node[first_id]
        value = pooled_values_by_node[first_id]
        reference_key, reference_value = gathered_by_node[first_id]
        groups[
            (
                tuple(int(size) for size in key.shape),
                tuple(int(size) for size in value.shape),
                str(key.device),
                str(value.device),
                str(reference_key.dtype),
                str(reference_value.dtype),
                len(recipe.member_ids),
            )
        ].append(recipe_index)

    for signature in sorted(groups, key=str):
        recipe_indices = tuple(groups[signature])
        member_count = len(recipes[recipe_indices[0]].member_ids)
        normalized_rows = tuple(
            tuple(
                float(weight) / float(sum(recipes[index].raw_weights))
                for weight in recipes[index].raw_weights
            )
            for index in recipe_indices
        )
        if any(len(row) != member_count for row in normalized_rows):
            raise RuntimeError("prototype count grouping mixed cluster sizes")
        if any(
            not math.isfinite(weight) or weight <= 0 for row in normalized_rows for weight in row
        ):
            raise PrototypeConstructionError("prototype weights must be finite and positive")
        stacked_keys = torch.stack(
            tuple(
                torch.stack(
                    tuple(pooled_keys_by_node[node_id] for node_id in recipes[index].member_ids),
                    dim=0,
                )
                for index in recipe_indices
            ),
            dim=0,
        )
        stacked_values = torch.stack(
            tuple(
                torch.stack(
                    tuple(pooled_values_by_node[node_id] for node_id in recipes[index].member_ids),
                    dim=0,
                )
                for index in recipe_indices
            ),
            dim=0,
        )
        normalized = torch.as_tensor(
            normalized_rows,
            dtype=stacked_keys.dtype,
            device=stacked_keys.device,
        )
        broadcast = normalized.reshape((*normalized.shape,) + (1,) * (stacked_keys.ndim - 2))
        key_averages = torch.sum(stacked_keys * broadcast, dim=1)
        value_broadcast = normalized.to(
            device=stacked_values.device,
            dtype=stacked_values.dtype,
        ).reshape((*normalized.shape,) + (1,) * (stacked_values.ndim - 2))
        value_averages = torch.sum(stacked_values * value_broadcast, dim=1)

        reduction_axes = tuple(range(2, stacked_keys.ndim))
        key_errors = torch.mean(
            (stacked_keys - key_averages.unsqueeze(1)) ** 2,
            dim=reduction_axes,
        )
        value_errors = torch.mean(
            (stacked_values - value_averages.unsqueeze(1)) ** 2,
            dim=tuple(range(2, stacked_values.ndim)),
        )
        value_norms = torch.sqrt(
            torch.mean(
                stacked_values * stacked_values,
                dim=tuple(range(2, stacked_values.ndim)),
            ).clamp_min(0)
        )
        key_dispersion = torch.sum(key_errors * normalized, dim=1)
        value_dispersion = torch.sum(
            value_errors * normalized.to(device=value_errors.device, dtype=value_errors.dtype),
            dim=1,
        )

        first_reference_key = gathered_by_node[recipes[recipe_indices[0]].member_ids[0]][0]
        first_reference_value = gathered_by_node[recipes[recipe_indices[0]].member_ids[0]][1]
        if key_transform is None:
            key_batch = key_averages.to(
                dtype=first_reference_key.dtype,
                device=first_reference_key.device,
            )
        else:
            key_batch = key_transform.apply_anchor_phase_batch(
                key_averages,
                tuple(recipes[index].prototype_logical_position for index in recipe_indices),
                first_reference_key,
            )
        value_batch = value_averages.to(
            dtype=first_reference_value.dtype,
            device=first_reference_value.device,
        )
        for local_cluster, recipe_index in enumerate(recipe_indices):
            prototype_keys[recipe_index] = key_batch[local_cluster]
            prototype_values[recipe_index] = value_batch[local_cluster]
            statistics[recipe_index] = _DeviceMemberStatistics(
                torch.stack(
                    (
                        0.5 * (key_errors[local_cluster] + value_errors[local_cluster]),
                        value_norms[local_cluster],
                    ),
                    dim=-1,
                ),
                torch.stack((key_dispersion[local_cluster], value_dispersion[local_cluster])),
            )
    if any(value is None for value in (*prototype_keys, *prototype_values, *statistics)):
        raise RuntimeError("batched prototype reduction left an uninitialized payload")
    return (
        tuple(value for value in prototype_keys if value is not None),
        tuple(value for value in prototype_values if value is not None),
        tuple(value for value in statistics if value is not None),
    )


def _member_statistics(
    pooled_keys: Sequence[Any],
    key_average: Any,
    pooled_values: Sequence[Any],
    value_average: Any,
    weights: Sequence[float],
) -> _DeviceMemberStatistics | _ResolvedMemberStatistics:
    """Compute member reductions while deferring device synchronization."""

    if pooled_keys[0].__class__.__module__.startswith("torch"):
        import torch

        torch_keys = torch.stack(tuple(pooled_keys), dim=0)
        torch_values = torch.stack(tuple(pooled_values), dim=0)
        axes = tuple(range(1, torch_keys.ndim))
        key_errors = torch.mean((torch_keys - key_average.unsqueeze(0)) ** 2, dim=axes)
        value_errors = torch.mean((torch_values - value_average.unsqueeze(0)) ** 2, dim=axes)
        value_norms = torch.sqrt(torch.mean(torch_values * torch_values, dim=axes).clamp_min(0))
        packed = torch.stack((0.5 * (key_errors + value_errors), value_norms), dim=-1)
        torch_normalized = torch.as_tensor(
            tuple(weight / sum(weights) for weight in weights),
            dtype=key_errors.dtype,
            device=key_errors.device,
        )
        cluster = torch.stack(
            (
                torch.sum(key_errors * torch_normalized),
                torch.sum(value_errors * torch_normalized),
            )
        )
        return _DeviceMemberStatistics(packed, cluster)
    numpy_keys = np.stack(tuple(np.asarray(value) for value in pooled_keys), axis=0)
    numpy_values = np.stack(tuple(np.asarray(value) for value in pooled_values), axis=0)
    axes = tuple(range(1, numpy_keys.ndim))
    key_errors = np.mean((numpy_keys - np.expand_dims(key_average, 0)) ** 2, axis=axes)
    value_errors = np.mean((numpy_values - np.expand_dims(value_average, 0)) ** 2, axis=axes)
    value_norms = np.sqrt(np.maximum(0.0, np.mean(numpy_values * numpy_values, axis=axes)))
    numpy_normalized = np.asarray(weights, dtype=np.float64)
    numpy_normalized /= float(numpy_normalized.sum())
    return (
        tuple(float(value) for value in 0.5 * (key_errors + value_errors)),
        tuple(float(value) for value in value_norms),
        max(0.0, float(np.dot(key_errors, numpy_normalized))),
        max(0.0, float(np.dot(value_errors, numpy_normalized))),
    )


def _resolve_member_statistics(
    values: Sequence[_DeviceMemberStatistics | _ResolvedMemberStatistics],
) -> tuple[_ResolvedMemberStatistics, ...]:
    """Resolve all device statistics through one synchronized host transfer."""

    if not values:
        return ()
    device_indices = tuple(
        index for index, value in enumerate(values) if isinstance(value, _DeviceMemberStatistics)
    )
    if not device_indices:
        return tuple(value for value in values if not isinstance(value, _DeviceMemberStatistics))
    if len(device_indices) != len(values):
        raise PrototypeConstructionError(
            "prototype statistics cannot mix NumPy and torch payloads in one cache"
        )
    import torch

    device_values = tuple(
        cast("_DeviceMemberStatistics", values[index]) for index in device_indices
    )
    if not all(isinstance(value, _DeviceMemberStatistics) for value in device_values):
        raise PrototypeConstructionError("deferred prototype statistics are malformed")
    counts = tuple(int(value.packed_members.shape[0]) for value in device_values)
    members = torch.cat(tuple(value.packed_members for value in device_values), dim=0)
    clusters = torch.stack(tuple(value.packed_cluster for value in device_values), dim=0)
    member_scalar_count = int(members.numel())
    combined = torch.cat((members.reshape(-1), clusters.reshape(-1)), dim=0)
    host = combined.detach().cpu().tolist()
    member_host = host[:member_scalar_count]
    cluster_host = host[member_scalar_count:]
    resolved: list[_ResolvedMemberStatistics] = []
    member_offset = 0
    cluster_offset = 0
    for count in counts:
        member_rows = tuple(
            (
                max(0.0, float(member_host[member_offset + 2 * index])),
                max(0.0, float(member_host[member_offset + 2 * index + 1])),
            )
            for index in range(count)
        )
        resolved.append(
            (
                tuple(row[0] for row in member_rows),
                tuple(row[1] for row in member_rows),
                max(0.0, float(cluster_host[cluster_offset])),
                max(0.0, float(cluster_host[cluster_offset + 1])),
            )
        )
        member_offset += 2 * count
        cluster_offset += 2
    return tuple(resolved)


def _prototype_payloads(
    full_state: FullKVState,
    graph: SparseEvidenceGraph,
    assignments: Mapping[int, tuple[int, float]],
    selection: SelectionResult,
    key_transform: RotaryPrototypeKeyTransform | None,
    config: PrototypeConfig,
    prototype_weight_by_node: Mapping[int, float] | None,
    prototype_token_weight_by_head: Mapping[tuple[int, int], Any] | None,
) -> tuple[PrototypeTier, tuple[PrototypeRecord, ...], dict[int, int], ExactTier]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for node_id, (anchor_id, _weight) in assignments.items():
        grouped[anchor_id].append(node_id)
    ordered_member_ids = tuple(
        node_id for anchor_id in sorted(grouped) for node_id in sorted(grouped[anchor_id])
    )
    ordered_member_blocks = tuple(graph.nodes[node_id].block for node_id in ordered_member_ids)
    # Gather every unselected source block once per layer/head.  Gathering one
    # prototype cluster at a time launches thousands of small indexing kernels
    # for long documents and videos.  The views below retain the identical
    # block order and feed both prototype construction and residual storage.
    gathered_sources = full_state.gather_exact_blocks(
        ordered_member_blocks,
        gather_mode=config.source_gather_mode,
    )
    gathered_by_node = {
        node_id: (key, value)
        for node_id, key, value in zip(
            ordered_member_ids,
            gathered_sources.key_blocks,
            gathered_sources.value_blocks,
            strict=True,
        )
    }
    recipes: list[_PrototypeRecipe] = []
    position_weights_by_node: dict[int, tuple[float, ...]] = {}
    for prototype_id, anchor_id in enumerate(sorted(grouped)):
        member_ids = tuple(sorted(grouped[anchor_id]))
        member_blocks = tuple(graph.nodes[node_id].block for node_id in member_ids)
        affinity_weights = tuple(assignments[node_id][1] for node_id in member_ids)
        position_weights_and_mass = tuple(
            _normalized_position_weights(
                block,
                (
                    prototype_token_weight_by_head
                    if config.pooling_mode == "forecast_attention_token"
                    else None
                ),
            )
            for block in member_blocks
        )
        position_weights = tuple(item[0] for item in position_weights_and_mass)
        token_attention_masses = tuple(item[1] for item in position_weights_and_mass)
        attention_values = tuple(
            max(0.0, float((prototype_weight_by_node or {}).get(node_id, 0.0)))
            for node_id in member_ids
        )
        if config.pooling_mode == "forecast_attention_token":
            if prototype_token_weight_by_head is None:
                raise PrototypeConstructionError(
                    "forecast-attention token pooling requires per-head token evidence"
                )
            raw_weights = token_attention_masses
        else:
            maximum_attention = max(attention_values, default=0.0)
            raw_weights = tuple(
                affinity
                * (
                    (1.0 - config.forecast_weight)
                    + config.forecast_weight
                    * (1e-6 + (attention / maximum_attention if maximum_attention > 0 else 0.0))
                )
                for affinity, attention in zip(affinity_weights, attention_values, strict=True)
            )
        anchor = graph.nodes[anchor_id].block
        prototype_logical_position = _prototype_logical_position(
            anchor,
            member_blocks,
            raw_weights,
            position_weights,
            config.anchor_position_strategy,
        )
        recipes.append(
            _PrototypeRecipe(
                prototype_id=prototype_id,
                anchor=anchor,
                anchor_id=anchor_id,
                prototype_logical_position=prototype_logical_position,
                member_ids=member_ids,
                member_blocks=member_blocks,
                raw_weights=raw_weights,
                position_weights=position_weights,
            )
        )
        position_weights_by_node.update(zip(member_ids, position_weights, strict=True))

    blocks_by_node = {node_id: graph.nodes[node_id].block for node_id in ordered_member_ids}
    pooled_keys_by_node, pooled_values_by_node = _batched_pool_source_members(
        full_state,
        gathered_by_node,
        blocks_by_node,
        position_weights_by_node,
        key_transform,
    )
    first_pooled = pooled_keys_by_node[ordered_member_ids[0]]
    prototype_keys_tuple: tuple[Any, ...]
    prototype_values_tuple: tuple[Any, ...]
    statistics_tuple: tuple[_DeviceMemberStatistics | _ResolvedMemberStatistics, ...]
    if first_pooled.__class__.__module__.startswith("torch"):
        (
            prototype_keys_tuple,
            prototype_values_tuple,
            device_statistics,
        ) = _batched_torch_prototypes(
            recipes,
            pooled_keys_by_node,
            pooled_values_by_node,
            gathered_by_node,
            key_transform,
        )
        statistics_tuple = tuple(device_statistics)
    else:
        prototype_keys: list[Any] = []
        prototype_values: list[Any] = []
        numpy_statistics: list[_ResolvedMemberStatistics] = []
        for recipe in recipes:
            pooled_keys = tuple(pooled_keys_by_node[node_id] for node_id in recipe.member_ids)
            pooled_values = tuple(pooled_values_by_node[node_id] for node_id in recipe.member_ids)
            reference_key, reference_value = gathered_by_node[recipe.member_ids[0]]
            key_average_fp32, prototype_key = _weighted_average(
                pooled_keys,
                recipe.raw_weights,
                reference_key,
            )
            if key_transform is not None:  # pragma: no cover - transform requires torch
                prototype_key = key_transform.apply_anchor_phase(
                    key_average_fp32,
                    recipe.prototype_logical_position,
                    reference_key,
                )
            value_average_fp32, prototype_value = _weighted_average(
                pooled_values,
                recipe.raw_weights,
                reference_value,
            )
            prototype_keys.append(prototype_key)
            prototype_values.append(prototype_value)
            numpy_statistics.append(
                cast(
                    "_ResolvedMemberStatistics",
                    _member_statistics(
                        pooled_keys,
                        key_average_fp32,
                        pooled_values,
                        value_average_fp32,
                        recipe.raw_weights,
                    ),
                )
            )
        prototype_keys_tuple = tuple(prototype_keys)
        prototype_values_tuple = tuple(prototype_values)
        statistics_tuple = tuple(numpy_statistics)

    source_blocks: list[KVBlockDescriptor] = []
    tier_assignments: list[int] = []
    source_to_prototype: dict[int, int] = {}
    pending_records: list[_PendingPrototypeRecord] = []
    for recipe, prototype_key, prototype_value, statistics in zip(
        recipes,
        prototype_keys_tuple,
        prototype_values_tuple,
        statistics_tuple,
        strict=True,
    ):
        pending_records.append(
            _PendingPrototypeRecord(
                prototype_id=recipe.prototype_id,
                anchor=recipe.anchor,
                anchor_id=recipe.anchor_id,
                prototype_logical_position=recipe.prototype_logical_position,
                member_ids=recipe.member_ids,
                member_blocks=recipe.member_blocks,
                raw_weights=recipe.raw_weights,
                position_weights=recipe.position_weights,
                prototype_key=prototype_key,
                prototype_value=prototype_value,
                statistics=statistics,
            )
        )
        for node_id, block in zip(recipe.member_ids, recipe.member_blocks, strict=True):
            source_blocks.append(block)
            tier_assignments.append(recipe.prototype_id)
            source_to_prototype[node_id] = recipe.prototype_id
    resolved_statistics = _resolve_member_statistics(
        tuple(pending.statistics for pending in pending_records)
    )
    records: list[PrototypeRecord] = []
    for pending, statistics in zip(pending_records, resolved_statistics, strict=True):
        member_errors, value_norms, key_dispersion, value_dispersion = statistics
        total_weight = sum(pending.raw_weights)
        members = tuple(
            PrototypeMember(
                node_id,
                raw_weight,
                raw_weight / total_weight,
                reconstruction_error,
                value_norm,
                local_position_weights,
            )
            for (
                node_id,
                raw_weight,
                reconstruction_error,
                value_norm,
                local_position_weights,
            ) in zip(
                pending.member_ids,
                pending.raw_weights,
                member_errors,
                value_norms,
                pending.position_weights,
                strict=True,
            )
        )
        modalities = Counter(block.modality for block in pending.member_blocks)
        logical_positions = tuple(
            position
            for block in pending.member_blocks
            for position in block.original_logical_positions
        )
        source_bytes = sum(block.byte_size for block in pending.member_blocks)
        prototype_bytes = tensor_storage_bytes(pending.prototype_key) + tensor_storage_bytes(
            pending.prototype_value
        )
        records.append(
            PrototypeRecord(
                prototype_id=pending.prototype_id,
                layer=pending.anchor.layer,
                kv_head=pending.anchor.kv_head,
                anchor_node_id=pending.anchor_id,
                anchor_logical_position=pending.prototype_logical_position,
                members=members,
                assigned_node_ids=pending.member_ids,
                diagnostics=PrototypeDiagnostics(
                    cluster_size=len(pending.member_ids),
                    key_dispersion=key_dispersion,
                    value_dispersion=value_dispersion,
                    modality_composition=tuple(
                        sorted(modalities.items(), key=lambda item: item[0].value)
                    ),
                    minimum_logical_position=min(logical_positions),
                    maximum_logical_position=max(logical_positions),
                    position_span=max(logical_positions) - min(logical_positions),
                    source_member_bytes=source_bytes,
                    prototype_bytes=prototype_bytes,
                    active_bytes_saved=source_bytes - prototype_bytes,
                ),
                represented_source_positions=sum(
                    block.position_count for block in pending.member_blocks
                ),
                attention_log_bias_scale=config.attention_log_bias_scale,
                attention_bias_basis=config.attention_bias_basis,
                pooling_mode=config.pooling_mode,
                anchor_position_strategy=config.anchor_position_strategy,
                eviction_utility=selection.decisions[pending.anchor_id].marginal_gain,
                utility_provenance=(
                    "selected_anchor_marginal_gain;prototype_weight="
                    + (
                        "forecast_attention_token_mass;"
                        if config.pooling_mode == "forecast_attention_token"
                        else (
                            f"graph_affinity_plus_{config.forecast_weight:.6f}_forecast_attention;"
                        )
                    )
                    + "attention_mass_correction="
                    f"{config.attention_log_bias_scale:.6f}_times_"
                    f"{config.attention_bias_basis};anchor_position="
                    f"{config.anchor_position_strategy}"
                ),
            )
        )
    tier = PrototypeTier(
        tuple(source_blocks),
        prototype_keys_tuple,
        prototype_values_tuple,
        tuple(tier_assignments),
    )
    return tier, tuple(records), source_to_prototype, gathered_sources


def _selection_budget_violation(
    exact: ExactTier,
    prototype_tier: PrototypeTier,
    selection: SelectionResult,
) -> str | None:
    unit = selection.budget.unit
    if unit is BudgetUnit.BLOCKS:
        active = len(exact.blocks) + len(prototype_tier.prototype_keys)
    elif unit is BudgetUnit.RETAINED_SLOTS:
        active = sum(block.position_count for block in exact.blocks) + len(
            prototype_tier.prototype_keys
        )
    elif unit is BudgetUnit.BYTES:
        active = exact.active_bytes + prototype_tier.active_bytes
    else:  # pragma: no cover - guarded by the enum
        raise PrototypeConstructionError(f"unsupported selection budget unit: {unit}")
    if active <= selection.budget.value:
        return None
    return (
        f"exact plus prototype active {unit.value} ({active}) exceed the "
        f"configured selection budget ({selection.budget.value})"
    )


def estimate_prototype_active_cost(
    graph: SparseEvidenceGraph,
    selection: SelectionResult,
    config: PrototypeConfig,
    *,
    position_limits: Mapping[tuple[int, int], int] | None = None,
    prototype_anchor_node_ids: Sequence[int] | None = None,
) -> tuple[int | None, str | None]:
    """Validate assignments and estimate exact prototype cost without tensor copies.

    Candidate-budget search can inspect dozens of selected sets. Materializing
    and casting every candidate K/V tier made planning proportional to the
    complete cache size times the number of search points. Fixed-size prototype
    slots have an exact descriptor-derived cost, so rejected candidates need no
    device gather at all.
    """

    assignments, reason = _assign_to_anchors(
        graph,
        frozenset(selection.selected_node_ids),
        config,
        prototype_anchor_node_ids=(
            None
            if prototype_anchor_node_ids is None
            else frozenset(int(node_id) for node_id in prototype_anchor_node_ids)
        ),
    )
    if reason is not None:
        return None, reason
    prototype_anchor_ids = tuple(sorted({anchor_id for anchor_id, _weight in assignments.values()}))
    if position_limits is not None:
        active_positions: dict[tuple[int, int], int] = defaultdict(int)
        for block in selection.selected_blocks:
            active_positions[(block.layer, block.kv_head)] += block.position_count
        for anchor_id in prototype_anchor_ids:
            anchor = graph.nodes[anchor_id].block
            active_positions[(anchor.layer, anchor.kv_head)] += 1
        violations = {
            identity: (active_positions[identity], limit)
            for identity, limit in position_limits.items()
            if active_positions[identity] > limit
        }
        if violations:
            return None, f"prototype slots exceed dense layer/head capacities: {violations}"
    if selection.budget.unit is BudgetUnit.BLOCKS:
        return len(selection.selected_blocks) + len(prototype_anchor_ids), None
    if selection.budget.unit is BudgetUnit.RETAINED_SLOTS:
        return (
            sum(block.position_count for block in selection.selected_blocks)
            + len(prototype_anchor_ids),
            None,
        )
    if selection.budget.unit is BudgetUnit.BYTES:
        prototype_bytes = 0
        for anchor_id in prototype_anchor_ids:
            anchor = graph.nodes[anchor_id].block
            if anchor.byte_size % anchor.position_count:
                return None, "prototype anchor bytes are not divisible by its position count"
            prototype_bytes += anchor.byte_size // anchor.position_count
        return selection.active_bytes + prototype_bytes, None
    raise PrototypeConstructionError(f"unsupported selection budget unit: {selection.budget.unit}")


def construct_three_tier_cache(
    full_state: FullKVState,
    graph: SparseEvidenceGraph,
    selection: SelectionResult,
    capabilities: AdapterCapabilities,
    *,
    prototype_config: PrototypeConfig | None = None,
    residual_config: ResidualConfig | None = None,
    active_byte_budget: int | None = None,
    total_storage_budget_bytes: int | None = None,
    retention_ratio: float | None = None,
    key_transform: RotaryPrototypeKeyTransform | None = None,
    prototype_weight_by_node: Mapping[int, float] | None = None,
    prototype_token_weight_by_head: Mapping[tuple[int, int], Any] | None = None,
    exact_gather_mode: str = "batched_layer",
    prototype_anchor_node_ids: Sequence[int] | None = None,
) -> ThreeTierCacheConstruction:
    """Build exact, prototype, and CPU residual tiers or safely select exact-only.

    The current policy deliberately rejects all post-RoPE prototype averaging.
    A model family must explicitly advertise prototype support, and its observed
    source-cache metadata must agree with the adapter declaration.
    """

    prototypes = prototype_config or PrototypeConfig()
    residuals = residual_config or ResidualConfig()
    if exact_gather_mode not in {"per_head_reference", "batched_layer"}:
        raise ValueError(
            "exact_gather_mode must be 'per_head_reference' or 'batched_layer'"
        )
    _validate_inputs(full_state, graph, selection)
    if retention_ratio is not None and (
        not math.isfinite(retention_ratio) or not 0 < retention_ratio <= 1
    ):
        raise ValueError("retention_ratio must be finite and in the interval (0, 1]")
    if active_byte_budget is not None and active_byte_budget < 1:
        raise ValueError("active_byte_budget must be positive when provided")
    if total_storage_budget_bytes is not None and total_storage_budget_bytes < 1:
        raise ValueError("total_storage_budget_bytes must be positive when provided")
    if selection.budget.unit is BudgetUnit.BYTES:
        active_byte_budget = (
            selection.budget.value
            if active_byte_budget is None
            else min(active_byte_budget, selection.budget.value)
        )
    if active_byte_budget is not None and selection.active_bytes > active_byte_budget:
        raise PrototypeConstructionError(
            "selected exact blocks already exceed active_byte_budget: "
            f"exact={selection.active_bytes}, budget={active_byte_budget}"
        )
    if (
        total_storage_budget_bytes is not None
        and selection.active_bytes > total_storage_budget_bytes
    ):
        raise PrototypeConstructionError(
            "selected exact blocks already exceed total_storage_budget_bytes: "
            f"exact={selection.active_bytes}, budget={total_storage_budget_bytes}"
        )
    safety = assess_prototype_safety(full_state, capabilities, key_transform)
    all_node_ids = tuple(range(len(graph.nodes)))
    if retention_ratio == 1.0 and selection.selected_node_ids != all_node_ids:
        raise PrototypeConstructionError(
            "retention_ratio=1.0 requires every source block to be selected exact"
        )
    if selection.selected_node_ids == all_node_ids:
        return _retention_one(
            full_state,
            safety,
            selection,
            capabilities.supports_residual_repair,
            exact_gather_mode=exact_gather_mode,
        )
    if not prototypes.enabled:
        return _exact_only(
            full_state,
            selection,
            safety,
            capabilities.supports_residual_repair,
            mode=TierConstructionMode.EXACT_ONLY_DISABLED,
            reason="prototype construction is disabled by configuration",
            exact_gather_mode=exact_gather_mode,
        )
    if not safety.safe:
        return _exact_only(
            full_state,
            selection,
            safety,
            capabilities.supports_residual_repair,
            mode=TierConstructionMode.EXACT_ONLY_UNSAFE,
            reason=safety.reason,
            exact_gather_mode=exact_gather_mode,
        )

    selected = frozenset(selection.selected_node_ids)
    assignments, incompatibility = _assign_to_anchors(
        graph,
        selected,
        prototypes,
        prototype_anchor_node_ids=(
            None
            if prototype_anchor_node_ids is None
            else frozenset(int(node_id) for node_id in prototype_anchor_node_ids)
        ),
    )
    if incompatibility is not None:
        return _exact_only(
            full_state,
            selection,
            safety,
            capabilities.supports_residual_repair,
            mode=TierConstructionMode.EXACT_ONLY_INCOMPATIBLE,
            reason=incompatibility,
            exact_gather_mode=exact_gather_mode,
        )
    exact = selection.to_exact_tier(full_state, gather_mode=exact_gather_mode)
    prototype_tier, records, source_to_prototype, gathered_sources = _prototype_payloads(
        full_state,
        graph,
        assignments,
        selection,
        key_transform,
        prototypes,
        prototype_weight_by_node,
        prototype_token_weight_by_head,
    )
    excessive_dispersion = tuple(
        record.prototype_id
        for record in records
        if (
            prototypes.max_key_dispersion is not None
            and record.diagnostics.key_dispersion > prototypes.max_key_dispersion
        )
        or (
            prototypes.max_value_dispersion is not None
            and record.diagnostics.value_dispersion > prototypes.max_value_dispersion
        )
    )
    if excessive_dispersion:
        return _exact_only(
            full_state,
            selection,
            safety,
            capabilities.supports_residual_repair,
            mode=TierConstructionMode.EXACT_ONLY_INCOMPATIBLE,
            reason=(
                f"prototype dispersion guard rejected cluster IDs {list(excessive_dispersion)}"
            ),
            exact_gather_mode=exact_gather_mode,
        )
    budget_violation = _selection_budget_violation(exact, prototype_tier, selection)
    if budget_violation is not None:
        return _exact_only(
            full_state,
            selection,
            safety,
            capabilities.supports_residual_repair,
            mode=TierConstructionMode.EXACT_ONLY_INCOMPATIBLE,
            reason=budget_violation,
            exact_gather_mode=exact_gather_mode,
        )
    if active_byte_budget is not None:
        active_bytes = exact.active_bytes + prototype_tier.active_bytes
        if active_bytes > active_byte_budget:
            return _exact_only(
                full_state,
                selection,
                safety,
                capabilities.supports_residual_repair,
                mode=TierConstructionMode.EXACT_ONLY_INCOMPATIBLE,
                reason=(
                    f"exact plus prototype active bytes ({active_bytes}) exceed the "
                    f"configured budget ({active_byte_budget})"
                ),
                exact_gather_mode=exact_gather_mode,
            )
    residual_node_order = tuple(
        node_id for record in records for node_id in record.assigned_node_ids
    )
    active_storage_bytes = exact.active_bytes + prototype_tier.active_bytes
    if (
        total_storage_budget_bytes is not None
        and active_storage_bytes > total_storage_budget_bytes
    ):
        raise PrototypeConstructionError(
            "active exact/prototype cache exceeds total storage budget: "
            f"active={active_storage_bytes}, total={total_storage_budget_bytes}"
        )
    residual_byte_limit = residuals.max_bytes
    if total_storage_budget_bytes is not None:
        remaining = total_storage_budget_bytes - active_storage_bytes
        residual_byte_limit = (
            remaining
            if residual_byte_limit is None
            else min(residual_byte_limit, remaining)
        )
    bounded_residuals = replace(residuals, max_bytes=residual_byte_limit)
    residual_priority_by_node = {
        member.node_id: (
            member.normalized_weight
            * member.reconstruction_error
            * (1.0 + member.value_output_norm)
        )
        for record in records
        for member in record.members
    }
    residual_report = build_residual_storage(
        full_state,
        source_to_prototype,
        bounded_residuals,
        node_order=residual_node_order,
        gathered=gathered_sources,
        priority_by_node=residual_priority_by_node,
    )
    state = MosaicKVState.create(
        full_state,
        exact=exact,
        prototypes=prototype_tier,
        residuals=residual_report.tier,
    )
    return ThreeTierCacheConstruction(
        state=state,
        mode=TierConstructionMode.THREE_TIER,
        reason=(
            "all unselected blocks were assigned to compatible selected anchors; "
            f"residuals stored for {len(residual_report.payloads)}/"
            f"{len(residual_node_order)} prototyped source blocks"
        ),
        safety=safety,
        adapter_declares_residual_repair=capabilities.supports_residual_repair,
        active_budget=selection.budget,
        exact_node_ids=selection.selected_node_ids,
        prototypes=records,
        residual_storage=residual_report,
        active_layouts=_active_layouts(full_state, exact, records),
        original_logical_sequence_length=full_state.original_logical_sequence_length,
        next_decode_position=full_state.next_decode_position,
    )


__all__ = [
    "PrototypeConstructionError",
    "assess_prototype_safety",
    "construct_three_tier_cache",
    "estimate_prototype_active_cost",
]
