"""Conservative exact/prototype/residual cache construction."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

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
        reason = (
            "prototype phase transform does not match the source cached-key representation"
        )
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
) -> ThreeTierCacheConstruction:
    exact = selection.to_exact_tier(full_state)
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
) -> ThreeTierCacheConstruction:
    exact = full_state.gather_exact_blocks(full_state.blocks)
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
    return (
        config.max_position_span is None or _pair_span(source, anchor) <= config.max_position_span
    )


def _assign_to_anchors(
    graph: SparseEvidenceGraph,
    selected: frozenset[int],
    config: PrototypeConfig,
) -> tuple[dict[int, tuple[int, float]], str | None]:
    weights = _undirected_anchor_weights(graph)
    neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (first, second), weight in weights.items():
        neighbors[first].append((second, weight))
        if first != second:
            neighbors[second].append((first, weight))
    selected_by_identity: dict[tuple[int, int, object], list[tuple[float, int]]] = defaultdict(
        list
    )
    for anchor_id in selected:
        anchor = graph.nodes[anchor_id]
        selected_by_identity[
            (anchor.block.layer, anchor.block.kv_head, anchor.block.modality)
        ].append((anchor.logical_center, anchor_id))
    for anchors in selected_by_identity.values():
        anchors.sort()
    allowed_anchor_modalities: dict[str, set[str]] = defaultdict(set)
    for pair in config.allowed_modality_pairs:
        source_modality, anchor_modality = pair.split(":", maxsplit=1)
        allowed_anchor_modalities[source_modality].add(anchor_modality)

    def positional_candidates(node_id: int) -> tuple[tuple[int, float], ...]:
        """Return a bounded compatibility scaffold independent of evidence scores."""

        source = graph.nodes[node_id]
        candidates: list[tuple[float, int]] = []
        window = max(1, 2 * config.group_size)
        for anchor_modality in sorted(
            allowed_anchor_modalities[source.block.modality.value]
        ):
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
                if not _compatible(source.block, anchor.block, config):
                    continue
                distance = abs(source.logical_center - center)
                span = config.max_position_span
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
            if anchor_id not in selected:
                continue
            anchor = graph.nodes[anchor_id]
            if (
                weight > 0
                and weight >= config.min_anchor_weight
                and _compatible(node.block, anchor.block, config)
            ):
                weights_by_anchor[anchor_id] = max(
                    weight, weights_by_anchor.get(anchor_id, 0.0)
                )
        for anchor_id, weight in positional_candidates(node_id):
            weights_by_anchor[anchor_id] = max(
                weight, weights_by_anchor.get(anchor_id, 0.0)
            )
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
            anchors = frozenset(node_id for node_id in member_ids if node_id in selected)
            if not anchors:
                return f"prototype anchor-cover segment {tuple(member_ids)} has no selected anchor"
            for node_id in member_ids:
                if node_id in selected:
                    continue
                candidates = tuple(
                    item for item in preferences[node_id] if item[1] in anchors
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
                exceeds_capacity = len(trial) > config.group_size + 1
                exceeds_span = bool(
                    config.max_position_span is not None
                    and max(positions) - min(positions) > config.max_position_span
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
        if config.max_position_span is None:
            return True
        positions = [
            position
            for candidate_id in (anchor_id, *member_ids)
            for position in graph.nodes[candidate_id].block.original_logical_positions
        ]
        return max(positions) - min(positions) <= config.max_position_span

    def assign(node_id: int, visited_nodes: set[int], visited_anchors: set[int]) -> bool:
        if node_id in visited_nodes:
            return False
        visited_nodes.add(node_id)
        for weight, anchor_id in preferences[node_id]:
            if anchor_id in visited_anchors:
                continue
            visited_anchors.add(anchor_id)
            members = groups[anchor_id]
            if len(members) < config.group_size and cluster_fits(
                anchor_id, (*members, node_id)
            ):
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
                f"prototypes.group_size={config.group_size}; {fallback_reason}"
            )

    for anchor_id, member_ids in groups.items():
        if config.max_position_span is not None:
            blocks = [graph.nodes[anchor_id].block]
            blocks.extend(graph.nodes[node_id].block for node_id in member_ids)
            positions = [
                position for block in blocks for position in block.original_logical_positions
            ]
            if max(positions) - min(positions) > config.max_position_span:
                fallback, fallback_reason = segmented_assignments()
                if fallback_reason is None:
                    return fallback, None
                return {}, (
                    f"anchor {anchor_id} cluster exceeds prototypes.max_position_span="
                    f"{config.max_position_span}; {fallback_reason}"
                )
    return assignments, None


def _pool_fp32(tensor: Any, axis: int) -> Any:
    if tensor.__class__.__module__.startswith("torch") and hasattr(tensor, "float"):
        return tensor.detach().float().mean(dim=axis, keepdim=True)
    return np.asarray(tensor, dtype=np.float32).mean(axis=axis, keepdims=True, dtype=np.float32)


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
    average = pooled[0] * normalized[0]
    for tensor, weight in zip(pooled[1:], normalized[1:], strict=True):
        average = average + tensor * weight
    return average, _cast_like(average, reference)


def _dispersion(pooled: Sequence[Any], average: Any, weights: Sequence[float]) -> float:
    total = float(sum(weights))
    result = 0.0
    for tensor, weight in zip(pooled, weights, strict=True):
        squared = (tensor - average) * (tensor - average)
        mean = squared.mean()
        value = float(mean.item()) if hasattr(mean, "item") else float(mean)
        result += (weight / total) * value
    return max(0.0, result)


def _prototype_payloads(
    full_state: FullKVState,
    graph: SparseEvidenceGraph,
    assignments: Mapping[int, tuple[int, float]],
    selection: SelectionResult,
    key_transform: RotaryPrototypeKeyTransform | None,
) -> tuple[PrototypeTier, tuple[PrototypeRecord, ...], dict[int, int], ExactTier]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for node_id, (anchor_id, _weight) in assignments.items():
        grouped[anchor_id].append(node_id)
    ordered_member_ids = tuple(
        node_id
        for anchor_id in sorted(grouped)
        for node_id in sorted(grouped[anchor_id])
    )
    ordered_member_blocks = tuple(graph.nodes[node_id].block for node_id in ordered_member_ids)
    # Gather every unselected source block once per layer/head.  Gathering one
    # prototype cluster at a time launches thousands of small indexing kernels
    # for long documents and videos.  The views below retain the identical
    # block order and feed both prototype construction and residual storage.
    gathered_sources = full_state.gather_exact_blocks(ordered_member_blocks)
    gathered_by_node = {
        node_id: (key, value)
        for node_id, key, value in zip(
            ordered_member_ids,
            gathered_sources.key_blocks,
            gathered_sources.value_blocks,
            strict=True,
        )
    }
    source_blocks: list[KVBlockDescriptor] = []
    tier_assignments: list[int] = []
    prototype_keys: list[Any] = []
    prototype_values: list[Any] = []
    records: list[PrototypeRecord] = []
    source_to_prototype: dict[int, int] = {}

    for prototype_id, anchor_id in enumerate(sorted(grouped)):
        member_ids = tuple(sorted(grouped[anchor_id]))
        member_blocks = tuple(graph.nodes[node_id].block for node_id in member_ids)
        gathered_keys = tuple(gathered_by_node[node_id][0] for node_id in member_ids)
        gathered_values = tuple(gathered_by_node[node_id][1] for node_id in member_ids)
        layer = full_state.layers[graph.nodes[anchor_id].block.layer]
        raw_weights = tuple(assignments[node_id][1] for node_id in member_ids)
        if key_transform is None:
            pooled_keys = tuple(
                _pool_fp32(tensor, layer.key_sequence_dimension) for tensor in gathered_keys
            )
            key_average_fp32, prototype_key = _weighted_average(
                pooled_keys, raw_weights, gathered_keys[0]
            )
        else:
            transformed = key_transform.merge(
                gathered_keys,
                member_blocks,
                raw_weights,
                sequence_dimension=layer.key_sequence_dimension,
                anchor_logical_position=(
                    graph.nodes[anchor_id].block.original_logical_positions[0]
                ),
            )
            pooled_keys = transformed.pooled_keys_fp32
            key_average_fp32 = transformed.average_key_fp32
            prototype_key = transformed.prototype_key
        pooled_values = tuple(
            _pool_fp32(tensor, layer.value_sequence_dimension) for tensor in gathered_values
        )
        value_average_fp32, prototype_value = _weighted_average(
            pooled_values, raw_weights, gathered_values[0]
        )
        prototype_keys.append(prototype_key)
        prototype_values.append(prototype_value)
        total_weight = sum(raw_weights)
        members = tuple(
            PrototypeMember(node_id, raw_weight, raw_weight / total_weight)
            for node_id, raw_weight in zip(member_ids, raw_weights, strict=True)
        )
        modalities = Counter(block.modality for block in member_blocks)
        logical_positions = tuple(
            position for block in member_blocks for position in block.original_logical_positions
        )
        source_bytes = sum(block.byte_size for block in member_blocks)
        prototype_bytes = tensor_storage_bytes(prototype_key) + tensor_storage_bytes(
            prototype_value
        )
        anchor = graph.nodes[anchor_id].block
        records.append(
            PrototypeRecord(
                prototype_id=prototype_id,
                layer=anchor.layer,
                kv_head=anchor.kv_head,
                anchor_node_id=anchor_id,
                anchor_logical_position=anchor.original_logical_positions[0],
                members=members,
                assigned_node_ids=member_ids,
                diagnostics=PrototypeDiagnostics(
                    cluster_size=len(member_ids),
                    key_dispersion=_dispersion(pooled_keys, key_average_fp32, raw_weights),
                    value_dispersion=_dispersion(pooled_values, value_average_fp32, raw_weights),
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
                eviction_utility=selection.decisions[anchor_id].marginal_gain,
                utility_provenance="selected_anchor_marginal_gain",
            )
        )
        for node_id, block in zip(member_ids, member_blocks, strict=True):
            source_blocks.append(block)
            tier_assignments.append(prototype_id)
            source_to_prototype[node_id] = prototype_id
    tier = PrototypeTier(
        tuple(source_blocks),
        tuple(prototype_keys),
        tuple(prototype_values),
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
    raise PrototypeConstructionError(
        f"unsupported selection budget unit: {selection.budget.unit}"
    )


def construct_three_tier_cache(
    full_state: FullKVState,
    graph: SparseEvidenceGraph,
    selection: SelectionResult,
    capabilities: AdapterCapabilities,
    *,
    prototype_config: PrototypeConfig | None = None,
    residual_config: ResidualConfig | None = None,
    active_byte_budget: int | None = None,
    retention_ratio: float | None = None,
    key_transform: RotaryPrototypeKeyTransform | None = None,
) -> ThreeTierCacheConstruction:
    """Build exact, prototype, and CPU residual tiers or safely select exact-only.

    The current policy deliberately rejects all post-RoPE prototype averaging.
    A model family must explicitly advertise prototype support, and its observed
    source-cache metadata must agree with the adapter declaration.
    """

    prototypes = prototype_config or PrototypeConfig()
    residuals = residual_config or ResidualConfig()
    _validate_inputs(full_state, graph, selection)
    if retention_ratio is not None and (
        not math.isfinite(retention_ratio) or not 0 < retention_ratio <= 1
    ):
        raise ValueError("retention_ratio must be finite and in the interval (0, 1]")
    if active_byte_budget is not None and active_byte_budget < 1:
        raise ValueError("active_byte_budget must be positive when provided")
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
        )
    if not prototypes.enabled:
        return _exact_only(
            full_state,
            selection,
            safety,
            capabilities.supports_residual_repair,
            mode=TierConstructionMode.EXACT_ONLY_DISABLED,
            reason="prototype construction is disabled by configuration",
        )
    if not safety.safe:
        return _exact_only(
            full_state,
            selection,
            safety,
            capabilities.supports_residual_repair,
            mode=TierConstructionMode.EXACT_ONLY_UNSAFE,
            reason=safety.reason,
        )

    selected = frozenset(selection.selected_node_ids)
    assignments, incompatibility = _assign_to_anchors(graph, selected, prototypes)
    if incompatibility is not None:
        return _exact_only(
            full_state,
            selection,
            safety,
            capabilities.supports_residual_repair,
            mode=TierConstructionMode.EXACT_ONLY_INCOMPATIBLE,
            reason=incompatibility,
        )
    exact = selection.to_exact_tier(full_state)
    prototype_tier, records, source_to_prototype, gathered_sources = _prototype_payloads(
        full_state, graph, assignments, selection, key_transform
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
            )
    residual_node_order = tuple(
        node_id for record in records for node_id in record.assigned_node_ids
    )
    residual_report = build_residual_storage(
        full_state,
        source_to_prototype,
        residuals,
        node_order=residual_node_order,
        gathered=gathered_sources,
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
        reason="all unselected blocks were assigned to compatible selected anchors",
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
