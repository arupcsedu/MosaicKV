"""Conservative exact/prototype/residual cache construction."""

from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
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
from mosaickv.selection.types import SelectionBudget, SelectionResult
from mosaickv.types import BudgetUnit


class PrototypeConstructionError(RuntimeError):
    """Raised for malformed graph/selection inputs or invalid prototype tensors."""


def _source_node_id(graph: SparseEvidenceGraph, node_id: int) -> int:
    return graph.nodes[node_id].source_id


def _source_node_ids(
    graph: SparseEvidenceGraph,
    node_ids: Sequence[int],
) -> tuple[int, ...]:
    return tuple(_source_node_id(graph, node_id) for node_id in node_ids)


@dataclass(frozen=True, slots=True)
class PrototypeAssignmentPlan:
    """Opaque, validated descriptor-only prototype assignment transaction.

    Assignment is one of the largest host-side construction costs.  Budget
    planning, cost validation, and tier materialization previously recomputed
    the same deterministic matching three times.  This object binds one result
    to the exact graph object, selected IDs, prototype configuration, and anchor
    allowlist so downstream stages can safely reuse it without accepting an
    assignment produced for different inputs.
    """

    graph: SparseEvidenceGraph = field(repr=False, compare=False)
    selected_node_ids: tuple[int, ...]
    config: PrototypeConfig
    prototype_anchor_node_ids: tuple[int, ...] | None
    assignment_rows: tuple[tuple[int, int, float], ...]
    incompatibility: str | None

    @property
    def assignments(self) -> dict[int, tuple[int, float]]:
        return {
            node_id: (anchor_id, weight)
            for node_id, anchor_id, weight in self.assignment_rows
        }

    @property
    def materialized_anchor_ids(self) -> tuple[int, ...]:
        return tuple(sorted({anchor_id for _node_id, anchor_id, _weight in self.assignment_rows}))


def plan_prototype_assignments(
    graph: SparseEvidenceGraph,
    selection: SelectionResult,
    config: PrototypeConfig,
    *,
    prototype_anchor_node_ids: Sequence[int] | None = None,
) -> PrototypeAssignmentPlan:
    """Compute one reusable deterministic assignment for a fixed candidate."""

    normalized_anchors = (
        None
        if prototype_anchor_node_ids is None
        else tuple(sorted(int(node_id) for node_id in prototype_anchor_node_ids))
    )
    assignments, reason = _assign_to_anchors(
        graph,
        frozenset(selection.selected_node_ids),
        config,
        prototype_anchor_node_ids=(
            None if normalized_anchors is None else frozenset(normalized_anchors)
        ),
    )
    return PrototypeAssignmentPlan(
        graph=graph,
        selected_node_ids=selection.selected_node_ids,
        config=config,
        prototype_anchor_node_ids=normalized_anchors,
        assignment_rows=tuple(
            (node_id, anchor_id, weight)
            for node_id, (anchor_id, weight) in sorted(assignments.items())
        ),
        incompatibility=reason,
    )


def _assignments_from_plan(
    plan: PrototypeAssignmentPlan,
    graph: SparseEvidenceGraph,
    selection: SelectionResult,
    config: PrototypeConfig,
    prototype_anchor_node_ids: Sequence[int] | None,
) -> tuple[dict[int, tuple[int, float]], str | None]:
    normalized_anchors = (
        None
        if prototype_anchor_node_ids is None
        else tuple(sorted(int(node_id) for node_id in prototype_anchor_node_ids))
    )
    if plan.graph is not graph:
        raise PrototypeConstructionError("prototype assignment plan belongs to another graph")
    if plan.selected_node_ids != selection.selected_node_ids:
        raise PrototypeConstructionError("prototype assignment plan selection does not match")
    if plan.config != config:
        raise PrototypeConstructionError("prototype assignment plan configuration does not match")
    if plan.prototype_anchor_node_ids != normalized_anchors:
        raise PrototypeConstructionError(
            "prototype assignment plan anchor allowlist does not match"
        )
    return plan.assignments, plan.incompatibility


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
    *,
    exact_medoid_key: bool = False,
) -> PrototypeSafetyAssessment:
    """Reject post-RoPE merging unless an exact audited phase transform exists."""

    source_state = _coerce_key_state(full_state.cached_key_state)
    adapter_state = _coerce_key_state(capabilities.cached_key_state)
    declares_support = capabilities.supports_prototype_merge
    transform_provenance = (
        "exact_post_rope_medoid_key_no_phase_average_v1"
        if exact_medoid_key and source_state is CachedKeyState.POST_ROPE
        else (None if key_transform is None else key_transform.provenance)
    )
    transform_matches = bool(
        exact_medoid_key
        or (key_transform is not None and key_transform.source_key_state is source_state)
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
                f" through phase-safe key policy {transform_provenance}"
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
    exact_positions: dict[tuple[int, int], list[int]] = defaultdict(list)
    for block in exact.blocks:
        exact_positions[(block.layer, block.kv_head)].extend(
            block.original_logical_positions
        )
    records_by_identity: dict[tuple[int, int], list[PrototypeRecord]] = defaultdict(list)
    for record in records:
        records_by_identity[(record.layer, record.kv_head)].append(record)
    result: list[ActiveHeadLayout] = []
    for layer, head in sorted(identities):
        matching = records_by_identity.get((layer, head), ())
        result.append(
            ActiveHeadLayout(
                layer=layer,
                kv_head=head,
                exact_logical_positions=tuple(sorted(exact_positions.get((layer, head), ()))),
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
    exact_node_ids: tuple[int, ...] | None = None,
    active_budget: SelectionBudget | None = None,
) -> ThreeTierCacheConstruction:
    resolved_exact_node_ids = (
        selection.selected_node_ids if exact_node_ids is None else exact_node_ids
    )
    exact = full_state.gather_exact_blocks(
        tuple(full_state.blocks[node_id] for node_id in resolved_exact_node_ids),
        gather_mode=exact_gather_mode,
    )
    residual_report = empty_residual_storage()
    state = MosaicKVState.create(full_state, exact=exact, residuals=residual_report.tier)
    return ThreeTierCacheConstruction(
        state=state,
        mode=mode,
        reason=reason,
        safety=safety,
        adapter_declares_residual_repair=adapter_declares_residual_repair,
        active_budget=selection.budget if active_budget is None else active_budget,
        exact_node_ids=resolved_exact_node_ids,
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
    source_ids = tuple(node.source_id for node in graph.nodes)
    if len(source_ids) != len(set(source_ids)) or tuple(sorted(source_ids)) != source_ids:
        raise PrototypeConstructionError(
            "evidence graph source IDs must be unique and source-ordered"
        )
    if any(node_id < 0 or node_id >= len(full_state.blocks) for node_id in source_ids):
        raise PrototypeConstructionError("evidence graph source ID lies outside FullKV blocks")
    graph_blocks = tuple(node.block for node in graph.nodes)
    expected_graph_blocks = tuple(full_state.blocks[node_id] for node_id in source_ids)
    if graph_blocks != expected_graph_blocks:
        raise PrototypeConstructionError(
            "evidence graph nodes must align with their mapped FullKV source blocks"
        )
    if len(selection.decisions) != len(graph.nodes):
        raise PrototypeConstructionError(
            "selection decisions must cover the same node table as the evidence graph"
        )
    expected = tuple(graph.nodes[node_id].block for node_id in selection.selected_node_ids)
    if selection.selected_blocks != expected:
        raise PrototypeConstructionError(
            "selected block descriptors do not align with their graph node IDs"
        )


def _selected_anchor_neighbors(
    graph: SparseEvidenceGraph,
    anchors: frozenset[int],
    selected: frozenset[int],
) -> Mapping[int, tuple[tuple[int, float], ...]]:
    """Index only graph edges that can route an unselected node to an anchor.

    The prior implementation materialized an undirected copy of every graph
    edge and then discarded all neighbors that were not selected anchors.
    Selection ratios are small, so most of that allocation and traversal could
    never affect a prototype assignment. Pairwise maximum weights retain the
    exact undirected pairwise-maximum semantics of the previous full index.
    """

    weights_by_source: dict[int, dict[int, float]] = defaultdict(dict)
    for source, target, weight in zip(
        graph.row_indices,
        graph.column_indices,
        graph.weights,
        strict=True,
    ):
        numeric_weight = float(weight)
        if source in anchors and target not in selected:
            current = weights_by_source[target].get(source, 0.0)
            weights_by_source[target][source] = max(current, numeric_weight)
        if target in anchors and source not in selected:
            current = weights_by_source[source].get(target, 0.0)
            weights_by_source[source][target] = max(current, numeric_weight)
    return {
        source_id: tuple(anchor_weights.items())
        for source_id, anchor_weights in weights_by_source.items()
    }


def _pair_span(first: KVBlockDescriptor, second: KVBlockDescriptor) -> int:
    # Descriptor invariants guarantee strictly monotonic positions. Prototype
    # assignment calls this millions of times for long documents, so scanning
    # and allocating both complete position tuples here is unnecessarily
    # expensive; their endpoints determine the exact same union span.
    return max(
        first.original_logical_positions[-1],
        second.original_logical_positions[-1],
    ) - min(
        first.original_logical_positions[0],
        second.original_logical_positions[0],
    )


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


def _assign_to_anchors_unchecked(
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
    neighbors = _selected_anchor_neighbors(graph, anchors, selected)

    if config.budget_planning_mode in {
        "one_pass_anchor_cover",
        "one_pass_bounded_anchor_cover",
    } and prototype_anchor_node_ids is not None:
        # The runtime planner reserves an anchor cover over these exact same
        # segments. Resolve assignments locally instead of building a global
        # preference table and then discarding every out-of-segment entry.
        # For each member this preserves the original score: maximum real
        # graph affinity versus the weak positional compatibility scaffold,
        # followed by the lower-node-ID tie break.
        strata: dict[tuple[int, int, object], list[int]] = defaultdict(list)
        for node in graph.nodes:
            block = node.block
            if not config.uses_prototype_block(block.layer, block.kv_head, block.modality):
                continue
            strata[(block.layer, block.kv_head, block.modality)].append(node.node_id)
        result: dict[int, tuple[int, float]] = {}
        bounded_cover = config.budget_planning_mode == "one_pass_bounded_anchor_cover"

        def assign_segment(member_ids: Sequence[int]) -> str | None:
            segment_anchors = tuple(node_id for node_id in member_ids if node_id in anchors)
            if not segment_anchors:
                if bounded_cover:
                    # A bounded cover intentionally represents only the most
                    # prompt-relevant compatible segments. Uncovered source
                    # blocks remain pruned; they are not silently assigned
                    # across group-size or position-span boundaries.
                    return None
                return f"prototype anchor-cover segment {tuple(member_ids)} has no selected anchor"
            segment_anchor_set = frozenset(segment_anchors)
            for node_id in member_ids:
                if node_id in selected:
                    continue
                source = graph.nodes[node_id]
                maximum_span = config.position_span_for(
                    source.block.layer,
                    source.block.kv_head,
                )
                graph_weights = {
                    anchor_id: weight
                    for anchor_id, weight in neighbors.get(node_id, ())
                    if anchor_id in segment_anchor_set
                    and weight > 0
                    and weight >= config.min_anchor_weight
                    and _compatible(source.block, graph.nodes[anchor_id].block, config)
                    and _descriptor_compatible(graph, node_id, anchor_id, config)
                }
                candidates: list[tuple[float, int]] = []
                for anchor_id in segment_anchors:
                    anchor = graph.nodes[anchor_id]
                    if not _compatible(source.block, anchor.block, config) or not (
                        _descriptor_compatible(graph, node_id, anchor_id, config)
                    ):
                        continue
                    distance = abs(source.logical_center - anchor.logical_center)
                    normalized = (
                        1.0 / (1.0 + distance)
                        if maximum_span is None
                        else 1.0 - distance / (maximum_span + 1)
                    )
                    positional_weight = max(
                        config.min_anchor_weight,
                        0.01 * max(normalized, 1e-12),
                    )
                    candidates.append(
                        (max(positional_weight, graph_weights.get(anchor_id, 0.0)), anchor_id)
                    )
                if not candidates:
                    return (
                        f"unselected node {node_id} has no compatible selected anchor "
                        "inside its anchor-cover segment"
                    )
                weight, anchor_id = min(candidates, key=lambda item: (-item[0], item[1]))
                result[node_id] = anchor_id, weight
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
                first = graph.nodes[trial[0]].block
                last = graph.nodes[node_id].block
                local_group_size = config.group_size_for(first.layer, first.kv_head)
                local_position_span = config.position_span_for(first.layer, first.kv_head)
                exceeds_capacity = len(trial) > local_group_size + 1
                exceeds_span = bool(
                    local_position_span is not None
                    and last.original_logical_positions[-1]
                    - first.original_logical_positions[0]
                    > local_position_span
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

    selected_by_identity: dict[tuple[int, int, object], list[tuple[float, int]]] = defaultdict(list)
    for anchor_id in anchors:
        anchor = graph.nodes[anchor_id]
        selected_by_identity[
            (anchor.block.layer, anchor.block.kv_head, anchor.block.modality)
        ].append((anchor.logical_center, anchor_id))
    for identity_anchors in selected_by_identity.values():
        identity_anchors.sort()
    selected_centers_by_identity = {
        identity: tuple(center for center, _anchor_id in identity_anchors)
        for identity, identity_anchors in selected_by_identity.items()
    }
    allowed_anchor_modalities: dict[str, set[object]] = defaultdict(set)
    modality_type = type(graph.nodes[0].block.modality) if graph.nodes else None
    for pair in config.allowed_modality_pairs:
        source_modality, anchor_modality = pair.split(":", maxsplit=1)
        if modality_type is None:  # pragma: no cover - a cache always has graph nodes
            continue
        allowed_anchor_modalities[source_modality].add(modality_type(anchor_modality))
    ordered_anchor_modalities = {
        source: tuple(sorted(values, key=str))
        for source, values in allowed_anchor_modalities.items()
    }
    head_group_sizes = {
        (node.block.layer, node.block.kv_head): config.group_size_for(
            node.block.layer,
            node.block.kv_head,
        )
        for node in graph.nodes
    }
    head_position_spans = {
        (node.block.layer, node.block.kv_head): config.position_span_for(
            node.block.layer,
            node.block.kv_head,
        )
        for node in graph.nodes
    }

    def positional_candidates(node_id: int) -> tuple[tuple[int, float], ...]:
        """Return a bounded compatibility scaffold independent of evidence scores."""

        source = graph.nodes[node_id]
        candidates: list[tuple[float, int]] = []
        identity = (source.block.layer, source.block.kv_head)
        local_group_size = head_group_sizes[identity]
        maximum_span = head_position_spans[identity]
        window = max(1, 2 * local_group_size)
        for modality in ordered_anchor_modalities[source.block.modality.value]:
            anchors = selected_by_identity.get(
                (source.block.layer, source.block.kv_head, modality),
                (),
            )
            if not anchors:
                continue
            centers = selected_centers_by_identity[
                (source.block.layer, source.block.kv_head, modality)
            ]
            insertion = bisect_left(centers, source.logical_center)
            first = max(0, insertion - window)
            last = min(len(anchors), insertion + window)
            for center, anchor_id in anchors[first:last]:
                anchor = graph.nodes[anchor_id]
                if maximum_span is not None and _pair_span(
                    source.block,
                    anchor.block,
                ) > maximum_span:
                    continue
                if config.adaptive_grouping and not _descriptor_compatible(
                    graph,
                    node_id,
                    anchor_id,
                    config,
                ):
                    continue
                distance = abs(source.logical_center - center)
                normalized = (
                    1.0 / (1.0 + distance)
                    if maximum_span is None
                    else 1.0 - distance / (maximum_span + 1)
                )
                # Compatibility edges are deliberately weaker than evidence
                # edges. They make representation feasible without claiming
                # semantic evidence or affecting which real graph edge wins.
                weight = max(config.min_anchor_weight, 0.01 * max(normalized, 1e-12))
                candidates.append((weight, anchor_id))
        return tuple((anchor_id, weight) for weight, anchor_id in candidates)

    preferences: dict[int, tuple[tuple[float, int], ...]] = {}
    for node_id, node in enumerate(graph.nodes):
        if node_id in selected or not config.uses_prototype_block(
            node.block.layer,
            node.block.kv_head,
            node.block.modality,
        ):
            continue
        weights_by_anchor: dict[int, float] = {}
        for anchor_id, weight in neighbors.get(node_id, ()):
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
            if not config.uses_prototype_block(block.layer, block.kv_head, block.modality):
                continue
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
                first = graph.nodes[trial[0]].block
                last = graph.nodes[node_id].block
                local_group_size = config.group_size_for(first.layer, first.kv_head)
                local_position_span = config.position_span_for(first.layer, first.kv_head)
                exceeds_capacity = len(trial) > local_group_size + 1
                exceeds_span = bool(
                    local_position_span is not None
                    and last.original_logical_positions[-1]
                    - first.original_logical_positions[0]
                    > local_position_span
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
        candidate_ids = (anchor_id, *member_ids)
        minimum = min(
            graph.nodes[candidate_id].block.original_logical_positions[0]
            for candidate_id in candidate_ids
        )
        maximum = max(
            graph.nodes[candidate_id].block.original_logical_positions[-1]
            for candidate_id in candidate_ids
        )
        return maximum - minimum <= maximum_span and all(
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
            if (
                max(block.original_logical_positions[-1] for block in blocks)
                - min(block.original_logical_positions[0] for block in blocks)
                > maximum_span
            ):
                fallback, fallback_reason = segmented_assignments()
                if fallback_reason is None:
                    return fallback, None
                return {}, (
                    f"anchor {anchor_id} cluster exceeds its position span="
                    f"{maximum_span}; {fallback_reason}"
                )
    return assignments, None


def _assign_to_anchors(
    graph: SparseEvidenceGraph,
    selected: frozenset[int],
    config: PrototypeConfig,
    *,
    prototype_anchor_node_ids: frozenset[int] | None = None,
) -> tuple[dict[int, tuple[int, float]], str | None]:
    """Assign prototyped heads and enforce the declared per-head launch cap."""

    assignments, reason = _assign_to_anchors_unchecked(
        graph,
        selected,
        config,
        prototype_anchor_node_ids=prototype_anchor_node_ids,
    )
    if reason is not None or config.maximum_prototypes_per_head is None:
        return assignments, reason
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for anchor_id in {anchor_id for anchor_id, _weight in assignments.values()}:
        block = graph.nodes[anchor_id].block
        counts[(block.layer, block.kv_head)] += 1
    violations = {
        identity: count
        for identity, count in counts.items()
        if count > config.maximum_prototypes_per_head
    }
    if violations:
        return {}, (
            "prototype count exceeds maximum_prototypes_per_head="
            f"{config.maximum_prototypes_per_head}: {dict(sorted(violations.items()))}"
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
    if positions[-1] - positions[0] + 1 == len(positions):
        values = source[positions[0] : positions[-1] + 1]
    else:
        values = np.asarray(source.take(positions), dtype=np.float64)
    if not bool(np.all(np.isfinite(values))) or bool(np.any(values < 0)):
        raise PrototypeConstructionError("prototype token attention must be finite and nonnegative")
    tiny = float(np.finfo(np.float64).tiny)
    values = np.maximum(values, tiny)
    mass = float(values.sum())
    return tuple(float(value / mass) for value in values), mass


def _normalized_position_weights_by_node(
    graph: SparseEvidenceGraph,
    node_ids: Sequence[int],
    token_attention_by_head: Mapping[tuple[int, int], Any] | None,
) -> dict[int, tuple[tuple[float, ...], float, float]]:
    """Batch the exact within-block normalization by layer/head and width.

    Prototype construction applies this reduction to tens of thousands of
    small blocks. Calling ``np.asarray``, finiteness checks, and reductions
    once per block made Python dispatch dominate the actual arithmetic. This
    grouped path performs the same float64 clamp and normalization in bounded
    matrices and materializes only the public per-member tuples afterward.
    """

    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for node_id in node_ids:
        block = graph.nodes[node_id].block
        grouped[(block.layer, block.kv_head, block.position_count)].append(node_id)
    result: dict[int, tuple[tuple[float, ...], float, float]] = {}
    tiny = float(np.finfo(np.float64).tiny)
    for (layer, head, width), local_ids in grouped.items():
        if token_attention_by_head is None:
            weights = tuple(1.0 / width for _position in range(width))
            entropy = math.log(width)
            result.update((node_id, (weights, 1.0, entropy)) for node_id in local_ids)
            continue
        try:
            source = np.asarray(
                token_attention_by_head[(layer, head)],
                dtype=np.float64,
            ).reshape(-1)
        except KeyError as error:
            raise PrototypeConstructionError(
                f"prototype token attention is missing layer/head {(layer, head)}"
            ) from error
        positions = np.asarray(
            tuple(graph.nodes[node_id].block.physical_cache_indices for node_id in local_ids),
            dtype=np.int64,
        )
        if positions.shape != (len(local_ids), width):
            raise RuntimeError("prototype position batch has an inconsistent width")
        if int(positions.max(initial=-1)) >= len(source):
            raise PrototypeConstructionError(
                "prototype token attention is shorter than a source cache block"
            )
        values = source[positions]
        if not bool(np.all(np.isfinite(values))) or bool(np.any(values < 0)):
            raise PrototypeConstructionError(
                "prototype token attention must be finite and nonnegative"
            )
        values = np.maximum(values, tiny)
        masses = values.sum(axis=1, dtype=np.float64)
        normalized = values / masses[:, None]
        entropies = np.asarray(
            -np.sum(normalized * np.log(normalized), axis=1, dtype=np.float64),
            dtype=np.float64,
        )
        for row, node_id in enumerate(local_ids):
            result[node_id] = (
                tuple(float(value) for value in normalized[row]),
                float(masses[row]),
                float(entropies[row]),
            )
    return result


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
    weighted_center = 0.0
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
        weighted_center += sum(
            position * block_weight * local_weight / total
            for position, local_weight in zip(
                block.original_logical_positions,
                within,
                strict=True,
            )
        )
    # Find the nearest exact logical position in each monotonic block rather
    # than scanning every token in every member. Sparse descriptors remain
    # exact because bisect examines both neighbors around the insertion point.
    candidates: list[int] = []
    for block in member_blocks:
        positions = block.original_logical_positions
        insertion = bisect_left(positions, weighted_center)
        if insertion < len(positions):
            candidates.append(positions[insertion])
        if insertion:
            candidates.append(positions[insertion - 1])
    return min(candidates, key=lambda position: (abs(position - weighted_center), position))


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
    position_entropies: tuple[float, ...]
    key_member_index: int
    key_position_index: int


def _batched_pool_source_members(
    full_state: FullKVState,
    blocks_by_node: Mapping[int, KVBlockDescriptor],
    position_weights_by_node: Mapping[int, tuple[float, ...]],
    key_transform: RotaryPrototypeKeyTransform | None,
) -> tuple[dict[int, Any], dict[int, Any]]:
    """Pool source blocks directly from fused layer/head tensors.

    Long visual/document prompts can contain tens of thousands of block/head
    pairs. Materializing one tensor per block, casting each to FP32, and then
    stacking those tensors is launch- and Python-bound. Torch caches are
    instead indexed once per layer and block width, then reduced as one
    ``[block, position, ...]`` tensor. NumPy keeps the simple reference path.
    """

    node_ids = tuple(sorted(blocks_by_node))
    if not node_ids:
        return {}, {}
    first_block = blocks_by_node[node_ids[0]]
    first_key = full_state.layers[first_block.layer].key
    if not first_key.__class__.__module__.startswith("torch"):
        gathered = full_state.gather_exact_blocks(
            tuple(blocks_by_node[node_id] for node_id in node_ids),
            gather_mode="batched_layer",
        )
        numpy_pooled_keys: dict[int, Any] = {}
        numpy_pooled_values: dict[int, Any] = {}
        for node_id, key, value in zip(
            node_ids,
            gathered.key_blocks,
            gathered.value_blocks,
            strict=True,
        ):
            block = blocks_by_node[node_id]
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
    shape_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for node_id in node_ids:
        block = blocks_by_node[node_id]
        shape_groups[(block.layer, block.position_count)].append(node_id)
    for layer_index, width in sorted(shape_groups):
        members = tuple(shape_groups[(layer_index, width)])
        layer = full_state.layers[layer_index]
        key_head_axis = layer.key_head_dimension % int(layer.key.ndim)
        key_sequence_axis = layer.key_sequence_dimension % int(layer.key.ndim)
        value_head_axis = layer.value_head_dimension % int(layer.value.ndim)
        value_sequence_axis = layer.value_sequence_dimension % int(layer.value.ndim)
        key_source = layer.key.movedim((key_head_axis, key_sequence_axis), (0, 1))
        value_source = layer.value.movedim((value_head_axis, value_sequence_axis), (0, 1))
        heads = torch.as_tensor(
            tuple(blocks_by_node[node_id].kv_head for node_id in members),
            dtype=torch.long,
            device=layer.key.device,
        )
        positions = torch.as_tensor(
            tuple(blocks_by_node[node_id].physical_cache_indices for node_id in members),
            dtype=torch.long,
            device=layer.key.device,
        )
        # Advanced indices consume the canonical head/sequence axes and leave
        # ``[member, position, remaining source axes...]``.
        keys = key_source[heads[:, None], positions].detach().float()
        value_positions = positions.to(device=layer.value.device)
        value_heads = heads.to(device=layer.value.device)
        values = value_source[value_heads[:, None], value_positions].detach().float()
        local_position_weights_batch = tuple(
            position_weights_by_node[node_id] for node_id in members
        )
        weights = torch.as_tensor(
            local_position_weights_batch,
            dtype=keys.dtype,
            device=keys.device,
        )
        key_weight = weights.reshape((len(members), width) + (1,) * (keys.ndim - 2))

        if key_transform is None:
            pooled_key_batch = torch.sum(keys * key_weight, dim=1, keepdim=True)
        else:
            if key_sequence_axis != int(layer.key.ndim) - 2:
                raise PrototypeConstructionError(
                    "batched RoPE prototype pooling requires the cache sequence axis at -2"
                )
            logical_rows = tuple(
                blocks_by_node[node_id].original_logical_positions for node_id in members
            )
            if min(min(row) for row in logical_rows) < 0 or max(
                max(row) for row in logical_rows
            ) >= int(key_transform.cos.shape[-2]):
                raise PrototypeConstructionError(
                    "prototype logical position lies outside the captured prefill RoPE phase"
                )
            logical_positions = torch.as_tensor(
                logical_rows,
                dtype=torch.long,
                device=key_transform.cos.device,
            )
            cos = key_transform.cos[0, 0][logical_positions].to(
                device=keys.device,
                dtype=torch.float32,
            )
            sin = key_transform.sin[0, 0][logical_positions].to(
                device=keys.device,
                dtype=torch.float32,
            )
            phase_shape = (len(members), width) + (1,) * (keys.ndim - 3) + (int(cos.shape[-1]),)
            cos = cos.reshape(phase_shape)
            sin = sin.reshape(phase_shape)
            denominator = (cos * cos + sin * sin).clamp_min(1e-12)
            neutral = (keys * cos - key_transform._rotate_half(keys) * sin) / denominator
            pooled_key_batch = torch.sum(neutral * key_weight, dim=1, keepdim=True)

        value_weight = weights.to(device=values.device, dtype=values.dtype).reshape(
            (len(members), width) + (1,) * (values.ndim - 2)
        )
        pooled_value_batch = torch.sum(values * value_weight, dim=1, keepdim=True)
        # Restore one singleton head axis and return each payload in the exact
        # source cache axis order. The leading dimension remains the member
        # batch until rows are exposed below.
        canonical_keys = pooled_key_batch.unsqueeze(1)
        canonical_values = pooled_value_batch.unsqueeze(1)
        restored_keys = canonical_keys.movedim(
            (1, 2),
            (key_head_axis + 1, key_sequence_axis + 1),
        )
        restored_values = canonical_values.movedim(
            (1, 2),
            (value_head_axis + 1, value_sequence_axis + 1),
        )
        for index, node_id in enumerate(members):
            pooled_keys[node_id] = restored_keys[index]
            pooled_values[node_id] = restored_values[index]
    return pooled_keys, pooled_values


def _batched_torch_prototypes(
    full_state: FullKVState,
    recipes: Sequence[_PrototypeRecipe],
    pooled_keys_by_node: Mapping[int, Any],
    pooled_values_by_node: Mapping[int, Any],
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
        layer = full_state.layers[recipe.anchor.layer]
        groups[
            (
                recipe.anchor.layer,
                tuple(int(size) for size in key.shape),
                tuple(int(size) for size in value.shape),
                str(key.device),
                str(value.device),
                str(layer.key.dtype),
                str(layer.value.dtype),
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

        reference_layer = full_state.layers[recipes[recipe_indices[0]].anchor.layer]
        first_reference_key = reference_layer.key
        first_reference_value = reference_layer.value
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


def _fused_torch_prototypes(
    full_state: FullKVState,
    recipes: Sequence[_PrototypeRecipe],
    key_transform: RotaryPrototypeKeyTransform | None,
    pooling_mode: str,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[_DeviceMemberStatistics, ...]]:
    """Pool source positions and reduce prototype members in one layer batch.

    The reference batched path first exposed one Python tensor view for every
    source block and then stacked those views again by prototype. A document
    cache contains tens of thousands of layer/head blocks, so Python object
    creation and the second stack dominated otherwise small GPU reductions.
    This path indexes the source cache once per layer/member-count group and
    applies the exact same normalized position and member weights directly to
    one ``[prototype, member, position, ...]`` tensor per layer. Ragged member
    counts and tail blocks are padded with zero-weight copies and therefore
    cannot affect the result.
    """

    import torch

    if not recipes:
        return (), (), ()
    prototype_keys: list[Any | None] = [None] * len(recipes)
    prototype_values: list[Any | None] = [None] * len(recipes)
    statistics: list[_DeviceMemberStatistics | None] = [None] * len(recipes)
    groups: dict[int, list[int]] = defaultdict(list)
    for recipe_index, recipe in enumerate(recipes):
        groups[recipe.anchor.layer].append(recipe_index)

    for layer_index in sorted(groups):
        recipe_indices = tuple(groups[layer_index])
        layer = full_state.layers[layer_index]
        key_head_axis = layer.key_head_dimension % int(layer.key.ndim)
        key_sequence_axis = layer.key_sequence_dimension % int(layer.key.ndim)
        value_head_axis = layer.value_head_dimension % int(layer.value.ndim)
        value_sequence_axis = layer.value_sequence_dimension % int(layer.value.ndim)
        key_source = layer.key.movedim((key_head_axis, key_sequence_axis), (0, 1))
        value_source = layer.value.movedim((value_head_axis, value_sequence_axis), (0, 1))
        maximum_width = max(
            block.position_count
            for recipe_index in recipe_indices
            for block in recipes[recipe_index].member_blocks
        )
        maximum_members = max(
            len(recipes[recipe_index].member_blocks) for recipe_index in recipe_indices
        )

        head_rows: list[tuple[int, ...]] = []
        physical_rows: list[tuple[tuple[int, ...], ...]] = []
        logical_rows: list[tuple[tuple[int, ...], ...]] = []
        position_weight_rows: list[tuple[tuple[float, ...], ...]] = []
        member_weight_rows: list[tuple[float, ...]] = []
        member_counts: list[int] = []
        for recipe_index in recipe_indices:
            recipe = recipes[recipe_index]
            total = float(sum(recipe.raw_weights))
            if not math.isfinite(total) or total <= 0:
                raise PrototypeConstructionError(
                    "prototype weights must sum to a finite positive value"
                )
            member_count = len(recipe.member_blocks)
            member_counts.append(member_count)
            padded_blocks = (
                *recipe.member_blocks,
                *(recipe.member_blocks[-1],) * (maximum_members - member_count),
            )
            head_rows.append(tuple(block.kv_head for block in padded_blocks))
            physical_members: list[tuple[int, ...]] = []
            logical_members: list[tuple[int, ...]] = []
            weight_members: list[tuple[float, ...]] = []
            for block, local_weights in zip(
                recipe.member_blocks,
                recipe.position_weights,
                strict=True,
            ):
                padding = maximum_width - block.position_count
                physical_members.append(
                    (*block.physical_cache_indices, *(block.physical_cache_indices[-1],) * padding)
                )
                logical_members.append(
                    (
                        *block.original_logical_positions,
                        *(block.original_logical_positions[-1],) * padding,
                    )
                )
                weight_members.append((*local_weights, *(0.0,) * padding))
            for block in padded_blocks[member_count:]:
                padding = maximum_width - block.position_count
                physical_members.append(
                    (*block.physical_cache_indices, *(block.physical_cache_indices[-1],) * padding)
                )
                logical_members.append(
                    (
                        *block.original_logical_positions,
                        *(block.original_logical_positions[-1],) * padding,
                    )
                )
                weight_members.append((0.0,) * maximum_width)
            physical_rows.append(tuple(physical_members))
            logical_rows.append(tuple(logical_members))
            position_weight_rows.append(tuple(weight_members))
            member_weight_rows.append(
                (
                    *(weight / total for weight in recipe.raw_weights),
                    *(0.0,) * (maximum_members - member_count),
                )
            )

        heads = torch.as_tensor(head_rows, dtype=torch.long, device=layer.key.device)
        physical = torch.as_tensor(
            physical_rows,
            dtype=torch.long,
            device=layer.key.device,
        )
        # Advanced indices consume canonical head/sequence axes and yield
        # [prototype, member, position, remaining source axes...].
        keys = key_source[heads[..., None], physical].detach().float()
        values = value_source[
            heads.to(device=layer.value.device)[..., None],
            physical.to(device=layer.value.device),
        ].detach().float()
        position_weights = torch.as_tensor(
            position_weight_rows,
            dtype=keys.dtype,
            device=keys.device,
        )
        key_position_weight = position_weights.reshape(
            (*position_weights.shape,) + (1,) * (keys.ndim - 3)
        )
        if key_transform is None:
            neutral_keys = keys
        else:
            if key_sequence_axis != int(layer.key.ndim) - 2:
                raise PrototypeConstructionError(
                    "fused RoPE prototype pooling requires the cache sequence axis at -2"
                )
            minimum_logical = min(
                min(member) for prototype in logical_rows for member in prototype
            )
            maximum_logical = max(
                max(member) for prototype in logical_rows for member in prototype
            )
            if minimum_logical < 0 or maximum_logical >= int(key_transform.cos.shape[-2]):
                raise PrototypeConstructionError(
                    "prototype logical position lies outside the captured prefill RoPE phase"
                )
            logical = torch.as_tensor(
                logical_rows,
                dtype=torch.long,
                device=key_transform.cos.device,
            )
            cos = key_transform.cos[0, 0][logical].to(device=keys.device, dtype=torch.float32)
            sin = key_transform.sin[0, 0][logical].to(device=keys.device, dtype=torch.float32)
            phase_shape = tuple(logical.shape) + (1,) * (keys.ndim - 4) + (
                int(cos.shape[-1]),
            )
            cos = cos.reshape(phase_shape)
            sin = sin.reshape(phase_shape)
            denominator = (cos * cos + sin * sin).clamp_min(1e-12)
            neutral_keys = (
                keys * cos - key_transform._rotate_half(keys) * sin
            ) / denominator
        pooled_keys = torch.sum(
            neutral_keys * key_position_weight,
            dim=2,
            keepdim=True,
        )
        value_position_weight = position_weights.to(
            device=values.device,
            dtype=values.dtype,
        ).reshape((*position_weights.shape,) + (1,) * (values.ndim - 3))
        pooled_values = torch.sum(
            values * value_position_weight,
            dim=2,
            keepdim=True,
        )
        member_weights = torch.as_tensor(
            member_weight_rows,
            dtype=pooled_keys.dtype,
            device=pooled_keys.device,
        )
        key_member_weight = member_weights.reshape(
            (*member_weights.shape,) + (1,) * (pooled_keys.ndim - 2)
        )
        if pooling_mode == "exact_medoid_key_forecast_value":
            recipe_rows = torch.arange(len(recipe_indices), device=keys.device)
            key_members = torch.as_tensor(
                tuple(recipes[index].key_member_index for index in recipe_indices),
                dtype=torch.long,
                device=keys.device,
            )
            key_positions = torch.as_tensor(
                tuple(recipes[index].key_position_index for index in recipe_indices),
                dtype=torch.long,
                device=keys.device,
            )
            # Preserve one source key exactly at its original RoPE phase. No
            # key average or inverse/reapplication is performed in this mode.
            key_averages_canonical = keys[
                recipe_rows,
                key_members,
                key_positions,
            ].unsqueeze(1)
        else:
            key_averages_canonical = torch.sum(
                pooled_keys * key_member_weight,
                dim=1,
            )
        value_member_weight = member_weights.to(
            device=pooled_values.device,
            dtype=pooled_values.dtype,
        ).reshape((*member_weights.shape,) + (1,) * (pooled_values.ndim - 2))
        value_averages_canonical = torch.sum(
            pooled_values * value_member_weight,
            dim=1,
        )

        key_error_axes = tuple(range(2, pooled_keys.ndim))
        value_error_axes = tuple(range(2, pooled_values.ndim))
        key_errors = torch.mean(
            (pooled_keys - key_averages_canonical.unsqueeze(1)) ** 2,
            dim=key_error_axes,
        )
        value_errors = torch.mean(
            (pooled_values - value_averages_canonical.unsqueeze(1)) ** 2,
            dim=value_error_axes,
        )
        value_norms = torch.sqrt(
            torch.mean(pooled_values * pooled_values, dim=value_error_axes).clamp_min(0)
        )
        key_dispersion = torch.sum(key_errors * member_weights, dim=1)
        value_dispersion = torch.sum(
            value_errors
            * member_weights.to(device=value_errors.device, dtype=value_errors.dtype),
            dim=1,
        )

        canonical_keys = key_averages_canonical.unsqueeze(1)
        canonical_values = value_averages_canonical.unsqueeze(1)
        restored_key_averages = canonical_keys.movedim(
            (1, 2),
            (key_head_axis + 1, key_sequence_axis + 1),
        )
        restored_value_averages = canonical_values.movedim(
            (1, 2),
            (value_head_axis + 1, value_sequence_axis + 1),
        )
        if pooling_mode == "exact_medoid_key_forecast_value" or key_transform is None:
            key_batch = restored_key_averages.to(
                dtype=layer.key.dtype,
                device=layer.key.device,
            )
        else:
            key_batch = key_transform.apply_anchor_phase_batch(
                restored_key_averages,
                tuple(
                    recipes[index].prototype_logical_position for index in recipe_indices
                ),
                layer.key,
            )
        value_batch = restored_value_averages.to(
            dtype=layer.value.dtype,
            device=layer.value.device,
        )
        for local_cluster, recipe_index in enumerate(recipe_indices):
            member_count = member_counts[local_cluster]
            prototype_keys[recipe_index] = key_batch[local_cluster]
            prototype_values[recipe_index] = value_batch[local_cluster]
            statistics[recipe_index] = _DeviceMemberStatistics(
                torch.stack(
                    (
                        0.5 * (key_errors[local_cluster] + value_errors[local_cluster]),
                        value_norms[local_cluster],
                    ),
                    dim=-1,
                )[:member_count],
                torch.stack(
                    (key_dispersion[local_cluster], value_dispersion[local_cluster])
                ),
            )
    if any(value is None for value in (*prototype_keys, *prototype_values, *statistics)):
        raise RuntimeError("fused prototype reduction left an uninitialized payload")
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
) -> tuple[PrototypeTier, tuple[PrototypeRecord, ...], dict[int, int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for node_id, (anchor_id, _weight) in assignments.items():
        grouped[anchor_id].append(node_id)
    if not grouped:
        return PrototypeTier(), (), {}
    ordered_member_ids = tuple(
        node_id for anchor_id in sorted(grouped) for node_id in sorted(grouped[anchor_id])
    )
    position_weights_and_mass_by_node = _normalized_position_weights_by_node(
        graph,
        ordered_member_ids,
        (
            prototype_token_weight_by_head
            if config.pooling_mode
            in {"forecast_attention_token", "exact_medoid_key_forecast_value"}
            else None
        ),
    )
    recipes: list[_PrototypeRecipe] = []
    position_weights_by_node: dict[int, tuple[float, ...]] = {}
    for prototype_id, anchor_id in enumerate(sorted(grouped)):
        member_ids = tuple(sorted(grouped[anchor_id]))
        member_blocks = tuple(graph.nodes[node_id].block for node_id in member_ids)
        affinity_weights = tuple(assignments[node_id][1] for node_id in member_ids)
        position_weights_and_mass = tuple(
            position_weights_and_mass_by_node[node_id] for node_id in member_ids
        )
        position_weights = tuple(item[0] for item in position_weights_and_mass)
        token_attention_masses = tuple(item[1] for item in position_weights_and_mass)
        position_entropies = tuple(item[2] for item in position_weights_and_mass)
        attention_values = tuple(
            max(
                0.0,
                float(
                    (prototype_weight_by_node or {}).get(
                        _source_node_id(graph, node_id),
                        0.0,
                    )
                ),
            )
            for node_id in member_ids
        )
        if config.pooling_mode in {
            "forecast_attention_token",
            "exact_medoid_key_forecast_value",
        }:
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
        if config.pooling_mode == "exact_medoid_key_forecast_value":
            key_member_index = next(
                index
                for index, block in enumerate(member_blocks)
                if prototype_logical_position in block.original_logical_positions
            )
            key_position_index = member_blocks[
                key_member_index
            ].original_logical_positions.index(prototype_logical_position)
        else:
            # Unused by averaging modes; retain stable scalar fields so the
            # fused recipe remains a compact, uniform representation.
            key_member_index = 0
            key_position_index = 0
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
                position_entropies=position_entropies,
                key_member_index=key_member_index,
                key_position_index=key_position_index,
            )
        )
        position_weights_by_node.update(zip(member_ids, position_weights, strict=True))

    blocks_by_node = {node_id: graph.nodes[node_id].block for node_id in ordered_member_ids}
    first_source = full_state.layers[recipes[0].anchor.layer].key
    prototype_keys_tuple: tuple[Any, ...]
    prototype_values_tuple: tuple[Any, ...]
    statistics_tuple: tuple[_DeviceMemberStatistics | _ResolvedMemberStatistics, ...]
    if first_source.__class__.__module__.startswith("torch"):
        (
            prototype_keys_tuple,
            prototype_values_tuple,
            device_statistics,
        ) = _fused_torch_prototypes(
            full_state,
            recipes,
            key_transform,
            config.pooling_mode,
        )
        statistics_tuple = tuple(device_statistics)
    else:
        pooled_keys_by_node, pooled_values_by_node = _batched_pool_source_members(
            full_state,
            blocks_by_node,
            position_weights_by_node,
            key_transform,
        )
        prototype_keys: list[Any] = []
        prototype_values: list[Any] = []
        numpy_statistics: list[_ResolvedMemberStatistics] = []
        gathered_sources = full_state.gather_exact_blocks(
            tuple(blocks_by_node[node_id] for node_id in ordered_member_ids),
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
        for recipe in recipes:
            pooled_keys = tuple(pooled_keys_by_node[node_id] for node_id in recipe.member_ids)
            pooled_values = tuple(pooled_values_by_node[node_id] for node_id in recipe.member_ids)
            reference_key, reference_value = gathered_by_node[recipe.member_ids[0]]
            if config.pooling_mode == "exact_medoid_key_forecast_value":
                medoid_node_id = recipe.member_ids[recipe.key_member_index]
                medoid_key = gathered_by_node[medoid_node_id][0]
                sequence_axis = (
                    full_state.layers[recipe.anchor.layer].key_sequence_dimension
                    % int(np.asarray(medoid_key).ndim)
                )
                prototype_key = np.take(
                    np.asarray(medoid_key),
                    (recipe.key_position_index,),
                    axis=sequence_axis,
                ).copy()
                key_average_fp32 = prototype_key.astype(np.float32, copy=True)
            else:
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
            source_to_prototype[_source_node_id(graph, node_id)] = recipe.prototype_id
    resolved_statistics = _resolve_member_statistics(
        tuple(pending.statistics for pending in pending_records)
    )
    records: list[PrototypeRecord] = []
    for pending, statistics in zip(pending_records, resolved_statistics, strict=True):
        member_errors, value_norms, key_dispersion, value_dispersion = statistics
        total_weight = sum(pending.raw_weights)
        recipe = recipes[pending.prototype_id]
        members = tuple(
            PrototypeMember._from_validated(
                node_id=_source_node_id(graph, node_id),
                raw_weight=raw_weight,
                normalized_weight=raw_weight / total_weight,
                reconstruction_error=reconstruction_error,
                value_output_norm=value_norm,
                position_weights=local_position_weights,
                within_block_attention_entropy=position_entropy,
            )
            for (
                node_id,
                raw_weight,
                reconstruction_error,
                value_norm,
                local_position_weights,
                position_entropy,
            ) in zip(
                pending.member_ids,
                pending.raw_weights,
                member_errors,
                value_norms,
                pending.position_weights,
                recipe.position_entropies,
                strict=True,
            )
        )
        modalities = Counter(block.modality for block in pending.member_blocks)
        minimum_logical_position = min(
            block.original_logical_positions[0] for block in pending.member_blocks
        )
        maximum_logical_position = max(
            block.original_logical_positions[-1] for block in pending.member_blocks
        )
        source_bytes = sum(block.byte_size for block in pending.member_blocks)
        prototype_bytes = tensor_storage_bytes(pending.prototype_key) + tensor_storage_bytes(
            pending.prototype_value
        )
        records.append(
            PrototypeRecord._from_validated(
                prototype_id=pending.prototype_id,
                layer=pending.anchor.layer,
                kv_head=pending.anchor.kv_head,
                anchor_node_id=_source_node_id(graph, pending.anchor_id),
                anchor_logical_position=pending.prototype_logical_position,
                members=members,
                assigned_node_ids=tuple(
                    sorted(_source_node_ids(graph, pending.member_ids))
                ),
                diagnostics=PrototypeDiagnostics(
                    cluster_size=len(pending.member_ids),
                    key_dispersion=key_dispersion,
                    value_dispersion=value_dispersion,
                    modality_composition=tuple(
                        sorted(modalities.items(), key=lambda item: item[0].value)
                    ),
                    minimum_logical_position=minimum_logical_position,
                    maximum_logical_position=maximum_logical_position,
                    position_span=maximum_logical_position - minimum_logical_position,
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
    return tier, tuple(records), source_to_prototype


def _selection_budget_violation(
    exact: ExactTier,
    prototype_tier: PrototypeTier,
    budget: SelectionBudget,
) -> str | None:
    unit = budget.unit
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
    if active <= budget.value:
        return None
    return (
        f"exact plus prototype active {unit.value} ({active}) exceed the "
        f"configured selection budget ({budget.value})"
    )


def estimate_prototype_active_cost(
    graph: SparseEvidenceGraph,
    selection: SelectionResult,
    config: PrototypeConfig,
    *,
    position_limits: Mapping[tuple[int, int], int] | None = None,
    prototype_anchor_node_ids: Sequence[int] | None = None,
    assignment_plan: PrototypeAssignmentPlan | None = None,
) -> tuple[int | None, str | None]:
    """Validate assignments and estimate exact prototype cost without tensor copies.

    Candidate-budget search can inspect dozens of selected sets. Materializing
    and casting every candidate K/V tier made planning proportional to the
    complete cache size times the number of search points. Fixed-size prototype
    slots have an exact descriptor-derived cost, so rejected candidates need no
    device gather at all.
    """

    assignments, reason = (
        _assign_to_anchors(
            graph,
            frozenset(selection.selected_node_ids),
            config,
            prototype_anchor_node_ids=(
                None
                if prototype_anchor_node_ids is None
                else frozenset(int(node_id) for node_id in prototype_anchor_node_ids)
            ),
        )
        if assignment_plan is None
        else _assignments_from_plan(
            assignment_plan,
            graph,
            selection,
            config,
            prototype_anchor_node_ids,
        )
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


def estimate_prototype_assignment_anchor_ids(
    graph: SparseEvidenceGraph,
    selection: SelectionResult,
    config: PrototypeConfig,
    *,
    prototype_anchor_node_ids: Sequence[int] | None = None,
) -> tuple[tuple[int, ...] | None, str | None]:
    """Return only anchors that would materialize a non-empty prototype.

    An anchor allowlist is a feasibility scaffold, not a prototype count.
    Mandatory terminal blocks, in particular, may anchor an otherwise empty
    tail segment. Reserving an active prototype slot for every allowed anchor
    therefore overstates the layout. This descriptor-only query uses the exact
    construction assignment rules without gathering or copying K/V tensors.
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
    return tuple(sorted({anchor_id for anchor_id, _weight in assignments.values()})), None


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
    assignment_plan: PrototypeAssignmentPlan | None = None,
    additional_exact_source_node_ids: Sequence[int] = (),
    active_budget: SelectionBudget | None = None,
) -> ThreeTierCacheConstruction:
    """Build exact, prototype, and CPU residual tiers or safely select exact-only.

    The current policy deliberately rejects all post-RoPE prototype averaging.
    A model family must explicitly advertise prototype support, and its observed
    source-cache metadata must agree with the adapter declaration.
    """

    prototypes = prototype_config or PrototypeConfig()
    residuals = residual_config or ResidualConfig()
    if exact_gather_mode not in {
        "per_head_reference",
        "batched_layer",
        "source_views",
    }:
        raise ValueError(
            "exact_gather_mode must be 'per_head_reference', 'batched_layer', or 'source_views'"
        )
    _validate_inputs(full_state, graph, selection)
    external_exact_node_ids = tuple(
        sorted(set(int(node_id) for node_id in additional_exact_source_node_ids))
    )
    if len(external_exact_node_ids) != len(additional_exact_source_node_ids):
        raise PrototypeConstructionError(
            "additional exact source node IDs must be unique"
        )
    if any(
        node_id < 0 or node_id >= len(full_state.blocks)
        for node_id in external_exact_node_ids
    ):
        raise PrototypeConstructionError(
            "additional exact source node ID lies outside FullKV blocks"
        )
    graph_source_set = {node.source_id for node in graph.nodes}
    if graph_source_set.intersection(external_exact_node_ids):
        raise PrototypeConstructionError(
            "additional exact blocks must be outside the routed planner graph"
        )
    resolved_active_budget = selection.budget if active_budget is None else active_budget
    if resolved_active_budget.unit is not selection.budget.unit:
        raise PrototypeConstructionError(
            "planner and total active budgets must use the same accounting unit"
        )
    if retention_ratio is not None and (
        not math.isfinite(retention_ratio) or not 0 < retention_ratio <= 1
    ):
        raise ValueError("retention_ratio must be finite and in the interval (0, 1]")
    if active_byte_budget is not None and active_byte_budget < 1:
        raise ValueError("active_byte_budget must be positive when provided")
    if total_storage_budget_bytes is not None and total_storage_budget_bytes < 1:
        raise ValueError("total_storage_budget_bytes must be positive when provided")
    if resolved_active_budget.unit is BudgetUnit.BYTES:
        active_byte_budget = (
            resolved_active_budget.value
            if active_byte_budget is None
            else min(active_byte_budget, resolved_active_budget.value)
        )
    selected_exact_bytes = selection.active_bytes + sum(
        full_state.blocks[node_id].byte_size for node_id in external_exact_node_ids
    )
    if active_byte_budget is not None and selected_exact_bytes > active_byte_budget:
        raise PrototypeConstructionError(
            "selected exact blocks already exceed active_byte_budget: "
            f"exact={selected_exact_bytes}, budget={active_byte_budget}"
        )
    if (
        total_storage_budget_bytes is not None
        and selected_exact_bytes > total_storage_budget_bytes
    ):
        raise PrototypeConstructionError(
            "selected exact blocks already exceed total_storage_budget_bytes: "
            f"exact={selected_exact_bytes}, budget={total_storage_budget_bytes}"
        )
    safety = assess_prototype_safety(
        full_state,
        capabilities,
        key_transform,
        exact_medoid_key=(
            prototypes.pooling_mode == "exact_medoid_key_forecast_value"
        ),
    )
    graph_source_node_ids = tuple(node.source_id for node in graph.nodes)
    exact_source_node_ids = tuple(
        sorted(
            {*_source_node_ids(graph, selection.selected_node_ids), *external_exact_node_ids}
        )
    )
    all_graph_node_ids = tuple(range(len(graph.nodes)))
    all_source_node_ids = tuple(range(len(full_state.blocks)))
    graph_covers_full_source = graph_source_node_ids == all_source_node_ids
    if retention_ratio == 1.0 and exact_source_node_ids != all_source_node_ids:
        raise PrototypeConstructionError(
            "retention_ratio=1.0 requires every source block to be selected exact"
        )
    if graph_covers_full_source and selection.selected_node_ids == all_graph_node_ids:
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
            exact_node_ids=exact_source_node_ids,
            active_budget=resolved_active_budget,
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
            exact_node_ids=exact_source_node_ids,
            active_budget=resolved_active_budget,
        )

    selected = frozenset(selection.selected_node_ids)
    assignments, incompatibility = (
        _assign_to_anchors(
            graph,
            selected,
            prototypes,
            prototype_anchor_node_ids=(
                None
                if prototype_anchor_node_ids is None
                else frozenset(int(node_id) for node_id in prototype_anchor_node_ids)
            ),
        )
        if assignment_plan is None
        else _assignments_from_plan(
            assignment_plan,
            graph,
            selection,
            prototypes,
            prototype_anchor_node_ids,
        )
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
            exact_node_ids=exact_source_node_ids,
            active_budget=resolved_active_budget,
        )
    exact = full_state.gather_exact_blocks(
        tuple(full_state.blocks[node_id] for node_id in exact_source_node_ids),
        gather_mode=exact_gather_mode,
    )
    prototype_tier, records, source_to_prototype = _prototype_payloads(
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
            exact_node_ids=exact_source_node_ids,
            active_budget=resolved_active_budget,
        )
    budget_violation = _selection_budget_violation(
        exact,
        prototype_tier,
        resolved_active_budget,
    )
    if budget_violation is not None:
        return _exact_only(
            full_state,
            selection,
            safety,
            capabilities.supports_residual_repair,
            mode=TierConstructionMode.EXACT_ONLY_INCOMPATIBLE,
            reason=budget_violation,
            exact_gather_mode=exact_gather_mode,
            exact_node_ids=exact_source_node_ids,
            active_budget=resolved_active_budget,
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
                exact_node_ids=exact_source_node_ids,
                active_budget=resolved_active_budget,
            )
    residual_node_order = tuple(
        node_id for record in records for node_id in record.assigned_node_ids
    )
    active_storage_bytes = exact.active_bytes + prototype_tier.active_bytes
    if total_storage_budget_bytes is not None and active_storage_bytes > total_storage_budget_bytes:
        raise PrototypeConstructionError(
            "active exact/prototype cache exceeds total storage budget: "
            f"active={active_storage_bytes}, total={total_storage_budget_bytes}"
        )
    residual_byte_limit = residuals.max_bytes
    if total_storage_budget_bytes is not None:
        remaining = total_storage_budget_bytes - active_storage_bytes
        residual_byte_limit = (
            remaining if residual_byte_limit is None else min(residual_byte_limit, remaining)
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
        priority_by_node=residual_priority_by_node,
    )
    state = MosaicKVState.create(
        full_state,
        exact=exact,
        prototypes=prototype_tier,
        residuals=residual_report.tier,
    )
    eligible_unselected = tuple(
        node.node_id
        for node in graph.nodes
        if node.node_id not in selected
        and prototypes.uses_prototype_block(
            node.block.layer,
            node.block.kv_head,
            node.block.modality,
        )
    )
    represented_unselected = len(assignments)
    bounded_partial_cover = (
        prototypes.budget_planning_mode == "one_pass_bounded_anchor_cover"
    )
    return ThreeTierCacheConstruction(
        state=state,
        mode=(
            TierConstructionMode.THREE_TIER
            if records
            else TierConstructionMode.EXACT_ONLY_ROUTED
        ),
        reason=(
            (
                (
                    "bounded prototype anchor cover assigned "
                    f"{represented_unselected}/{len(eligible_unselected)} eligible "
                    "unselected source blocks to compatible selected anchors; "
                )
                if bounded_partial_cover
                else (
                    "all unselected prototype-routed blocks were assigned to compatible "
                    "selected anchors; "
                )
            )
            + (
                f"residuals stored for {len(residual_report.payloads)}/"
                f"{len(residual_node_order)} prototyped source blocks"
            )
            if records
            else "all layer/head routes use exact selection without prototype merging"
        ),
        safety=safety,
        adapter_declares_residual_repair=capabilities.supports_residual_repair,
        active_budget=resolved_active_budget,
        exact_node_ids=exact_source_node_ids,
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
