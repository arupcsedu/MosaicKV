"""Conservative exact/prototype/residual cache construction."""

from __future__ import annotations

import math
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
) -> PrototypeSafetyAssessment:
    """Apply the initial no-post-RoPE-averaging safety policy."""

    source_state = _coerce_key_state(full_state.cached_key_state)
    adapter_state = _coerce_key_state(capabilities.cached_key_state)
    declares_support = capabilities.supports_prototype_merge
    safe = (
        declares_support
        and source_state == adapter_state
        and source_state in {CachedKeyState.PRE_ROPE, CachedKeyState.NOT_APPLICABLE}
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
    elif source_state is CachedKeyState.POST_ROPE:
        reason = (
            "cached keys are post-RoPE; the initial implementation never averages "
            "different RoPE phases"
        )
    elif not declares_support:
        reason = "adapter does not declare prototype merging safe"
    else:
        reason = (
            f"adapter explicitly supports prototype merging for {source_state.value} cached keys"
        )
    return PrototypeSafetyAssessment(
        model_family=capabilities.model_family,
        adapter_cached_key_state=adapter_state,
        source_cached_key_state=source_state,
        adapter_declares_support=declares_support,
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
    assignments: dict[int, tuple[int, float]] = {}
    for node_id, node in enumerate(graph.nodes):
        if node_id in selected:
            continue
        candidates: list[tuple[float, int]] = []
        for anchor_id in selected:
            anchor = graph.nodes[anchor_id]
            identity = (min(node_id, anchor_id), max(node_id, anchor_id))
            weight = weights.get(identity, 0.0)
            if (
                weight > 0
                and weight >= config.min_anchor_weight
                and _compatible(node.block, anchor.block, config)
            ):
                candidates.append((weight, anchor_id))
        if not candidates:
            return {}, f"unselected node {node_id} has no compatible selected graph anchor"
        # Higher graph weight wins; lower node ID is the deterministic tie-break.
        weight, anchor_id = min(candidates, key=lambda item: (-item[0], item[1]))
        assignments[node_id] = (anchor_id, weight)

    groups: dict[int, list[int]] = defaultdict(list)
    for node_id, (anchor_id, _weight) in assignments.items():
        groups[anchor_id].append(node_id)
    for anchor_id, member_ids in groups.items():
        if len(member_ids) > config.group_size:
            return {}, (
                f"anchor {anchor_id} received {len(member_ids)} members, exceeding "
                f"prototypes.group_size={config.group_size}"
            )
        if config.max_position_span is not None:
            blocks = [graph.nodes[anchor_id].block]
            blocks.extend(graph.nodes[node_id].block for node_id in member_ids)
            positions = [
                position for block in blocks for position in block.original_logical_positions
            ]
            if max(positions) - min(positions) > config.max_position_span:
                return {}, (
                    f"anchor {anchor_id} cluster exceeds prototypes.max_position_span="
                    f"{config.max_position_span}"
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
) -> tuple[PrototypeTier, tuple[PrototypeRecord, ...], dict[int, int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for node_id, (anchor_id, _weight) in assignments.items():
        grouped[anchor_id].append(node_id)
    source_blocks: list[KVBlockDescriptor] = []
    tier_assignments: list[int] = []
    prototype_keys: list[Any] = []
    prototype_values: list[Any] = []
    records: list[PrototypeRecord] = []
    source_to_prototype: dict[int, int] = {}

    for prototype_id, anchor_id in enumerate(sorted(grouped)):
        member_ids = tuple(sorted(grouped[anchor_id]))
        member_blocks = tuple(graph.nodes[node_id].block for node_id in member_ids)
        gathered = full_state.gather_exact_blocks(member_blocks)
        layer = full_state.layers[graph.nodes[anchor_id].block.layer]
        pooled_keys = tuple(
            _pool_fp32(tensor, layer.key_sequence_dimension) for tensor in gathered.key_blocks
        )
        pooled_values = tuple(
            _pool_fp32(tensor, layer.value_sequence_dimension) for tensor in gathered.value_blocks
        )
        raw_weights = tuple(assignments[node_id][1] for node_id in member_ids)
        key_average_fp32, prototype_key = _weighted_average(
            pooled_keys, raw_weights, gathered.key_blocks[0]
        )
        value_average_fp32, prototype_value = _weighted_average(
            pooled_values, raw_weights, gathered.value_blocks[0]
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
    return tier, tuple(records), source_to_prototype


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
    safety = assess_prototype_safety(full_state, capabilities)
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
    prototype_tier, records, source_to_prototype = _prototype_payloads(
        full_state, graph, assignments, selection
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
    residual_report = build_residual_storage(full_state, source_to_prototype, residuals)
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
]
