from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from mosaickv.baselines import (
    select_exact_baseline,
    select_prompt_attention_topk_ids,
    select_prompt_attention_topk_ids_by_head,
)
from mosaickv.baselines.selectors import (
    _atomic_group_multiple_choice_knapsack,
    _AtomicLayoutGroupCandidate,
    _best_prompt_subset_excluding_group,
    _best_prompt_subset_for_exact_positions,
    _build_prompt_subset_interval_exclusion_index,
    _gain_risk_pareto_frontier,
)
from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import ConfigurationError, GraphConfig, SelectionConfig, UtilityConfig
from mosaickv.graph import SparseEvidenceGraph, build_evidence_graph
from mosaickv.selection import (
    BlockUtilityTable,
    BudgetedObjective,
    InfeasibleSelectionBudgetError,
    SelectionBudget,
    SelectionReason,
    audit_objective_properties,
    compare_greedy_to_exhaustive,
    compute_block_utilities,
    exhaustive_select,
    lazy_greedy_select,
    select_all_exact,
)
from mosaickv.types import BudgetUnit, MosaicKVMethod


def _isolated_graph_config(**changes: object) -> GraphConfig:
    values: dict[str, object] = {
        "semantic_weight": 0.0,
        "attention_weight": 0.0,
        "spatial_weight": 0.0,
        "layout_weight": 0.0,
        "temporal_weight": 0.0,
        "same_region_weight": 0.0,
        "cross_modal_weight": 0.0,
        "fallback_weight": 0.0,
    }
    values.update(changes)
    return GraphConfig(**values)  # type: ignore[arg-type]


def _full(
    modalities: tuple[Modality, ...],
    *,
    mandatory: tuple[int, ...] = (),
    block_size: int = 1,
    values: np.ndarray | None = None,
    kv_heads: int = 1,
    image_grid_shape: tuple[int, int] | None = None,
) -> FullKVState:
    sequence_length = len(modalities)
    key = np.asarray(
        [
            [
                [
                    [(index + 1) / sequence_length, 1.0 - index / sequence_length]
                    for index in range(sequence_length)
                ]
            ]
        ],
        dtype=np.float32,
    ).reshape(1, 1, sequence_length, 2)
    key = np.repeat(key, kv_heads, axis=1)
    if values is None:
        value = key.copy()
    elif values.ndim == 2:
        value = np.repeat(values.reshape(1, 1, sequence_length, -1), kv_heads, axis=1)
    elif values.ndim == 3 and values.shape[:2] == (kv_heads, sequence_length):
        value = values.reshape(1, kv_heads, sequence_length, -1)
    else:
        raise ValueError(
            "test values must have shape [sequence, feature] or [head, sequence, feature]"
        )
    spans: list[ModalitySpan] = []
    start = 0
    for index in range(1, sequence_length + 1):
        if index == sequence_length or modalities[index] is not modalities[start]:
            modality = modalities[start]
            spans.append(
                ModalitySpan(
                    start,
                    index,
                    modality,
                    image_index=0 if modality is Modality.IMAGE else None,
                    frame_index=start if modality is Modality.VIDEO else None,
                    page_index=0 if modality is Modality.IMAGE else None,
                    grid_shape=image_grid_shape if modality is Modality.IMAGE else None,
                )
            )
            start = index
    return FullKVState.from_tensors(
        ((key, value),),
        modality_spans=tuple(spans),
        block_size=block_size,
        mandatory_logical_positions=mandatory,
    )


def _utilities_and_objective(
    full: FullKVState,
    *,
    probabilities: tuple[float, ...] | None = None,
    graph_config: GraphConfig | None = None,
    utility_config: UtilityConfig | None = None,
    selection_config: SelectionConfig | None = None,
) -> tuple[SparseEvidenceGraph, BlockUtilityTable, BudgetedObjective]:
    graph = build_evidence_graph(full, graph_config or _isolated_graph_config())
    values = probabilities or tuple(1.0 for _node in graph.nodes)
    utilities = compute_block_utilities(
        graph,
        utility_config or UtilityConfig(lambda_q=1.0, lambda_v=0.0, lambda_o=0.0),
        forecast_attention_by_node=values,
        attention_provenance="synthetic_rope_aware_fixture",
        rope_aware=True,
    )
    objective = BudgetedObjective(
        graph,
        utilities,
        selection_config or SelectionConfig(lambda_g=-0.5, lambda_m=-0.25),
    )
    return graph, utilities, objective


def _atomic_candidate(
    identity: tuple[int, int],
    group: tuple[int, ...],
    *,
    gain: float,
    risk: float,
    byte_size: int,
) -> _AtomicLayoutGroupCandidate:
    return _AtomicLayoutGroupCandidate(
        identity=identity,
        group=group,
        selected_optional_node_ids=group,
        normalized_query_gain=gain,
        normalized_prompt_risk=risk,
        output_impact_risk=risk,
        group_score=gain,
        group_bytes=byte_size,
        baseline_coverage=0.5,
        selected_coverage=0.5 + gain,
        baseline_prompt_mass=1.0,
        selected_prompt_mass=1.0 - risk,
        donor_node_ids=(),
        recipient_node_ids=group,
    )


def test_atomic_gain_risk_frontier_and_multiple_choice_knapsack_are_deterministic() -> None:
    route_zero = (
        _atomic_candidate((0, 0), (0, 1), gain=0.30, risk=0.05, byte_size=32),
        _atomic_candidate((0, 0), (2, 3), gain=0.20, risk=0.10, byte_size=32),
        _atomic_candidate((0, 0), (4, 5), gain=0.35, risk=0.20, byte_size=16),
    )
    route_one = (
        _atomic_candidate((0, 1), (6, 7), gain=0.25, risk=0.02, byte_size=32),
        _atomic_candidate((0, 1), (8, 9), gain=0.20, risk=0.03, byte_size=16),
    )

    frontier_zero = _gain_risk_pareto_frontier(route_zero)
    frontier_one = _gain_risk_pareto_frontier(route_one)
    assert tuple(candidate.group for candidate in frontier_zero) == ((0, 1), (4, 5))
    assert tuple(candidate.group for candidate in frontier_one) == ((6, 7), (8, 9))

    first = _atomic_group_multiple_choice_knapsack(
        {(0, 0): frontier_zero, (0, 1): frontier_one},
        byte_budget=48,
        gain_to_risk_ratio=1.01,
    )
    second = _atomic_group_multiple_choice_knapsack(
        {(0, 0): frontier_zero, (0, 1): frontier_one},
        byte_budget=48,
        gain_to_risk_ratio=1.01,
    )

    assert first == second
    assert tuple(candidate.group for candidate in first) == ((0, 1), (8, 9))
    assert sum(candidate.group_bytes for candidate in first) == 48
    assert len({candidate.identity for candidate in first}) == len(first)


def test_prompt_subset_interval_index_matches_exact_fill_with_partial_block() -> None:
    full = _full((Modality.IMAGE,) * 10, block_size=4)
    prompt_scores = np.asarray((0.2, 0.9, 0.8), dtype=np.float64)
    index = _build_prompt_subset_interval_exclusion_index(
        full,
        (0, 1, 2),
        maximum_positions=10,
        prompt_score_array=prompt_scores,
    )

    indexed = _best_prompt_subset_excluding_group(
        full,
        index,
        (1,),
        target_positions=6,
        prompt_score_array=prompt_scores,
    )
    reference = _best_prompt_subset_for_exact_positions(
        full,
        (0, 2),
        target_positions=6,
        prompt_score_array=prompt_scores,
    )
    fallback = _best_prompt_subset_excluding_group(
        full,
        index,
        (0, 2),
        target_positions=4,
        prompt_score_array=prompt_scores,
    )

    assert indexed == reference == (0, 2)
    assert fallback == (1,)


def test_block_utility_records_every_requested_signal_and_exact_formula() -> None:
    values = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float32,
    )
    full = _full(
        (Modality.TEXT, Modality.TEXT, Modality.IMAGE),
        mandatory=(0,),
        values=values,
    )
    graph = build_evidence_graph(
        full,
        _isolated_graph_config(semantic_weight=1.0, max_neighbors=2),
    )
    weights = UtilityConfig(lambda_q=2.0, lambda_v=0.5, lambda_o=0.25)

    table = compute_block_utilities(
        graph,
        weights,
        forecast_attention_by_node=(0.5, 0.3, 0.2),
        attention_provenance="synthetic_rope_aware_fixture",
        rope_aware=True,
    )

    first = table.for_node(0)
    assert first.forecast_attention_probability == pytest.approx(0.5)
    assert first.value_novelty == pytest.approx(0.0)
    assert first.redundancy_penalty == pytest.approx(1.0)
    assert first.expected_attention_output_contribution == pytest.approx(0.5)
    assert first.mandatory_priority == 1.0
    assert first.local_utility == pytest.approx(2.0 * 0.5 - 0.5 * 0.5)
    assert all(0 <= item.graph_centrality <= 1 for item in table.blocks)
    assert all(0 <= item.singleton_coverage_gain <= 1 for item in table.blocks)
    assert table.for_node(2).modality_rarity > table.for_node(0).modality_rarity


def test_signed_value_coefficient_rewards_expected_output_contribution() -> None:
    full = _full(
        (Modality.TEXT, Modality.TEXT),
        values=np.asarray([[10.0, 0.0], [1.0, 0.0]], dtype=np.float32),
    )
    graph = build_evidence_graph(full, _isolated_graph_config())
    table = compute_block_utilities(
        graph,
        UtilityConfig(lambda_q=0.0, lambda_v=-1.0, lambda_o=0.0),
        forecast_attention_by_node=(0.5, 0.5),
        attention_provenance="synthetic_signed_preservation_fixture",
        rope_aware=True,
    )

    assert table.for_node(0).value_contribution > table.for_node(1).value_contribution
    assert table.for_node(0).local_utility > table.for_node(1).local_utility


def test_value_aware_utility_only_materializes_routed_nodes() -> None:
    full = _full(
        (Modality.TEXT,) * 4,
        values=np.asarray(
            [[10.0, 0.0], [4.0, 0.0], [0.0, 3.0], [0.0, 1.0]],
            dtype=np.float32,
        ),
    )
    graph = build_evidence_graph(
        full,
        _isolated_graph_config(semantic_weight=1.0, max_neighbors=2),
    )

    table = compute_block_utilities(
        graph,
        UtilityConfig(lambda_q=1.0, lambda_v=-1.0, lambda_o=-1.0),
        forecast_attention_by_node=(0.4, 0.3, 0.2, 0.1),
        attention_provenance="synthetic_routed_utility_fixture",
        rope_aware=True,
        value_aware_node_ids=(2, 3),
    )

    for node_id in (0, 1):
        block = table.for_node(node_id)
        assert block.value_novelty == 0.0
        assert block.expected_attention_output_contribution == 0.0
        assert block.redundancy_penalty == 1.0
        assert block.local_utility == pytest.approx(block.forecast_attention_probability)
    assert table.for_node(2).expected_attention_output_contribution > 0
    assert table.for_node(3).expected_attention_output_contribution > 0


def test_token_attention_is_aggregated_and_normalized_by_layer_and_head() -> None:
    full = _full((Modality.TEXT,) * 4, block_size=2)
    graph = build_evidence_graph(full, _isolated_graph_config())

    table = compute_block_utilities(
        graph,
        UtilityConfig(lambda_q=1.0, lambda_v=0.0, lambda_o=0.0),
        forecast_attention_by_head={(0, 0): np.asarray([0.1, 0.2, 0.3, 0.4])},
        attention_provenance="synthetic_post_rope_attention",
        rope_aware=True,
    )

    assert [item.forecast_attention_probability for item in table.blocks] == pytest.approx(
        [0.3, 0.7]
    )
    with pytest.raises(ValueError, match="RoPE-aware"):
        compute_block_utilities(
            graph,
            UtilityConfig(),
            forecast_attention_by_node=(0.5, 0.5),
            attention_provenance="pre_rope_invalid",
            rope_aware=False,
        )


def test_value_signals_share_the_attention_l1_scale_within_each_head() -> None:
    values = np.asarray(
        [[1000.0, 0.0], [1.0, 0.0], [0.0, 2.0]],
        dtype=np.float32,
    )
    full = _full((Modality.TEXT,) * 3, values=values)
    graph = build_evidence_graph(
        full,
        _isolated_graph_config(semantic_weight=1.0, max_neighbors=2),
    )

    table = compute_block_utilities(
        graph,
        UtilityConfig(),
        forecast_attention_by_node=(0.5, 0.3, 0.2),
        attention_provenance="synthetic_rope_aware_fixture",
        rope_aware=True,
    )

    assert sum(item.forecast_attention_probability for item in table.blocks) == pytest.approx(1.0)
    assert sum(item.value_contribution for item in table.blocks) == pytest.approx(1.0)
    assert sum(item.uniqueness for item in table.blocks) == pytest.approx(1.0)
    assert all(0 <= item.value_contribution <= 1 for item in table.blocks)
    assert all(0 <= item.uniqueness <= 1 for item in table.blocks)


def test_vectorized_value_novelty_matches_sparse_reference() -> None:
    rng = np.random.default_rng(29)
    values = rng.normal(size=(31, 19)).astype(np.float32)
    full = _full((Modality.TEXT,) * len(values), values=values)
    graph = build_evidence_graph(
        full,
        _isolated_graph_config(semantic_weight=1.0, max_neighbors=6),
    )
    probabilities = tuple(float(value) for value in rng.uniform(0.1, 1.0, len(values)))
    common = {
        "forecast_attention_by_node": probabilities,
        "attention_provenance": "synthetic_vectorized_parity_fixture",
        "rope_aware": True,
    }

    reference = compute_block_utilities(
        graph,
        UtilityConfig(execution_mode="reference_sparse"),
        **common,
    )
    vectorized = compute_block_utilities(
        graph,
        UtilityConfig(execution_mode="vectorized_edges"),
        **common,
    )

    for left, right in zip(reference.blocks, vectorized.blocks, strict=True):
        assert left.node_id == right.node_id
        assert left.value_novelty == pytest.approx(right.value_novelty, abs=1e-12)
        assert left.redundancy_penalty == pytest.approx(
            right.redundancy_penalty,
            abs=1e-12,
        )
        assert left.value_contribution == pytest.approx(right.value_contribution, abs=1e-12)
        assert left.local_utility == pytest.approx(right.local_utility, abs=1e-12)


def test_lazy_greedy_is_deterministic_and_selects_mandatory_blocks_first() -> None:
    full = _full((Modality.TEXT,) * 5, mandatory=(2,))
    graph, _utilities, objective = _utilities_and_objective(full)
    budget = SelectionBudget(3, BudgetUnit.BLOCKS)

    first = lazy_greedy_select(objective, budget)
    second = lazy_greedy_select(objective, budget)

    assert first == second
    mandatory_id = next(node.node_id for node in graph.nodes if node.block.mandatory)
    assert first.selection_order[0] == mandatory_id
    assert mandatory_id in first.selected_node_ids
    assert first.decisions[mandatory_id].reason is SelectionReason.MANDATORY_EXACT
    assert len(first.decisions) == len(graph.nodes)
    assert first.budget_spent <= budget.value
    assert first.marginal_recomputations > 0
    assert first.to_exact_tier(full).active_bytes == first.active_bytes


def test_required_prototype_anchors_are_selected_before_lazy_greedy() -> None:
    full = _full((Modality.TEXT,) * 6, mandatory=(5,))
    _graph, _utilities, objective = _utilities_and_objective(full)

    result = lazy_greedy_select(
        objective,
        SelectionBudget(3, BudgetUnit.BLOCKS),
        required_node_ids=(3, 1, 3),
    )

    assert result.selection_order[:3] == (5, 1, 3)
    assert result.decisions[1].reason is SelectionReason.PROTOTYPE_ANCHOR
    assert result.decisions[3].reason is SelectionReason.PROTOTYPE_ANCHOR


def test_prompt_and_evidence_reservations_are_auditable_and_budgeted() -> None:
    full = _full((Modality.TEXT,) * 6, mandatory=(5,))
    _graph, _utilities, objective = _utilities_and_objective(full)

    result = lazy_greedy_select(
        objective,
        SelectionBudget(4, BudgetUnit.BLOCKS),
        prompt_anchor_node_ids=(3,),
        evidence_node_ids=(1,),
    )

    assert result.selection_order[:3] == (5, 3, 1)
    assert result.decisions[3].reason is SelectionReason.PROMPT_ANCHOR
    assert result.decisions[1].reason is SelectionReason.EVIDENCE_PRESERVATION
    assert result.budget_spent <= result.budget.value


@pytest.mark.parametrize("with_position_limits", (False, True))
def test_prompt_topk_id_only_path_matches_full_baseline(
    with_position_limits: bool,
) -> None:
    full = _full(
        (Modality.TEXT, Modality.TEXT, Modality.IMAGE, Modality.IMAGE, Modality.TEXT),
        mandatory=(4,),
    )
    budget = SelectionBudget(3, BudgetUnit.BLOCKS)
    scores = {0: 0.4, 1: 0.3, 2: 0.9, 3: 0.2, 4: 0.0}
    limits = {(0, 0): 3} if with_position_limits else None
    reference = select_exact_baseline(
        full,
        MosaicKVMethod.PROMPT_ATTENTION_TOPK,
        budget,
        seed=7,
        prompt_attention_by_node=scores,
        position_limits=limits,
    )

    assert (
        select_prompt_attention_topk_ids(
            full,
            budget,
            prompt_attention_by_node=scores,
            position_limits=limits,
        )
        == reference.selected_node_ids
    )


def test_vectorized_head_local_prompt_topk_matches_canonical_dense_selection() -> None:
    key = np.arange(2 * 6 * 2, dtype=np.float32).reshape(1, 2, 6, 2) + 1
    full = FullKVState.from_tensors(
        ((key, key.copy()),),
        block_size=1,
        mandatory_logical_positions=(5,),
    )
    scores = {
        node_id: float((10 if block.kv_head == 1 else 1) * (12 - node_id))
        for node_id, block in enumerate(full.blocks)
    }
    budget = SelectionBudget(6, BudgetUnit.BLOCKS)
    limits = {(0, 0): 3, (0, 1): 3}
    canonical = select_prompt_attention_topk_ids(
        full,
        budget,
        prompt_attention_by_node=scores,
        position_limits=limits,
    )

    local = select_prompt_attention_topk_ids_by_head(
        full,
        budget,
        prompt_attention_by_node=scores,
        head_routes=((0, 1),),
        position_limits=limits,
    )
    contiguous = select_prompt_attention_topk_ids_by_head(
        full,
        budget,
        prompt_attention_by_node={},
        prompt_attention_scores=tuple(scores[index] for index in range(len(scores))),
        head_routes=((0, 1),),
        position_limits=limits,
    )

    assert local.selected_node_ids == tuple(
        node_id for node_id in canonical if full.blocks[node_id].kv_head == 1
    )
    assert contiguous == local
    assert local.total_shortfall_positions == 0


def test_head_local_prompt_topk_solves_the_previous_64_of_75_stranding_case() -> None:
    sizes = (16, 16, 16, 16, 15, 15, 15, 15, 15)
    sequence = sum(sizes)
    key = np.arange(sequence * 2, dtype=np.float32).reshape(1, 1, sequence, 2)
    spans: list[ModalitySpan] = []
    offset = 0
    for image_index, size in enumerate(sizes):
        spans.append(
            ModalitySpan(
                offset,
                offset + size,
                Modality.IMAGE,
                image_index=image_index,
            )
        )
        offset += size
    full = FullKVState.from_tensors(
        ((key, key.copy()),),
        modality_spans=tuple(spans),
        block_size=16,
    )
    scores = {
        node_id: (10.0 if block.position_count == 16 else 9.0)
        for node_id, block in enumerate(full.blocks)
    }

    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(full.active_bytes, BudgetUnit.BYTES),
        prompt_attention_by_node=scores,
        head_routes=((0, 0),),
        position_limits={(0, 0): 75},
    )

    assert sum(full.blocks[node_id].position_count for node_id in result.selected_node_ids) == 75
    assert sum(scores[node_id] for node_id in result.selected_node_ids) == 45.0
    assert all(full.blocks[node_id].position_count == 15 for node_id in result.selected_node_ids)
    assert result.capacity_records[0].shortfall_positions == 0


def test_head_local_prompt_topk_fills_capacity_before_breaking_ties_by_score() -> None:
    sizes = (6, 6, 8, 5)
    sequence = sum(sizes)
    key = np.arange(sequence * 2, dtype=np.float32).reshape(1, 1, sequence, 2)
    spans: list[ModalitySpan] = []
    offset = 0
    for image_index, size in enumerate(sizes):
        spans.append(
            ModalitySpan(
                offset,
                offset + size,
                Modality.IMAGE,
                image_index=image_index,
            )
        )
        offset += size
    full = FullKVState.from_tensors(
        ((key, key.copy()),),
        modality_spans=tuple(spans),
        block_size=8,
    )
    scores = {
        node_id: (100.0 - node_id if block.position_count == 6 else 1.0)
        for node_id, block in enumerate(full.blocks)
    }

    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(full.active_bytes, BudgetUnit.BYTES),
        prompt_attention_by_node=scores,
        head_routes=((0, 0),),
        position_limits={(0, 0): 13},
    )

    selected_sizes = tuple(
        sorted(full.blocks[node_id].position_count for node_id in result.selected_node_ids)
    )
    assert selected_sizes == (5, 8)
    assert result.capacity_records[0].selected_positions == 13
    assert result.capacity_records[0].shortfall_positions == 0


def test_head_local_prompt_topk_records_unavoidable_capacity_shortfall() -> None:
    sequence = 80
    key = np.arange(sequence * 2, dtype=np.float32).reshape(1, 1, sequence, 2)
    full = FullKVState.from_tensors(
        ((key, key.copy()),),
        block_size=16,
    )

    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(full.active_bytes, BudgetUnit.BYTES),
        prompt_attention_by_node={node_id: float(10 - node_id) for node_id in range(5)},
        head_routes=((0, 0),),
        position_limits={(0, 0): 75},
    )

    assert len(result.selected_node_ids) == 4
    assert result.capacity_records[0].selected_positions == 64
    assert result.capacity_records[0].shortfall_positions == 11
    assert result.total_shortfall_positions == 11


def test_head_local_prompt_topk_reserves_visual_capacity_without_underfill() -> None:
    full = _full(
        (
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.TEXT,
            Modality.TEXT,
            Modality.TEXT,
            Modality.TEXT,
        ),
        mandatory=(7,),
    )
    scores = {
        node_id: (100.0 - node_id if block.modality is Modality.TEXT else 1.0)
        for node_id, block in enumerate(full.blocks)
    }

    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(4, BudgetUnit.BLOCKS),
        prompt_attention_by_node=scores,
        head_routes=((0, 0),),
        position_limits={(0, 0): 4},
        minimum_visual_fraction=0.5,
    )

    selected = [full.blocks[node_id] for node_id in result.selected_node_ids]
    assert sum(block.position_count for block in selected) == 4
    assert sum(block.modality is Modality.IMAGE for block in selected) == 2
    assert result.capacity_records[0].selected_visual_positions == 2
    assert result.capacity_records[0].target_visual_positions == 2
    assert result.capacity_records[0].shortfall_positions == 0


def test_head_local_prompt_topk_adaptive_swap_preserves_capacity_and_text_floor() -> None:
    full = _full(
        (
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.TEXT,
            Modality.TEXT,
            Modality.TEXT,
            Modality.TEXT,
        ),
        mandatory=(6,),
    )
    scores = {
        0: 0.10,
        1: 0.09,
        2: 0.00,
        3: 0.20,
        4: 0.19,
        5: 0.18,
        6: 100.0,
    }

    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(4, BudgetUnit.BLOCKS),
        prompt_attention_by_node=scores,
        head_routes=((0, 0),),
        position_limits={(0, 0): 4},
        adaptive_visual_swaps=True,
        minimum_text_mass_fraction=0.995,
        visual_gain_ratio=1.0,
        maximum_visual_swaps=2,
    )

    record = result.capacity_records[0]
    assert record.selected_positions == 4
    assert record.shortfall_positions == 0
    assert record.adaptive_swap_count == 2
    assert record.selected_visual_positions == 2
    assert record.selected_text_prompt_mass >= record.baseline_text_prompt_mass * 0.995
    assert record.normalized_visual_evidence_gain > record.normalized_text_prompt_mass_loss
    assert 6 in result.selected_node_ids

    with_unused_forecast = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(4, BudgetUnit.BLOCKS),
        prompt_attention_by_node=scores,
        forecast_attention_by_node={node_id: float(node_id + 1) for node_id in scores},
        head_routes=((0, 0),),
        position_limits={(0, 0): 4},
        adaptive_visual_swaps=True,
        minimum_text_mass_fraction=0.995,
        visual_gain_ratio=1.0,
        maximum_visual_swaps=2,
    )
    assert with_unused_forecast.selected_node_ids == result.selected_node_ids
    assert with_unused_forecast.capacity_records[0].adaptive_swap_pairs == (
        result.capacity_records[0].adaptive_swap_pairs
    )


def test_head_local_prompt_topk_adaptive_swap_rejects_text_floor_violation() -> None:
    full = _full(
        (Modality.IMAGE, Modality.IMAGE, Modality.TEXT, Modality.TEXT),
        mandatory=(3,),
    )
    scores = {0: 0.10, 1: 0.09, 2: 0.20, 3: 1.0}

    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(2, BudgetUnit.BLOCKS),
        prompt_attention_by_node=scores,
        head_routes=((0, 0),),
        position_limits={(0, 0): 2},
        adaptive_visual_swaps=True,
        minimum_text_mass_fraction=0.99,
        visual_gain_ratio=1.0,
        maximum_visual_swaps=1,
    )

    record = result.capacity_records[0]
    assert record.adaptive_swap_count == 0
    assert record.selected_visual_positions == 0
    assert result.selected_node_ids == (2, 3)


def test_head_local_prompt_topk_forecast_risk_gate_rejects_future_text_loss() -> None:
    full = _full(
        (Modality.IMAGE, Modality.IMAGE, Modality.TEXT, Modality.TEXT),
        mandatory=(3,),
    )
    prompt_scores = {0: 0.10, 1: 0.09, 2: 0.20, 3: 100.0}
    forecast_scores = {0: 0.01, 1: 0.01, 2: 1.0, 3: 1.0}

    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(2, BudgetUnit.BLOCKS),
        prompt_attention_by_node=prompt_scores,
        forecast_attention_by_node=forecast_scores,
        head_routes=((0, 0),),
        position_limits={(0, 0): 2},
        adaptive_visual_swaps=True,
        minimum_text_mass_fraction=0.997,
        visual_gain_ratio=1.0,
        maximum_visual_swaps=1,
        swap_risk_gate=True,
        maximum_normalized_text_loss=0.002,
        forecast_gain_ratio=1.5,
    )

    record = result.capacity_records[0]
    assert record.adaptive_swap_count == 0
    assert record.adaptive_swap_candidate_count >= 1
    assert record.adaptive_swap_risk_rejected_count >= 1
    assert record.adaptive_swap_pairs == ()
    assert result.selected_node_ids == (2, 3)


def test_head_local_prompt_topk_forecast_risk_gate_records_safe_per_head_pair() -> None:
    full = _full(
        (Modality.IMAGE, Modality.IMAGE, Modality.TEXT, Modality.TEXT),
        mandatory=(3,),
    )
    prompt_scores = {0: 0.10, 1: 0.09, 2: 0.20, 3: 100.0}
    forecast_scores = {0: 1.0, 1: 0.01, 2: 0.001, 3: 1.0}

    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(2, BudgetUnit.BLOCKS),
        prompt_attention_by_node=prompt_scores,
        forecast_attention_by_node=forecast_scores,
        head_routes=((0, 0),),
        position_limits={(0, 0): 2},
        adaptive_visual_swaps=True,
        minimum_text_mass_fraction=0.997,
        visual_gain_ratio=1.0,
        maximum_visual_swaps=1,
        swap_risk_gate=True,
        maximum_normalized_text_loss=0.002,
        forecast_gain_ratio=1.5,
    )

    record = result.capacity_records[0]
    assert record.adaptive_swap_count == 1
    assert record.adaptive_swap_pairs == ((2, 0),)
    assert record.normalized_forecast_visual_gain > record.normalized_forecast_text_loss
    assert record.selected_positions == 2
    assert result.selected_node_ids == (0, 3)


def test_head_local_future_query_reranking_is_same_byte_prompt_anchored_and_deterministic() -> None:
    full = _full(
        (
            Modality.TEXT,
            Modality.TEXT,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.TEXT,
        ),
        mandatory=(5,),
    )
    prompt_scores = {0: 10.0, 1: 9.0, 2: 1.0, 3: 8.0, 4: 7.0, 5: 100.0}
    forecast_scores = {0: 0.2, 1: 0.1, 2: 0.0, 3: 10.0, 4: 9.0, 5: 1.0}
    kwargs = {
        "prompt_attention_by_node": prompt_scores,
        "forecast_attention_by_node": forecast_scores,
        "head_routes": ((0, 0),),
        "position_limits": {(0, 0): 3},
        "future_query_reranking": True,
        "future_minimum_prompt_mass_fraction": 0.99,
        "future_gain_ratio": 1.0,
        "maximum_future_replacements": 2,
        "future_divergence_fail_closed": True,
        "future_minimum_prompt_forecast_tv": 0.01,
    }

    first = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(3, BudgetUnit.BLOCKS),
        **kwargs,
    )
    second = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(3, BudgetUnit.BLOCKS),
        **kwargs,
    )

    assert first == second
    assert first.selected_node_ids == (0, 3, 5)
    record = first.capacity_records[0]
    assert record.selected_positions == 3
    assert record.selected_bytes == sum(full.blocks[node_id].byte_size for node_id in (0, 3, 5))
    assert record.future_rerank_count == 1
    assert record.future_rerank_pairs == ((1, 3),)
    assert record.selected_prompt_mass_after_future_rerank >= (
        record.baseline_selected_prompt_mass * 0.99
    )
    assert record.normalized_future_query_gain > record.normalized_future_prompt_mass_loss
    assert record.prompt_forecast_total_variation >= 0.01
    assert record.forecast_divergence_eligible
    assert 5 in first.selected_node_ids


def test_future_query_reranking_rejects_visual_swap_combination() -> None:
    full = _full((Modality.TEXT, Modality.IMAGE), mandatory=(1,))
    with pytest.raises(ValueError, match="mutually exclusive"):
        select_prompt_attention_topk_ids_by_head(
            full,
            SelectionBudget(1, BudgetUnit.BLOCKS),
            prompt_attention_by_node={0: 1.0, 1: 2.0},
            forecast_attention_by_node={0: 2.0, 1: 1.0},
            head_routes=((0, 0),),
            position_limits={(0, 0): 1},
            adaptive_visual_swaps=True,
            maximum_visual_swaps=1,
            future_query_reranking=True,
            maximum_future_replacements=1,
        )


def test_future_query_reranking_risk_gate_rejects_large_prompt_loss() -> None:
    full = _full(
        (
            Modality.TEXT,
            Modality.TEXT,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.TEXT,
        ),
        mandatory=(5,),
    )
    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(3, BudgetUnit.BLOCKS),
        prompt_attention_by_node={0: 10.0, 1: 9.0, 2: 1.0, 3: 8.0, 4: 7.0, 5: 100.0},
        forecast_attention_by_node={0: 0.2, 1: 0.1, 2: 0.0, 3: 10.0, 4: 9.0, 5: 1.0},
        head_routes=((0, 0),),
        position_limits={(0, 0): 3},
        future_query_reranking=True,
        future_minimum_prompt_mass_fraction=0.99,
        future_gain_ratio=1.0,
        maximum_future_replacements=2,
        future_divergence_fail_closed=True,
        future_minimum_prompt_forecast_tv=0.01,
        future_risk_gate=True,
        future_maximum_normalized_prompt_loss=0.005,
    )

    assert result.selected_node_ids == (0, 1, 5)
    record = result.capacity_records[0]
    assert record.future_rerank_count == 0
    assert record.future_rerank_risk_rejected_count == 2
    assert record.future_rerank_rejected_count == 2


def test_future_query_reranking_fails_closed_when_forecast_matches_prompt() -> None:
    full = _full((Modality.TEXT,) * 4, mandatory=(3,))
    scores = {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}

    with pytest.raises(ValueError, match="did not diverge"):
        select_prompt_attention_topk_ids_by_head(
            full,
            SelectionBudget(2, BudgetUnit.BLOCKS),
            prompt_attention_by_node=scores,
            forecast_attention_by_node=scores,
            head_routes=((0, 0),),
            position_limits={(0, 0): 2},
            future_query_reranking=True,
            future_minimum_prompt_mass_fraction=0.99,
            future_gain_ratio=1.0,
            maximum_future_replacements=1,
            future_divergence_fail_closed=True,
            future_minimum_prompt_forecast_tv=0.01,
        )


def test_prompt_query_coverage_reranking_preserves_same_bytes_and_distinct_evidence() -> None:
    full = _full(
        (Modality.IMAGE, Modality.IMAGE, Modality.IMAGE, Modality.TEXT),
        mandatory=(3,),
    )
    prompt_scores = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.0}
    query_attention = {
        0: (0.9, 0.1),
        1: (0.8, 0.2),
        2: (0.1, 0.9),
        3: (0.0, 0.0),
    }
    kwargs = {
        "prompt_attention_by_node": prompt_scores,
        "prompt_query_attention_by_node": query_attention,
        "head_routes": ((0, 0),),
        "position_limits": {(0, 0): 3},
        "query_coverage_reranking": True,
        "query_coverage_minimum_prompt_mass_fraction": 1.0,
        "query_coverage_maximum_replacements": 1,
        "query_coverage_minimum_normalized_gain": 0.1,
    }

    first = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(3, BudgetUnit.BLOCKS),
        **kwargs,
    )
    second = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(3, BudgetUnit.BLOCKS),
        **kwargs,
    )

    assert first == second
    assert first.selected_node_ids == (0, 2, 3)
    record = first.capacity_records[0]
    assert record.selected_positions == 3
    assert record.shortfall_positions == 0
    assert record.query_coverage_rerank_count == 1
    assert record.query_coverage_pairs == ((1, 2),)
    assert record.selected_query_coverage > record.baseline_query_coverage
    assert record.normalized_query_coverage_gain > 0.1
    assert record.query_coverage_selected_prompt_mass == (
        record.query_coverage_baseline_prompt_mass
    )
    assert record.query_coverage_normalized_prompt_loss == 0.0
    assert record.selected_bytes == sum(
        full.blocks[node_id].byte_size for node_id in first.selected_node_ids
    )


def test_prompt_query_coverage_reranking_never_crosses_modalities() -> None:
    full = _full(
        (Modality.IMAGE, Modality.IMAGE, Modality.TEXT, Modality.TEXT),
        mandatory=(3,),
    )
    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(2, BudgetUnit.BLOCKS),
        prompt_attention_by_node={0: 0.5, 1: 0.4, 2: 0.1, 3: 0.0},
        prompt_query_attention_by_node={
            0: (0.9, 0.0),
            1: (0.8, 0.0),
            2: (0.0, 1.0),
            3: (0.0, 0.0),
        },
        head_routes=((0, 0),),
        position_limits={(0, 0): 2},
        query_coverage_reranking=True,
        query_coverage_minimum_prompt_mass_fraction=0.5,
        query_coverage_maximum_replacements=1,
        query_coverage_minimum_normalized_gain=0.01,
    )

    assert result.selected_node_ids == (0, 3)
    record = result.capacity_records[0]
    assert record.query_coverage_rerank_count == 0
    assert record.query_coverage_pairs == ()
    assert record.query_coverage_candidate_count == 1


def test_prompt_query_coverage_reranking_fails_closed_on_invalid_evidence() -> None:
    full = _full((Modality.IMAGE, Modality.IMAGE, Modality.TEXT), mandatory=(2,))
    with pytest.raises(ValueError, match="must cover every source block"):
        select_prompt_attention_topk_ids_by_head(
            full,
            SelectionBudget(2, BudgetUnit.BLOCKS),
            prompt_attention_by_node={0: 0.5, 1: 0.4, 2: 0.0},
            prompt_query_attention_by_node={0: (0.9, 0.1), 2: (0.0, 0.0)},
            head_routes=((0, 0),),
            position_limits={(0, 0): 2},
            query_coverage_reranking=True,
            query_coverage_maximum_replacements=1,
            query_coverage_minimum_normalized_gain=0.01,
        )


def test_prompt_query_coverage_initial_selection_reserves_diverse_evidence() -> None:
    full = _full(
        (Modality.IMAGE, Modality.IMAGE, Modality.IMAGE, Modality.IMAGE, Modality.TEXT),
        mandatory=(4,),
    )
    prompt_scores = {0: 0.6, 1: 0.5, 2: 0.4, 3: 0.3, 4: 0.0}
    query_attention = {
        0: (1.0, 0.0),
        1: (0.9, 0.0),
        2: (0.0, 1.0),
        3: (0.0, 0.8),
        4: (0.0, 0.0),
    }
    kwargs = {
        "prompt_attention_by_node": prompt_scores,
        "prompt_query_attention_by_node": query_attention,
        "head_routes": ((0, 0),),
        "position_limits": {(0, 0): 3},
        "query_coverage_initial_selection": True,
        "query_coverage_initial_fraction": 1.0,
        "query_coverage_initial_prompt_weight": 0.1,
        "query_coverage_initial_minimum_prompt_mass_fraction": 0.8,
    }

    first = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(3, BudgetUnit.BLOCKS),
        **kwargs,
    )
    second = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(3, BudgetUnit.BLOCKS),
        **kwargs,
    )

    assert first == second
    assert first.selected_node_ids == (0, 2, 4)
    record = first.capacity_records[0]
    assert record.selected_positions == 3
    assert record.shortfall_positions == 0
    assert record.query_coverage_initial_reserved_count == 2
    assert record.query_coverage_initial_changed_count == 1
    assert record.query_coverage_initial_added_node_ids == (2,)
    assert record.query_coverage_initial_selected > record.query_coverage_initial_baseline
    assert record.query_coverage_initial_normalized_gain > 0
    assert record.query_coverage_initial_selected_prompt_mass >= (
        0.8 * record.query_coverage_initial_baseline_prompt_mass
    )
    assert record.selected_bytes == sum(
        full.blocks[node_id].byte_size for node_id in first.selected_node_ids
    )


def test_prompt_query_coverage_initial_selection_falls_back_at_prompt_mass_floor() -> None:
    full = _full(
        (Modality.IMAGE, Modality.IMAGE, Modality.IMAGE, Modality.TEXT),
        mandatory=(3,),
    )
    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(3, BudgetUnit.BLOCKS),
        prompt_attention_by_node={0: 0.6, 1: 0.5, 2: 0.1, 3: 0.0},
        prompt_query_attention_by_node={
            0: (1.0, 0.0),
            1: (0.9, 0.0),
            2: (0.0, 1.0),
            3: (0.0, 0.0),
        },
        head_routes=((0, 0),),
        position_limits={(0, 0): 3},
        query_coverage_initial_selection=True,
        query_coverage_initial_fraction=1.0,
        query_coverage_initial_prompt_weight=0.0,
        query_coverage_initial_minimum_prompt_mass_fraction=1.0,
    )

    assert result.selected_node_ids == (0, 1, 3)
    record = result.capacity_records[0]
    assert record.query_coverage_initial_reserved_count == 0
    assert record.query_coverage_initial_changed_count == 0


def test_capped_cumulative_query_coverage_preserves_additional_query_mass() -> None:
    full = _full(
        (Modality.IMAGE, Modality.IMAGE, Modality.IMAGE, Modality.IMAGE, Modality.TEXT),
        mandatory=(4,),
    )
    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(3, BudgetUnit.BLOCKS),
        prompt_attention_by_node={0: 0.6, 1: 0.5, 2: 0.4, 3: 0.3, 4: 0.0},
        prompt_query_attention_by_node={
            0: (0.6, 0.0),
            1: (0.4, 0.0),
            2: (0.0, 0.6),
            3: (0.0, 0.4),
            4: (0.0, 0.0),
        },
        head_routes=((0, 0),),
        position_limits={(0, 0): 3},
        query_coverage_initial_selection=True,
        query_coverage_initial_fraction=1.0,
        query_coverage_initial_prompt_weight=0.1,
        query_coverage_initial_minimum_prompt_mass_fraction=0.9,
        query_coverage_initial_objective="capped_cumulative",
        query_coverage_initial_cap_fraction=0.5,
    )

    assert result.selected_node_ids == (0, 2, 4)
    record = result.capacity_records[0]
    assert record.query_coverage_initial_changed_count == 1
    assert record.query_coverage_initial_added_node_ids == (2,)
    assert record.query_coverage_initial_selected == pytest.approx(1.0)
    assert record.query_coverage_initial_baseline == pytest.approx(0.5)


def test_cross_head_query_byte_allocation_prioritizes_high_risk_head() -> None:
    full = _full(
        (Modality.IMAGE, Modality.IMAGE, Modality.IMAGE, Modality.IMAGE, Modality.TEXT),
        mandatory=(4,),
        kv_heads=2,
    )
    prompt_scores = {
        0: 0.6,
        1: 0.5,
        2: 0.4,
        3: 0.3,
        4: 0.0,
        5: 0.6,
        6: 0.5,
        7: 0.4,
        8: 0.3,
        9: 0.0,
    }
    query_attention = {
        0: (0.5, 0.5),
        1: (0.5, 0.5),
        2: (0.0, 0.0),
        3: (0.0, 0.0),
        4: (0.0, 0.0),
        5: (0.6, 0.0),
        6: (0.4, 0.0),
        7: (0.0, 0.6),
        8: (0.0, 0.4),
        9: (0.0, 0.0),
    }
    kwargs = {
        "prompt_attention_by_node": prompt_scores,
        "prompt_query_attention_by_node": query_attention,
        "head_routes": ((0, 0), (0, 1)),
        "position_limits": {(0, 0): 3, (0, 1): 3},
        "query_coverage_initial_selection": True,
        "query_coverage_initial_fraction": 0.5,
        "query_coverage_initial_prompt_weight": 0.1,
        "query_coverage_initial_minimum_prompt_mass_fraction": 0.8,
        "query_coverage_initial_objective": "capped_cumulative",
        "query_coverage_initial_cap_fraction": 0.5,
        "query_coverage_cross_head_allocation": True,
    }

    first = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(6, BudgetUnit.BLOCKS),
        **kwargs,
    )
    second = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(6, BudgetUnit.BLOCKS),
        **kwargs,
    )

    assert first == second
    assert first.selected_node_ids == (0, 1, 4, 5, 7, 9)
    by_head = {record.kv_head: record for record in first.capacity_records}
    assert by_head[0].query_coverage_initial_global_allocation
    assert by_head[1].query_coverage_initial_global_allocation
    assert by_head[0].query_coverage_initial_reserved_bytes == 0
    assert by_head[1].query_coverage_initial_reserved_bytes == 2 * full.blocks[5].byte_size
    assert by_head[0].query_coverage_initial_allocated_bytes == 0
    assert by_head[1].query_coverage_initial_allocated_bytes == 2 * full.blocks[5].byte_size
    assert by_head[1].query_coverage_initial_cross_head_risk > (
        by_head[0].query_coverage_initial_cross_head_risk
    )
    assert by_head[1].query_coverage_initial_selected > (by_head[1].query_coverage_initial_baseline)
    assert sum(record.selected_bytes for record in first.capacity_records) == (
        full.active_bytes * 3 // 5
    )


def test_cross_head_layout_group_reservation_protects_complete_spatial_group() -> None:
    full = _full(
        (
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.TEXT,
        ),
        mandatory=(6,),
        image_grid_shape=(1, 6),
    )
    kwargs = {
        "prompt_attention_by_node": {
            0: 0.9,
            1: 0.8,
            2: 0.7,
            3: 0.6,
            4: 0.5,
            5: 0.4,
            6: 0.0,
        },
        "prompt_query_attention_by_node": {
            0: (0.1, 0.0),
            1: (0.1, 0.0),
            2: (0.1, 0.0),
            3: (0.1, 0.0),
            4: (0.0, 1.0),
            5: (0.0, 1.0),
            6: (0.0, 0.0),
        },
        "head_routes": ((0, 0),),
        "position_limits": {(0, 0): 5},
        "query_coverage_initial_selection": True,
        "query_coverage_initial_fraction": 0.5,
        "query_coverage_initial_prompt_weight": 0.5,
        "query_coverage_initial_minimum_prompt_mass_fraction": 0.8,
        "query_coverage_initial_objective": "capped_cumulative",
        "query_coverage_initial_cap_fraction": 0.5,
        "query_coverage_cross_head_allocation": True,
        "query_coverage_layout_group_reservation": True,
        "query_coverage_layout_group_fraction": 1.0,
        "query_coverage_layout_group_maximum_blocks": 2,
        "query_coverage_layout_group_row_bins": 1,
    }

    first = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(5, BudgetUnit.BLOCKS),
        **kwargs,
    )
    second = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(5, BudgetUnit.BLOCKS),
        **kwargs,
    )

    assert first == second
    assert first.selected_node_ids == (0, 1, 4, 5, 6)
    record = first.capacity_records[0]
    assert record.query_coverage_initial_layout_group_candidate_count == 3
    assert record.query_coverage_initial_layout_group_protected_group_count == 1
    assert record.query_coverage_initial_layout_group_protected_block_count == 2
    assert record.query_coverage_initial_layout_group_node_ids == ((4, 5),)
    assert record.query_coverage_initial_layout_group_protected_bytes == (
        full.blocks[4].byte_size + full.blocks[5].byte_size
    )
    assert record.query_coverage_initial_layout_group_protected_score > 0
    assert record.query_coverage_initial_output_impact_protected_count == 0
    assert record.selected_positions == 5
    assert record.shortfall_positions == 0


def test_cross_head_layout_group_reservation_fails_closed_without_regions() -> None:
    full = _full(
        (Modality.IMAGE, Modality.IMAGE, Modality.IMAGE, Modality.TEXT),
        mandatory=(3,),
    )
    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(3, BudgetUnit.BLOCKS),
        prompt_attention_by_node={0: 0.9, 1: 0.8, 2: 0.1, 3: 0.0},
        prompt_query_attention_by_node={
            0: (1.0, 0.0),
            1: (0.8, 0.0),
            2: (0.0, 1.0),
            3: (0.0, 0.0),
        },
        head_routes=((0, 0),),
        position_limits={(0, 0): 3},
        query_coverage_initial_selection=True,
        query_coverage_initial_fraction=0.5,
        query_coverage_initial_prompt_weight=0.5,
        query_coverage_initial_minimum_prompt_mass_fraction=0.95,
        query_coverage_initial_objective="capped_cumulative",
        query_coverage_initial_cap_fraction=0.5,
        query_coverage_cross_head_allocation=True,
        query_coverage_layout_group_reservation=True,
        query_coverage_layout_group_fraction=1.0,
        query_coverage_layout_group_maximum_blocks=2,
        query_coverage_layout_group_row_bins=1,
    )

    assert result.selected_node_ids == (0, 1, 3)
    record = result.capacity_records[0]
    assert record.query_coverage_initial_layout_group_candidate_count == 0
    assert record.query_coverage_initial_layout_group_protected_group_count == 0
    assert record.query_coverage_initial_layout_group_node_ids == ()


def test_atomic_layout_group_survives_final_capacity_selection() -> None:
    full = _full(
        (
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.TEXT,
        ),
        mandatory=(6,),
        image_grid_shape=(1, 6),
    )
    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(5, BudgetUnit.BLOCKS),
        prompt_attention_by_node={
            0: 0.9,
            1: 0.8,
            2: 0.7,
            3: 0.6,
            4: 0.5,
            5: 0.4,
            6: 0.0,
        },
        prompt_query_attention_by_node={
            0: (0.1, 0.0),
            1: (0.1, 0.0),
            2: (0.1, 0.0),
            3: (0.1, 0.0),
            4: (0.0, 1.0),
            5: (0.0, 1.0),
            6: (0.0, 0.0),
        },
        head_routes=((0, 0),),
        position_limits={(0, 0): 5},
        query_coverage_initial_selection=True,
        query_coverage_initial_fraction=0.5,
        query_coverage_initial_prompt_weight=0.5,
        query_coverage_initial_minimum_prompt_mass_fraction=0.8,
        query_coverage_initial_objective="capped_cumulative",
        query_coverage_initial_cap_fraction=0.5,
        query_coverage_cross_head_allocation=True,
        query_coverage_cross_head_stability_gate=True,
        query_coverage_cross_head_stability_risk_mode="output_impact",
        query_coverage_cross_head_minimum_normalized_gain=0.01,
        query_coverage_cross_head_gain_to_prompt_loss_ratio=1.01,
        query_coverage_layout_group_reservation=True,
        query_coverage_layout_group_fraction=1.0,
        query_coverage_layout_group_maximum_blocks=2,
        query_coverage_layout_group_row_bins=1,
        query_coverage_layout_group_atomic_selection=True,
        query_coverage_layout_group_atomic_selection_mode="pareto_knapsack",
        query_coverage_layout_group_minimum_utilization_fraction=1.0,
        query_coverage_layout_group_minimum_normalized_gain=0.01,
        query_coverage_layout_group_gain_to_risk_ratio=1.01,
    )

    assert result.selected_node_ids == (0, 1, 4, 5, 6)
    record = result.capacity_records[0]
    assert record.query_coverage_initial_layout_group_node_ids == ((4, 5),)
    assert set(record.query_coverage_initial_layout_group_node_ids[0]) <= set(
        result.selected_node_ids
    )
    assert record.query_coverage_initial_layout_group_accepted_candidate_count == 1
    assert record.query_coverage_initial_layout_group_pareto_candidate_count > 0
    assert record.query_coverage_initial_layout_group_knapsack_selected_count == 1
    assert record.query_coverage_initial_layout_group_target_bytes == (
        full.blocks[4].byte_size + full.blocks[5].byte_size
    )
    assert record.query_coverage_initial_layout_group_protected_bytes == (
        record.query_coverage_initial_layout_group_target_bytes
    )
    assert record.query_coverage_initial_layout_group_normalized_query_gain > 0
    assert record.selected_positions == 5
    assert record.shortfall_positions == 0


def test_joint_group_donor_selection_avoids_high_output_impact_donors() -> None:
    values = np.asarray(
        (
            (1.0, 0.0),
            (1.0, 0.0),
            (-1.0, 0.0),
            (-1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
        ),
        dtype=np.float32,
    )
    full = _full(
        (Modality.IMAGE,) * 6 + (Modality.TEXT,),
        mandatory=(6,),
        image_grid_shape=(1, 6),
        values=values,
    )
    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(5, BudgetUnit.BLOCKS),
        prompt_attention_by_node={
            0: 0.90,
            1: 0.89,
            2: 0.88,
            3: 0.87,
            4: 0.86,
            5: 0.85,
            6: 0.0,
        },
        prompt_query_attention_by_node={
            0: (0.25, 0.0),
            1: (0.25, 0.0),
            2: (0.25, 0.0),
            3: (0.25, 0.0),
            4: (0.0, 1.0),
            5: (0.0, 1.0),
            6: (0.0, 0.0),
        },
        head_routes=((0, 0),),
        position_limits={(0, 0): 5},
        query_coverage_initial_selection=True,
        query_coverage_initial_fraction=0.5,
        query_coverage_initial_prompt_weight=0.5,
        query_coverage_initial_minimum_prompt_mass_fraction=0.95,
        query_coverage_initial_objective="capped_cumulative",
        query_coverage_initial_cap_fraction=0.5,
        query_coverage_cross_head_allocation=True,
        query_coverage_cross_head_stability_gate=True,
        query_coverage_cross_head_stability_risk_mode="output_impact",
        query_coverage_cross_head_minimum_normalized_gain=0.01,
        query_coverage_cross_head_gain_to_prompt_loss_ratio=1.01,
        query_coverage_layout_group_reservation=True,
        query_coverage_layout_group_fraction=1.0,
        query_coverage_layout_group_maximum_blocks=2,
        query_coverage_layout_group_row_bins=1,
        query_coverage_layout_group_atomic_selection=True,
        query_coverage_layout_group_atomic_selection_mode="pareto_joint_donor_knapsack",
        query_coverage_layout_group_minimum_utilization_fraction=1.0,
        query_coverage_layout_group_minimum_normalized_gain=0.01,
        query_coverage_layout_group_gain_to_risk_ratio=1.01,
    )

    assert result.selected_node_ids == (2, 3, 4, 5, 6)
    record = result.capacity_records[0]
    assert record.query_coverage_initial_layout_group_node_ids == ((4, 5),)
    assert record.query_coverage_initial_layout_group_joint_donor_candidate_count == 1
    assert record.query_coverage_initial_layout_group_pareto_candidate_count == 1
    assert record.query_coverage_initial_layout_group_knapsack_selected_count == 1
    assert record.query_coverage_initial_layout_group_output_impact_risk == pytest.approx(0.0)
    assert record.query_coverage_initial_layout_group_normalized_query_gain > 0
    assert record.selected_positions == record.position_limit == 5


def test_short_horizon_logit_selection_emits_two_same_byte_candidates() -> None:
    full = _full(
        (
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.TEXT,
        ),
        mandatory=(6,),
        image_grid_shape=(1, 6),
    )
    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(5, BudgetUnit.BLOCKS),
        prompt_attention_by_node={index: float(6 - index) for index in range(7)},
        prompt_query_attention_by_node={
            0: (0.25, 0.0),
            1: (0.25, 0.0),
            2: (0.25, 0.0),
            3: (0.25, 0.0),
            4: (0.0, 1.0),
            5: (0.0, 1.0),
            6: (0.0, 0.0),
        },
        head_routes=((0, 0),),
        position_limits={(0, 0): 5},
        query_coverage_initial_selection=True,
        query_coverage_initial_fraction=0.5,
        query_coverage_initial_prompt_weight=0.5,
        query_coverage_initial_minimum_prompt_mass_fraction=0.8,
        query_coverage_initial_objective="capped_cumulative",
        query_coverage_initial_cap_fraction=0.5,
        query_coverage_cross_head_allocation=True,
        query_coverage_layout_group_reservation=True,
        query_coverage_layout_group_fraction=1.0,
        query_coverage_layout_group_maximum_blocks=2,
        query_coverage_layout_group_row_bins=1,
        query_coverage_layout_group_atomic_selection=True,
        query_coverage_layout_group_atomic_selection_mode="pareto_joint_donor_knapsack",
        query_coverage_layout_group_minimum_utilization_fraction=1.0,
        query_coverage_layout_group_minimum_normalized_gain=0.01,
        query_coverage_layout_group_gain_to_risk_ratio=1.01,
        short_horizon_logit_stability=True,
    )

    assert tuple(item.candidate_id for item in result.logit_stability_shortlist) == (
        "pareto_joint_group_donor",
        "scalar_prompt_topk_anchor",
    )
    costs = [
        sum(full.blocks[node_id].byte_size for node_id in item.selected_node_ids)
        for item in result.logit_stability_shortlist
    ]
    assert costs[0] == costs[1]
    assert result.logit_stability_shortlist[0].selected_node_ids == result.selected_node_ids


def test_progressive_logit_safe_eviction_defers_pareto_surrogate_gate() -> None:
    """V99 reaches runtime probing without requiring the retired surrogate gate."""

    full = _full(
        (
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.TEXT,
        ),
        mandatory=(6,),
        image_grid_shape=(1, 6),
    )
    common = {
        "prompt_attention_by_node": {index: float(6 - index) for index in range(7)},
        "prompt_query_attention_by_node": {
            0: (0.25, 0.0),
            1: (0.25, 0.0),
            2: (0.25, 0.0),
            3: (0.25, 0.0),
            4: (0.0, 1.0),
            5: (0.0, 1.0),
            6: (0.0, 0.0),
        },
        "head_routes": ((0, 0),),
        "position_limits": {(0, 0): 5},
        "query_coverage_initial_selection": True,
        "query_coverage_initial_fraction": 0.25,
        "query_coverage_initial_prompt_weight": 0.5,
        "query_coverage_initial_minimum_prompt_mass_fraction": 0.95,
        "query_coverage_initial_objective": "capped_cumulative",
        "query_coverage_initial_cap_fraction": 0.5,
        "query_coverage_cross_head_allocation": True,
        "query_coverage_cross_head_stability_gate": False,
        "query_coverage_layout_group_reservation": True,
        "query_coverage_layout_group_fraction": 0.5,
        "query_coverage_layout_group_maximum_blocks": 4,
        "query_coverage_layout_group_row_bins": 4,
        "query_coverage_layout_group_atomic_selection": True,
        "query_coverage_layout_group_atomic_selection_mode": (
            "pareto_joint_donor_knapsack"
        ),
        "query_coverage_layout_group_minimum_utilization_fraction": 0.75,
        "query_coverage_layout_group_minimum_normalized_gain": 0.000001,
        "query_coverage_layout_group_gain_to_risk_ratio": 1.01,
    }
    with pytest.raises(
        ValueError,
        match="Pareto-knapsack atomic groups require non-group stability gating",
    ):
        select_prompt_attention_topk_ids_by_head(
            full,
            SelectionBudget(5, BudgetUnit.BLOCKS),
            **common,
        )

    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(5, BudgetUnit.BLOCKS),
        progressive_logit_safe_eviction=True,
        **common,
    )

    assert result.selected_node_ids
    assert result.logit_stability_shortlist == ()
    assert result.capacity_records[0].position_limit == 5


def test_hierarchical_logit_safe_eviction_defers_pareto_surrogate_gate() -> None:
    """V100 reaches bounded runtime probes without requiring the retired surrogate gate."""

    full = _full(
        (
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.TEXT,
        ),
        mandatory=(6,),
        image_grid_shape=(1, 6),
    )
    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(5, BudgetUnit.BLOCKS),
        prompt_attention_by_node={index: float(6 - index) for index in range(7)},
        prompt_query_attention_by_node={
            0: (0.25, 0.0),
            1: (0.25, 0.0),
            2: (0.25, 0.0),
            3: (0.25, 0.0),
            4: (0.0, 1.0),
            5: (0.0, 1.0),
            6: (0.0, 0.0),
        },
        head_routes=((0, 0),),
        position_limits={(0, 0): 5},
        query_coverage_initial_selection=True,
        query_coverage_initial_fraction=0.25,
        query_coverage_initial_prompt_weight=0.5,
        query_coverage_initial_minimum_prompt_mass_fraction=0.95,
        query_coverage_initial_objective="capped_cumulative",
        query_coverage_initial_cap_fraction=0.5,
        query_coverage_cross_head_allocation=True,
        query_coverage_layout_group_reservation=True,
        query_coverage_layout_group_fraction=0.5,
        query_coverage_layout_group_maximum_blocks=4,
        query_coverage_layout_group_row_bins=4,
        query_coverage_layout_group_atomic_selection=True,
        query_coverage_layout_group_atomic_selection_mode="pareto_joint_donor_knapsack",
        query_coverage_layout_group_minimum_utilization_fraction=0.75,
        query_coverage_layout_group_minimum_normalized_gain=0.000001,
        query_coverage_layout_group_gain_to_risk_ratio=1.01,
        hierarchical_logit_safe_eviction=True,
    )

    assert result.selected_node_ids
    assert result.logit_stability_shortlist == ()
    assert result.capacity_records[0].position_limit == 5


def test_atomic_layout_group_rejects_insufficient_query_gain() -> None:
    full = _full(
        (
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.IMAGE,
            Modality.TEXT,
        ),
        mandatory=(6,),
        image_grid_shape=(1, 6),
    )
    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(5, BudgetUnit.BLOCKS),
        prompt_attention_by_node={index: float(6 - index) for index in range(7)},
        prompt_query_attention_by_node={
            0: (0.1, 0.0),
            1: (0.1, 0.0),
            2: (0.1, 0.0),
            3: (0.1, 0.0),
            4: (0.0, 1.0),
            5: (0.0, 1.0),
            6: (0.0, 0.0),
        },
        head_routes=((0, 0),),
        position_limits={(0, 0): 5},
        query_coverage_initial_selection=True,
        query_coverage_initial_fraction=0.5,
        query_coverage_initial_prompt_weight=0.5,
        query_coverage_initial_minimum_prompt_mass_fraction=0.8,
        query_coverage_initial_objective="capped_cumulative",
        query_coverage_initial_cap_fraction=0.5,
        query_coverage_cross_head_allocation=True,
        query_coverage_layout_group_reservation=True,
        query_coverage_layout_group_fraction=1.0,
        query_coverage_layout_group_maximum_blocks=2,
        query_coverage_layout_group_row_bins=1,
        query_coverage_layout_group_atomic_selection=True,
        query_coverage_layout_group_minimum_utilization_fraction=1.0,
        query_coverage_layout_group_minimum_normalized_gain=0.9,
        query_coverage_layout_group_gain_to_risk_ratio=1.01,
    )

    assert not {4, 5} <= set(result.selected_node_ids)
    record = result.capacity_records[0]
    assert record.query_coverage_initial_layout_group_protected_group_count == 0
    assert record.query_coverage_initial_layout_group_accepted_candidate_count == 0
    assert record.query_coverage_initial_layout_group_risk_rejected_count > 0
    assert record.query_coverage_initial_layout_group_target_bytes > 0


def test_cross_head_stability_gate_falls_back_to_scalar_selection() -> None:
    full = _full(
        (Modality.IMAGE, Modality.IMAGE, Modality.IMAGE, Modality.IMAGE, Modality.TEXT),
        mandatory=(4,),
        kv_heads=2,
    )
    result = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(6, BudgetUnit.BLOCKS),
        prompt_attention_by_node={
            0: 0.6,
            1: 0.5,
            2: 0.4,
            3: 0.3,
            4: 0.0,
            5: 0.6,
            6: 0.5,
            7: 0.4,
            8: 0.3,
            9: 0.0,
        },
        prompt_query_attention_by_node={
            0: (0.5, 0.5),
            1: (0.5, 0.5),
            2: (0.0, 0.0),
            3: (0.0, 0.0),
            4: (0.0, 0.0),
            5: (0.6, 0.0),
            6: (0.4, 0.0),
            7: (0.0, 0.6),
            8: (0.0, 0.4),
            9: (0.0, 0.0),
        },
        head_routes=((0, 0), (0, 1)),
        position_limits={(0, 0): 3, (0, 1): 3},
        query_coverage_initial_selection=True,
        query_coverage_initial_fraction=0.5,
        query_coverage_initial_prompt_weight=0.1,
        query_coverage_initial_minimum_prompt_mass_fraction=0.8,
        query_coverage_initial_objective="capped_cumulative",
        query_coverage_initial_cap_fraction=0.5,
        query_coverage_cross_head_allocation=True,
        query_coverage_cross_head_stability_gate=True,
        query_coverage_cross_head_minimum_normalized_gain=0.1,
        query_coverage_cross_head_gain_to_prompt_loss_ratio=10.0,
    )

    assert result.selected_node_ids == (0, 1, 4, 5, 6, 9)
    by_head = {record.kv_head: record for record in result.capacity_records}
    assert not by_head[0].query_coverage_initial_stability_rejected
    assert by_head[1].query_coverage_initial_stability_rejected
    assert by_head[1].query_coverage_initial_proposed_normalized_gain > 0
    assert by_head[1].query_coverage_initial_proposed_normalized_prompt_loss > 0
    assert by_head[1].query_coverage_initial_changed_count == 0
    assert by_head[1].query_coverage_initial_reserved_bytes == 0
    assert by_head[1].query_coverage_initial_allocated_bytes == 2 * full.blocks[5].byte_size


def test_cross_head_output_impact_risk_rejects_value_deviating_donor() -> None:
    values = np.ones((2, 5, 2), dtype=np.float32)
    values[1, 1] = np.asarray([1.0, 0.0], dtype=np.float32)
    values[1, 2] = np.asarray([-1.0, 0.0], dtype=np.float32)
    full = _full(
        (Modality.IMAGE, Modality.IMAGE, Modality.IMAGE, Modality.IMAGE, Modality.TEXT),
        mandatory=(4,),
        kv_heads=2,
        values=values,
    )
    common = {
        "prompt_attention_by_node": {
            0: 0.6,
            1: 0.5,
            2: 0.4,
            3: 0.3,
            4: 0.0,
            5: 0.6,
            6: 0.5,
            7: 0.4,
            8: 0.3,
            9: 0.0,
        },
        "prompt_query_attention_by_node": {
            0: (0.5, 0.5),
            1: (0.5, 0.5),
            2: (0.0, 0.0),
            3: (0.0, 0.0),
            4: (0.0, 0.0),
            5: (0.6, 0.0),
            6: (0.4, 0.0),
            7: (0.0, 0.6),
            8: (0.0, 0.4),
            9: (0.0, 0.0),
        },
        "head_routes": ((0, 0), (0, 1)),
        "position_limits": {(0, 0): 3, (0, 1): 3},
        "query_coverage_initial_selection": True,
        "query_coverage_initial_fraction": 0.5,
        "query_coverage_initial_prompt_weight": 0.1,
        "query_coverage_initial_minimum_prompt_mass_fraction": 0.8,
        "query_coverage_initial_objective": "capped_cumulative",
        "query_coverage_initial_cap_fraction": 0.5,
        "query_coverage_cross_head_allocation": True,
        "query_coverage_cross_head_stability_gate": True,
        "query_coverage_cross_head_minimum_normalized_gain": 0.01,
        "query_coverage_cross_head_gain_to_prompt_loss_ratio": 2.0,
    }

    prompt_mass = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(6, BudgetUnit.BLOCKS),
        query_coverage_cross_head_stability_risk_mode="prompt_mass",
        **common,  # type: ignore[arg-type]
    )
    output_impact = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(6, BudgetUnit.BLOCKS),
        query_coverage_cross_head_stability_risk_mode="output_impact",
        **common,  # type: ignore[arg-type]
    )

    assert prompt_mass.selected_node_ids == (0, 1, 4, 5, 7, 9)
    assert output_impact.selected_node_ids == (0, 1, 4, 5, 6, 9)
    record = {item.kv_head: item for item in output_impact.capacity_records}[1]
    assert record.query_coverage_initial_stability_rejected
    assert record.query_coverage_initial_proposed_output_impact_risk == pytest.approx(5 / 18)
    assert record.query_coverage_initial_output_impact_donor_count == 1
    assert record.query_coverage_initial_proposed_normalized_gain == pytest.approx(0.5)


def test_cross_head_output_impact_reservation_protects_before_query_coverage() -> None:
    values = np.ones((2, 5, 2), dtype=np.float32)
    values[1, 0] = np.asarray([1.0, 1.0], dtype=np.float32)
    values[1, 1] = np.asarray([-1.0, 0.0], dtype=np.float32)
    full = _full(
        (Modality.IMAGE, Modality.IMAGE, Modality.IMAGE, Modality.IMAGE, Modality.TEXT),
        mandatory=(4,),
        kv_heads=2,
        values=values,
    )
    common = {
        "prompt_attention_by_node": {
            0: 0.6,
            1: 0.5,
            2: 0.4,
            3: 0.3,
            4: 0.0,
            5: 0.5,
            6: 0.6,
            7: 0.4,
            8: 0.3,
            9: 0.0,
        },
        "prompt_query_attention_by_node": {
            0: (0.5, 0.5),
            1: (0.5, 0.5),
            2: (0.0, 0.0),
            3: (0.0, 0.0),
            4: (0.0, 0.0),
            5: (0.6, 0.0),
            6: (0.4, 0.0),
            7: (0.0, 0.6),
            8: (0.0, 0.4),
            9: (0.0, 0.0),
        },
        "head_routes": ((0, 0), (0, 1)),
        "position_limits": {(0, 0): 3, (0, 1): 3},
        "query_coverage_initial_selection": True,
        "query_coverage_initial_fraction": 0.5,
        "query_coverage_initial_prompt_weight": 0.1,
        "query_coverage_initial_minimum_prompt_mass_fraction": 0.8,
        "query_coverage_initial_objective": "capped_cumulative",
        "query_coverage_initial_cap_fraction": 0.5,
        "query_coverage_cross_head_allocation": True,
    }

    unprotected = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(6, BudgetUnit.BLOCKS),
        **common,  # type: ignore[arg-type]
    )
    protected = select_prompt_attention_topk_ids_by_head(
        full,
        SelectionBudget(6, BudgetUnit.BLOCKS),
        query_coverage_output_impact_reservation=True,
        query_coverage_output_impact_fraction=0.5,
        **common,  # type: ignore[arg-type]
    )

    assert unprotected.selected_node_ids == (0, 1, 4, 5, 7, 9)
    assert protected.selected_node_ids == (0, 1, 4, 6, 7, 9)
    record = {item.kv_head: item for item in protected.capacity_records}[1]
    assert record.query_coverage_initial_output_impact_protected_count == 1
    assert record.query_coverage_initial_output_impact_protected_bytes == full.blocks[6].byte_size
    assert record.query_coverage_initial_output_impact_protected_score > 0
    assert record.query_coverage_initial_added_node_ids == (7,)
    assert record.selected_positions == record.position_limit


def test_cross_head_query_allocation_requires_capped_cumulative_objective() -> None:
    full = _full((Modality.IMAGE, Modality.IMAGE, Modality.TEXT), mandatory=(2,))
    with pytest.raises(ValueError, match="requires capped cumulative"):
        select_prompt_attention_topk_ids_by_head(
            full,
            SelectionBudget(2, BudgetUnit.BLOCKS),
            prompt_attention_by_node={0: 0.6, 1: 0.4, 2: 0.0},
            prompt_query_attention_by_node={0: (0.6,), 1: (0.4,), 2: (0.0,)},
            head_routes=((0, 0),),
            position_limits={(0, 0): 2},
            query_coverage_initial_selection=True,
            query_coverage_initial_fraction=0.5,
            query_coverage_cross_head_allocation=True,
        )


def test_max_query_coverage_rejects_nonidentity_cap() -> None:
    full = _full((Modality.IMAGE, Modality.IMAGE, Modality.TEXT), mandatory=(2,))
    with pytest.raises(ValueError, match=r"cap fraction 1\.0"):
        select_prompt_attention_topk_ids_by_head(
            full,
            SelectionBudget(2, BudgetUnit.BLOCKS),
            prompt_attention_by_node={0: 0.6, 1: 0.4, 2: 0.0},
            prompt_query_attention_by_node={
                0: (0.6,),
                1: (0.4,),
                2: (0.0,),
            },
            head_routes=((0, 0),),
            position_limits={(0, 0): 2},
            query_coverage_initial_selection=True,
            query_coverage_initial_fraction=1.0,
            query_coverage_initial_objective="max",
            query_coverage_initial_cap_fraction=0.5,
        )


def test_required_prototype_anchor_budget_fails_closed() -> None:
    full = _full((Modality.TEXT,) * 4, mandatory=(3,))
    _graph, _utilities, objective = _utilities_and_objective(full)

    with pytest.raises(InfeasibleSelectionBudgetError, match="prototype anchors"):
        lazy_greedy_select(
            objective,
            SelectionBudget(2, BudgetUnit.BLOCKS),
            required_node_ids=(0, 1),
        )


def test_retention_one_selects_all_exact_in_linear_mandatory_first_path() -> None:
    full = _full((Modality.TEXT,) * 5, mandatory=(2,))
    graph, _utilities, objective = _utilities_and_objective(full)

    result = select_all_exact(
        objective,
        SelectionBudget(len(graph.nodes), BudgetUnit.BLOCKS),
    )

    mandatory_id = next(node.node_id for node in graph.nodes if node.block.mandatory)
    assert result.selected_node_ids == tuple(range(len(graph.nodes)))
    assert result.selection_order[0] == mandatory_id
    assert result.budget_spent == len(graph.nodes)
    assert result.heap_pops == 0
    assert result.marginal_recomputations == 0
    assert result.decisions[mandatory_id].reason is SelectionReason.MANDATORY_EXACT
    assert all(
        decision.reason is SelectionReason.RETENTION_ONE_EXACT
        for decision in result.decisions
        if decision.node_id != mandatory_id
    )


def test_incremental_marginals_match_exact_objective_differences() -> None:
    full = _full((Modality.TEXT, Modality.TEXT, Modality.IMAGE, Modality.VIDEO))
    _graph, _utilities, objective = _utilities_and_objective(
        full,
        probabilities=(0.4, 0.3, 0.2, 0.1),
        graph_config=_isolated_graph_config(semantic_weight=1.0, max_neighbors=2),
    )
    incremental = objective.incremental_state()
    selected: set[int] = set()

    for candidate in (2, 0, 3):
        remaining = tuple(node_id for node_id in range(4) if node_id not in selected)
        assert incremental.marginal_gains(remaining) == pytest.approx(
            tuple(objective.marginal_gain(selected, node_id) for node_id in remaining),
            abs=1e-12,
        )
        assert incremental.marginal_gain(candidate) == pytest.approx(
            objective.marginal_gain(selected, candidate), abs=1e-12
        )
        incremental.select(candidate)
        selected.add(candidate)


def test_atomic_reservation_matches_sequential_final_objective_state() -> None:
    full = _full((Modality.TEXT, Modality.TEXT, Modality.IMAGE, Modality.VIDEO))
    _graph, _utilities, objective = _utilities_and_objective(
        full,
        probabilities=(0.4, 0.3, 0.2, 0.1),
        graph_config=_isolated_graph_config(semantic_weight=1.0, max_neighbors=2),
    )
    sequential = objective.incremental_state()
    atomic = objective.incremental_state()

    snapshot_gains = tuple(objective.marginal_gain((), node_id) for node_id in (2, 0))
    assert atomic.reserve_many((2, 0)) == pytest.approx(snapshot_gains, abs=1e-12)
    sequential.select(2)
    sequential.select(0)

    assert atomic.marginal_gain(3) == pytest.approx(
        sequential.marginal_gain(3),
        abs=1e-12,
    )


def test_atomic_reservation_rejects_duplicates() -> None:
    full = _full((Modality.TEXT, Modality.IMAGE))
    _graph, _utilities, objective = _utilities_and_objective(full)

    with pytest.raises(ValueError, match="must be unique"):
        objective.incremental_state().reserve_many((0, 0))


def test_deterministic_tie_breaking_prefers_lower_node_id() -> None:
    full = _full((Modality.TEXT,) * 4)
    _graph, _utilities, objective = _utilities_and_objective(
        full,
        selection_config=SelectionConfig(lambda_g=0.0, lambda_m=0.0),
    )

    result = lazy_greedy_select(objective, SelectionBudget(1, BudgetUnit.BLOCKS))

    assert result.selected_node_ids == (0,)
    assert result.selection_order == (0,)


def test_vectorized_coverage_selection_is_deterministic_and_budget_safe() -> None:
    full = _full(
        (Modality.TEXT, Modality.TEXT, Modality.IMAGE, Modality.IMAGE, Modality.VIDEO),
        mandatory=(0,),
    )
    _graph, _utilities, objective = _utilities_and_objective(
        full,
        probabilities=(0.35, 0.25, 0.2, 0.15, 0.05),
        graph_config=_isolated_graph_config(semantic_weight=1.0, max_neighbors=2),
        selection_config=SelectionConfig(
            algorithm="vectorized_coverage",
            coverage_correction_rounds=2,
            lambda_g=-0.5,
            lambda_m=-0.25,
        ),
    )
    budget = SelectionBudget(3, BudgetUnit.BLOCKS)

    first = lazy_greedy_select(objective, budget)
    second = lazy_greedy_select(objective, budget)

    assert first.selected_node_ids == second.selected_node_ids
    assert first.selection_order == second.selection_order
    assert first.budget_spent <= budget.value
    assert 0 in first.selected_node_ids
    assert first.decisions[0].reason is SelectionReason.MANDATORY_EXACT
    assert all(
        decision.reason
        in {
            SelectionReason.MANDATORY_EXACT,
            SelectionReason.VECTORIZED_COVERAGE,
            SelectionReason.NOT_SELECTED_LOWER_GAIN,
            SelectionReason.BUDGET_EXHAUSTED,
        }
        for decision in first.decisions
    )


def test_routed_selector_never_optimizes_stable_non_anchor_nodes() -> None:
    full = _full(
        (Modality.TEXT, Modality.TEXT, Modality.TEXT, Modality.IMAGE, Modality.IMAGE),
        mandatory=(0,),
    )
    _graph, _utilities, objective = _utilities_and_objective(
        full,
        probabilities=(0.01, 0.01, 0.95, 0.02, 0.01),
        selection_config=SelectionConfig(
            algorithm="vectorized_coverage",
            coverage_correction_rounds=0,
            minimum_stratum_fraction=0.0,
            lambda_g=-0.5,
            lambda_m=-0.25,
        ),
    )

    result = lazy_greedy_select(
        objective,
        SelectionBudget(3, BudgetUnit.BLOCKS),
        prompt_anchor_node_ids=(1,),
        optimization_node_ids=(3, 4),
    )

    assert result.selected_node_ids == (0, 1, 3)
    assert not result.decisions[2].selected
    assert result.decisions[1].reason is SelectionReason.PROMPT_ANCHOR
    assert result.decisions[3].reason is SelectionReason.VECTORIZED_COVERAGE
    assert len(result.decisions) == len(full.blocks)


def test_byte_budget_never_exceeds_hard_limit_for_variable_size_blocks() -> None:
    full = _full(
        (Modality.TEXT, Modality.TEXT, Modality.IMAGE, Modality.VIDEO),
        block_size=2,
    )
    graph, _utilities, objective = _utilities_and_objective(full)
    sizes = sorted(node.block.byte_size for node in graph.nodes)
    byte_limit = sizes[0] + sizes[1]

    result = lazy_greedy_select(
        objective,
        SelectionBudget(byte_limit, BudgetUnit.BYTES),
    )

    assert len(set(node.block.byte_size for node in graph.nodes)) > 1
    assert result.active_bytes == result.budget_spent
    assert result.active_bytes <= byte_limit
    assert all(
        decision.cost == graph.nodes[decision.node_id].block.byte_size
        for decision in result.decisions
    )


def test_dense_cache_position_limit_is_enforced_and_recorded() -> None:
    full = _full((Modality.TEXT,) * 6, mandatory=(5,))
    _graph, _utilities, objective = _utilities_and_objective(
        full,
        selection_config=SelectionConfig(
            lambda_g=-0.5,
            lambda_m=-0.25,
            stop_on_nonpositive_gain=False,
        ),
    )

    result = lazy_greedy_select(
        objective,
        SelectionBudget(6, BudgetUnit.BLOCKS),
        position_limits={(0, 0): 2},
    )

    assert len(result.selected_node_ids) == 2
    assert 5 in result.selected_node_ids
    assert any(
        decision.reason is SelectionReason.DENSE_CACHE_CAPACITY for decision in result.decisions
    )


def test_dense_selector_reserves_each_layer_head_stratum_before_global_greedy() -> None:
    sequence = 8
    key = np.arange(2 * sequence * 2, dtype=np.float32).reshape(1, 2, sequence, 2)
    full = FullKVState.from_tensors(
        ((key, key.copy()),),
        modality_spans=(ModalitySpan(0, sequence, Modality.TEXT),),
        block_size=1,
        mandatory_logical_positions=(sequence - 1,),
    )
    graph = build_evidence_graph(full, _isolated_graph_config())
    probabilities = {
        node.node_id: (100.0 if node.block.kv_head == 0 else 1.0) for node in graph.nodes
    }
    utilities = compute_block_utilities(
        graph,
        UtilityConfig(lambda_v=0.0, lambda_o=0.0),
        forecast_attention_by_node=probabilities,
        attention_provenance="synthetic_rope_aware_fixture",
        rope_aware=True,
    )
    objective = BudgetedObjective(
        graph,
        utilities,
        SelectionConfig(
            lambda_g=0.0,
            lambda_m=0.0,
            stop_on_nonpositive_gain=False,
            minimum_stratum_fraction=0.5,
        ),
    )

    result = lazy_greedy_select(
        objective,
        SelectionBudget(8, BudgetUnit.BLOCKS),
        position_limits={(0, 0): 4, (0, 1): 4},
    )

    selected_per_head = {
        head: sum(
            graph.nodes[node_id].block.position_count
            for node_id in result.selected_node_ids
            if graph.nodes[node_id].block.kv_head == head
        )
        for head in (0, 1)
    }
    assert selected_per_head == {0: 4, 1: 4}
    assert any(decision.reason is SelectionReason.STRATUM_RESERVE for decision in result.decisions)


def test_joint_reservation_transaction_relaxes_optional_blocks_for_all_strata() -> None:
    sequence = 64
    key = np.arange(3 * sequence * 2, dtype=np.float32).reshape(1, 3, sequence, 2)
    full = FullKVState.from_tensors(
        ((key, key.copy()),),
        modality_spans=(ModalitySpan(0, sequence, Modality.TEXT),),
        block_size=4,
        mandatory_logical_positions=(sequence - 1,),
    )
    graph = build_evidence_graph(full, _isolated_graph_config())
    utilities = compute_block_utilities(
        graph,
        UtilityConfig(lambda_v=0.0, lambda_o=0.0),
        forecast_attention_by_node={node.node_id: 1.0 for node in graph.nodes},
        attention_provenance="synthetic_joint_reservation_fixture",
        rope_aware=True,
    )
    objective = BudgetedObjective(
        graph,
        utilities,
        SelectionConfig(
            lambda_g=0.0,
            lambda_m=0.0,
            stop_on_nonpositive_gain=False,
            minimum_stratum_fraction=0.15,
        ),
    )
    by_head = {
        head: tuple(
            node.node_id for node in graph.nodes if node.block.kv_head == head
        )
        for head in range(3)
    }
    required = (by_head[0][0],)
    prompt = (*by_head[0][1:10], *by_head[1][1:6])
    evidence = (*by_head[1][6:10], *by_head[2][1:10])

    first = lazy_greedy_select(
        objective,
        SelectionBudget(40, BudgetUnit.RETAINED_SLOTS),
        position_limits={(0, 0): 61, (0, 1): 55, (0, 2): 54},
        required_node_ids=required,
        prompt_anchor_node_ids=prompt,
        evidence_node_ids=evidence,
        # Reproduce routed planning where the ordinary optimization shortlist
        # is too small to satisfy every head on its own.
        optimization_node_ids=(by_head[0][1], by_head[1][1], by_head[2][1]),
    )
    second = lazy_greedy_select(
        objective,
        SelectionBudget(40, BudgetUnit.RETAINED_SLOTS),
        position_limits={(0, 0): 61, (0, 1): 55, (0, 2): 54},
        required_node_ids=required,
        prompt_anchor_node_ids=prompt,
        evidence_node_ids=evidence,
        optimization_node_ids=(by_head[0][1], by_head[1][1], by_head[2][1]),
    )

    selected_positions = {
        head: sum(
            graph.nodes[node_id].block.position_count
            for node_id in first.selected_node_ids
            if graph.nodes[node_id].block.kv_head == head
        )
        for head in range(3)
    }
    assert selected_positions[0] >= math.floor(61 * 0.15)
    assert selected_positions[1] >= math.floor(55 * 0.15)
    assert selected_positions[2] >= math.floor(54 * 0.15)
    assert first.budget_spent <= 40
    assert first.selected_node_ids == second.selected_node_ids
    assert set(required).issubset(first.selected_node_ids)
    assert not set((*prompt, *evidence)).issubset(first.selected_node_ids)


def test_joint_reservation_transaction_fills_observed_54_to_61_position_lattice() -> None:
    sequence = 64
    key = np.arange(3 * sequence * 2, dtype=np.float32).reshape(1, 3, sequence, 2)
    full = FullKVState.from_tensors(
        ((key, key.copy()),),
        modality_spans=(ModalitySpan(0, sequence, Modality.TEXT),),
        block_size=4,
        mandatory_logical_positions=(sequence - 1,),
    )
    _graph, _utilities, objective = _utilities_and_objective(
        full,
        selection_config=SelectionConfig(
            lambda_g=0.0,
            lambda_m=0.0,
            stop_on_nonpositive_gain=False,
            minimum_stratum_fraction=0.15,
        ),
    )

    result = lazy_greedy_select(
        objective,
        SelectionBudget(170, BudgetUnit.RETAINED_SLOTS),
        position_limits={(0, 0): 61, (0, 1): 55, (0, 2): 54},
        fill_position_limits=True,
        position_fill_quantum=4,
    )

    positions = {
        head: sum(
            block.position_count for block in result.selected_blocks if block.kv_head == head
        )
        for head in range(3)
    }
    assert positions == {0: 60, 1: 52, 2: 52}
    assert result.budget_spent == 164
    assert result.budget_spent <= result.budget.value


@pytest.mark.parametrize(
    ("limit", "boundary_split", "mandatory_position", "minimum_positions"),
    (
        (56, False, 63, 8),
        (51, True, 0, 7),
        (53, True, 63, 9),
    ),
)
def test_dense_feasibility_fill_uses_graph_beyond_routed_shortlist(
    limit: int,
    boundary_split: bool,
    mandatory_position: int,
    minimum_positions: int,
) -> None:
    sequence = 64
    key = np.arange(sequence * 2, dtype=np.float32).reshape(1, 1, sequence, 2)
    spans = (
        (
            ModalitySpan(0, 3, Modality.TEXT),
            ModalitySpan(3, sequence, Modality.IMAGE),
        )
        if boundary_split
        else (ModalitySpan(0, sequence, Modality.TEXT),)
    )
    full = FullKVState.from_tensors(
        ((key, key.copy()),),
        modality_spans=spans,
        block_size=4,
        mandatory_logical_positions=(mandatory_position,),
    )
    graph, _utilities, objective = _utilities_and_objective(
        full,
        selection_config=SelectionConfig(
            lambda_g=0.0,
            lambda_m=0.0,
            stop_on_nonpositive_gain=False,
            minimum_stratum_fraction=0.15,
        ),
    )
    mandatory_ids = {
        node.node_id for node in graph.nodes if node.block.mandatory
    }
    aligned_optional = tuple(
        node.node_id
        for node in graph.nodes
        if not node.block.mandatory and node.block.position_count == 4
    )
    shortlist_count = 2 if minimum_positions == 9 else 1
    optimization_shortlist = aligned_optional[:shortlist_count]

    result = lazy_greedy_select(
        objective,
        SelectionBudget(limit, BudgetUnit.RETAINED_SLOTS),
        position_limits={(0, 0): limit},
        fill_position_limits=True,
        position_fill_quantum=4,
        optimization_node_ids=optimization_shortlist,
    )

    reserved_positions = sum(
        graph.nodes[decision.node_id].block.position_count
        for decision in result.decisions
        if decision.reason
        in {SelectionReason.MANDATORY_EXACT, SelectionReason.STRATUM_RESERVE}
    )
    assert reserved_positions == minimum_positions
    assert result.budget_spent == limit
    assert sum(block.position_count for block in result.selected_blocks) == limit
    assert any(
        decision.selected
        and decision.node_id not in mandatory_ids
        and decision.node_id not in optimization_shortlist
        and decision.reason is SelectionReason.DENSE_CACHE_FILL
        for decision in result.decisions
    )
    assert all(
        decision.node_id in optimization_shortlist
        for decision in result.decisions
        if decision.reason
        in {SelectionReason.LAZY_GREEDY, SelectionReason.VECTORIZED_COVERAGE}
    )


def test_dense_fill_removes_padding_even_after_nonpositive_stop() -> None:
    sequence = 6
    key = np.arange(2 * sequence * 2, dtype=np.float32).reshape(1, 2, sequence, 2)
    full = FullKVState.from_tensors(
        ((key, key.copy()),),
        modality_spans=(ModalitySpan(0, sequence, Modality.TEXT),),
        block_size=1,
        mandatory_logical_positions=(sequence - 1,),
    )
    graph, utilities, _objective = _utilities_and_objective(full)
    zero_utilities = replace(
        utilities,
        blocks=tuple(replace(item, local_utility=0.0) for item in utilities.blocks),
        weights=UtilityConfig(lambda_q=0.0, lambda_v=0.0, lambda_o=0.0),
    )
    objective = BudgetedObjective(
        graph,
        zero_utilities,
        SelectionConfig(lambda_g=0.0, lambda_m=0.0, stop_on_nonpositive_gain=True),
    )

    result = lazy_greedy_select(
        objective,
        SelectionBudget(6, BudgetUnit.BLOCKS),
        position_limits={(0, 0): 3, (0, 1): 3},
        fill_position_limits=True,
    )

    assert result.budget_spent == 6
    assert (
        sum(decision.reason is SelectionReason.DENSE_CACHE_FILL for decision in result.decisions)
        == 4
    )


def test_dense_fill_completes_partial_prototype_anchor_on_block_lattice() -> None:
    key = np.arange(10 * 2, dtype=np.float32).reshape(1, 1, 10, 2)
    full = FullKVState.from_tensors(
        ((key, key.copy()),),
        modality_spans=(
            ModalitySpan(0, 2, Modality.TEXT),
            ModalitySpan(2, 5, Modality.IMAGE),
            ModalitySpan(5, 9, Modality.VIDEO),
            ModalitySpan(9, 10, Modality.TEXT),
        ),
        block_size=4,
        mandatory_logical_positions=(9,),
    )
    _graph, _utilities, objective = _utilities_and_objective(full)

    result = lazy_greedy_select(
        objective,
        SelectionBudget(10, BudgetUnit.RETAINED_SLOTS),
        position_limits={(0, 0): 10},
        fill_position_limits=True,
        position_fill_quantum=4,
        required_node_ids=(0,),
    )

    assert result.selected_node_ids == (0, 1, 2, 3)
    assert result.decisions[0].reason is SelectionReason.PROTOTYPE_ANCHOR
    assert result.decisions[1].reason is SelectionReason.DENSE_CACHE_FILL
    assert sum(block.position_count for block in result.selected_blocks) == 10


def test_dense_fill_completes_partial_prompt_anchor_on_block_lattice() -> None:
    key = np.arange(10 * 2, dtype=np.float32).reshape(1, 1, 10, 2)
    full = FullKVState.from_tensors(
        ((key, key.copy()),),
        modality_spans=(
            ModalitySpan(0, 2, Modality.TEXT),
            ModalitySpan(2, 5, Modality.IMAGE),
            ModalitySpan(5, 9, Modality.VIDEO),
            ModalitySpan(9, 10, Modality.TEXT),
        ),
        block_size=4,
        mandatory_logical_positions=(9,),
    )
    _graph, _utilities, objective = _utilities_and_objective(full)

    result = lazy_greedy_select(
        objective,
        SelectionBudget(10, BudgetUnit.RETAINED_SLOTS),
        position_limits={(0, 0): 10},
        fill_position_limits=True,
        position_fill_quantum=4,
        prompt_anchor_node_ids=(0,),
    )

    assert result.selected_node_ids == (0, 1, 2, 3)
    assert result.decisions[0].reason is SelectionReason.PROMPT_ANCHOR
    assert result.decisions[1].reason is SelectionReason.DENSE_CACHE_FILL
    assert sum(block.position_count for block in result.selected_blocks) == 10


def test_dense_fill_lowers_unattainable_44_position_limit_to_32() -> None:
    key = np.arange(48 * 2, dtype=np.float32).reshape(1, 1, 48, 2)
    full = FullKVState.from_tensors(
        ((key, key.copy()),),
        block_size=16,
    )
    _graph, _utilities, objective = _utilities_and_objective(full)

    result = lazy_greedy_select(
        objective,
        SelectionBudget(44, BudgetUnit.RETAINED_SLOTS),
        position_limits={(0, 0): 44},
        fill_position_limits=True,
        position_fill_quantum=16,
        prompt_anchor_node_ids=(0,),
    )

    assert sum(block.position_count for block in result.selected_blocks) == 32
    assert result.budget_spent == 32
    assert result.remaining_budget == 12


def test_mandatory_blocks_that_exceed_budget_fail_closed() -> None:
    full = _full((Modality.TEXT,) * 3, mandatory=(0, 1))
    _graph, _utilities, objective = _utilities_and_objective(full)

    with pytest.raises(InfeasibleSelectionBudgetError, match="mandatory exact blocks"):
        lazy_greedy_select(objective, SelectionBudget(1, BudgetUnit.BLOCKS))


def test_greedy_is_compared_with_exhaustive_tiny_graph_optimum() -> None:
    full = _full((Modality.TEXT, Modality.TEXT, Modality.IMAGE, Modality.VIDEO))
    _graph, _utilities, objective = _utilities_and_objective(
        full,
        probabilities=(0.4, 0.3, 0.2, 0.1),
    )
    budget = SelectionBudget(2, BudgetUnit.BLOCKS)

    greedy = lazy_greedy_select(objective, budget)
    exact = exhaustive_select(objective, budget)
    comparison = compare_greedy_to_exhaustive(greedy, objective)

    assert exact.objective.total >= greedy.objective.total - 1e-12
    assert comparison.optimal_objective == exact.objective.total
    assert comparison.absolute_gap == pytest.approx(exact.objective.total - greedy.objective.total)
    assert comparison.evaluated_subsets > 1


def test_objective_is_monotone_and_has_diminishing_returns_when_assumptions_hold() -> None:
    full = _full((Modality.TEXT, Modality.IMAGE, Modality.VIDEO))
    _graph, _utilities, objective = _utilities_and_objective(
        full,
        probabilities=(1.0, 1.0, 1.0),
        graph_config=_isolated_graph_config(semantic_weight=1.0, max_neighbors=2),
        selection_config=SelectionConfig(lambda_g=-0.5, lambda_m=-0.5),
    )

    report = audit_objective_properties(objective)

    assert objective.is_submodular
    assert report.monotone
    assert report.diminishing_returns
    assert objective.marginal_gain((), 0) >= objective.marginal_gain((1,), 0)
    selected = objective.evaluate((0, 2))
    assert selected.total == pytest.approx(
        selected.local_utility_sum
        - objective.config.lambda_g * selected.facility_location_coverage
        - objective.config.lambda_m * selected.modality_coverage
    )


def test_lazy_selector_rejects_coverage_signs_without_submodular_guarantee() -> None:
    full = _full((Modality.TEXT, Modality.IMAGE, Modality.VIDEO))
    _graph, _utilities, objective = _utilities_and_objective(
        full,
        selection_config=SelectionConfig(lambda_g=0.5, lambda_m=0.5),
    )

    assert not objective.is_submodular
    with pytest.raises(ValueError, match="submodular objective"):
        lazy_greedy_select(objective, SelectionBudget(2, BudgetUnit.BLOCKS))


def test_selector_records_nonpositive_and_budget_rejection_reasons() -> None:
    full = _full((Modality.TEXT,) * 3)
    graph, utilities, _objective = _utilities_and_objective(full)
    zero_utilities = replace(
        utilities,
        blocks=tuple(replace(item, local_utility=0.0) for item in utilities.blocks),
        weights=UtilityConfig(lambda_q=0.0, lambda_v=0.0, lambda_o=0.0),
    )
    objective = BudgetedObjective(
        graph,
        zero_utilities,
        SelectionConfig(lambda_g=0.0, lambda_m=0.0),
    )

    result = lazy_greedy_select(objective, SelectionBudget(1, BudgetUnit.BLOCKS))

    assert not result.selected_node_ids
    assert all(
        decision.reason is SelectionReason.NONPOSITIVE_MARGINAL_GAIN
        for decision in result.decisions
    )


def test_selection_configuration_validation_is_strict() -> None:
    with pytest.raises(ConfigurationError, match=r"utility\.lambda_q"):
        UtilityConfig(lambda_q=-1.0)
    with pytest.raises(ConfigurationError, match=r"selection\.lambda_g"):
        SelectionConfig(lambda_g=float("inf"))
    with pytest.raises(ConfigurationError, match="minimum_stratum_fraction"):
        SelectionConfig(minimum_stratum_fraction=1.1)
    with pytest.raises(ValueError, match="budget value"):
        SelectionBudget(0)
