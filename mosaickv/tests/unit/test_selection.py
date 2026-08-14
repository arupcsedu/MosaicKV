from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mosaickv.baselines import (
    select_exact_baseline,
    select_prompt_attention_topk_ids,
    select_prompt_attention_topk_ids_by_head,
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
    value = key.copy() if values is None else values.reshape(1, 1, sequence_length, -1)
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
        assert block.local_utility == pytest.approx(
            block.forecast_attention_probability
        )
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

    assert select_prompt_attention_topk_ids(
        full,
        budget,
        prompt_attention_by_node=scores,
        position_limits=limits,
    ) == reference.selected_node_ids


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

    assert local.selected_node_ids == tuple(
        node_id for node_id in canonical if full.blocks[node_id].kv_head == 1
    )
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
