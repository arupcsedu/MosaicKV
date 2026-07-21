from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

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
from mosaickv.types import BudgetUnit


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
        assert incremental.marginal_gain(candidate) == pytest.approx(
            objective.marginal_gain(selected, candidate), abs=1e-12
        )
        incremental.select(candidate)
        selected.add(candidate)


def test_deterministic_tie_breaking_prefers_lower_node_id() -> None:
    full = _full((Modality.TEXT,) * 4)
    _graph, _utilities, objective = _utilities_and_objective(
        full,
        selection_config=SelectionConfig(lambda_g=0.0, lambda_m=0.0),
    )

    result = lazy_greedy_select(objective, SelectionBudget(1, BudgetUnit.BLOCKS))

    assert result.selected_node_ids == (0,)
    assert result.selection_order == (0,)


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
    with pytest.raises(ValueError, match="budget value"):
        SelectionBudget(0)
