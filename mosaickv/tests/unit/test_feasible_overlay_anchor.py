from types import SimpleNamespace

import pytest
import torch

from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import GraphConfig, PrototypeConfig
from mosaickv.evaluation.feasible_overlay_anchor import choose
from mosaickv.graph import build_evidence_graph
from mosaickv.prototypes.construction import plan_prototype_assignments


def fixture(n=9):
    g = torch.Generator().manual_seed(17)
    pair = tuple(torch.randn(1, 1, n, 128, generator=g).bfloat16() for _ in range(2))
    state = FullKVState.from_tensors(
        (pair,),
        block_size=1,
        modality_spans=(ModalitySpan(0, n, Modality.IMAGE),),
        mandatory_logical_positions=(0,),
    )
    graph = build_evidence_graph(state, GraphConfig(max_neighbors=2))
    cfg = PrototypeConfig(
        group_size=2,
        max_position_span=1024,
        maximum_prototypes_per_head=1,
        budget_planning_mode="one_pass_bounded_anchor_cover",
        source_modalities=("image",),
    )
    return graph, cfg


def test_highest_score_incomplete_group_is_replaced_by_complete_feasible_group():
    graph, cfg = fixture(6)
    selected = SimpleNamespace(selected_node_ids=(0, 1, 3))
    winner, plan, proof = choose(graph, selected, cfg, (0, 1, 3), [10, 9, 0, 1, 0, 0])
    assert winner == 3
    assert [node for node, _, _ in plan.assignment_rows] == [4, 5]
    assert proof["initial_member_count"] == 1
    assert proof["original_best_anchor_preserved"] is False
    assert selected.selected_node_ids == (0, 1, 3)


def test_every_previous_valid_best_anchor_and_its_assignments_are_preserved():
    graph, cfg = fixture()
    selected = SimpleNamespace(selected_node_ids=(0, 3, 6))
    scores = [1, 0, 0, 10, 0, 0, 2, 0, 0]
    expected = plan_prototype_assignments(graph, selected, cfg, prototype_anchor_node_ids=(3,))
    winner, observed, proof = choose(graph, selected, cfg, (0, 3, 6), scores)
    assert winner == 3
    assert observed.assignment_rows == expected.assignment_rows
    assert proof["assignment_plans_computed"] == 1
    assert proof["original_best_anchor_preserved"] is True


def test_tie_breaking_and_unavailable_groups_fail_closed():
    graph, cfg = fixture()
    selected = SimpleNamespace(selected_node_ids=(0, 1, 3, 6))
    winner, _, _ = choose(graph, selected, cfg, (0, 1, 3, 6), [20, 1, 0, 10, 0, 0, 10, 0, 0])
    assert winner == 3
    selected = SimpleNamespace(selected_node_ids=(0, 1, 3, 4, 6, 7))
    with pytest.raises(RuntimeError, match="no selected image anchor"):
        choose(graph, selected, cfg, selected.selected_node_ids, [1] * 9)
    with pytest.raises(ValueError, match="already be selected"):
        choose(graph, selected, cfg, (2,), [1] * 9)
