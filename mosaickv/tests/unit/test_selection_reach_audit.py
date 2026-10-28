"""Selection activity must not be mistaken for quality or fresh evidence."""

import inspect

import evicastkv_selection_reach as audit
import pytest


def test_one_exchange_is_not_counted_twice():
    result = audit.selection_delta([1, 2, 3, 4], [1, 2, 3, 5])
    assert result["replacements"] == 1
    assert result["replacement_fraction"] == 0.25
    assert result["jaccard"] == 3 / 5
    assert result["changed"] is True


def test_reordered_identical_subset_is_not_a_change():
    result = audit.selection_delta([1, 2], [2, 1])
    assert result["replacements"] == 0
    assert result["jaccard"] == 1
    assert result["changed"] is False


@pytest.mark.parametrize(
    "left,right", [([], []), ([1], [1, 2]), ([1, 1], [1, 2]), ([1, 2], [1, 1])]
)
def test_invalid_capacity_and_duplicates_rejected(left, right):
    with pytest.raises(ValueError):
        audit.selection_delta(left, right)


def test_no_benchmark_label_or_model_access_in_audit():
    source = inspect.getsource(audit.build)
    for field in ('["answer"]', '["references"]', '["score"]', "sbatch", "load_hf_adapter"):
        assert field not in source
    assert "algorithm_changed=False" in source
    assert "fresh_confirmation=False" in source


def test_zero_edges_does_not_invent_cross_modal_activity():
    result = audit.summarize(
        [
            dict(
                workload="synthetic",
                variant="control",
                budget_bytes=100,
                changed=False,
                replacement_fraction=0,
                replacements=0,
                graph_nodes=0,
                graph_edges=0,
                routed_heads=0,
                edge_sources={},
                cross_modal_edges_removed=None,
            )
        ]
    )[0]
    assert result["changed_inputs"] == 0
    assert result["cross_modal_edges_removed"] is None
    assert result["edge_sources"] == {}
