import numpy as np
import pytest

from mosaickv.evaluation.hybrid_shortlist import shortlist
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy


def test_shortlist_preserves_all_mandatory_nodes_and_deterministic_ties():
    attention = np.ones(603, dtype=np.float32)
    required = (0, 99, 602)
    ids = shortlist(attention, required, 64)
    assert set(required) <= set(ids)
    assert len(ids) == 67
    assert ids == tuple(sorted((*range(64), 99, 602, 64)))
    assert ids == shortlist(attention.copy(), required, 64)


def test_attention_changes_scope_without_looking_at_any_label():
    a = np.arange(603, dtype=np.float32)
    ids = shortlist(a, (0, 1), 128)
    assert len(ids) == 130
    assert set(range(475, 603)) <= set(ids)
    assert ids != shortlist(a[::-1], (0, 1), 128)


@pytest.mark.parametrize(
    "scores,required,limit",
    [
        ([float("nan")], (), 64),
        ([-1], (), 64),
        ([1], (1,), 64),
        ([1], (0, 0), 64),
        ([1], (), 63),
    ],
)
def test_invalid_or_unregistered_policies_fail(scores, required, limit):
    with pytest.raises(ValueError):
        shortlist(scores, required, limit)


def test_original_six_policy_defaults_unchanged_and_variant_is_explicit():
    assert OverlayPolicy().planner_similarity_limit == 0
    assert OverlayPolicy().planner_host_transfer == "deferred_layer"
    assert OverlayPolicy(
        planner_similarity_limit=128, planner_host_transfer="deferred_layer_vectorized"
    )
    with pytest.raises(ValueError):
        OverlayPolicy(planner_similarity_limit=1)
    with pytest.raises(ValueError):
        OverlayPolicy(planner_host_transfer="guess")
    with pytest.raises(ValueError):
        OverlayPolicy(planner_similarity_limit=128, demand_weighted_coverage=True)
