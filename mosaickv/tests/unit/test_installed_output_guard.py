from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mosaickv.backends.hf_runtime import _prompt_query_token_attention_by_head
from mosaickv.config import EvidencePolicyConfig
from mosaickv.selection.installed_output_guard import decide, output_errors


def test_improving_installed_subset_is_kept_without_changing_values_or_capacity():
    a = np.array([[0.4, 0.25, 0.25, 0.1], [0.3, 0.3, 0.3, 0.1]])
    v = np.array([[0.0], [0.0], [10.0], [0.0]])
    original = v.copy()
    ids, trace = decide(a, v, (0, 1), (0, 2), (0,), minimum_query_mass_fraction=0.95)
    assert ids == (0, 2) and trace["proposal_accepted"]
    assert trace["changed_positions"] == 1
    assert trace["installed_error"] < trace["anchor_error"]
    np.testing.assert_array_equal(v, original)


def test_mean_improvement_cannot_hide_a_single_query_regression():
    a = np.array([[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]])
    v = np.array([[0.0], [10.0], [2.0], [0.0]])
    _, before = output_errors(a, v, (0, 1))
    _, after = output_errors(a, v, (0, 2))
    assert after.mean() < before.mean() and (after > before).any()
    ids, trace = decide(a, v, (0, 1), (0, 2), (0,), minimum_query_mass_fraction=0.1)
    assert ids == (0, 1) and trace["rolled_back"]
    assert trace["installed_error"] == trace["anchor_error"]


@pytest.mark.parametrize("proposal", [(1, 2), (0,), (0, 0)])
def test_mandatory_capacity_and_unique_position_fail_closed(proposal):
    with pytest.raises(ValueError):
        decide(
            np.ones((2, 4)),
            np.ones((4, 2)),
            (0, 1),
            proposal,
            (0,),
            minimum_query_mass_fraction=0.95,
        )


@pytest.mark.parametrize("as_numpy", [False, True])
def test_individual_gqa_heads_survive_query_capture(as_numpy):
    data = torch.arange(1, 49, dtype=torch.float32).reshape(1, 4, 3, 4)
    source = data.numpy() if as_numpy else data
    state = SimpleNamespace(layers=[SimpleNamespace(kv_heads=2, sequence_length=4)])
    prefill = SimpleNamespace(attention_weights=[source])
    kept = _prompt_query_token_attention_by_head(
        state, prefill, prompt_window=3, preserve_query_heads=True
    )
    pooled = _prompt_query_token_attention_by_head(state, prefill, prompt_window=3)
    assert kept[0, 0].shape == (6, 4) and pooled[0, 0].shape == (3, 4)
    np.testing.assert_array_equal(kept[0, 0], data[0, :2].reshape(6, 4).numpy())
    np.testing.assert_array_equal(pooled[0, 0], data[0, :2].mean(0).numpy())
    assert EvidencePolicyConfig(variant="output_guarded").opportunity_routing is False
