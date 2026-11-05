import copy

import pytest

from mosaickv.evaluation.joint_graph_scope import graph_scope, reconcile


@pytest.mark.parametrize("policy,used", [("none", []), ("keys", ["key"]), ("kv", ["key", "value"])])
def test_incoming_projection_graphs_are_not_mislabeled_as_immutable_prompt_only(policy, used):
    record = graph_scope(policy)
    assert record["only_immutable_prompt_captured"] is (policy == "none")
    assert record["reused_incoming_components"] == used
    assert not record["attention_softmax_and_model_layers_captured"]
    assert record["captured_incoming_components"] == ([] if policy == "none" else ["key", "value"])


def test_derived_reconciliation_never_changes_old_measurements_or_raw_records():
    original = dict(
        joint_incoming_projection=dict(policy="kv", eligible_layers=32),
        prompt_graph_reader=dict(only_immutable_prompt_captured=True, replays=8128),
        logits_bitwise=True,
        total_wall_seconds=10.0,
        initial_encoded_bytes=12345,
    )
    untouched = copy.deepcopy(original)
    derived = reconcile(original)
    assert original == untouched
    assert not derived["prompt_graph_reader"]["only_immutable_prompt_captured"]
    assert derived["graph_scope_metadata_correction"]["correction_required"]
    for key in ("logits_bitwise", "total_wall_seconds", "initial_encoded_bytes"):
        assert derived[key] == original[key]


def test_source_less_baseline_control_is_still_an_immutable_prompt_graph():
    derived = reconcile(
        dict(
            joint_incoming_projection=dict(policy="kv", eligible_layers=0),
            prompt_graph_reader=dict(only_immutable_prompt_captured=True),
        )
    )
    assert derived["prompt_graph_reader"]["only_immutable_prompt_captured"]
    assert not derived["graph_scope_metadata_correction"]["correction_required"]


@pytest.mark.parametrize("policy", [True, None, "unknown", "qkv"])
def test_unknown_scope_is_not_silently_reinterpreted(policy):
    with pytest.raises(ValueError):
        graph_scope(policy)
