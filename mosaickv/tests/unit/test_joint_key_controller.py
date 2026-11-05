import copy

import evicastkv_joint_key_transaction as controller
import pytest


def measured(active):
    return dict(
        logits_bitwise=True,
        no_FullKV_fallback=True,
        peak_measurement_admissible=True,
        prompt_graph_reader=dict(
            captures=64,
            replays=8128,
            graph_private_live_bytes=0,
            active_workspace_bytes=0,
            retired=True,
            lt_workspace_peak_bytes=2**21 + (8192 if active else 0),
        ),
        joint_incoming_projection=dict(
            policy="keys" if active else "none",
            eligible_layers=32 if active else 0,
            value_projection_unchanged=True,
            no_placeholder_values=True,
            transactions=[
                dict(
                    key_calls=4064,
                    folded_keys=4032,
                    original_cold_keys=32,
                    pending=False,
                    all_value_projections_original=True,
                )
            ]
            if active
            else [],
        ),
    )


def test_genuine_key_only_engagement_is_required_and_original_controls_are_unchanged():
    controller.validate(measured(True), True)
    controller.validate(measured(False), False)


@pytest.mark.parametrize("corruption", ["logits", "owners", "replays", "folds", "bytes", "values"])
def test_a_partial_or_unaccounted_transaction_cannot_be_called_passed(corruption):
    data = copy.deepcopy(measured(True))
    if corruption == "logits":
        data["logits_bitwise"] = False
    elif corruption == "owners":
        data["prompt_graph_reader"]["active_workspace_bytes"] = 8192
    elif corruption == "replays":
        data["prompt_graph_reader"]["replays"] += 1
    elif corruption == "folds":
        data["joint_incoming_projection"]["transactions"][0]["folded_keys"] -= 1
    elif corruption == "bytes":
        data["prompt_graph_reader"]["lt_workspace_peak_bytes"] = 2**21
    else:
        data["joint_incoming_projection"]["no_placeholder_values"] = False
    with pytest.raises(ValueError):
        controller.validate(data, True)


def test_the_original_control_cannot_silently_use_the_new_engine():
    with pytest.raises(ValueError, match="control"):
        controller.validate(measured(True), False)


def test_the_next_stage_cannot_run_on_incomplete_or_negative_all_layer_proof(monkeypatch, tmp_path):
    monkeypatch.setattr(
        controller.driver,
        "read",
        lambda *args: dict(complete=True, failures=[], all_layer_exact_both_inputs=False),
    )
    called = []
    monkeypatch.setattr(controller, "ORIGINAL_PREPARE", lambda *args: called.append(args))
    with pytest.raises(ValueError, match="all-layer"):
        controller.prepare(tmp_path, "source")
    assert not called
