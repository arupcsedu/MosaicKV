import copy

import evicastkv_joint_kv_transaction as controller
import pytest


@pytest.fixture
def configured(monkeypatch):
    control = controller.control
    for key, value in dict(
        POLICY="kv",
        CANDIDATE_ARM="joint-kv",
        VARIANT=controller.VARIANT,
        SEQUENCE=controller.SEQUENCE,
        EXPECTED_TRANSACTION=dict(
            key_calls=4064,
            folded_keys=4032,
            original_cold_keys=32,
            pending=False,
            all_value_projections_original=False,
            value_commits=4064,
            fake_raw_value_tensors=0,
        ),
    ).items():
        monkeypatch.setattr(control, key, value)
    return control


def test_real_value_commit_is_required_and_not_mislabeled_as_original_projection(configured):
    from test_joint_key_controller import measured

    value = measured(True)
    value["joint_incoming_projection"].update(
        policy="kv",
        value_projection_unchanged=False,
        transactions=[copy.deepcopy(configured.EXPECTED_TRANSACTION)],
    )
    configured.validate(value, True)
    value["joint_incoming_projection"]["transactions"][0]["value_commits"] -= 1
    with pytest.raises(ValueError, match="actually share"):
        configured.validate(value, True)


@pytest.mark.parametrize(
    "key_pass,timing_complete,allowed",
    [(True, True, False), (False, False, False), (False, True, True)],
)
def test_gpu_preparation_waits_for_negative_key_screen_and_positive_parity(
    monkeypatch, tmp_path, key_pass, timing_complete, allowed
):
    timing = dict(complete=timing_complete, failures=[], four_percent_POINT_only=key_pass)
    parity = dict(complete=True, failures=[], joint_key_128_vector_parity_and_isolation=True)
    monkeypatch.setattr(
        controller.driver,
        "read",
        lambda p: timing if p.parent.parent == controller.WAIT else parity,
    )
    called = []
    monkeypatch.setattr(controller, "ORIGINAL_PREPARE", lambda *args: called.append(args))
    if allowed:
        controller.prepare(tmp_path, "source")
        assert called
    else:
        with pytest.raises(ValueError, match="key-only"):
            controller.prepare(tmp_path, "source")
        assert not called
