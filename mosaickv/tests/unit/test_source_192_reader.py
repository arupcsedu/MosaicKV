from types import SimpleNamespace

import pytest

from mosaickv.evaluation import joint_incoming_projection as joint
from mosaickv.evaluation.source_192_reader import Engine192


@pytest.mark.parametrize("rows", [True, 0, 129, 192.0, "192", 256])
def test_only_explicit_capacities_accepted(rows):
    with pytest.raises(ValueError, match="capacity"):
        joint.validate_source_rows(rows, "kv", True, False)


@pytest.mark.parametrize(
    "policy,frozen,history", [("keys", True, False), ("kv", False, False), ("kv", True, True)]
)
def test_unsupported_combinations_fail_closed(policy, frozen, history):
    with pytest.raises(NotImplementedError):
        joint.validate_source_rows(192, policy, frozen, history)


def test_baseline_ineligibility_is_a_truthful_noop():
    joint.validate_source_rows(192, "none", True, False)
    with pytest.raises(ValueError, match="private"):
        Engine192(SimpleNamespace())


def test_historical_128_policy_is_unchanged():
    for policy in ("none", "keys", "kv"):
        joint.validate_source_rows(128, policy, False, True)


def test_metadata_alone_cannot_forge_new_workspace():
    import evicastkv_joint_key_transaction as control
    from test_joint_key_controller import measured

    value = measured(True)
    value["joint_incoming_projection"]["source_tile_rows"] = 192
    with pytest.raises(ValueError, match="actual corrected"):
        control.validate(value, True)
