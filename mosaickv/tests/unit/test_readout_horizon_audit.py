import pytest
import torch

from mosaickv.evaluation.readout_horizon_audit import audit, forced_plan


def test_fixed_plans_are_explicitly_controlled_and_bounded():
    assert forced_plan([dict(name="identity")], 0)["evaluation_only_forced_plan"]
    for invalid in (-1, 1, True):
        with pytest.raises(ValueError):
            forced_plan([dict(name="identity")], invalid)
    from storage_six_ablation import component_policies

    policies = component_policies(horizon_study=True)
    assert all(policies[f"controlled_plan_{i}"] == policies["readout_joint"] for i in range(10))


def test_oracle_diagnostic_separates_selection_from_later_predictions():
    reference = torch.tensor([[3.0, 1.0, 0.0]]).expand(128, -1).clone()
    identity = reference + torch.tensor([0.0, 0.1, 0.0])
    early_only = identity.clone()
    early_only[1] = reference[1]
    early_only[2:, 1] += 0.3
    stable = identity.clone()
    stable[2:] = reference[2:]
    result = audit(reference, dict(identity=identity, early_only=early_only, stable=stable))
    assert result["horizons"]["1"]["selected"] == "early_only"
    assert result["horizons"]["1"]["later_kl_difference"] > 0
    assert result["horizons"]["7"]["selected"] == "stable"
    assert result["horizons"]["7"]["later_kl_difference"] < 0
    assert result["oracle_future_logits_evaluation_only"] is True
    assert result["superiority_established"] is False


def test_failed_token_guard_is_not_a_useful_plan_and_incomplete_data_fail_closed():
    reference = torch.tensor([[3.0, 1.0, 0.0]]).expand(128, -1).clone()
    bad = reference.flip(-1)
    result = audit(reference, dict(identity=reference, bad=bad))
    assert not result["candidate_diagnostics"]["bad"]["absolute_32_token_guard_passed"]
    assert result["candidate_diagnostics"]["bad"]["kl_by_prediction"][1:] == [None] * 127
    assert all(r["selected"] == "identity" for r in result["horizons"].values())
    with pytest.raises(ValueError):
        audit(reference[:20], dict(identity=reference[:20]))
