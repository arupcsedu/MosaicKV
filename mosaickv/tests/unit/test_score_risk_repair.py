from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.score_risk_repair import ScoreRiskRepair, should_repair
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy


@pytest.mark.parametrize(
    "repaired,disabled,force,expected",
    [
        (False, False, False, True),
        (True, False, True, False),
        (False, True, True, False),
        (False, False, True, True),
    ],
)
def test_risk_policy_never_repeats_or_overrides_none(repaired, disabled, force, expected):
    assert (
        should_repair(
            mass=0.5,
            dispersion=0.5,
            threshold=0.1,
            repaired=repaired,
            disabled=disabled,
            force=force,
        )
        == expected
    )


def test_invalid_risk_is_failure_not_silent_veto():
    with pytest.raises(ValueError):
        should_repair(
            mass=float("nan"),
            dispersion=1,
            threshold=0.1,
            repaired=False,
            disabled=False,
            force=False,
        )


def test_score_risk_execution_is_separately_labeled_and_cannot_claim_parent_lock():
    with pytest.raises(ValueError):
        OverlayPolicy(repair_execution="score_risk")
    with pytest.raises(ValueError):
        OverlayPolicy(
            repair_execution="score_risk", repair_policy="prototype_risk", live_anchor_guard=True
        )
    p = OverlayPolicy(repair_execution="score_risk", repair_policy="prototype_risk")
    assert not p.live_anchor_guard and not p.inline_repair


def test_real_equal_positions_updated_once_without_incrementing_committed_score_count():
    policy = OverlayPolicy(
        repair_execution="score_risk", repair_policy="prototype_risk", prototype_risk_threshold=0.01
    )
    report = dict(repair_events=[])
    key = torch.zeros(1, 32, 8, 128, dtype=torch.bfloat16)
    positions = torch.tensor([2, 4])
    overlay = SimpleNamespace(
        head=4,
        positions=positions,
        dispersion=1.0,
        repair_count=0,
        anchor=1,
        promotion=(
            torch.full((2, 128), 7, dtype=torch.bfloat16),
            torch.full((2, 128), 9, dtype=torch.bfloat16),
        ),
    )
    promoted = []

    def promote():
        promoted.append(True)
        overlay.repair_count += 1
        return dict(restored_bytes=1024, restored_positions=[2, 4])

    overlay.promote = promote
    calls = []

    def replay(query, mask, length):
        assert bool((key[0, 4, positions] == 7).all())
        calls.append(length)
        return torch.full((1, 32, 1, 8), 0.125, dtype=torch.bfloat16)

    binding = SimpleNamespace(calls=32, core=SimpleNamespace(replay_pending_scores=replay))
    query = torch.zeros(1, 32, 1, 128, dtype=torch.bfloat16)
    scores = torch.full((1, 32, 1, 8), 0.125, dtype=torch.bfloat16)
    guard = ScoreRiskRepair(overlay, policy, report)
    guard.apply(binding, query, key, None, 128**-0.5, scores)
    guard.apply(binding, query, key, None, 128**-0.5, scores)
    assert len(promoted) == len(calls) == len(report["repair_events"]) == 1
    assert binding.calls == 32 and guard.steps == 2
    assert bool((key[0, :4] == 0).all()) and bool((key[0, 5:] == 0).all())
    assert report["repair_events"][0]["provisional_vocabulary_logits_computed"] is False
