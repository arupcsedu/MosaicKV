from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.live_anchor_guard import LiveAnchorGuard, repair_operator_controls
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy


def test_repair_attention_matches_full_native_shapes_on_cpu():
    assert repair_operator_controls("cpu")["passed"]


def test_risk_and_stress_policies_require_live_guard_and_are_explicitly_distinct():
    for cfg in ({"anchor_risk_repair": True}, {"anchor_repair_stress": True}):
        with pytest.raises(ValueError):
            OverlayPolicy(**cfg)
    p = OverlayPolicy(inline_repair=True, live_anchor_guard=True, anchor_risk_repair=True)
    assert not p.anchor_repair_stress
    assert OverlayPolicy(
        inline_repair=True,
        live_anchor_guard=True,
        anchor_risk_repair=True,
        anchor_repair_stress=True,
    )


def test_coexisting_inline_and_guard_kv_owners_are_observed_together():
    observed = []
    inline = SimpleNamespace(key=torch.zeros(7, 128).bfloat16())
    guard = LiveAnchorGuard(SimpleNamespace(), SimpleNamespace(), {}, inline_owner=inline)
    guard.layer = SimpleNamespace(observe=lambda tensors: observed.append(tensors))
    guard.active_key = torch.zeros(1, 1, 7, 128).bfloat16()
    guard.charge()
    assert guard.peak_temporary_kv_bytes == 7 * 128 * 2
    assert guard.coexisting_temporary_kv_peak_bytes == 2 * 7 * 128 * 2
    assert len(observed[-1]) == 2


def test_risk_panel_cannot_create_an_unreviewed_fresh_cohort(tmp_path):
    from storage_six_anchor_followup import prepare

    with pytest.raises(ValueError, match="fixed consumed panel"):
        prepare(tmp_path / "not_panel", risk_repair=True)
