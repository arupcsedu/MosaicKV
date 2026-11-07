from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.live_anchor_guard import LiveAnchorGuard, _current, decision
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy


def test_live_decision_keeps_registered_thresholds_and_greedy_tokens():
    original = torch.tensor([4.0, 1.0, 0.0])
    assert bool(decision(original, original)[0])
    assert bool(decision(original, original + 0.02)[0])
    assert bool(decision(original, original + 1.0)[0])
    assert not bool(decision(original, original + 1.01)[0])
    assert not bool(decision(original, torch.tensor([3.0, 1.9, 0.9]))[0])
    assert not bool(decision(original, torch.tensor([0.0, 4.0, 0.0]))[0])
    with pytest.raises(ValueError, match="nonfinite"):
        decision(original, torch.full((3,), float("nan")))
    with pytest.raises(ValueError):
        decision(original, original[None])


def test_live_guard_scope_restores_patches_and_parent_owners_on_failure():
    from mosaickv.evaluation.uniform_core_transport import Binding

    score = Binding.scores
    report = {}
    g = LiveAnchorGuard(SimpleNamespace(), SimpleNamespace(head=0), report)
    with pytest.raises(RuntimeError, match="controlled"), g:
        assert _current.get() is g
        raise RuntimeError("controlled")
    assert _current.get() is None and Binding.scores is score
    assert report["live_anchor_guard"]["steps"] == 0
    assert g.parent_key is g.parent_value is None
    with pytest.raises(ValueError):
        OverlayPolicy(live_anchor_guard=True)
    assert OverlayPolicy(inline_repair=True, live_anchor_guard=True)


def test_parent_snapshots_own_only_one_head_and_do_not_alias_the_live_lease():
    observed = []
    lease = torch.randn(1, 32, 9, 128).bfloat16()
    layer = SimpleNamespace(
        attention=SimpleNamespace(layer_idx=31),
        leased=(lease,),
        get_seq_length=lambda: 7,
        observe=lambda t: observed.append(t),
    )
    g = LiveAnchorGuard(SimpleNamespace(), SimpleNamespace(head=4), {})
    g.record_component(layer, 0)
    assert g.parent_key.shape == (1, 1, 7, 128)
    original = g.parent_key.clone()
    lease.zero_()
    assert torch.equal(g.parent_key, original)
    assert g.parent_key.untyped_storage().nbytes() == 7 * 128 * 2
    assert observed and g.peak_temporary_kv_bytes == 7 * 128 * 2
    with pytest.raises(RuntimeError, match="not consumed"):
        g.record_component(layer, 0)
