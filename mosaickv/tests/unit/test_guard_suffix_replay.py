from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.guard_suffix_replay import replay
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy


def fake():
    w = SimpleNamespace(output=torch.zeros(1, 4096, dtype=torch.bfloat16))
    w.fill = lambda residual, attention: setattr(w, "input", residual + attention)
    graph = SimpleNamespace(replay=lambda: w.output.copy_(w.input.view_as(w.output)))
    layer = SimpleNamespace(self_attn=SimpleNamespace(layer_idx=31, o_proj=lambda x: x * 2))
    t = SimpleNamespace(
        workspace=w,
        closed=False,
        calls=32,
        replays=32,
        captures=32,
        entries={31: graph},
        layers=(*range(31), layer),
    )
    return t, layer


def test_replay_keeps_active_state_independent_and_layer_order_unchanged():
    t, layer = fake()
    residual, heads = (torch.ones(1, 1, 4096, dtype=torch.bfloat16) * n for n in (3, 5))
    t.workspace.output.fill_(17)
    saved, parent, size = replay(t, layer, residual, heads, t.workspace.output.view(1, 1, -1))
    assert saved.eq(17).all() and parent.eq(13).all()
    assert saved.untyped_storage().data_ptr() != parent.untyped_storage().data_ptr()
    assert (t.calls, t.replays, t.captures) == (32, 32, 32)
    assert t.guard_counterfactual_replays == 1 and size == 8192
    t.workspace.output.zero_()
    assert saved.eq(17).all()
    assert residual.eq(3).all() and heads.eq(5).all()


@pytest.mark.parametrize("condition", ("ordering", "missing_graph", "alias", "wrong_layer"))
def test_unaudited_suffix_or_shared_residual_fails_closed(condition):
    t, layer = fake()
    x = torch.ones(1, 1, 4096, dtype=torch.bfloat16)
    r = x
    if condition == "ordering":
        t.calls = 31
    if condition == "missing_graph":
        t.entries = {}
    if condition == "alias":
        r = t.workspace.output.view(1, 1, -1)
    if condition == "wrong_layer":
        layer = SimpleNamespace(self_attn=layer.self_attn)
    with pytest.raises(ValueError, match="independent residual"):
        replay(t, layer, r, x, x)


def test_suffix_policy_does_not_silently_activate():
    assert OverlayPolicy().anchor_suffix == "reference"
    with pytest.raises(ValueError, match="live parent guard"):
        OverlayPolicy(anchor_suffix="shared_graph")
    with pytest.raises(ValueError, match="verification"):
        OverlayPolicy(verify_anchor_suffix=True)
    assert OverlayPolicy(
        live_anchor_guard=True,
        inline_repair=True,
        anchor_suffix="shared_graph",
        verify_anchor_suffix=True,
    )
