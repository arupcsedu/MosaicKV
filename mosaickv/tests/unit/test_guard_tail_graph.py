from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.guard_tail_graph import GuardTail, native_tail
from mosaickv.evaluation.live_anchor_guard import decision
from mosaickv.evaluation.storage_six_overlay import OverlayPolicy


@pytest.mark.parametrize("delta", [0.0, 0.01, 1.0, 2.0])
def test_native_guard_tail_retains_exact_reference_decision(delta):
    g = torch.Generator().manual_seed(149)
    parent = torch.randn(1, 1, 16, generator=g).bfloat16()
    active = parent + delta
    head = torch.nn.Linear(16, 23, bias=False).bfloat16()
    with torch.no_grad():
        head.weight.copy_(torch.randn(23, 16, generator=g).bfloat16())
        output, logits, stats = native_tail(torch.nn.Identity(), head, parent, active)
        reference = head(parent).float().reshape(-1)
        passed, record = decision(reference, head(active).float().reshape(-1))
    assert torch.equal(stats[:4], record)
    assert bool(stats[4])
    assert torch.equal(logits, reference)
    assert torch.equal(output, torch.where(passed, active, parent))


def test_nonfinite_tail_never_silently_passes():
    p = torch.tensor([[[float("nan"), 0.0]]])
    _, _, stats = native_tail(torch.nn.Identity(), torch.nn.Identity(), p, p)
    assert not bool(stats[4]) and not bool(stats[3])


def test_guard_tail_scope_rejects_cpu_shapes_and_releases_owners():
    g = GuardTail(SimpleNamespace(), SimpleNamespace())
    with pytest.raises(ValueError, match="isolated"):
        g.run(torch.zeros(1, 1, 4096).bfloat16(), torch.zeros(1, 1, 4096).bfloat16())
    g.close()
    assert g.closed and g.norm is g.head is g.inputs is g.outputs is None
    assert g.report()["additional_persistent_KV_bytes"] == 0


def test_graph_tail_is_opt_in_and_does_not_silently_disable_repair():
    assert OverlayPolicy().anchor_tail == "reference"
    with pytest.raises(ValueError, match="guard-tail"):
        OverlayPolicy(anchor_tail="cuda_graph")
    with pytest.raises(ValueError, match="guard-tail"):
        OverlayPolicy(
            inline_repair=True,
            live_anchor_guard=True,
            anchor_tail="cuda_graph",
            anchor_risk_repair=True,
        )
    assert OverlayPolicy(inline_repair=True, live_anchor_guard=True, anchor_tail="cuda_graph")
