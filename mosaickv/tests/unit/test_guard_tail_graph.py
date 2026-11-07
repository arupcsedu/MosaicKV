from contextlib import nullcontext
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


def test_manual_capture_does_not_flush_allocator_or_collect_garbage(monkeypatch):
    from mosaickv.evaluation import post_attention_segment

    calls = []
    stream = SimpleNamespace(wait_stream=lambda other: calls.append("wait"))
    graph = SimpleNamespace(
        capture_begin=lambda **kw: calls.append(("begin", kw)),
        capture_end=lambda: calls.append("end"),
        reset=lambda: calls.append("reset"),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("manual capture may not invoke the allocator-flushing context")

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(torch.cuda, "graph", forbidden)
    monkeypatch.setattr(torch.cuda, "empty_cache", forbidden)
    monkeypatch.setattr(post_attention_segment, "capture_stream", lambda *args: stream)
    g = GuardTail(torch.nn.Identity(), torch.nn.Identity(), capture_mode="manual_shared_stream")
    p = torch.ones(1, 1, 4096).bfloat16()
    g._capture(p, p)
    assert ("begin", {"capture_error_mode": "global"}) in calls
    assert calls.count("end") == 1
    assert torch.equal(g.outputs[0], p)
    assert g.report()["capture_mode"] == "manual_shared_stream"
    g.close()
    assert calls[-1] == "reset"


def test_manual_capture_finishes_and_retires_graph_on_operator_failure(monkeypatch):
    from mosaickv.evaluation import guard_tail_graph, post_attention_segment

    calls = []
    stream = SimpleNamespace(wait_stream=lambda _: None)
    graph = SimpleNamespace(
        capture_begin=lambda **_: calls.append("begin"),
        capture_end=lambda: calls.append("end"),
        reset=lambda: calls.append("reset"),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda *args: nullcontext())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(post_attention_segment, "capture_stream", lambda *args: stream)
    n = 0

    def native(*args):
        nonlocal n
        n += 1
        if n == 3:
            raise RuntimeError("controlled capture failure")
        return torch.zeros(1)

    monkeypatch.setattr(guard_tail_graph, "native_tail", native)
    g = GuardTail(None, None, capture_mode="manual_shared_stream")
    p = torch.ones(1, 1, 4096).bfloat16()
    with pytest.raises(RuntimeError, match="controlled capture"):
        g._capture(p, p)
    assert calls == ["begin", "end", "reset"]


def test_capture_modes_remain_explicit():
    with pytest.raises(ValueError, match="registered"):
        GuardTail(None, None, capture_mode="approximate")
    assert OverlayPolicy(
        inline_repair=True, live_anchor_guard=True, anchor_tail="manual_shared_stream"
    )
