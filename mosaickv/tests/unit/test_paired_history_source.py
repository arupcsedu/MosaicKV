import weakref
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import mixed_history_projection as mixed
from mosaickv.evaluation import native_history_reader as native
from mosaickv.evaluation.lossless_bf16 import pack_cpu
from mosaickv.evaluation.paired_history_source import use


def fixture(monkeypatch):
    class State:
        closed = False
        graphs = None
        projections = 0

    state = State()

    def execute(source, weight, output, n):
        output[:n].copy_(source[:n] * weight)

    state.engine = SimpleNamespace(
        source=torch.empty(128, 4096, dtype=torch.bfloat16),
        output=torch.empty(128, 4096, dtype=torch.bfloat16),
        plans=SimpleNamespace(execute=execute),
    )
    monkeypatch.setattr(native, "current", lambda: state)
    monkeypatch.setattr("mosaickv.evaluation.shared_source_kernel.rotate_native", lambda *a: None)
    recent = tuple(
        SimpleNamespace(
            shape=(1, 32, 16, 128),
            archive=SimpleNamespace(source=pack_cpu(torch.randn(1, 1, 16, 4096).bfloat16())),
        )
        for _ in range(2)
    )
    cos, sin = torch.ones(1, 32, 128).bfloat16(), torch.zeros(1, 32, 128).bfloat16()
    packet = mixed.MixedHistory(
        None,
        recent,
        cos,
        sin,
        (),
        weakref.ref(state),
        (1, 32, 32, 128),
        (native.phase_token(cos), native.phase_token(sin)),
    )
    attention = SimpleNamespace(k_proj=SimpleNamespace(weight=1), v_proj=SimpleNamespace(weight=2))
    return packet, attention, state


def test_reuse_only_inside_immediate_pair_preserves_both_projections(monkeypatch):
    packet, attention, state = fixture(monkeypatch)
    expected = packet.materialize(attention)
    with use() as report:
        actual = packet.materialize(attention)
        assert all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(actual, expected, strict=True)
        )
        assert report["reused_sources"] == 1 and report["paired_transactions"] == 1
        # Model's ordinary separate K,V calls never reuse, including a source
        # overwrite between them. Reconstruct the V source again.
        packet.fill_component(attention, 0, actual[0])
        state.engine.source.fill_(91)
        packet.fill_component(attention, 1, actual[1])
        assert report["reused_sources"] == 1
        assert report["original_unpacks"] == 3
        assert actual[1].equal(expected[1])
    assert report["added_GPU_storage_bytes"] == 0 and state.projections == 6


def test_target_alias_or_nested_policy_is_rejected(monkeypatch):
    packet, attention, state = fixture(monkeypatch)
    pair = (
        state.engine.source[:32].view(1, 32, 32, 128),
        torch.empty(packet.shape, dtype=torch.bfloat16),
    )
    with use():
        with pytest.raises(RuntimeError, match="nonaliasing"):
            packet.fill(attention, pair, direct=True)
        with pytest.raises(RuntimeError, match="nonnested"), use():
            pass


def test_request_isolation_and_all_passes_recompute_source(monkeypatch):
    packet, attention, state = fixture(monkeypatch)
    expected = packet.materialize(attention)
    with use() as report:
        for _ in range(3):
            state.engine.source.fill_(float("nan"))
            actual = packet.materialize(attention)
            assert all(a.equal(b) for a, b in zip(actual, expected, strict=True))
    assert report["reused_sources"] == report["original_unpacks"] == 3
    with use() as other:
        packet.materialize(attention)
    assert other["reused_sources"] == other["original_unpacks"] == 1
