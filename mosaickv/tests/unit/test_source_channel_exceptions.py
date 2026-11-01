from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.source_channel_exceptions import (
    ChannelExceptions,
    construct,
    kernel_arguments,
    restore_cpu,
)


@pytest.mark.parametrize("budget", [0, 1, 63, 64, 65, 256, 4096])
def test_budget_determinism_and_coupled_source_columns(budget):
    x = torch.arange(8 * 64).reshape(1, 1, 8, 64).bfloat16()
    old = x - 1
    ex, proof = construct(x, old, torch.ones_like(x, dtype=torch.float32), budget_bytes=budget)
    if ex is None:
        assert proof["exception_bytes"] == 0
        return
    assert proof["exception_bytes"] <= budget
    expected = min(254, (budget - 64) // 16, 64)
    assert proof["channels"] == expected
    restored = restore_cpu(ex, old[:, 0].clone())
    ids = (ex.lookup != 255).nonzero().flatten()
    assert torch.equal(restored[0, :, ids], x[0, 0, :, ids])
    other = (ex.lookup == 255).nonzero().flatten()
    assert torch.equal(restored[0, :, other], old[0, 0, :, other])
    assert ex.values.data_ptr() != x.data_ptr()
    again, _ = construct(x, old, torch.ones_like(x, dtype=torch.float32), budget_bytes=budget)
    assert torch.equal(ex.lookup, again.lookup) and torch.equal(ex.values, again.values)


def test_no_positive_gain_needs_no_payload_and_invalid_maps_fail():
    x = torch.ones(1, 1, 8, 64).bfloat16()
    assert construct(x, x, torch.ones_like(x), budget_bytes=4096)[0] is None
    with pytest.raises(ValueError):
        ChannelExceptions(torch.tensor([0, 0], dtype=torch.uint8), torch.zeros(1, 8).bfloat16())
    with pytest.raises(ValueError):
        construct(x, x, -torch.ones_like(x), budget_bytes=4096)


def test_runtime_arguments_do_not_reduce_or_copy_tensors(monkeypatch):
    x = torch.ones(1, 1, 8, 64).bfloat16()
    ex, _ = construct(x, x * 0, torch.ones_like(x), budget_bytes=256)
    packet = SimpleNamespace(
        source=SimpleNamespace(shape=x.shape, codes=torch.ones(1, dtype=torch.uint8)),
        channel_exceptions=ex,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("decode may not scan GPU data")

    monkeypatch.setattr(torch, "equal", forbidden)
    monkeypatch.setattr(torch, "isfinite", forbidden)
    a, b, active = kernel_arguments(packet)
    assert active and a is ex.lookup and b is ex.values


def test_real_input_geometry_costs_are_included():
    for n in (600, 604):
        x = torch.ones(1, 1, n, 4096).bfloat16()
        _ex, p = construct(x, x * 0, torch.ones_like(x), budget_bytes=32768)
        assert p["channels"] == 23 and p["exception_bytes"] == 4096 + 23 * n * 2 <= 32768
