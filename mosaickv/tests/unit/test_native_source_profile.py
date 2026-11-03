from types import SimpleNamespace
from unittest.mock import patch

import evicastkv_native_operator_study as driver
import pytest
import torch

from mosaickv.evaluation import full_prompt_source as native
from mosaickv.evaluation.native_source_profile import aggregate, product_mode, wide_product


@pytest.mark.parametrize("length", (33, 128, 257))
@pytest.mark.parametrize("chunk", (None, 128))
def test_wide_product_charges_output_and_preserves_strided_history(length, chunk):
    h, d = 2, 128
    torch.manual_seed(31)
    source = torch.randn(1, length, h * d)
    weight = torch.randn(h * d, h * d)
    weights = weight.view(h, d, h * d).transpose(1, 2)
    owned = torch.full((h, length + 7, d), -71.0)
    output = owned[:, :length]
    buffer = torch.empty((min(length, chunk or length), h * d))
    wide_product(source, weights, output, buffer, chunk=chunk)
    expected = (source[0] @ weight.t()).view(length, h, d).transpose(0, 1)
    torch.testing.assert_close(output, expected)
    assert (owned[:, length:] == -71).all()
    assert buffer.numel() * buffer.element_size() == min(length, chunk or length) * h * d * 4


def test_invalid_buffer_and_mode_are_rejected():
    source = torch.zeros(1, 33, 256)
    weights = torch.zeros(256, 256).view(2, 128, 256).transpose(1, 2)
    output = torch.empty(2, 33, 128)
    with pytest.raises(ValueError, match="charged"):
        wide_product(source, weights, output, torch.empty(32, 256))
    with pytest.raises(ValueError, match="registered"), product_mode("guess"):
        pass


def test_product_injection_is_isolated_and_default_remains_bmm():
    original = native.projection_product
    with product_mode("wide_128") as buffers:
        assert native.projection_product is not original
        source = torch.zeros(1, 257, 256)
        weight = torch.zeros(256, 256).view(2, 128, 256).transpose(1, 2)
        output = torch.empty(2, 257, 128)
        native.projection_product(source, weight, output)
        assert len(buffers) == 1 and buffers[0].shape == (128, 256)
    assert native.projection_product is original
    with pytest.raises(RuntimeError), product_mode("wide"):
        raise RuntimeError("isolated failure")
    assert native.projection_product is original


def test_default_product_matches_preexisting_bmm_call():
    workspace = torch.randn(1, 3, 256)
    weights = torch.randn(2, 256, 128)
    output = torch.empty(2, 3, 128)
    original = torch.bmm
    with patch.object(torch, "bmm", wraps=original) as call:
        native.projection_product(workspace, weights, output)
    assert call.call_count == 1
    torch.testing.assert_close(output, original(workspace.expand(2, -1, -1), weights))


def test_aggregation_excludes_warmups_and_promotion_requires_every_geometry():
    rows = [
        dict(mode="bmm", warmup=True, cuda_interval_ms=999, host_seconds=99),
        dict(mode="bmm", warmup=False, cuda_interval_ms=2, host_seconds=1),
    ]
    assert aggregate(rows)["bmm"]["mean_cuda_ms"] == 2
    cases = [
        dict(
            layer=layer,
            padding=padding,
            aggregate={
                m: dict(mean_cuda_ms=1 if m == "original" else 0.8)
                for m in ("original", "wide", "wide_128")
            },
            correctness={
                m: dict(bitwise=True, extra_output_workspace_bytes=10, repair_bytes=2)
                for m in ("wide", "wide_128")
            },
        )
        for layer in (0, 16, 31)
        for padding in (0, 64, 127)
    ]
    assert driver.evaluate(cases)["wide"]["operator_improvement_gate"]
    cases[-1]["aggregate"]["wide"]["mean_cuda_ms"] = 0.91
    assert not driver.evaluate(cases)["wide"]["operator_improvement_gate"]
    with pytest.raises(ValueError, match="complete unique"):
        driver.evaluate(cases[:-1])


def test_wide_refitting_is_bitwise_and_does_not_replace_baseline_repairs():
    from test_full_prompt_source import example

    attention, packet = example(length=64)
    reference = packet.materialize(attention)
    for mode in ("wide", "wide_128"):
        with product_mode(mode):
            fitted = native.fit(packet, attention, (0, 1))
            actual = fitted.materialize(attention)
            assert all(
                torch.equal(a.view(torch.int16), b.view(torch.int16))
                for a, b in zip(reference, actual, strict=True)
            )
    assert type(attention) is SimpleNamespace
