"""CPU semantic and allocation controls; native CUDA parity is separately gated."""

from unittest.mock import patch

import pytest
import torch

from mosaickv.evaluation.bounded_native_attention import NativeHeadWorkspace
from mosaickv.evaluation.bounded_native_attention_broadcast import attention, selftest


@pytest.mark.parametrize("length", [98, 375, 600, 604, 727, 731])
def test_single_physical_head_full_axis_zero_stride_bmm(length):
    workspace = NativeHeadWorkspace(device="cpu", slots=1024)
    query = torch.ones(1, 2, 1, 128, dtype=torch.bfloat16)
    calls, bmm_calls = [], []
    native_bmm = torch.bmm

    def fill(first, stop, k, v):
        assert stop == first + 1
        assert k.shape == v.shape == (1, 1, length, 128)
        assert k.untyped_storage().data_ptr() == workspace.key.untyped_storage().data_ptr()
        assert v.untyped_storage().data_ptr() == workspace.value.untyped_storage().data_ptr()
        # Simulate V-as-RoPE scratch, then fill actual values.
        v.zero_()
        k.fill_(1)
        v.fill_(first + 1)
        calls.append(first)

    def bmm(a, b):
        assert a.shape[0] == b.shape[0] == 32
        assert a.stride(0) == b.stride(0) == 0
        bmm_calls.append((tuple(a.shape), tuple(b.shape)))
        return native_bmm(a, b)

    with patch.object(torch, "bmm", bmm):
        actual, probabilities, report = attention(
            query,
            fill,
            workspace,
            length=length,
            scaling=128**-0.5,
        )
    assert calls == [0, 1] and len(bmm_calls) == 4
    assert probabilities is None and "scores" not in report
    # Constant values make the semantic expectation independent of native
    # reduction-order ties. BF16 uniform probabilities need not sum to 1 exactly.
    key = torch.ones(1, 2, length, 128, dtype=torch.bfloat16)
    value = torch.ones_like(key)
    value[:, 1].fill_(2)
    p = (query @ key.transpose(-1, -2) * 128**-0.5).softmax(-1, dtype=torch.float32).bfloat16()
    expected = (p @ value).transpose(1, 2).contiguous()
    assert torch.equal(actual, expected)
    assert workspace.backing_bytes == report["workspace_backing_bytes"] == 524288
    assert report["physical_heads_per_fill"] == 1 and not report["KV_repeat_or_clone"]


def test_explicit_diagnostics_return_full_stages_and_default_is_bounded():
    workspace = NativeHeadWorkspace(device="cpu", slots=1024)
    query = torch.ones(1, 2, 1, 128, dtype=torch.bfloat16)

    def fill(_first, _stop, k, v):
        k.fill_(1)
        v.fill_(2)

    actual, p, report = attention(
        query,
        fill,
        workspace,
        length=8,
        scaling=0.5,
        capture_probabilities=True,
        capture_scores=True,
    )
    assert actual.shape == (1, 1, 2, 128)
    assert p.shape == report["scores"].shape == (1, 2, 1, 8)
    assert torch.equal(p, torch.full_like(p, 0.125))
    assert not actual.requires_grad


def test_invalid_inputs_fail_closed():
    workspace = NativeHeadWorkspace(device="cpu", slots=1024)
    query = torch.ones(1, 2, 1, 128, dtype=torch.bfloat16)

    def noop(*_args):
        return None

    for length in (0, 1025):
        with pytest.raises(ValueError, match="1024"):
            attention(query, noop, workspace, length=length, scaling=1.0)
    with pytest.raises(ValueError, match="BF16"):
        attention(query.float(), noop, workspace, length=8, scaling=1.0)
    with pytest.raises(NotImplementedError, match="GQA"):
        attention(query, noop, workspace, length=8, scaling=1.0, num_key_value_heads=1)
    with pytest.raises(NotImplementedError, match="nonzero"):
        attention(
            query,
            noop,
            workspace,
            length=8,
            scaling=1.0,
            attention_mask=torch.full((1, 1, 1, 8), -1, dtype=torch.bfloat16),
        )


def test_selftest_retains_every_cpu_stage_without_claiming_cuda_parity():
    report = selftest(device="cpu", lengths=(98, 600))
    assert report["measurement_type"] == "synthetic_cpu_operator_diagnostic"
    assert report["workspace_backing_bytes"] == 524288
    assert [row["length"] for row in report["rows"]] == [98, 600]
    assert all(
        "scores" in row and "probabilities" in row and "output" in row for row in report["rows"]
    )
