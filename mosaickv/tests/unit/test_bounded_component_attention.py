"""CPU layout, lifetime and padding semantics for the component-sized reader."""

from unittest.mock import patch

import pytest
import torch

from mosaickv.evaluation.bounded_component_attention import ComponentWorkspace, attention, selftest


@pytest.mark.parametrize("length", [98, 375, 600, 604, 727, 731])
def test_component_pool_layout_shared_owner_and_native_broadcast(length):
    workspace = ComponentWorkspace(device="cpu", heads=2)
    query = torch.ones(1, 2, 1, 128, dtype=torch.bfloat16)
    key = torch.ones(1, 2, length, 128, dtype=torch.bfloat16)
    value = key.clone()
    value[:, 1].fill_(2)
    owners, stages, bmm_stages = [], [], []
    native_bmm = torch.bmm

    def fill(component, target, scratch):
        assert target.shape == (1, 2, length, 128)
        assert target.transpose(1, 2).is_contiguous()
        assert target.stride(1) == 128 and target.stride(2) == 256
        assert scratch.shape == (1, 1, length, 128) and scratch.is_contiguous()
        owners.append(target.untyped_storage().data_ptr())
        stages.append(component)
        scratch.fill_(17)  # K construction may use the same one-head scratch.
        target.copy_((key, value)[component])

    def bmm(a, b):
        assert a.shape[0] == b.shape[0] == 32
        assert a.stride(0) == b.stride(0) == 0
        bmm_stages.append(stages[-1])
        return native_bmm(a, b)

    with patch.object(torch, "bmm", bmm):
        actual, probabilities, report = attention(
            query, fill, workspace, length=length, scaling=128**-0.5
        )
    p = (query @ key.transpose(-1, -2) * 128**-0.5).softmax(-1, dtype=torch.float32).bfloat16()
    expected = (p @ value).transpose(1, 2).contiguous()
    assert torch.equal(actual, expected)
    assert owners == [workspace.component.untyped_storage().data_ptr()] * 2
    assert stages == [0, 1] and bmm_stages == [0, 0, 1, 1]
    assert probabilities is None and "scores" not in report
    assert report["simultaneous_full_components"] == 1 and report["component_fills"] == 2


def test_registered_full_workspace_counts_every_allocated_slot():
    workspace = ComponentWorkspace(device="cpu")
    assert workspace.backing_bytes == 8650752
    assert len(workspace.owned_tensors()) == 2
    for length in (1, 98, 731, 1024):
        assert workspace.component_view(length).untyped_storage().nbytes() == 8388608
        assert workspace.head_view(length).untyped_storage().nbytes() == 262144
        assert workspace.backing_bytes == 8650752


def test_per_head_prompt_padding_excludes_padding_but_keeps_history():
    workspace = ComponentWorkspace(device="cpu", heads=2)
    query = torch.zeros(1, 2, 1, 128, dtype=torch.bfloat16)
    key = torch.zeros(1, 2, 7, 128, dtype=torch.bfloat16)
    value = torch.arange(7, dtype=torch.bfloat16).view(1, 1, 7, 1).expand(1, 2, 7, 128).clone()
    value[:, 0, 3:5].fill_(999)

    def fill(component, target, _scratch):
        target.copy_((key, value)[component])

    actual, p, report = attention(
        query,
        fill,
        workspace,
        length=7,
        scaling=1.0,
        prompt_lengths=(3, 5),
        prompt_max_length=5,
        capture_probabilities=True,
        capture_scores=True,
    )
    scores = torch.zeros(1, 2, 1, 7, dtype=torch.bfloat16)
    scores[:, 0, :, 3:5] = torch.finfo(torch.bfloat16).min
    expected_p = scores.softmax(-1, dtype=torch.float32).bfloat16()
    assert torch.equal(p, expected_p)
    assert torch.equal(actual, (p @ value).transpose(1, 2).contiguous())
    assert not p[:, 0, :, 3:5].any() and p[:, 0, :, 5:].all()
    assert report["prompt_padding_masked_positions"] == 2
    assert torch.equal(report["scores"], scores)


def test_invalid_capacity_and_padding_fail_closed():
    workspace = ComponentWorkspace(device="cpu", heads=2)
    query = torch.zeros(1, 2, 1, 128, dtype=torch.bfloat16)

    def noop(*_args):
        return None

    with pytest.raises(ValueError, match="1024"):
        attention(query, noop, workspace, length=1025, scaling=1.0)
    for lengths, maximum in (((3,), 5), ((3, 6), 5), ((3, 5), None)):
        with pytest.raises(ValueError, match="prompt"):
            attention(
                query,
                noop,
                workspace,
                length=7,
                scaling=1.0,
                prompt_lengths=lengths,
                prompt_max_length=maximum,
            )
    with pytest.raises(NotImplementedError, match="GQA"):
        attention(query, noop, workspace, length=7, scaling=1.0, num_key_value_heads=1)


def test_cpu_diagnostic_reports_all_stages_without_cuda_claim():
    report = selftest(device="cpu", lengths=(98, 600))
    assert report["measurement_type"] == "synthetic_cpu_operator_diagnostic"
    assert report["workspace_backing_bytes"] == 8650752
    assert [row["length"] for row in report["rows"]] == [98, 600]
    assert all(
        "scores" in row and "probabilities" in row and "output" in row for row in report["rows"]
    )
