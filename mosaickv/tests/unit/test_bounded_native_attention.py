"""CPU-only grouping, accounting and native complete-axis reader controls."""

import pytest
import torch

from mosaickv.evaluation.bounded_native_attention import (
    NativeHeadWorkspace,
    attention,
    plan_groups,
    selftest,
)


@pytest.mark.parametrize("length", [98, 375, 600, 604, 727, 731])
def test_complete_axis_groups_have_no_singleton(length):
    groups = plan_groups(32, length)
    assert [head for first, stop in groups for head in range(first, stop)] == list(range(32))
    assert all(stop - first >= 2 and (stop - first) * length <= 2048 for first, stop in groups)
    assert plan_groups(32, length) == groups


@pytest.mark.parametrize("heads,length,slots", [(1, 98, 2048), (32, 1025, 2048), (3, 100, 200)])
def test_unsupported_singleton_is_explicit(heads, length, slots):
    with pytest.raises(NotImplementedError, match="non-singleton"):
        plan_groups(heads, length, slots)


def test_backing_cost_is_constant_for_all_groups():
    workspace = NativeHeadWorkspace(device="cpu")
    pointers = [t.untyped_storage().data_ptr() for t in workspace.owned_tensors()]
    assert workspace.backing_bytes == 2 * 2048 * 128 * 2
    for length in (98, 375, 731):
        for first, stop in plan_groups(32, length):
            views = workspace.group_views(stop - first, length)
            assert [t.untyped_storage().data_ptr() for t in views] == pointers
            assert all(t.is_contiguous() and t.shape[-2] == length for t in views)
            assert workspace.backing_bytes == 1048576


@pytest.mark.parametrize("length", [98, 600, 731])
def test_native_complete_axis_output_and_callback(length):
    generator = torch.Generator().manual_seed(123)
    query = torch.randn((1, 8, 1, 128), generator=generator).bfloat16()
    key, value = tuple(
        torch.randn((1, 8, length, 128), generator=generator).bfloat16() for _ in range(2)
    )
    workspace = NativeHeadWorkspace(device="cpu")
    calls, observations = [], []

    def fill(first, stop, k, v):
        calls.append((first, stop))
        assert k.shape == v.shape == (1, stop - first, length, 128)
        # V-as-scratch is permitted; it must be filled with final values last.
        v.zero_()
        k.copy_(key[:, first:stop])
        v.copy_(value[:, first:stop])

    actual, probabilities, record = attention(
        query,
        fill,
        workspace,
        length=length,
        scaling=128**-0.5,
        observe=lambda tensors: observations.append(tensors),
    )
    reference_p = (
        (query @ key.transpose(-1, -2) * 128**-0.5).softmax(-1, dtype=torch.float32).bfloat16()
    )
    reference = (reference_p @ value).transpose(1, 2).contiguous()
    assert torch.equal(actual, reference)
    assert probabilities is None and "scores" not in record
    assert calls == list(plan_groups(8, length))
    assert record["token_reduction_is_unsplit"] and observations
    again, _, _ = attention(query, fill, workspace, length=length, scaling=128**-0.5)
    assert torch.equal(actual, again)


def test_unsupported_dtype_mask_and_gqa_fail_closed():
    workspace = NativeHeadWorkspace(device="cpu")
    query = torch.ones(1, 2, 1, 128, dtype=torch.bfloat16)

    def noop(*_args):
        return None

    with pytest.raises(ValueError, match="BF16"):
        attention(query.float(), noop, workspace, length=10, scaling=1.0)
    with pytest.raises(NotImplementedError, match="GQA"):
        attention(query, noop, workspace, length=10, scaling=1.0, num_key_value_heads=1)
    with pytest.raises(NotImplementedError, match="masked"):
        attention(
            query,
            noop,
            workspace,
            length=10,
            scaling=1.0,
            attention_mask=torch.full((1, 1, 1, 10), -1, dtype=torch.bfloat16),
        )


def test_selftest_collects_every_length():
    report = selftest(device="cpu", lengths=(98, 375, 600))
    assert [row["length"] for row in report["rows"]] == [98, 375, 600]
    assert report["measurement_type"] == "synthetic_cpu_operator_diagnostic"
    assert all(set(("scores", "probabilities", "output")) <= row.keys() for row in report["rows"])
    assert report["workspace_backing_bytes"] == 1048576
