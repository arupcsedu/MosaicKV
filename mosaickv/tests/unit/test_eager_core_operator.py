import pytest
import torch

from mosaickv.evaluation.eager_core_operator import (
    Core,
    opportunities,
    reference_scores,
    same_words,
)


@pytest.mark.parametrize("mask_dtype", [None, torch.bfloat16, torch.float32])
@torch.inference_mode()
def test_out_operations_match_original_cpu_api_and_keep_one_lease(mask_dtype):
    lease = torch.full((1, 2, 17, 128), -71.0, dtype=torch.bfloat16)
    core = Core(lease, mask_dtype=mask_dtype, graph=False, frequencies={})
    for length in (7, 13, 7):
        q = torch.randn(1, 2, 1, 128).bfloat16()
        k = torch.randn(1, 2, length, 128).bfloat16()
        v = torch.randn_like(k)
        mask = None if mask_dtype is None else torch.zeros(1, 1, 1, length, dtype=mask_dtype)
        if mask is not None:
            mask[..., 0] = torch.finfo(mask_dtype).min
        expected_weights = reference_scores(q, k, mask, 128**-0.5)
        expected_output = torch.matmul(expected_weights, v).transpose(1, 2).contiguous()
        lease[..., :length, :].copy_(k)
        got = core.scores_for(q, mask, length)
        assert same_words(got, expected_weights)
        assert same_words(lease[..., :length, :], k)
        with pytest.raises(RuntimeError, match="matching QK"):
            core.values_for(length + 1)
        lease[..., :length, :].copy_(v)
        assert same_words(core.values_for(length), expected_output)
        assert same_words(lease[..., :length, :], v)
    assert core.captures == 0 and core.direct_calls == 6
    assert all(
        t.untyped_storage().data_ptr() != lease.untyped_storage().data_ptr() for t in core.tensors()
    )
    core.close()
    assert core.closed and not core.tensors()


def test_cpu_graph_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match="real CUDA"):
        Core(
            torch.empty(1, 2, 17, 128, dtype=torch.bfloat16),
            mask_dtype=None,
            graph=True,
            frequencies={},
        )


def test_registered_opportunity_rule_is_label_blind_and_bounded():
    counts = opportunities((600,) * 32, 127)
    assert len(counts) == 127 and set(counts.values()) == {32}
    assert counts[601] == 32 and counts[727] == 32
    assert sum(opportunities((100, 200), 4).values()) == 8
    with pytest.raises(ValueError, match="bounded"):
        opportunities((600,), 128)


@torch.inference_mode()
def test_pending_transaction_cannot_be_overwritten_or_retired():
    lease = torch.zeros(1, 2, 17, 128, dtype=torch.bfloat16)
    core = Core(lease, mask_dtype=None, graph=False, frequencies={})
    query = torch.zeros(1, 2, 1, 128, dtype=torch.bfloat16)
    core.scores_for(query, None, 8)
    with pytest.raises(ValueError, match="transaction"):
        core.scores_for(query, None, 8)
    with pytest.raises(RuntimeError, match="incomplete"):
        core.close()
    core.values_for(8)
    core.close()
