import pytest
import torch

from mosaickv.evaluation.fused_transport_predicate import LIMIT, original, stage, validate


def test_original_fixture_keeps_nonfinite_and_signed_zero_semantics():
    left = torch.tensor([0, -0.0, float("inf"), float("nan"), 1], dtype=torch.bfloat16)
    right = torch.tensor([-0.0, 0, float("inf"), float("nan"), 2], dtype=torch.bfloat16)
    mask, finite = original(left, right)
    assert mask.tolist() == [False, False, False, True, True]
    assert not finite
    ids, words, status = stage(original, left, right)
    assert ids.tolist() == [3, 4] and ids.dtype == torch.int32
    assert torch.equal(words.view(torch.int16), left[ids.long()].view(torch.int16))
    assert not status


def test_cpu_is_explicitly_unsupported_not_a_fake_kernel():
    x = torch.ones(3, dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="CUDA-only"):
        validate(x, x)


@pytest.mark.parametrize("fault", ["dtype", "shape", "stride", "empty", "bound"])
def test_invalid_operator_inputs_fail_before_dispatch(fault):
    x = torch.ones(4, dtype=torch.bfloat16)
    y = x.clone()
    if fault == "dtype":
        y = y.float()
    elif fault == "shape":
        y = y[:2]
    elif fault == "stride":
        x, y = x[::2], y[::2]
    elif fault == "empty":
        x, y = x[:0], y[:0]
    else:
        x = torch.empty(LIMIT + 1, dtype=torch.bfloat16, device="meta")
        y = torch.empty_like(x)
    with pytest.raises(ValueError, match="bounded nonempty BF16"):
        validate(x, y)
