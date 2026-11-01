import pytest
import torch

from mosaickv.evaluation.source_projection_audit import (
    comparison,
    frozen_projection_weight,
    projection_precision,
)


def test_checkpoint_fp16_weight_follows_registered_bf16_runtime_load():
    weight = torch.randn(16, 32).half()
    loaded = frozen_projection_weight(weight, device="cpu")
    assert loaded.dtype == torch.bfloat16
    assert torch.equal(loaded, weight.bfloat16())
    assert weight.dtype == torch.float16


def test_existing_operator_tolerance_not_relaxed_and_failures_preserved():
    a = torch.ones(3).bfloat16()
    b = a.clone()
    b[1] += 0.125
    record = comparison(a, b)
    assert not record["registered_tolerance_pass"] and record["mismatched_elements"] == 1
    assert record["examples"][0]["index"] == [1]
    assert comparison(a, a)["registered_tolerance_pass"]


def test_diagnostic_backend_flags_restored_on_failure():
    old = (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_tf32,
    )
    with pytest.raises(RuntimeError), projection_precision(reduced=False, fp32=True):
        assert not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        assert not torch.backends.cuda.matmul.allow_tf32
        raise RuntimeError("diagnostic only")
    assert old == (
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        torch.backends.cuda.matmul.allow_tf32,
    )
