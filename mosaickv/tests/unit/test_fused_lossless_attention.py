import pytest
import torch

from mosaickv.evaluation.fused_lossless_attention import reference, run, validate
from mosaickv.evaluation.lossless_bf16 import pack_cpu


def test_reference_retains_eager_math_and_rejects_unsupported_layout():
    torch.manual_seed(1)
    q = torch.randn(1, 2, 1, 128).bfloat16()
    k, v = (torch.randn(1, 2, 17, 128).bfloat16() for _ in range(2))
    key, value = pack_cpu(k), pack_cpu(v)
    out, weights = reference(q, key, value, scale=128**-0.5)
    expected = torch.softmax(
        (q @ k.transpose(2, 3)) * 128**-0.5, dim=-1, dtype=torch.float32
    ).bfloat16()
    assert torch.equal(weights, expected)
    assert torch.equal(out, expected @ v)
    with pytest.raises(NotImplementedError, match="actual CUDA"):
        run(q, key, value, scale=128**-0.5)
    with pytest.raises(ValueError, match="MHA"):
        validate(q.expand(2, -1, -1, -1), key, value)
