from dataclasses import replace

import pytest
import torch

from mosaickv.evaluation.compact_quantization_metadata import compact
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import pack, unpack_cpu


@pytest.mark.parametrize("axis", (2, 3))
@pytest.mark.parametrize("bits", (3, 4, 8))
def test_actual_header_storage_and_unchanged_codes(axis, bits):
    torch.manual_seed(8814)
    x = torch.randn(1, 2, 33, 128).bfloat16()
    original = pack(x, bits=bits, axis=axis)
    candidate = compact(original)
    assert candidate.codes is original.codes
    assert candidate.minimum.dtype == candidate.scale.dtype == torch.float16
    assert original.minimum.dtype == original.scale.dtype == torch.float32
    expected = 2 * (original.minimum.numel() + original.scale.numel())
    assert backing_bytes(original.tensors()) - backing_bytes(candidate.tensors()) == expected
    assert torch.isfinite(unpack_cpu(candidate)).all()
    assert candidate.shape == original.shape and candidate.axis == original.axis


def test_no_clipping_or_silent_raw_fallback():
    body = pack(torch.ones(1, 1, 16, 128).bfloat16(), bits=3, axis=2)
    for broken in (
        replace(body, minimum=torch.full_like(body.minimum, 1e20)),
        replace(body, scale=torch.full_like(body.scale, 1e-20)),
    ):
        with pytest.raises(ValueError, match="unsupported"):
            compact(broken)
    with pytest.raises(ValueError, match="FP32"):
        compact(compact(body))
