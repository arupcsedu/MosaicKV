import pytest
import torch

from mosaickv.evaluation.affine_refit import refit
from mosaickv.evaluation.precision_tier import pack, unpack_cpu


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("axis", [2, 3])
def test_frozen_two_step_refit_never_increases_BF16_group_error_or_bytes(bits, axis):
    torch.manual_seed(407)
    x = torch.randn(1, 2, 136, 128).bfloat16()
    old = pack(x, bits=bits, axis=axis)
    candidate, proof = refit(old, x)
    assert proof["active_source_SSE_after"] <= proof["active_source_SSE_before"] + 1e-12
    assert proof["same_payload_bytes"] and proof["steps"] == 2
    assert (
        torch.sum((unpack_cpu(candidate).float() - x.float()).square())
        <= torch.sum((unpack_cpu(old).float() - x.float()).square()) + 1e-5
    )
    again, _ = refit(old, x)
    assert all(torch.equal(a, b) for a, b in zip(candidate.tensors(), again.tensors(), strict=True))


def test_empty_exact_groups_retain_original_codes_and_headers():
    x = torch.randn(1, 1, 16, 128).bfloat16()
    body = pack(x, bits=3, axis=2)
    candidate, proof = refit(body, x, exact_rows=torch.ones(1, 16, dtype=torch.bool))
    assert proof["accepted_groups"] == [0, 0]
    assert all(torch.equal(a, b) for a, b in zip(candidate.tensors(), body.tensors(), strict=True))


def test_wrong_geometry_and_dtype_fail_closed():
    x = torch.randn(1, 1, 16, 128).bfloat16()
    body = pack(x, bits=3, axis=2)
    with pytest.raises(ValueError):
        refit(body, x.float())
    with pytest.raises(ValueError):
        refit(body, x, exact_rows=torch.ones(1, 16))
