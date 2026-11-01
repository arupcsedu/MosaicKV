import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from mosaickv.evaluation.precision_tier import pack


def original(values, bits, axis):
    x = values.float().movedim(axis, -1).contiguous()
    length = x.shape[-1]
    if length % 128:
        x = torch.cat((x, x[..., -1:].expand(*x.shape[:-1], (-length) % 128)), -1)
    shaped = x.reshape(*x.shape[:-1], -1, 128)
    minimum, maximum = shaped.amin(-1), shaped.amax(-1)
    scale = (maximum - minimum) / ((1 << bits) - 1)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = ((shaped - minimum[..., None]) / scale[..., None]).round().clamp(0, (1 << bits) - 1).long()
    q = q.flatten(-2)[..., :length].contiguous().reshape(-1, 8)
    words = (q << (torch.arange(8, dtype=torch.long, device=x.device) * bits)).sum(-1)
    codes = torch.stack(
        tuple(((words >> (8 * i)) & 255).to(torch.uint8) for i in range(bits)), -1
    ).flatten()
    return codes, minimum, scale


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("axis", [2, 3])
@pytest.mark.parametrize("kind", ["random", "constant", "extrema"])
def test_smaller_integer_temporaries_keep_every_code_and_header_bit(bits, axis, kind):
    torch.manual_seed(349)
    x = torch.randn(1, 3, 17, 128).bfloat16()
    if kind == "constant":
        x.fill_(-0.0)
    elif kind == "extrema":
        # Includes maximal 0xffffffff four-bit and signed int64 eight-bit words.
        x = (torch.arange(x.numel()).reshape_as(x) % 16 == 0).bfloat16()
    expected = original(x, bits, axis)
    body = pack(x, bits=bits, axis=axis)
    for a, b in zip(expected, (body.codes, body.minimum, body.scale), strict=True):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


class IntegerAudit(TorchDispatchMode):
    def __init__(self):
        self.largest = 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        for t in result if isinstance(result, tuple) else (result,):
            if isinstance(t, torch.Tensor) and t.dtype in (torch.int64, torch.int32, torch.uint8):
                self.largest = max(self.largest, t.numel() * t.element_size())
        return result


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_integer_result_footprint_decreases_without_changing_serialized_storage(bits):
    x = torch.randn(1, 4, 128, 128).bfloat16()
    before, after = IntegerAudit(), IntegerAudit()
    with before:
        expected = original(x, bits, 3)
    with after:
        actual = pack(x, bits=bits, axis=3)
    assert after.largest <= before.largest // (8 if bits == 8 else 2)
    assert torch.equal(expected[0], actual.codes)
