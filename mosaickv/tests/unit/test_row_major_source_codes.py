from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.row_major_source_codes import geometry, lookup, reference, use


def packed(values):
    bits = values.flatten().long()
    words = (bits.reshape(-1, 8) << (torch.arange(8) * 3)).sum(-1)
    return torch.stack(tuple(((words >> (i * 8)) & 255).byte() for i in range(3)), 1).flatten()


@pytest.mark.parametrize("rows,width", [(1, 8), (7, 16), (17, 512), (32, 32)])
def test_exact_integer_layout_transpose_and_nibble_pairs(rows, width):
    q = (torch.arange(rows * width) % 8).reshape(width, rows)
    codes = packed(q)
    out = reference(codes, rows, width)
    actual = torch.stack((out & 15, out >> 4), -1).reshape(rows, width)
    assert torch.equal(actual.long(), q.t())
    assert out.numel() == geometry(rows, width)


def test_binding_is_private_exact_and_reset_even_after_error():
    a = SimpleNamespace(codes=torch.zeros(24, dtype=torch.uint8), shape=(1, 1, 8, 8))
    b = SimpleNamespace(codes=a.codes.clone(), shape=a.shape)
    scratch = torch.zeros(32, dtype=torch.uint8)
    assert lookup(a) is None
    with pytest.raises(RuntimeError), use(a, scratch):
        assert lookup(a) is scratch
        with pytest.raises(ValueError):
            lookup(b)
        with use(a, scratch):
            pass
    assert lookup(a) is None


@pytest.mark.parametrize("rows,width", [(0, 8), (1, 7), (True, 8), (2**30, 8)])
def test_invalid_or_unbounded_layout_rejected(rows, width):
    with pytest.raises(ValueError):
        geometry(rows, width)
