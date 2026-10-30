import pytest
import torch

from mosaickv.evaluation.precision_tier import construct, pack, unpack_cpu


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("n", [1, 9, 129, 300])
def test_precision_roundtrip_bounds_and_originals(bits, n):
    generator = torch.Generator().manual_seed(42)
    pair = tuple(torch.randn((1, 2, n, 128), generator=generator).bfloat16() for _ in range(2))
    originals = tuple(t.clone() for t in pair)
    fmt = construct(pair, bits=bits)
    assert all(b.codes.numel() == pair[0].numel() * bits // 8 for b in fmt.pair)
    restored = fmt.materialize()
    for actual, reference in zip(restored, pair, strict=True):
        assert torch.isfinite(actual).all()
        assert torch.equal(
            actual[..., fmt.exact_indices.long(), :], reference[..., fmt.exact_indices.long(), :]
        )
    assert all(torch.equal(a, b) for a, b in zip(originals, pair, strict=True))
    assert fmt.component_bytes == sum(t.untyped_storage().nbytes() for t in fmt.tensors())
    if n == 300:
        assert fmt.component_bytes < sum(t.numel() * t.element_size() for t in pair) * 0.4


@pytest.mark.parametrize("axis", [2, 3])
def test_constant_partial_group_no_nan(axis):
    x = torch.full((1, 2, 131, 128), 2.5, dtype=torch.bfloat16)
    b = pack(x, bits=3, axis=axis)
    assert torch.equal(unpack_cpu(b), x)


def test_strided_destination_and_protected_rows():
    pair = tuple(torch.arange(2 * 19 * 128).reshape(1, 2, 19, 128).bfloat16() for _ in range(2))
    f = construct(pair)
    dest = tuple(torch.full((1, 2, 23, 128), -5, dtype=torch.bfloat16) for _ in range(2))
    f.fill(None, dest, direct=True)
    assert all(bool((t[..., 19:, :] == -5).all()) for t in dest)
    assert all(torch.equal(t[..., :19, :], r) for t, r in zip(dest, f.materialize(), strict=True))


@pytest.mark.parametrize("bad", [0, 2, 5, True])
def test_invalid_bits_rejected(bad):
    with pytest.raises(ValueError, match="3/4 bits"):
        pack(torch.ones(1, 2, 32, 128, dtype=torch.bfloat16), bits=bad, axis=2)


def test_nonfinite_rejected():
    x = torch.ones(1, 2, 32, 128, dtype=torch.bfloat16)
    x[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        pack(x, bits=4, axis=2)


def test_additional_text_rows_exact_and_map_charged():
    pair = tuple(torch.randn(1, 2, 141, 128).bfloat16() for _ in range(2))
    f = construct(pair, protected=(9, 64, 100))
    assert all(int(f.exact_map[i]) >= 0 for i in (9, 64, 100))
    assert int(f.exact_map[65]) == -1
    for a, b in zip(f.materialize(), pair, strict=True):
        assert torch.equal(a[..., (9, 64, 100), :], b[..., (9, 64, 100), :])
    with pytest.raises(ValueError, match="in bounds"):
        construct(pair, protected=(141,))


def test_decode_length_and_growing_stride_never_specialize():
    from mosaickv.evaluation.precision_tier_kernel import _decode

    parameters = {p.name: p for p in _decode.params}
    for name in ("N", "SH", "PROTECTED"):
        assert not parameters[name].is_constexpr
        assert parameters[name].do_not_specialize
