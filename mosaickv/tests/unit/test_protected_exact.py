import pytest
import torch

from mosaickv.evaluation.lossless_bf16 import pack_cpu
from mosaickv.evaluation.precision_tier import construct
from mosaickv.evaluation.protected_exact import scatter_into


@pytest.mark.parametrize("bits", [3, 4])
@pytest.mark.parametrize("n", [1, 19, 129, 300])
def test_encoding_exact_text_never_changes_precision_outputs(bits, n):
    generator = torch.Generator().manual_seed(19)
    pair = tuple(torch.randn(1, 2, n, 128, generator=generator).bfloat16() for _ in range(2))
    raw = construct(pair, bits=bits)
    encoded = construct(pair, bits=bits, encoded_exact=True)
    assert encoded.exact_map.numel() == 0
    assert all(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
        for a, b in zip(raw.materialize(), encoded.materialize(), strict=True)
    )
    assert encoded.component_bytes <= raw.component_bytes


def test_all_bf16_patterns_and_nonprotected_history_are_exact():
    words = torch.arange(65536).short()
    x = words.view(torch.bfloat16).reshape(1, 2, 256, 128)
    body = pack_cpu(x)
    positions = torch.arange(256, dtype=torch.int32) * 2 + 3
    dest = torch.full((1, 2, 521, 128), -5, dtype=torch.bfloat16)
    scatter_into(body, positions, dest)
    assert torch.equal(dest[..., positions.long(), :].view(torch.int16), x.view(torch.int16))
    mask = torch.ones(521, dtype=torch.bool)
    mask[positions.long()] = False
    assert (dest[..., mask, :] == -5).all()


def test_growth_geometry_never_triggers_per_token_compilation():
    from mosaickv.evaluation.protected_exact_kernel import _scatter

    p = {p.name: p for p in _scatter.params}
    assert all(
        not p[name].is_constexpr and p[name].do_not_specialize
        for name in ("COUNT", "P", "HEAD_OUT")
    )
