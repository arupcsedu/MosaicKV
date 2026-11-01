from dataclasses import replace

import pytest
import torch

from mosaickv.evaluation.exact_minimum_metadata import compact, compact_format
from mosaickv.evaluation.head_local_precision import localize
from mosaickv.evaluation.precision_tier import construct, pack, unpack_cpu


@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("axis", [2, 3])
def test_only_exact_minimum_storage_changes_not_scale_codes_or_decoded_values(bits, axis):
    torch.manual_seed(452)
    original = pack(torch.randn(1, 3, 137, 128).bfloat16(), bits=bits, axis=axis)
    smaller = compact(original)
    assert smaller.minimum.dtype == torch.bfloat16
    assert smaller.scale is original.scale and smaller.codes is original.codes
    assert torch.equal(
        smaller.minimum.float().view(torch.int32), original.minimum.view(torch.int32)
    )
    assert torch.equal(
        unpack_cpu(smaller).view(torch.int16), unpack_cpu(original).view(torch.int16)
    )
    assert (
        sum(t.numel() * t.element_size() for t in smaller.tensors())
        == sum(t.numel() * t.element_size() for t in original.tensors())
        - original.minimum.numel() * 2
    )
    assert compact(smaller) is smaller


def test_nonrepresentable_minimum_is_rejected_instead_of_approximated():
    body = pack(torch.randn(1, 2, 32, 128).bfloat16(), bits=3, axis=2)
    invalid = replace(body, minimum=body.minimum.clone())
    invalid.minimum.flatten()[0] = 0.1
    with pytest.raises(ValueError, match="exactly BF16"):
        compact(invalid)
    with pytest.raises(ValueError, match="FP32 scale"):
        compact(replace(body, scale=body.scale.half()))


def test_exact_rows_and_corrections_are_preserved_inside_head_local_format():
    pair = tuple(torch.randn(1, 2, 64, 128).bfloat16() for _ in range(2))
    packet = construct(pair, bits=3, encoded_exact=True)
    heads = (tuple(range(64)),) * 2
    packet = localize(packet, pair, heads, {0, 63}, policy="head3_boundary8")
    converted = compact_format(packet)
    assert converted.base.exact_values is packet.base.exact_values
    assert converted.base.corrections is packet.base.corrections
    assert converted.component_bytes < packet.component_bytes
    assert all(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
        for a, b in zip(packet.materialize(), converted.materialize(), strict=True)
    )


def test_shared_source_projection_and_transport_corrections_remain_bitwise():
    from types import SimpleNamespace

    from mosaickv.evaluation.exact_storage import SparseBitPatch
    from mosaickv.evaluation.shared_source_format import construct as source_construct

    torch.manual_seed(902)
    n, h, d = 32, 2, 128
    x = torch.randn(1, n, h * d).bfloat16()
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
    )
    phase = (torch.ones(1, n, d).bfloat16(), torch.zeros(1, n, d).bfloat16())
    with torch.inference_mode():
        pair = tuple(
            p(x).reshape(1, n, h, d).transpose(1, 2) for p in (attention.k_proj, attention.v_proj)
        )
        source = source_construct(
            x,
            pair,
            ((*range(15), 31), tuple(range(16, 32))),
            (0, 31),
            phase,
            (
                torch.arange(n, dtype=torch.int32),
                torch.arange(n, dtype=torch.int16)[None].expand(h, n),
            ),
        )
        patches = tuple(
            SparseBitPatch(torch.empty(0, dtype=torch.int32), torch.empty(0, dtype=torch.bfloat16))
            for _ in range(2)
        )
        source = replace(source, transport_patches=patches)
        smaller = compact_format(source)
        assert smaller.transport_patches is source.transport_patches
        assert smaller.source.codes is source.source.codes
        assert smaller.source.scale is source.source.scale
        assert smaller.component_bytes < source.component_bytes
        assert all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(
                source.materialize(attention), smaller.materialize(attention), strict=True
            )
        )
