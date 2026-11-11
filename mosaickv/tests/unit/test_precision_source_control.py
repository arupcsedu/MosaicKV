from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.precision_source_control import (
    BITS,
    LABEL,
    PrecisionSourceControl,
    construct,
    decode_source,
    validate,
)
from mosaickv.evaluation.precision_tier import pack, unpack_cpu
from mosaickv.evaluation.shared_source_format import construct as parent_construct


def fixture(n=8):
    generator = torch.Generator().manual_seed(123)
    x = torch.randn((1, n, 256), generator=generator).bfloat16()
    k = torch.nn.Linear(256, 256, bias=False).bfloat16()
    v = torch.nn.Linear(256, 256, bias=False).bfloat16()
    with torch.no_grad():
        for block in (k, v):
            block.weight.copy_(torch.randn(block.weight.shape, generator=generator).bfloat16())
    attention = SimpleNamespace(k_proj=k, v_proj=v)
    pair = tuple(proj(x).reshape(1, n, 2, 128).transpose(1, 2).contiguous() for proj in (k, v))
    positions = torch.arange(n, dtype=torch.int32)
    parent = parent_construct(
        x,
        pair,
        ((0, n - 1), (0, n - 1)),
        (1,),
        (torch.ones(1, n, 128).bfloat16(), torch.zeros(1, n, 128).bfloat16()),
        (positions, positions.short()[None].expand(2, n).contiguous()),
    )
    return x, attention, pair, parent


@pytest.mark.parametrize("n", [8, 13, 128, 131])
def test_native_source_decode_matches_quantizer_oracle(n):
    x = torch.randn((1, 1, n, 256), generator=torch.Generator().manual_seed(n)).bfloat16()
    body = pack(x, bits=BITS, axis=2)
    assert torch.equal(
        decode_source(body).view(torch.int16), unpack_cpu(body)[:, 0].view(torch.int16)
    )


def test_construction_is_label_blind_deterministic_and_preserves_protected_rows():
    x, attention, pair, parent = fixture()
    old_x = x.clone()
    old_weights = tuple(p.weight.clone() for p in (attention.k_proj, attention.v_proj))
    first, report = construct(parent, x, attention)
    second, again = construct(parent, x, attention)
    assert first.kind == LABEL and first.source.bits == 8
    assert parent.source.bits == 3 and not parent.transport_patches
    assert report == again
    assert torch.equal(x, old_x)
    assert all(
        torch.equal(p.weight, old)
        for p, old in zip((attention.k_proj, attention.v_proj), old_weights, strict=True)
    )
    assert all(torch.equal(a, b) for a, b in zip(first.tensors(), second.tensors(), strict=True))
    ids = parent.row_positions.long()
    output = first.materialize(attention)
    for actual, original in zip(output, pair, strict=True):
        assert torch.equal(actual[0, ids // 8, ids % 8], original[0, ids // 8, ids % 8])
    assert first.shared() == parent.shared()
    assert first.base.row_positions is parent.row_positions
    assert first.base.exact is parent.exact
    assert report["persistent_bytes"] == first.component_bytes
    assert report["production_three_bit_path_changed"] is False
    assert report["no_benchmark_labels_or_reference_logits"]
    assert first.source.codes.data_ptr() != x.data_ptr()
    assert first.source.codes.data_ptr() != parent.source.codes.data_ptr()


@pytest.mark.parametrize("field,value", [("input_overlay", object()), ("source_head_group", 2)])
def test_unsupported_policy_is_rejected(field, value):
    x, attention, _, parent = fixture()
    modified = replace(parent, **{field: value})
    with pytest.raises(ValueError):
        construct(modified, x, attention)


def test_invalid_source_or_strides_are_not_silently_accepted():
    x, attention, _, parent = fixture()
    with pytest.raises(ValueError, match="unpatched"):
        construct(replace(parent, source=pack(x[:, None], bits=4, axis=2)), x, attention)
    first, _ = construct(parent, x, attention)
    targets = tuple(torch.empty(1, 2, 8, 128).bfloat16() for _ in range(2))
    validate(first.base, attention, targets)
    with pytest.raises(ValueError, match="eight-bit"):
        validate(parent, attention, targets)
    with pytest.raises(ValueError, match="strided"):
        validate(first.base, attention, (targets[0][..., ::2], targets[1]))
    with pytest.raises(ValueError, match="uncompacted"):
        decode_source(parent.source)
    assert PrecisionSourceControl(first.base).source is first.source
