import pytest
import torch

from mosaickv.evaluation.conditional_precision import exact_rows, pack_unprotected
from mosaickv.evaluation.head_local_precision import localize
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.phase_capacity import recode
from mosaickv.evaluation.precision_tier import construct, pack, unpack_cpu


def example(n=256):
    x = torch.linspace(-1, 1, n)[None, None, :, None].expand(1, 2, n, 128).contiguous().bfloat16()
    x[..., :4, :] = 100
    x[..., -8:, :] = -100
    # Put both signs in each exact boundary group: their discarded extrema
    # otherwise leave no quantization level near the useful [-1,1] interval.
    x[..., 1, :] = -100
    x[..., -1, :] = 100
    pair = x, torch.randn_like(x)
    ids = (tuple(range(n)),) * 2
    packet = localize(
        construct(pair, bits=3, encoded_exact=True), pair, ids, set(), policy="head3_boundary8"
    )
    return packet, pair, ids


def test_exact_outliers_no_longer_waste_key_levels_and_body_bytes_do_not_increase():
    packet, pair, _ = example()
    rows = exact_rows(packet)
    before = pack(pair[0], bits=2, axis=2)
    after = pack_unprotected(pair[0], rows, bits=2, axis=2)
    old_error = (unpack_cpu(before)[0][~rows].float() - pair[0][0][~rows].float()).square().mean()
    new_error = (unpack_cpu(after)[0][~rows].float() - pair[0][0][~rows].float()).square().mean()
    assert new_error < old_error * 0.01
    assert backing_bytes(before.tensors()) == backing_bytes(after.tensors())


@pytest.mark.parametrize("axis", [2, 3])
def test_empty_mask_is_bitwise_identical_and_all_masked_groups_are_finite(axis):
    _, pair, _ = example()
    rows = torch.zeros(2, 256, dtype=torch.bool)
    a, b = pack(pair[0], bits=3, axis=axis), pack_unprotected(pair[0], rows, bits=3, axis=axis)
    assert all(torch.equal(x, y) for x, y in zip(a.tensors(), b.tensors(), strict=True))
    c = pack_unprotected(pair[0], ~rows, bits=3, axis=axis)
    assert torch.isfinite(unpack_cpu(c)).all()


def test_exact_post_rope_rows_remain_bitwise_and_no_persistent_mask_is_added():
    packet, pair, ids = example()
    n = pair[0].shape[-2]
    phases = torch.ones(1, n, 128).bfloat16(), torch.zeros(1, n, 128).bfloat16()
    a = recode(packet, pair, ids, phases, key_bits=2, value_bits=2, pre_rotary=True)
    b = recode(
        packet,
        pair,
        ids,
        phases,
        key_bits=2,
        value_bits=2,
        pre_rotary=True,
        exclude_exact_rows=True,
    )
    assert a.component_bytes == b.component_bytes
    rows = exact_rows(packet)
    for actual, reference in zip(b.materialize(None), pair, strict=True):
        assert torch.equal(actual[0][rows], reference[0][rows])
    with pytest.raises(ValueError):
        pack_unprotected(pair[0], rows.float(), bits=2, axis=2)


def test_value_groups_do_not_change_unprotected_tokens():
    packet, pair, _ = example()
    rows = exact_rows(packet)
    a, b = pack(pair[1], bits=2, axis=3), pack_unprotected(pair[1], rows, bits=2, axis=3)
    assert torch.equal(unpack_cpu(a)[0][~rows], unpack_cpu(b)[0][~rows])


def test_source_text_metadata_is_used_but_scalar_only_patches_do_not_exclude_a_row():
    from dataclasses import replace
    from types import SimpleNamespace

    from mosaickv.evaluation.exact_storage import SparseBitPatch
    from mosaickv.evaluation.text_source_storage import TextSourceFormat

    packet, _, _ = example()
    patch = SparseBitPatch(
        torch.tensor([10 * 128 + 7], dtype=torch.int32), torch.tensor([1.0]).bfloat16()
    )
    local = replace(packet, base=replace(packet.base, corrections=(patch, patch)))
    outer = TextSourceFormat(
        local, SimpleNamespace(slots=torch.tensor([[10, -1], [11, 12]], dtype=torch.int16))
    )
    assert not exact_rows(local)[0, 10]
    mask = exact_rows(outer)
    assert mask[0, 10] and not mask[0, 11] and mask[1, 11] and mask[1, 12]
