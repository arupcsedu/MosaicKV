import pytest
import torch

from mosaickv.evaluation.packed_prototypes import construct, selected_codes
from mosaickv.evaluation.precision_tier import pack
from mosaickv.evaluation.shared_head_prototypes import construct as reference
from mosaickv.evaluation.shared_head_prototypes import integers


@pytest.mark.parametrize("rows,width", ((128, 64), (129, 128), (603, 128)))
@pytest.mark.parametrize("groups", (((8, 9, 10),), ((8, 9), (40, 41, 42, 43))))
def test_packed_constructor_matches_reference_words_metadata_and_repair(rows, width, groups):
    torch.manual_seed(rows)
    body = pack(torch.randn(1, 1, rows, width).bfloat16(), bits=3, axis=2)
    actual, expected = construct(body, groups, heads=4), reference(body, groups, heads=4)
    for name in ("codes", "row_map", "minimum", "scale"):
        assert torch.equal(getattr(actual.body, name), getattr(expected.body, name))
    assert torch.equal(actual.residual_codes, expected.residual_codes)
    assert actual.changed_code_symbols == expected.changed_code_symbols
    assert actual.record() == expected.record()
    assert torch.equal(actual.promote().codes, body.codes)


def test_selected_code_gather_uses_original_channel_major_bit_order():
    body = pack(torch.randn(1, 1, 603, 128).bfloat16(), bits=3, axis=2)
    assert torch.equal(
        selected_codes(body.codes, 603, 128, (0, 17, 602)),
        integers(body.codes, 603, 128)[[0, 17, 602]],
    )
    assert construct(body, (), heads=4, retention_one=True) is body
    with pytest.raises(ValueError):
        construct(body, ((0, 1),), heads=4, mandatory=(0,))
