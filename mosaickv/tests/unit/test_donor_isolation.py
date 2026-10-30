"""Synthetic donor-availability controls; not measured benchmark results."""

from copy import deepcopy

import pytest
import torch

from mosaickv.evaluation.donor_isolation import donor_slots, make_masks


def _transaction():
    return dict(
        mandatory_positions=[3],
        delimiter_positions=[2, 3],
        changed_routes=1,
        total_replaced_bytes=64,
        events=[
            dict(
                layer=0,
                kv_head=0,
                removed_positions=[0],
                added_positions=[2],
                position_bytes=64,
                replaced_bytes=64,
            )
        ],
    )


def test_saved_original_donor_maps_to_physical_slot_without_new_selection():
    assert donor_slots([[[0, 1, 3], [1, 2, 3]]], _transaction()) == [[[0], []]]


@pytest.mark.parametrize("change", ["mandatory", "missing", "repeated", "bytes", "restore"])
def test_saved_transaction_fails_closed(change):
    tx = deepcopy(_transaction())
    if change == "mandatory":
        tx["events"][0]["removed_positions"] = [3]
    elif change == "missing":
        tx["events"][0]["removed_positions"] = [7]
    elif change == "repeated":
        tx["events"] *= 2
    elif change == "bytes":
        tx["total_replaced_bytes"] += 1
    else:
        tx["events"][0]["added_positions"] = [1]
    with pytest.raises(ValueError):
        donor_slots([[[0, 1, 3]]], tx)


def test_gqa_mask_isolation_zero_control_and_physical_gather_equivalence():
    torch.manual_seed(19)
    original = (torch.zeros((1, 1, 1, 5)),)
    saved = original[0].clone()
    slots = [[[0], [1]]]
    zero = make_masks(original, slots, [4], block=False)[0]
    mask = make_masks(original, slots, [4], block=True)[0]
    assert torch.equal(original[0], saved) and torch.equal(zero, saved.expand_as(zero))
    assert mask.shape == (1, 4, 1, 5)
    assert torch.isneginf(mask[0, :2, 0, 0]).all()
    assert torch.isneginf(mask[0, 2:, 0, 1]).all()
    assert torch.equal(mask[..., 3:], torch.zeros_like(mask[..., 3:]))
    scores, values = torch.randn((1, 4, 1, 5)), torch.randn((1, 4, 5, 3))
    probabilities = (scores + mask).softmax(-1)
    actual = probabilities @ values
    for h in range(4):
        keep = [i for i in range(5) if i != (0 if h < 2 else 1)]
        expected = scores[:, h, :, keep].softmax(-1) @ values[:, h, keep, :]
        torch.testing.assert_close(actual[:, h], expected, rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(probabilities[0, :2, 0, 0]) == 0
    assert torch.count_nonzero(probabilities[0, 2:, 0, 1]) == 0


@pytest.mark.parametrize("heads,slots", [(3, [[[0], []]]), (2, [[[6]]]), (2, [[[0, 0]]])])
def test_invalid_gqa_or_slots_rejected(heads, slots):
    with pytest.raises(ValueError):
        make_masks((torch.zeros((1, 1, 1, 5)),), slots, [heads], block=True)
