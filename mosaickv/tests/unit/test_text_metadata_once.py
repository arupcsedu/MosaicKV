import random

import pytest

from mosaickv.evaluation.text_source_storage import text_slots


def legacy(positions, text):
    return tuple(tuple({p: j for j, p in enumerate(h)}.get(t, -1) for t in text) for h in positions)


@pytest.mark.parametrize("n", [1, 17, 300, 600, 604])
def test_same_slots_and_boolean_mask_as_original_for_different_head_orders(n):
    rng = random.Random(113)
    heads = tuple(tuple(rng.sample(range(n * 2), n)) for _ in range(8))
    text = tuple(sorted(rng.sample(range(n * 3), min(n, 28))))
    assert text_slots(heads, text) == legacy(heads, text)
    assert [[p in text for p in h] for h in heads] == [
        [p in frozenset(text) for p in h] for h in heads
    ]


@pytest.mark.parametrize(
    "positions,text",
    [
        ([], ()),
        (((1, 1),), (1,)),
        (((-1,),), (1,)),
        (((True,),), (1,)),
        (((1,), (1, 2)), (1,)),
        (((1,),), (2, 1)),
    ],
)
def test_original_validation_remains_fail_closed(positions, text):
    with pytest.raises(ValueError):
        text_slots(positions, text)


@pytest.mark.parametrize("merged", [False, True])
def test_common_constructor_packet_bits_unchanged_with_legacy_indexer(monkeypatch, merged):
    import torch
    from test_text_source_storage import setup

    from mosaickv.evaluation import text_source_storage

    packet, positions, text, record, attention = setup(merged)
    candidate, proof = text_source_storage.optimize(packet, record, positions, text, attention)
    monkeypatch.setattr(text_source_storage, "text_slots", legacy)
    reference, original_proof = text_source_storage.optimize(
        packet, record, positions, text, attention
    )
    assert proof == original_proof
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(candidate.tensors(), reference.tensors(), strict=True)
    )
