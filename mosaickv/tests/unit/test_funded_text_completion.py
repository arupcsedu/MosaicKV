from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import funded_text_completion as completion
from mosaickv.evaluation.head_local_precision import localize
from mosaickv.evaluation.precision_tier import construct


def test_additions_preserve_every_token_without_filler_or_donors():
    positions = ((0, 1, 7, 8, 9), (0, 1, 2, 7, 9))
    assert completion.additions(positions, (7, 8, 9), 10) == ((), (8,))


def test_invalid_logical_positions_or_missing_mandatory_are_rejected():
    with pytest.raises(ValueError):
        completion.additions(((0, 1, 2, 11),), (8, 9), 10)
    with pytest.raises(ValueError):
        completion.additions(((0, 1, 2, 8),), (8, 9), 10)


def test_generation_delimiter_requires_the_exact_tokenized_suffix():
    processor = SimpleNamespace(encode=lambda text, **kwargs: [5, 6, 7])
    prepared = SimpleNamespace(
        model_inputs={"input_ids": torch.tensor([[2, 3, 5, 6, 7]])},
        modality_map=[SimpleNamespace(modality="text", start=0, end=5)],
    )
    assert completion.generation_delimiter(prepared, processor) == (2, 3, 4)
    prepared.model_inputs["input_ids"][0, -1] = 8
    with pytest.raises(NotImplementedError):
        completion.generation_delimiter(prepared, processor)


def fixture():
    torch.manual_seed(426)
    h, n, d = 4, 64, 128
    hidden = torch.randn(1, n, h * d).bfloat16()
    attention = SimpleNamespace(
        head_dim=d,
        k_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
        v_proj=torch.nn.Linear(h * d, h * d, bias=False).bfloat16(),
    )
    with torch.inference_mode():
        source = tuple(
            p(hidden).reshape(1, n, h, d).transpose(1, 2).contiguous()
            for p in (attention.k_proj, attention.v_proj)
        )
    text = tuple((*range(2), *range(52, 64)))
    original = tuple((*range(36), *range(52, 64)))
    changed = tuple(sorted((set(original) - {62}) | {40}))
    positions = (original, changed, original, original)
    index = torch.tensor(positions)[None, :, :, None].expand(1, h, 48, d)
    selected = tuple(t.gather(2, index) for t in source)
    base = construct(selected, bits=3, encoded_exact=True)
    packet = localize(base, selected, positions, set(text), policy="head3_boundary8")
    record = dict(
        normalized=hidden[:, text].contiguous(),
        cos=torch.ones(1, len(text), d).bfloat16(),
        sin=torch.zeros(1, len(text), d).bfloat16(),
    )
    p = torch.ones(1, h, 2, n)
    p[:, :, :, 36] = 9
    return packet, record, positions, text, attention, source, p


def test_old_decoded_values_are_bitwise_preserved_and_physical_rows_sorted():
    packet, record, positions, text, attention, source, _p = fixture()
    result, final_positions, proof = completion.build_layer(
        packet, record, positions, text, (60, 61, 62, 63), attention, source
    )
    assert proof["extra_physical_capacity_per_head"] == 1
    assert proof["physical_padding_slots"] == 3
    assert proof["no_donor_removal"]
    assert proof["virtual_zero_padding_materialized_and_counted_in_decode_peak"]
    assert all(
        set(a).issubset(b) and {60, 61, 62, 63}.issubset(b)
        for a, b in zip(positions, final_positions, strict=True)
    )
    assert all(
        len(p) == 49
        and tuple(sorted(x for x in p if x is not None)) == tuple(x for x in p if x is not None)
        for p in final_positions
    )
    before, after = packet.materialize(attention), result.materialize(attention)
    old_indices = torch.tensor(
        [[row.index(v) for v in old] for row, old in zip(final_positions, positions, strict=True)]
    )[None, :, :, None].expand_as(before[0])
    for a, b in zip(before, after, strict=True):
        assert torch.equal(a.view(torch.int16), b.gather(2, old_indices).view(torch.int16))
    targets = tuple(torch.full((1, 4, 60, 128), -73).bfloat16() for _ in range(2))
    result.fill(attention, targets, direct=True)
    assert all(
        torch.equal(t[:, :, :49].view(torch.int16), a.view(torch.int16))
        and (t[:, :, 49:] == -73).all()
        for t, a in zip(targets, after, strict=True)
    )


def test_global_cap_allows_shared_funding_but_never_exceeds_anchor(monkeypatch):
    class Format:
        def __init__(self, n):
            self.body = torch.zeros(n, dtype=torch.uint8)

        def tensors(self):
            return (self.body,)

    packets = (Format(10), Format(20))
    proposed = [Format(12), Format(10)]
    monkeypatch.setattr(completion, "build_layer", lambda *args: (proposed.pop(0), ((0,),), {}))
    selected, proof = completion.transaction(
        packets, {0: None, 1: None}, [[], []], (), (), [None, None], [None, None]
    )
    assert proof["proposed_completion_accepted"] and proof["final_physical_bytes"] == 22
    proposed[:] = [Format(20), Format(20)]
    monkeypatch.setattr(completion, "optimize", lambda packet, *args: (packet, {}))
    selected, proof = completion.transaction(
        packets, {0: None, 1: None}, [[], []], (), (), [None, None], [None, None]
    )
    assert not proof["proposed_completion_accepted"]
    assert selected == packets
    assert (
        proof["final_physical_bytes"] == 30
        and proof["fallback"] == "source_only_no_capacity_increase"
    )
