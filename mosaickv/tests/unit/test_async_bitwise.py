import pytest
import torch
from test_text_source_storage import setup

from mosaickv.evaluation import text_source_storage
from mosaickv.evaluation.async_bitwise import match_flags


@pytest.mark.parametrize("change", [False, True])
def test_same_bitwise_predicate_for_strided_bf16_and_special_patterns(change):
    a = torch.randn((1, 2, 34, 128)).bfloat16()[:, :, ::2]
    a[0, 0, 0, :4].view(torch.int16).copy_(torch.tensor([0, -32768, 32704, -64], dtype=torch.int16))
    b = a.clone()
    if change:
        b.view(torch.int16).flatten()[-1] ^= 1
    assert bool(match_flags(a, b).all()) == torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_empty_and_invalid_shapes_fail_explicitly():
    x = torch.empty((1, 2, 0, 128), dtype=torch.bfloat16)
    assert bool(match_flags(x, x).all())
    for wrong in (
        x.float(),
        torch.empty((1, 2, 1, 128), dtype=torch.bfloat16),
        torch.empty((2, 0, 128), dtype=torch.bfloat16),
    ):
        with pytest.raises(ValueError):
            match_flags(x, wrong)


@pytest.mark.parametrize("failure_pass", [0, 1, 2])
def test_every_independent_reconstruction_is_verified(monkeypatch, failure_pass):
    packet, positions, text, record, attention = setup()
    original = text_source_storage.TextSourceFormat.materialize
    calls = []

    def materialize(self, module):
        result = original(self, module)
        calls.append(len(calls))
        # Call zero builds the sparse patch. Calls one--three MUST independently verify.
        if len(calls) == failure_pass + 2:
            result[0].view(torch.int16).flatten()[-1] ^= 1
        return result

    monkeypatch.setattr(text_source_storage.TextSourceFormat, "materialize", materialize)
    with pytest.raises(RuntimeError, match="existing decoded K/V bit"):
        text_source_storage.optimize(packet, record, positions, text, attention)
    assert len(calls) == 4
