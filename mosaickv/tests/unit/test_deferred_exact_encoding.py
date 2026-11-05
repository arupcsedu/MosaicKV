import gc
import weakref
from dataclasses import dataclass, replace

import pytest
import torch

from mosaickv.evaluation import deferred_exact_encoding as deferred
from mosaickv.evaluation.exact_storage import decoded, encode


def test_bitwise_special_values_and_owned_raw_intermediate():
    original = (
        torch.arange(-32768, 32768, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(1, 1, 512, 128)
    )
    old = encode(original)
    with deferred.use(True) as state:
        temporary = encode(original)
        assert isinstance(temporary, deferred.DeferredBF16)
        assert temporary.buffers[0].data_ptr() != original.data_ptr()
        assert torch.equal(decoded(temporary).view(torch.int16), original.view(torch.int16))
        final = deferred.finalize(temporary)
        assert type(final) is type(old)
        assert final.raw is old.raw
        assert all(
            torch.equal(x.view(torch.uint8), y.view(torch.uint8))
            for x, y in zip(final.buffers, old.buffers, strict=True)
        )
        assert state["created"] == state["encoded"] == 1


def test_aliases_stay_shared_and_discarded_bodies_are_not_encoded():
    @dataclass(frozen=True)
    class Box:
        bodies: object
        tensor: object

    tensor = torch.ones(1024, dtype=torch.bfloat16)
    with deferred.use(True) as state:
        dead = encode(tensor)
        reference = weakref.ref(dead.buffers[0])
        del dead
        gc.collect()
        assert reference() is None
        a = encode(tensor)
        nested = Box(dict(a=a, b=(a, [a])), tensor)
        final = deferred.finalize(nested)
        assert final.bodies["a"] is final.bodies["b"][0] is final.bodies["b"][1][0]
        assert final.tensor is tensor
        assert deferred.finalize(final) is final
        assert state["created"] == 2 and state["encoded"] == 1
        # A later alias uses a weak cache, not a second encode.
        assert deferred.finalize(a) is final.bodies["a"]
        assert state["encoded"] == 1


def test_context_restoration_and_bound(monkeypatch):
    monkeypatch.setattr(deferred, "MAX_WORDS", 16)
    with pytest.raises(RuntimeError), deferred.use(True):
        assert not isinstance(encode(torch.ones(17, dtype=torch.bfloat16)), deferred.DeferredBF16)
        with pytest.raises(ValueError), deferred.use(True):
            pass
        raise RuntimeError("abort")
    assert not isinstance(encode(torch.ones(8, dtype=torch.bfloat16)), deferred.DeferredBF16)
    with pytest.raises(ValueError), deferred.use(1):
        pass


def test_funded_correction_budget_uses_actual_encoded_bytes():
    from mosaickv.evaluation.precision_corrections import fund_corrections
    from mosaickv.evaluation.precision_tier import construct

    torch.manual_seed(4)
    pair = tuple(torch.randn(1, 2, 140, 128).bfloat16() for _ in (0, 1))
    old, report = fund_corrections(
        construct(pair, bits=3, encoded_exact=True), pair, fraction=0.005
    )
    with deferred.use(True):
        pending = construct(pair, bits=3, encoded_exact=True)
        new, actual = fund_corrections(pending, pair, fraction=0.005)
        new = deferred.finalize(new)
    assert report == actual
    assert all(torch.equal(a, b) for a, b in zip(old.private(), new.private(), strict=True))


def test_retirement_finalizes_before_storage_validation():
    from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache, lossless_kv_encoder

    pair = tuple(torch.randn(1, 2, 16, 128).bfloat16() for _ in (0, 1))
    attention = torch.nn.Linear(2, 2)
    old = RetiringPrefillCache(1, audit=False)
    old.update(*pair, 0)
    old.seal(0, attention, lossless_kv_encoder)
    new = RetiringPrefillCache(1, audit=False)
    new.update(*pair, 0)
    with deferred.use(True) as state:
        new.seal(0, attention, lossless_kv_encoder)
        assert state["encoded"] == 2
        assert not any(isinstance(b, deferred.DeferredBF16) for b in new.formats()[0].pair)
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(old.formats()[0].tensors(), new.formats()[0].tensors(), strict=True)
    )
    assert new.record()["final_prefill_cache_bytes"] == old.record()["final_prefill_cache_bytes"]


def test_shared_source_discarded_overlay_has_identical_final_words():
    from test_full_prompt_source import example

    from mosaickv.evaluation.direct_text_construction import build
    from mosaickv.evaluation.selective_text_construction import use as selective

    attention, old = example(29)
    old = replace(old, protected_patches_disjoint=True)
    pair = old.materialize(attention)
    positions = (tuple(range(old.shape[2])),) * old.shape[1]
    text = (0,)
    old = build(old, pair, positions, text, ("encoded", "resident"), attention=attention)
    with deferred.use(True) as state, selective(True):
        attention, pending = example(29)
        pending = replace(pending, protected_patches_disjoint=True)
        new = build(pending, pair, positions, text, ("encoded", "resident"), attention=attention)
        new = deferred.finalize(new)
        assert state["created"] > state["encoded"] > 0
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(old.tensors(), new.tensors(), strict=True)
    )
