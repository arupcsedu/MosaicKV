import weakref
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import streaming_prefill as mod
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


def packet(*tensors):
    return SimpleNamespace(tensors=lambda: tensors)


def test_shared_views_extra_owners_and_layer_replacement_match_full_enumeration():
    cache = mod.RetiringPrefillCache(2, audit=False)
    body = torch.ones(128)
    extra = torch.ones(16)
    cache.layers[0].packet = packet(body[:32])
    cache.layer_owners_changed(0)
    cache.observe()
    cache.layers[1].packet = packet(body[32:])
    cache.layer_owners_changed(1)
    cache.observe((body[:1],))
    assert cache.peak_bytes == backing_bytes((body,))
    cache.extra_tensors = (extra, body[:4])
    cache.observe()
    assert cache.record(require_complete=False)["final_prefill_cache_bytes"] == backing_bytes(
        (body, extra)
    )
    replacement = torch.ones(7)
    cache.layers[0].packet = packet(replacement)
    cache.layer_owners_changed(0)
    cache.observe()
    assert cache.record(require_complete=False)["final_prefill_cache_bytes"] == backing_bytes(
        (body, extra, replacement)
    )


def test_unannounced_change_fails_and_generic_invalidation_restores_correctness():
    cache = mod.RetiringPrefillCache(1, audit=False)
    cache.layers[0].packet = packet(torch.ones(9))
    cache.owners_changed()
    cache.observe()
    cache.layers[0].packet = packet(torch.ones(10))
    with pytest.raises(RuntimeError, match="independent enumeration"):
        cache.record(require_complete=False)
    cache.owners_changed()
    cache.observe()
    assert cache.record(require_complete=False)["final_prefill_cache_bytes"] == 40


def test_cached_segments_do_not_keep_dead_tensor_owners():
    cache = mod.RetiringPrefillCache(1, audit=False)
    tensor = torch.ones(7)
    ref = weakref.ref(tensor)
    cache.layers[0].packet = packet(tensor)
    cache.layer_owners_changed(0)
    cache.observe()
    cache.layers[0].packet = None
    cache.layer_owners_changed(0)
    del tensor
    assert ref() is None
    cache.observe()
    assert cache.record(require_complete=False)["final_prefill_cache_bytes"] == 0


def test_layerwise_identity_work_is_linear_not_a_full_prefix_rescan(monkeypatch):
    original = mod.identity

    def count(layers):
        calls = 0

        def spy(tensor):
            nonlocal calls
            calls += 1
            return original(tensor)

        monkeypatch.setattr(mod, "identity", spy)
        cache = mod.RetiringPrefillCache(layers, audit=False)
        for i in range(layers):
            cache.layers[i].packet = packet(torch.ones(5), torch.ones(6))
            cache.layer_owners_changed(i)
            cache.observe()
        cache.record(require_complete=False)
        return calls

    short, long = count(4), count(12)
    assert long <= 3.1 * short


@pytest.mark.parametrize("index", [True, -1, 2, 1.0])
def test_invalid_changed_layer_rejected(index):
    with pytest.raises(ValueError, match="valid changed"):
        mod.RetiringPrefillCache(2).layer_owners_changed(index)
