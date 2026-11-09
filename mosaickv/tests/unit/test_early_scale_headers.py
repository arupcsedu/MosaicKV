import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
from test_exact_scale_replay import packets

from mosaickv.evaluation.early_scale_headers import (
    ArchivedScaleBody,
    Placement,
    archive_packets,
    restore_packets,
)
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


def test_prepared_native_share_exact_archive_and_restore_owner(monkeypatch):
    packet = packets(monkeypatch)[0]
    expected = [b.scale.view(torch.int32).clone() for b in packet.pair]
    refs = [weakref.ref(b.scale) for b in packet.pair]
    (a, b), records = archive_packets((packet, packet))
    assert len(records) == 2
    assert all(x is y for x, y in zip(a.pair, b.pair, strict=True))
    assert all(isinstance(x, ArchivedScaleBody) and x.scale.numel() == 0 for x in a.pair)
    assert backing_bytes(a.tensors()) < backing_bytes(packet.tensors())
    del packet
    gc.collect()
    assert all(r() is None for r in refs)
    (ra, rb), count = restore_packets((a, b))
    assert count == 2
    for i, (x, y) in enumerate(zip(ra.pair, rb.pair, strict=True)):
        assert x is y and type(x).__name__ == "QuantizedBody"
        assert torch.equal(x.scale.view(torch.int32), expected[i])
        assert x.codes is a.pair[i].codes and x.minimum is a.pair[i].minimum
    again, repeated = restore_packets((ra, rb))
    assert repeated == 0 and again[0].pair[0] is ra.pair[0]


def test_publication_and_final_restore_invalidate_storage_catalog(monkeypatch):
    originals = packets(monkeypatch)
    pipeline = SimpleNamespace(deferred_text=True, prepared={0: (originals[0], originals[0], 0)})
    changes = []
    cache = SimpleNamespace(
        layers=[SimpleNamespace(packet=originals[0])],
        layer_owners_changed=lambda i: changes.append(i),
    )
    state = Placement()
    state.before(pipeline, cache, False)
    state.after(pipeline, cache, False)
    assert state.report["packed_layers"] == 1
    assert cache.layers[0].packet is pipeline.prepared[0][0]
    state.after(pipeline, cache, False)
    assert state.report["packed_layers"] == 1  # No repeated recoding.
    state.before(pipeline, cache, True)
    assert state.report["downstream_headers_restored"]
    assert state.report["restored_bodies"] == 2 and not state.packed
    assert changes == [0, 0]
    assert cache.layers[0].packet is pipeline.prepared[0][0]
    assert cache.layers[0].packet.pair[0].scale.equal(originals[0].pair[0].scale)


def test_wrong_owner_or_packet_replacement_rejected(monkeypatch):
    original = packets(monkeypatch)[0]
    pipeline = SimpleNamespace(deferred_text=True, prepared={0: (original, original, 0)})
    cache = SimpleNamespace(
        layers=[SimpleNamespace(packet=original)], layer_owners_changed=lambda i: None
    )
    state = Placement()
    with pytest.raises(ValueError, match="deferred"):
        state.before(SimpleNamespace(deferred_text=False), cache, False)
    state.before(pipeline, cache, False)
    with pytest.raises(RuntimeError, match="cross request"):
        state.before(SimpleNamespace(deferred_text=True), cache, False)
    cache.layers[0].packet = object()
    with pytest.raises(RuntimeError, match="actually published"):
        state.after(pipeline, cache, False)
