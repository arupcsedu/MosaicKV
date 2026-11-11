"""CPU callbacks see exact live backing and phase names, without CUDA audit."""

import pytest
import torch

from mosaickv.evaluation import kv_allocator_ledger
from mosaickv.evaluation.bounded_component_cache import ComponentCache
from mosaickv.evaluation.bounded_ragged_raw import RaggedRawPacket
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


def _owners(tensors):
    return frozenset(
        (str(t.device), t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        for t in tensors
        if t.untyped_storage().nbytes()
    )


def _packet(length):
    pairs, positions = [], []
    for head in range(32):
        count = length - head % 2
        joint = torch.zeros((2, count, 128), dtype=torch.bfloat16)
        pairs.append((joint[0], joint[1]))
        positions.append(torch.arange(count, dtype=torch.long))
    return RaggedRawPacket(pairs, positions, original_length=length, prompt_capacity=length)


def test_external_observer_covers_exact_owner_sets_and_phases_with_allocator_disabled(monkeypatch):
    def no_internal_auditor(**kwargs):
        raise AssertionError("External CPU callback must not instantiate a CUDA allocator auditor")

    monkeypatch.setattr(kv_allocator_ledger, "AllocatorLedger", no_internal_auditor)
    records = []

    def observe(tensors, phase):
        # Never retain strong tensors or extend old storage lifetimes.
        records.append((phase, _owners(tensors), backing_bytes(tensors)))

    metadata = torch.arange(512, dtype=torch.long)
    cache = ComponentCache(
        (_packet(3), _packet(5)),
        updates=32,
        metadata=(metadata,),
        allocator_audit=False,
        ownership_observer=observe,
    )
    assert cache.allocator_ledger is None
    assert records == [("decode:initial_cache", _owners(cache.shared_tensors), cache.shared_bytes)]
    shared = _owners(cache.shared_tensors)
    cache._mask_dispatch_active = True  # CPU accounting only, not a model transport bypass.
    for step in range(1, 18):
        for index, layer in enumerate(cache.layers):
            before = len(records)
            old_history = _owners(layer.history)
            old_capacity = layer.history_capacity
            key = torch.full((1, 32, 1, 128), step, dtype=torch.bfloat16)
            value = torch.full_like(key, -step)
            incoming = _owners((key, value))
            other_history = _owners(
                t for other in cache.layers if other is not layer for t in other.owned_tensors()
            )
            layer.update(key, value)
            new_history = _owners(layer.history)
            events = records[before:]
            expected = shared | other_history | new_history | incoming
            if layer.history_capacity != old_capacity:
                assert len(events) == 2
                assert events[0][0] == f"decode:decode_update_{step}_layer_{index}_growth_overlap"
                assert events[0][1] == expected | old_history
                assert events[1][1] == expected
                assert not old_history & events[1][1]
            else:
                assert len(events) == 1
                assert events[0][1] == expected
            assert events[-1][0] == f"decode:decode_update_{step}_layer_{index}_copied_history"
            layer.release()
            cache.observe()
            assert records[-1][0] == "decode:final_cache"
            assert records[-1][1] == shared | other_history | new_history
            assert not incoming & records[-1][1]
    cache._mask_dispatch_active = False
    report = cache.storage_record()
    assert report["allocator_blocks"] is None
    assert report["observations"] == len(records)
    assert report["initial_active_kv_storage_bytes"] == records[0][2]
    assert report["final_active_kv_storage_bytes"] == records[-1][2]
    assert report["peak_active_kv_storage_bytes"] == max(row[2] for row in records)
    assert report["whole_request_peak_certified"] is False


def test_observer_failure_propagates_instead_of_silently_omitting_ownership():
    def fail(tensors, phase):
        raise RuntimeError("synthetic ownership callback failure")

    with pytest.raises(RuntimeError, match="callback failure"):
        ComponentCache((_packet(3),), allocator_audit=False, ownership_observer=fail)
