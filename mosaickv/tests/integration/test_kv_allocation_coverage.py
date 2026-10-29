"""Explicit old/new cache storage, weak ownership, and hook restoration."""

import gc

import torch
from test_hf_tiny_models import _llava
from transformers import DynamicCache

from mosaickv.adapters.huggingface.layer_local_cache import gather_layer_local_snapshot
from mosaickv.adapters.huggingface.preallocated_cache import build_preallocated_dynamic_cache
from mosaickv.measurements.kv_ownership import KVOwnershipLedger
from mosaickv.measurements.kv_transactions import DynamicKVTransactions


def test_preallocation_includes_complete_old_and_new_backing_storage():
    keys = torch.zeros(1, 1, 3, 4)
    values = torch.ones_like(keys)
    ledger = KVOwnershipLedger()
    with DynamicKVTransactions(ledger) as observer:
        initial = DynamicCache.from_legacy_cache(((keys, values),))
        observer.begin_decode(initial)
        cache = build_preallocated_dynamic_cache(((keys, values),), 5)
        observer.observe_decode_cache(cache)
        assert observer.decode_ledger.observe() == 8 * 4 * 4 * 2
        assert observer.decode_ledger.peak_storage_bytes == (3 + 8) * 4 * 4 * 2
    assert ledger.peak_storage_bytes == (3 + 8) * 4 * 4 * 2
    assert (
        observer.record()["explicit_cache_allocation_operations"][
            "append_preallocation_destination"
        ]
        == 1
    )
    torch.testing.assert_close(cache.layers[0].keys, keys)


def test_clone_snapshot_counts_both_owners_and_does_not_prevent_source_retirement():
    adapter = _llava()
    keys = torch.zeros(1, 1, 3, 4)
    values = torch.ones_like(keys)
    cache = DynamicCache.from_legacy_cache(((keys, values),))
    ledger = KVOwnershipLedger()
    ledger.track("prefill", ((keys, values),))
    with DynamicKVTransactions(ledger):
        snapshot = adapter.extract_past_key_values(cache)
    assert ledger.peak_storage_bytes == 3 * 4 * 4 * 2 * 2
    del cache, keys, values
    gc.collect()
    assert ledger.observe() == 3 * 4 * 4 * 2
    assert snapshot.layers[0].key.shape[-2] == 3


def test_layer_local_packing_counts_source_and_all_partial_destinations():
    adapter = _llava()
    pairs = tuple((torch.zeros(1, 1, 8, 4), torch.ones(1, 1, 8, 4)) for _ in range(2))
    cache = DynamicCache.from_legacy_cache(pairs)
    ledger = KVOwnershipLedger()
    with DynamicKVTransactions(ledger) as observer:
        packed = gather_layer_local_snapshot(adapter, cache, (((0, 2, 7),), ((0, 4),)))
    assert ledger.peak_storage_bytes == (8 + 8 + 3 + 2) * 4 * 4 * 2
    assert (
        observer.record()["explicit_cache_allocation_operations"]["layer_local_pack_destination"]
        == 2
    )
    assert [x.key.shape[-2] for x in packed.layers] == [3, 2]
