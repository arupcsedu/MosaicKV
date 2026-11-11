"""Real delta machinery over explicit synthetic CPU allocator fixtures."""

from copy import deepcopy

import pytest
import torch

from mosaickv.evaluation import live_owner_snapshot_cache as cache_module
from mosaickv.evaluation.kv_allocator_ledger import AllocatorLedger, verify_ownership_deltas


@pytest.fixture
def ledger(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    policy = "backend:native,expandable_segments:True"
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", policy)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", policy)
    first, second = torch.empty(64, dtype=torch.uint8), torch.empty(128, dtype=torch.uint8)
    locations = {first.untyped_storage()._cdata: 1000, second.untyped_storage()._cdata: 2000}
    snapshots = []

    def metadata(tensor):
        base = tensor if tensor._base is None else tensor._base
        storage = tensor.untyped_storage()
        return cache_module.OwnerMetadata(
            representative=base,
            device=0,
            storage_cdata=storage._cdata,
            pointer=locations[storage._cdata],
            nbytes=storage.nbytes(),
            view=(
                tensor.storage_offset(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                str(tensor.dtype),
                tensor.element_size(),
            ),
        )

    def snapshot():
        snapshots.append(True)
        return [
            dict(
                device=0,
                address=address,
                total_size=512,
                is_expandable=True,
                blocks=[dict(size=512, requested_size=size, state="active_allocated")],
            )
            for address, size in ((1000, 64), (2000, 128))
        ]

    original = cache_module.LiveOwnerSnapshotCache

    def factory(**kwargs):
        return original(**kwargs, owner_metadata_fn=metadata, snapshot_fn=snapshot)

    monkeypatch.setattr(cache_module, "LiveOwnerSnapshotCache", factory)
    result = AllocatorLedger(device="cuda:0", constructor_snapshot_reuse=True)
    return result, first, second, snapshots


def populate(fixture):
    ledger, first, second, snapshots = fixture
    phase = "prefill_layer_0_constructor_workspace"
    ledger.observe((first,), phase="request_workspace_created")
    ledger.observe((first,), phase=phase)
    for _ in range(3):
        ledger.observe((first,), phase=phase)
    ledger.observe((first, second), phase=phase)
    ledger.observe((first, second), phase=phase)
    ledger.observe((first,), phase=phase)
    ledger.observe((first,), phase="prefill_complete")
    return ledger, snapshots


def test_every_delta_and_exact_bytes_survive_constructor_only_reuse(ledger):
    instance, snapshots = populate(ledger)
    report = instance.summary()
    assert report["observations"] == 9
    assert len(report["ownership_deltas"]) == 9
    assert len(snapshots) == 5
    assert report["initial_allocator_block_bytes"] == 512
    assert report["final_allocator_block_bytes"] == 512
    assert report["peak_allocator_block_bytes"] == 1024
    assert report["peak_tensor_backing_bytes"] == 192
    assert report["snapshot_cache"]["fresh_native_snapshots"] == 5
    assert report["snapshot_cache"]["reused_actual_resolutions"] == 4
    checked = verify_ownership_deltas(report["ownership_deltas"], expected_summary=report)
    assert checked["ownership_deltas_verified"] is True
    assert checked["snapshot_cache"] == report["snapshot_cache"]


@pytest.mark.parametrize("index", [0, 2, 8])
def test_resolution_proof_cannot_disappear_from_any_observation(ledger, index):
    instance, _ = populate(ledger)
    report = instance.summary()
    deltas = deepcopy(report["ownership_deltas"])
    del deltas[index]["snapshot_cache"]
    with pytest.raises(ValueError, match="proof"):
        verify_ownership_deltas(deltas, expected_summary=report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_observation_index", 777),
        ("owner_view_fingerprint_sha256", "0" * 64),
        ("snapshot_calls", 100),
        ("prior_weak_representatives_alive", False),
    ],
)
def test_reused_map_provenance_is_rechecked_not_ignored(ledger, field, value):
    instance, _ = populate(ledger)
    report = instance.summary()
    deltas = deepcopy(report["ownership_deltas"])
    deltas[2]["snapshot_cache"][field] = value
    with pytest.raises(ValueError):
        verify_ownership_deltas(deltas, expected_summary=report)
