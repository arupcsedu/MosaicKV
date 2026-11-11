"""CPU-only fixtures for the unused opt-in constructor snapshot draft."""

import gc
import weakref
from dataclasses import replace

import pytest
import torch

from mosaickv.evaluation.live_owner_snapshot_cache import (
    LiveOwnerSnapshotCache,
    OwnerMetadata,
    cuda_metadata,
    resolution_mapping_sha256,
    selftest,
    storage_representative,
)

PHASE = "prefill_layer_0_constructor_workspace"


class Owner:
    def __init__(self, pointer=4096, nbytes=126, cdata=1):
        self.pointer, self.nbytes, self.cdata = pointer, nbytes, cdata


class View:
    def __init__(self, owner, *, offset=0, shape=(63,), stride=(1,)):
        self.owner, self.offset, self.shape, self.stride = owner, offset, shape, stride


def metadata(tensor):
    view = tensor if isinstance(tensor, View) else View(tensor)
    owner = view.owner
    return OwnerMetadata(
        owner,
        0,
        owner.cdata,
        owner.pointer,
        owner.nbytes,
        (view.offset, view.shape, view.stride, "torch.bfloat16", 2),
    )


class Snapshots:
    def __init__(self, *owners):
        self.calls = 0
        # Store scalar geometry, never the fixture owners themselves.
        self.geometry = [(owner.pointer, owner.nbytes) for owner in owners]
        self.state = "active_allocated"
        self.requested_override = None
        self.fail = False

    def __call__(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("injected snapshot failure")
        return [
            dict(
                device=0,
                address=pointer,
                total_size=512,
                blocks=[
                    dict(
                        size=512,
                        requested_size=(
                            nbytes if self.requested_override is None else self.requested_override
                        ),
                        state=self.state,
                    )
                ],
            )
            for pointer, nbytes in self.geometry
        ]


def configured(*owners, enabled=True, provider=metadata):
    snapshots = Snapshots(*owners)
    return LiveOwnerSnapshotCache(
        device=0,
        enabled=enabled,
        owner_metadata_fn=provider,
        snapshot_fn=snapshots,
    ), snapshots


def mode(record):
    return record["snapshot_cache"]["mode"]


def test_disabled_by_default_and_every_observation_is_preserved():
    owner = Owner()
    snapshots = Snapshots(owner)
    cache = LiveOwnerSnapshotCache(device=0, owner_metadata_fn=metadata, snapshot_fn=snapshots)
    for _ in range(4):
        assert mode(cache.resolve([owner], PHASE)) == "fresh_native_snapshot"
    assert snapshots.calls == len(cache.observations) == 4
    assert [r["observation_index"] for r in cache.observations] == list(range(4))
    assert not any(row["observation_skipped"] for row in cache.observations)


def test_identical_constructor_reuses_actual_block_not_rounded_owner_size():
    owner = Owner()
    cache, snapshots = configured(owner)
    first = cache.resolve([owner], PHASE)
    second = cache.resolve([owner], PHASE)
    assert snapshots.calls == 1
    assert first["current_tensor_backing_bytes"] == 126
    assert first["current_allocator_requested_bytes"] == 126
    assert second["current_allocator_block_bytes"] == 512
    assert second["blocks"] == first["blocks"]
    assert mode(second) == "reused_actual_resolution"
    assert second["snapshot_cache"]["source_observation_index"] == 0
    assert len(cache.observations) == 2


def test_alias_order_and_duplicates_do_not_change_owner_or_view_sets():
    owner = Owner()
    a, b = View(owner, shape=(20,)), View(owner, offset=20, shape=(20,))
    cache, snapshots = configured(owner)
    cache.resolve([a, b], PHASE)
    result = cache.resolve([b, a, a], PHASE)
    assert snapshots.calls == 1
    assert mode(result) == "reused_actual_resolution"
    assert result["unique_tensor_owners"] == 1


def test_new_equivalent_view_reuses_live_full_storage_base():
    owner = Owner()
    cache, snapshots = configured(owner)
    old_view = View(owner)
    cache.resolve([old_view], PHASE)
    reference = weakref.ref(old_view)
    del old_view
    gc.collect()
    assert reference() is None
    assert mode(cache.resolve([View(owner)], PHASE)) == "reused_actual_resolution"
    assert snapshots.calls == 1


@pytest.mark.parametrize("change", ["offset", "shape", "stride", "add", "remove"])
def test_view_descriptors_add_remove_and_offsets_force_snapshot(change):
    owner = Owner()
    a = View(owner, shape=(10,))
    cache, snapshots = configured(owner)
    original = [a, View(owner, offset=10, shape=(10,))] if change == "remove" else [a]
    cache.resolve(original, PHASE)
    changed = {
        "offset": [View(owner, offset=1, shape=(10,))],
        "shape": [View(owner, shape=(9,))],
        "stride": [View(owner, shape=(10,), stride=(2,))],
        "add": [a, View(owner, offset=10, shape=(10,))],
        "remove": [a],
    }[change]
    assert mode(cache.resolve(changed, PHASE)) == "fresh_native_snapshot"
    assert snapshots.calls == 2


def test_owner_add_remove_forces_fresh_even_when_total_bytes_identical():
    first, second = Owner(), Owner(8192, cdata=2)
    cache, snapshots = configured(first, second)
    cache.resolve([first], PHASE)
    assert mode(cache.resolve([second], PHASE)) == "fresh_native_snapshot"
    assert snapshots.calls == 2


def test_dead_owner_and_same_pointer_cdata_reuse_never_reuses_resolution():
    owner = Owner()
    cache, snapshots = configured(owner)
    cache.resolve([owner], PHASE)
    old = weakref.ref(owner)
    del owner
    gc.collect()
    assert old() is None, "cache must never extend backing-owner lifetime"
    replacement = Owner()
    result = cache.resolve([replacement], PHASE)
    assert snapshots.calls == 2
    assert result["snapshot_cache"]["reason"] == "previous_owner_died"


def test_distinct_live_representative_same_storage_also_forces_fresh():
    owner, alias = Owner(), Owner()
    cache, snapshots = configured(owner)
    cache.resolve([owner], PHASE)
    result = cache.resolve([alias], PHASE)
    assert snapshots.calls == 2
    assert result["snapshot_cache"]["reason"] == "owner_representative_or_storage_changed"


@pytest.mark.parametrize("field,new", [("cdata", 2), ("pointer", 8192), ("nbytes", 128)])
def test_in_place_storage_identity_changes_force_fresh(field, new):
    owner = Owner()
    cache, snapshots = configured(owner)
    cache.resolve([owner], PHASE)
    setattr(owner, field, new)
    snapshots.geometry = [(owner.pointer, owner.nbytes)]
    result = cache.resolve([owner], PHASE)
    assert mode(result) == "fresh_native_snapshot"
    assert snapshots.calls == 2


@pytest.mark.parametrize(
    "phase",
    [
        "prefill_layer_0_KV_published",
        "prefill_layer_0_native_retired",
        "prefill_complete",
        "decode_initial",
        "decode_final",
        "decode_growth_1",
        "prefill_layer_0_constructor_workspace_suffix",
        "prefill_layer_00_constructor_workspace",
    ],
)
def test_lifecycle_phases_always_fresh_and_invalidate_constructor_cache(phase):
    owner = Owner()
    cache, snapshots = configured(owner)
    cache.resolve([owner], PHASE)
    for _ in range(2):
        assert mode(cache.resolve([owner], phase)) == "fresh_native_snapshot"
    assert mode(cache.resolve([owner], PHASE)) == "fresh_native_snapshot"
    assert snapshots.calls == len(cache.observations) == 4


def test_constructor_layer_transition_requires_new_actual_snapshot():
    owner = Owner()
    cache, snapshots = configured(owner)
    cache.resolve([owner], PHASE)
    result = cache.resolve([owner], "prefill_layer_1_constructor_workspace")
    assert result["snapshot_cache"]["reason"] == "phase_transition"
    assert snapshots.calls == 2


@pytest.mark.parametrize("bad", ["missing", "inactive", "awaiting", "range", "geometry"])
def test_original_strict_snapshot_checks_fail_closed_and_discard_stale_state(bad):
    owner = Owner()
    cache, snapshots = configured(owner)
    cache.resolve([owner], PHASE)
    cache.invalidate()
    if bad == "missing":
        snapshots.geometry = []
    elif bad in {"inactive", "awaiting"}:
        snapshots.state = "inactive" if bad == "inactive" else "active_awaiting_free"
    elif bad == "range":
        snapshots.requested_override = 125
    else:
        snapshots.requested_override = 513
    with pytest.raises(ValueError):
        cache.resolve([owner], PHASE)
    assert cache.observations[-1]["outcome"] == "failed_closed"
    snapshots.geometry = [(4096, 126)]
    snapshots.state, snapshots.requested_override = "active_allocated", None
    assert mode(cache.resolve([owner], PHASE)) == "fresh_native_snapshot"
    assert len(cache.observations) == 3


def test_snapshot_exception_does_not_leave_a_reusable_resolution():
    owner = Owner()
    cache, snapshots = configured(owner)
    cache.resolve([owner], PHASE)
    snapshots.fail = True
    with pytest.raises(RuntimeError, match="injected"):
        cache.resolve([owner], "prefill_layer_0_KV_published")
    snapshots.fail = False
    assert mode(cache.resolve([owner], PHASE)) == "fresh_native_snapshot"
    assert len(cache.observations) == 3


def test_returned_evidence_and_journal_cannot_mutate_internal_resolution():
    owner = Owner()
    cache, snapshots = configured(owner)
    result = cache.resolve([owner], PHASE)
    result["blocks"][0]["size"] = 999999
    result["owners"][0][1] = 999999
    result["snapshot_cache"]["mode"] = "forged"
    journal = cache.observations
    journal[0]["phase"] = "forged"
    next_result = cache.resolve([owner], PHASE)
    assert next_result["blocks"][0]["size"] == 512
    assert next_result["owners"] == [[4096, 126]]
    assert next_result["snapshot_cache"]["resolution_mapping_sha256"] == resolution_mapping_sha256(
        next_result
    )
    assert cache.observations[0]["phase"] == PHASE
    assert snapshots.calls == 1


def test_resolution_hash_independent_of_phase_index_and_reuse_metadata():
    owner = Owner()
    cache, snapshots = configured(owner)
    first = cache.resolve([owner], PHASE)
    reused = cache.resolve([owner], PHASE)
    lifecycle = cache.resolve([owner], "prefill_layer_0_KV_published")
    hashes = [
        row["snapshot_cache"]["resolution_mapping_sha256"] for row in (first, reused, lifecycle)
    ]
    assert len(set(hashes)) == 1
    assert hashes[0] == resolution_mapping_sha256(first)
    assert snapshots.calls == 2
    lifecycle["blocks"][0]["requested_size"] -= 1
    assert resolution_mapping_sha256(lifecycle) != hashes[0]


def test_resolution_hash_includes_owner_geometry_and_all_measured_totals():
    owner = Owner()
    cache, _ = configured(owner)
    first = cache.resolve([owner], PHASE)
    original = resolution_mapping_sha256(first)
    first["owners"][0][1] -= 1
    assert resolution_mapping_sha256(first) != original
    first["owners"][0][1] += 1
    first["current_allocator_requested_bytes"] -= 1
    assert resolution_mapping_sha256(first) != original


def test_selftest_refuses_cpu_and_unavailable_cuda_without_fixture_allocations(monkeypatch):
    with pytest.raises(ValueError, match="available CUDA"):
        selftest(device="cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="available CUDA"):
        selftest(device="cuda")


def test_empty_ownership_retains_complete_zero_accounting():
    cache, snapshots = configured()
    cache.resolve([], PHASE)
    result = cache.resolve([], PHASE)
    assert result["current_allocator_block_bytes"] == result["current_tensor_backing_bytes"] == 0
    assert result["owners"] == result["blocks"] == []
    assert result["all_required_owners_resolved"] is True
    assert snapshots.calls == 1


@pytest.mark.parametrize("phase", [None, "", "x" * 257, 1])
def test_invalid_phase_is_a_recorded_failure(phase):
    cache, snapshots = configured()
    with pytest.raises(ValueError, match="phase"):
        cache.resolve([], phase)
    assert snapshots.calls == 0
    assert len(cache.observations) == 1
    assert cache.observations[0]["outcome"] == "failed_closed"


def test_metadata_failure_invalidates_cache_and_cannot_skip_observation():
    owner = Owner()
    invalid = False

    def provider(tensor):
        result = metadata(tensor)
        return replace(result, nbytes=True) if invalid else result

    cache, snapshots = configured(owner, provider=provider)
    cache.resolve([owner], PHASE)
    invalid = True
    with pytest.raises(ValueError, match="storage nbytes"):
        cache.resolve([owner], PHASE)
    invalid = False
    assert mode(cache.resolve([owner], PHASE)) == "fresh_native_snapshot"
    assert snapshots.calls == 2
    assert [r["outcome"] for r in cache.observations] == ["resolved", "failed_closed", "resolved"]


def cpu_metadata(tensor):
    storage = tensor.untyped_storage()
    return OwnerMetadata(
        storage_representative(tensor),
        0,
        storage._cdata,
        storage.data_ptr(),
        storage.nbytes(),
        (
            tensor.storage_offset(),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            str(tensor.dtype),
            tensor.element_size(),
        ),
    )


def test_actual_cpu_views_choose_full_base_and_do_not_keep_tensor_or_storage_alive():
    owner = torch.arange(32, dtype=torch.bfloat16)
    views = [owner[:16].view(2, 8), owner[16:]]
    assert all(storage_representative(view) is owner for view in views)
    pointer, nbytes = owner.untyped_storage().data_ptr(), owner.untyped_storage().nbytes()
    snapshots = Snapshots()
    snapshots.geometry = [(pointer, nbytes)]
    cache = LiveOwnerSnapshotCache(
        device=0,
        enabled=True,
        owner_metadata_fn=cpu_metadata,
        snapshot_fn=snapshots,
    )
    cache.resolve(views, PHASE)
    assert (
        mode(cache.resolve([owner[:16].view(2, 8), owner[16:]], PHASE))
        == "reused_actual_resolution"
    )
    references = [weakref.ref(tensor) for tensor in [owner, *views]]
    storage_reference = weakref.ref(owner.untyped_storage())
    del owner, views
    gc.collect()
    assert all(ref() is None for ref in references)
    assert storage_reference() is None


def test_actual_cpu_resize_changes_backing_and_forces_fresh():
    owner = torch.empty(16, dtype=torch.bfloat16)
    snapshots = Snapshots()
    snapshots.geometry = [(owner.untyped_storage().data_ptr(), owner.untyped_storage().nbytes())]
    cache = LiveOwnerSnapshotCache(
        device=0,
        enabled=True,
        owner_metadata_fn=cpu_metadata,
        snapshot_fn=snapshots,
    )
    cache.resolve([owner], PHASE)
    owner.resize_(32)
    snapshots.geometry = [(owner.untyped_storage().data_ptr(), owner.untyped_storage().nbytes())]
    assert mode(cache.resolve([owner], PHASE)) == "fresh_native_snapshot"
    assert snapshots.calls == 2


def test_production_default_rejects_cpu_and_never_calls_snapshot():
    snapshots = Snapshots()
    cache = LiveOwnerSnapshotCache(device=0, enabled=True, snapshot_fn=snapshots)
    with pytest.raises(ValueError, match="CUDA"):
        cache.resolve([torch.ones(1)], PHASE)
    with pytest.raises(ValueError, match="CUDA"):
        cuda_metadata(torch.ones(1), device=0)
    assert snapshots.calls == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("device", 1),
        ("storage_cdata", True),
        ("pointer", 0),
        ("view", (63, (1,), (1,), "torch.bfloat16", 2)),
        ("representative", object()),
    ],
)
def test_malformed_injected_metadata_fails_closed(field, value):
    owner = Owner()
    cache, snapshots = configured(owner, provider=lambda t: replace(metadata(t), **{field: value}))
    with pytest.raises(ValueError):
        cache.resolve([owner], PHASE)
    assert snapshots.calls == 0
    assert cache.observations[0]["outcome"] == "failed_closed"
