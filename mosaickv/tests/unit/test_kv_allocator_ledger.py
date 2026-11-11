"""Pure synthetic CPU mapping tests: no CUDA snapshot or allocator assumed."""

from copy import deepcopy

import pytest
import torch

from mosaickv.evaluation import kv_allocator_ledger as module
from mosaickv.evaluation.kv_allocator_ledger import (
    AllocatorLedger,
    resolve_snapshot,
    tensor_owners,
    verify_ownership_deltas,
)


def segment(*, device=0, explicit=False):
    blocks = [
        dict(size=512, requested_size=128, state="active_allocated"),
        dict(size=1024, requested_size=900, state="active_allocated"),
        dict(size=512, requested_size=0, state="inactive"),
    ]
    if explicit:
        for block, address in zip(blocks, (4096, 4608, 5632), strict=True):
            block["address"] = address
    return dict(device=device, address=4096, total_size=2048, blocks=blocks, is_expandable=True)


@pytest.mark.parametrize("explicit", [False, True])
def test_actual_blocks_not_guessed_rounding_and_alias_dedup(explicit):
    snapshot = [segment(explicit=explicit)]
    record = resolve_snapshot(snapshot, [(4096, 128), (4096, 128), (4608, 900)], device=0)
    assert record["current_tensor_backing_bytes"] == 1028
    assert record["current_allocator_block_bytes"] == 1536
    assert record["current_allocator_requested_bytes"] == 1028
    assert record["unique_tensor_owners"] == record["unique_allocator_blocks"] == 2
    assert record["all_required_owners_resolved"]
    assert [block["address"] for block in record["blocks"]] == [4096, 4608]
    assert [block["size"] for block in record["blocks"]] == [512, 1024]


def test_joint_subrange_owners_union_payload_but_charge_physical_block_once():
    record = resolve_snapshot([segment()], [(4608, 400), (4808, 400), (4908, 100)], device=0)
    assert record["current_tensor_backing_bytes"] == 600
    assert record["current_allocator_block_bytes"] == 1024
    assert record["unique_allocator_blocks"] == 1


@pytest.mark.parametrize("state", ["inactive", "active_awaiting_free", "unknown"])
def test_nonallocated_owner_is_rejected(state):
    snapshot = segment()
    snapshot["blocks"][0]["state"] = state
    with pytest.raises(ValueError, match="active_allocated"):
        resolve_snapshot([snapshot], [(4096, 1)], device=0)


@pytest.mark.parametrize("owner", [(1, 1), (6144, 1), (4220, 8), (4608, 901), (0, 1)])
def test_missing_out_of_requested_range_and_null_are_rejected(owner):
    with pytest.raises(ValueError):
        resolve_snapshot([segment()], [owner], device=0)


def test_cross_device_never_borrowed_and_current_device_selected_explicitly():
    with pytest.raises(ValueError, match="cross-device"):
        resolve_snapshot([segment(device=1)], [(4096, 128)], device=0)
    record = resolve_snapshot([segment(device=1), segment(device=0)], [(4096, 128)], device=0)
    assert record["blocks"][0]["device"] == 0


def test_wrapper_snapshot_empty_owners_and_zero_size_are_supported():
    record = resolve_snapshot({"segments": [segment()]}, [(0, 0)], device=0)
    assert record["current_tensor_backing_bytes"] == record["current_allocator_block_bytes"] == 0
    assert record["blocks"] == [] and record["all_required_owners_resolved"]
    assert tensor_owners((), device=0) == ()


def test_geometry_failures_and_request_rounding_are_not_hidden():
    bad = segment(explicit=True)
    bad["blocks"][1]["address"] = 4500
    with pytest.raises(ValueError, match="overlaps"):
        resolve_snapshot([bad], [(4096, 128)], device=0)
    bad = segment()
    bad["blocks"][0]["requested_size"] = 600
    with pytest.raises(ValueError, match="exceeds"):
        resolve_snapshot([bad], [(4096, 128)], device=0)


def test_cpu_tensor_owners_are_explicitly_unsupported():
    with pytest.raises(ValueError, match="real CUDA"):
        tensor_owners((torch.ones(1),), device=0)
    assert tensor_owners((torch.empty(0),), device=0) == ()


def test_stale_unowned_inactive_requested_size_does_not_change_charged_bytes():
    snapshot = segment()
    snapshot["blocks"][2]["requested_size"] = 8192
    record = resolve_snapshot([snapshot], [(4096, 128), (4608, 900)], device=0)
    assert record["current_tensor_backing_bytes"] == 1028
    assert record["current_allocator_block_bytes"] == 1536
    assert record["current_allocator_requested_bytes"] == 1028
    assert all(block["state"] == "active_allocated" for block in record["blocks"])


def test_stale_inactive_block_cannot_satisfy_a_required_owner():
    snapshot = segment()
    snapshot["blocks"][2]["requested_size"] = 8192
    with pytest.raises(ValueError, match="active_allocated"):
        resolve_snapshot([snapshot], [(5632, 1)], device=0)


@pytest.mark.parametrize("state", ["active_allocated", "active_awaiting_free"])
def test_active_oversized_requested_size_remains_a_hard_failure(state):
    snapshot = segment()
    snapshot["blocks"][0].update(state=state, requested_size=8192)
    with pytest.raises(ValueError, match="exceeds physical"):
        resolve_snapshot([snapshot], [(4608, 900)], device=0)


def test_inactive_physical_geometry_is_not_exempted():
    snapshot = segment()
    snapshot["blocks"][2].update(size=8192, requested_size=8192)
    with pytest.raises(ValueError, match="outside"):
        resolve_snapshot([snapshot], [(4096, 128)], device=0)


def mocked_ledger(monkeypatch, *, max_records=2):
    """Synthetic allocator boundaries, never a real GPU certification."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: "native")
    config = "backend:native,expandable_segments:True,roundup_power2_divisions:0"
    monkeypatch.setenv("PYTORCH_ALLOC_CONF", config)
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", config)
    monkeypatch.setattr(module, "tensor_owners", lambda tensors, *, device: tuple(tensors))
    calls = []

    def snapshot():
        calls.append(True)
        return [segment()]

    monkeypatch.setattr(torch.cuda, "memory_snapshot", snapshot)
    return AllocatorLedger(device="cuda:0", max_records=max_records), calls


def test_every_observation_snapshots_hashes_and_bounds_retained_events(monkeypatch):
    ledger, calls = mocked_ledger(monkeypatch)
    ledger.observe(((4096, 128),), phase="initial")
    ledger.observe(((4096, 128), (4608, 900)), phase="joint_peak")
    ledger.observe(((4608, 900),), phase="final")
    report = ledger.summary()
    assert len(calls) == report["observations"] == 3
    assert report["initial_allocator_block_bytes"] == 512
    assert report["final_allocator_block_bytes"] == 1024
    assert report["peak_allocator_block_bytes"] == 1536
    assert report["initial_tensor_backing_bytes"] == 128
    assert report["final_tensor_backing_bytes"] == 900
    assert report["peak_tensor_backing_bytes"] == 1028
    assert [event["index"] for event in report["retained_events"]] == [1, 2]
    assert len(report["observations_sha256"]) == 64
    assert all(
        "blocks" not in event and "owners" not in event for event in report["retained_events"]
    )
    assert report["all_required_owners_resolved"] and not report["whole_request_peak_certified"]
    assert report["backend"] == "native"
    assert len(calls) == 3  # summary must not acquire a new snapshot.
    replay, _ = mocked_ledger(monkeypatch)
    replay.observe(((4096, 128),), phase="initial")
    replay.observe(((4096, 128), (4608, 900)), phase="joint_peak")
    replay.observe(((4608, 900),), phase="final")
    assert replay.summary()["observations_sha256"] == report["observations_sha256"]


def test_failed_observation_cannot_be_hidden_by_later_summary(monkeypatch):
    ledger, _ = mocked_ledger(monkeypatch)
    ledger.observe(((4096, 128),), phase="initial")
    with pytest.raises(ValueError, match="missing"):
        ledger.observe(((1, 1),), phase="invalid")
    with pytest.raises(RuntimeError, match="unresolved"):
        ledger.summary()


def three_event_ledger(monkeypatch):
    ledger, _ = mocked_ledger(monkeypatch)
    ledger.observe(((4096, 128),), phase="initial")
    ledger.observe(((4096, 128), (4608, 900)), phase="joint_peak")
    ledger.observe(((4608, 900),), phase="final")
    return ledger


def test_complete_compact_deltas_independently_replay_bytes_hash_and_count(monkeypatch):
    ledger = three_event_ledger(monkeypatch)
    summary = ledger.summary()
    deltas = summary["ownership_deltas"]
    assert len(deltas) == 3 and len(summary["retained_events"]) == 2
    assert [block["address"] for block in deltas[0]["added_blocks"]] == [4096]
    assert deltas[0]["added_owners"] == [[4096, 128]]
    assert [block["address"] for block in deltas[1]["added_blocks"]] == [4608]
    assert deltas[2]["removed_blocks"] == [4096]
    assert deltas[2]["removed_owners"] == [[4096, 128]]
    replay = verify_ownership_deltas(deltas, expected_summary=summary)
    assert replay["ownership_deltas_verified"]
    assert replay["observations"] == 3 and replay["peak_allocator_block_bytes"] == 1536
    assert replay["observations_sha256"] == summary["observations_sha256"]
    ledger.observe(((4608, 900), (4608, 900)), phase="unchanged_alias")
    last = ledger.summary()["ownership_deltas"][-1]
    assert all(
        last[name] == []
        for name in (
            "added_blocks",
            "removed_blocks",
            "changed_blocks",
            "added_owners",
            "removed_owners",
        )
    )


@pytest.mark.parametrize("tamper", ["bytes", "requested_size", "owners", "timestamp"])
def test_tampered_delta_is_not_accepted(monkeypatch, tamper):
    summary = three_event_ledger(monkeypatch).summary()
    deltas = deepcopy(summary["ownership_deltas"])
    if tamper == "bytes":
        deltas[1]["current_allocator_block_bytes"] += 512
    elif tamper == "requested_size":
        deltas[0]["added_blocks"][0]["requested_size"] += 1
    elif tamper == "owners":
        deltas[1]["added_owners"][0][1] += 1
    else:
        deltas[1]["monotonic_ns"] = deltas[0]["monotonic_ns"] - 1
    with pytest.raises(ValueError):
        verify_ownership_deltas(deltas, expected_summary=summary)


def test_missing_middle_or_truncated_final_delta_fails_sealed_summary(monkeypatch):
    summary = three_event_ledger(monkeypatch).summary()
    deltas = summary["ownership_deltas"]
    with pytest.raises(ValueError, match="contiguous"):
        verify_ownership_deltas([deltas[0], deltas[2]], expected_summary=summary)
    with pytest.raises(ValueError, match="summary"):
        verify_ownership_deltas(deltas[:-1], expected_summary=summary)
    wrong = dict(summary, observations_sha256="0" * 64)
    with pytest.raises(ValueError, match="summary"):
        verify_ownership_deltas(deltas, expected_summary=wrong)


def test_actual_block_metadata_changes_are_explicit_and_replayable(monkeypatch):
    ledger, _ = mocked_ledger(monkeypatch)
    ledger.observe(((4096, 128),), phase="initial")
    updated = segment()
    updated["blocks"][0]["requested_size"] = 160
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: [updated])
    ledger.observe(((4096, 160),), phase="same_block_new_requested_range")
    summary = ledger.summary()
    delta = summary["ownership_deltas"][1]
    assert delta["added_blocks"] == [] and delta["removed_blocks"] == []
    assert delta["changed_blocks"][0]["requested_size"] == 160
    assert delta["removed_owners"] == [[4096, 128]]
    assert delta["added_owners"] == [[4096, 160]]
    replay = verify_ownership_deltas(summary["ownership_deltas"], expected_summary=summary)
    assert replay["final_allocator_block_bytes"] == 512
    assert replay["final_tensor_backing_bytes"] == 160
