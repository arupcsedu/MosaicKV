"""CPU mocks verify ownership-event integration, not CUDA allocator evidence."""

import pytest
import torch

from mosaickv.evaluation import kv_allocator_ledger
from mosaickv.evaluation.bounded_component_attention import ComponentWorkspace
from mosaickv.evaluation.bounded_component_cache import ComponentCache
from mosaickv.evaluation.bounded_ragged_raw import RaggedRawPacket
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


def owners(tensors):
    return frozenset(
        (str(t.device), t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
        for t in tensors
        if t.untyped_storage().nbytes()
    )


class MockAllocatorLedger:
    """Save scalar owner descriptors; do not extend tensor/storage lifetimes."""

    def __init__(self, *, device):
        self.device = torch.device(device)
        self.records = []

    def observe(self, tensors, *, phase):
        current = tuple(tensors)
        record = dict(
            phase=phase,
            owners=owners(current),
            current_tensor_backing_bytes=backing_bytes(current),
        )
        self.records.append(record)
        return record

    def summary(self):
        return dict(
            observations=len(self.records),
            initial_tensor_backing_bytes=self.records[0]["current_tensor_backing_bytes"],
            final_tensor_backing_bytes=self.records[-1]["current_tensor_backing_bytes"],
            peak_tensor_backing_bytes=max(r["current_tensor_backing_bytes"] for r in self.records),
            synthetic_CPU_mock=True,
            whole_request_peak_certified=False,
        )


class SourceFixture:
    def __init__(self, length, *, phases=None):
        self.shape = (1, 32, length, 128)
        self.source = torch.zeros((1, length, 4096), dtype=torch.bfloat16)
        self.phases = phases or tuple(
            torch.ones((1, length, 128), dtype=torch.bfloat16) for _ in range(2)
        )

    def tensors(self):
        return self.source, *self.phases


def raw_packet(length=3):
    pairs, positions = [], []
    for head in range(32):
        count = length - (head % 2 and length > 1)
        joint = torch.zeros((2, count, 128), dtype=torch.bfloat16)
        pairs.append((joint[0], joint[1]))
        positions.append(torch.arange(count, dtype=torch.long))
    return RaggedRawPacket(pairs, positions, original_length=length, prompt_capacity=length)


@pytest.fixture(autouse=True)
def mock_native_allocator(monkeypatch):
    monkeypatch.setattr(kv_allocator_ledger, "AllocatorLedger", MockAllocatorLedger)


def test_ledger_receives_complete_owners_on_every_observation_and_history_overlap():
    source = SourceFixture(5)
    metadata = torch.arange(512, dtype=torch.long)
    cache = ComponentCache(
        (raw_packet(3), source), metadata=(metadata,), updates=32, allocator_audit=True
    )
    ledger = cache.allocator_ledger
    assert isinstance(ledger, MockAllocatorLedger)
    shared = owners(cache.shared_tensors)
    assert shared == ledger.records[0]["owners"]
    assert owners((metadata, *source.tensors(), *cache.workspace.owned_tensors())) <= shared
    cache._mask_dispatch_active = True  # Accounting unit; no model kernels execute.
    for step in range(20):
        for layer in cache.layers:
            old_history = owners(layer.history)
            old_capacity = layer.history_capacity
            key = torch.full((1, 32, 1, 128), step, dtype=torch.bfloat16)
            value = torch.full_like(key, -step)
            incoming = owners((key, value))
            first = len(ledger.records)
            layer.update(key, value)
            observations = ledger.records[first:]
            assert observations and all(shared <= row["owners"] for row in observations)
            assert all(incoming <= row["owners"] for row in observations)
            new_history = owners(layer.history)
            if layer.history_capacity != old_capacity:
                assert len(observations) == 2
                # observe(newer) must see BOTH old live history and new extras.
                assert old_history | new_history | incoming <= observations[0]["owners"]
                assert new_history | incoming <= observations[1]["owners"]
                assert not old_history & observations[1]["owners"]
            else:
                assert len(observations) == 1
                assert old_history | incoming <= observations[0]["owners"]
            layer.release()
            cache.observe()
            assert not incoming & ledger.records[-1]["owners"]
    cache._mask_dispatch_active = False
    report = cache.storage_record()
    final_owners = owners(
        (*cache.shared_tensors, *(t for layer in cache.layers for t in layer.owned_tensors()))
    )
    assert ledger.records[-1]["owners"] == final_owners
    assert len(ledger.records) == cache.observations
    assert ledger.records[0]["phase"] == "initial_cache"
    assert ledger.records[-1]["phase"] == "final_cache"
    assert any(row["phase"].endswith("growth_overlap") for row in ledger.records)
    assert any(row["phase"].endswith("copied_history") for row in ledger.records)
    assert report["allocator_blocks"]["observations"] == cache.observations
    assert (
        report["allocator_blocks"]["peak_tensor_backing_bytes"]
        == report["peak_active_kv_storage_bytes"]
    )
    assert (
        report["allocator_blocks"]["final_tensor_backing_bytes"]
        == report["final_active_kv_storage_bytes"]
    )
    assert report["allocator_blocks"]["whole_request_peak_certified"] is False


def test_shared_workspace_identity_across_source_and_raw_caches_and_phase_aliases():
    workspace = ComponentWorkspace(device="cpu")
    source = SourceFixture(3)
    source_alias = SourceFixture(3, phases=source.phases)
    metadata = torch.arange(512, dtype=torch.long)
    first = ComponentCache(
        (source, source_alias),
        metadata=(metadata, *source.phases),
        workspace=workspace,
        allocator_audit=True,
    )
    second = ComponentCache(
        (raw_packet(3),), metadata=(metadata,), workspace=workspace, allocator_audit=True
    )
    assert first.workspace is second.workspace is workspace
    assert first.allocator_ledger is not second.allocator_ledger
    workspace_owners = owners(workspace.owned_tensors())
    assert workspace_owners <= first.allocator_ledger.records[0]["owners"]
    assert workspace_owners <= second.allocator_ledger.records[0]["owners"]
    assert first.shared_bytes == backing_bytes(
        (*workspace.owned_tensors(), metadata, source.source, source_alias.source, *source.phases)
    )
    assert first.storage_record()["allocator_blocks"]["synthetic_CPU_mock"] is True
    assert second.storage_record()["allocator_blocks"]["synthetic_CPU_mock"] is True


def test_default_no_ledger_remains_compatible_and_does_not_construct_an_auditor(monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError("Default CPU transport must not request a CUDA allocator ledger")

    monkeypatch.setattr(kv_allocator_ledger, "AllocatorLedger", forbidden)
    first = ComponentCache((SourceFixture(3),))
    second = ComponentCache((raw_packet(3),), allocator_audit=False)
    assert first.allocator_ledger is second.allocator_ledger is None
    assert first.workspace is not second.workspace
    assert first.storage_record()["allocator_blocks"] is None
    assert second.storage_record()["allocator_blocks"] is None


@pytest.mark.parametrize("bad", [None, "true", 1, 0])
def test_allocator_audit_flag_rejects_non_booleans(bad):
    with pytest.raises(ValueError, match="boolean"):
        ComponentCache((SourceFixture(3),), allocator_audit=bad)


@pytest.mark.parametrize("kind", ["type", "device", "heads", "dtype", "shape", "alias"])
def test_reused_workspace_requires_registered_geometry_dtype_device_and_disjoint_scratch(kind):
    if kind == "type":
        workspace = object()
    elif kind == "device":
        workspace = ComponentWorkspace(device="meta")  # No CUDA request or allocation.
    else:
        workspace = ComponentWorkspace(device="cpu", heads=1 if kind == "heads" else 32)
        if kind == "dtype":
            workspace.component = workspace.component.float()
        elif kind == "shape":
            workspace.component = workspace.component.T
        elif kind == "alias":
            workspace.head = workspace.component.reshape(-1)[: 1024 * 128].view(1024, 128)
    with pytest.raises(ValueError, match="workspace"):
        ComponentCache((SourceFixture(3),), workspace=workspace)
