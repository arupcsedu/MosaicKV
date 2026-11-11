"""Exact CUDA allocator-block mapping at declared decode-ownership events.

This ledger measures actual native-allocator blocks. It never substitutes
rounding formulas, reserved segments, weak references, or sampled allocator
envelopes for the strong owners supplied by the caller. Its scope is expressly
the declared decode-owned KV boundary, not construction or whole-request peak.
"""

import hashlib
import json
import os
import time
from bisect import bisect_right
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from itertools import pairwise
from typing import Any

import torch


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _union_bytes(ranges: Sequence[tuple[int, int]]) -> int:
    total, previous_end = 0, -1
    for start, stop in sorted(ranges):
        total += max(0, stop - max(start, previous_end))
        previous_end = max(previous_end, stop)
    return total


def resolve_snapshot(
    snapshot: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    owners: Iterable[tuple[int, int]],
    *,
    device: int,
) -> dict[str, Any]:
    """Resolve (storage pointer, backing nbytes) to unique active native blocks.

    Supports explicit block addresses and older snapshots whose block address
    is implied by the segment address plus preceding block sizes. Multiple
    storages within one active block charge that physical block only once.
    Every owner's complete range must fit the block's *requested* allocation,
    not merely its allocator-rounded size.
    """
    device = _integer(device, name="device")
    segments = snapshot.get("segments") if isinstance(snapshot, Mapping) else snapshot
    if not isinstance(segments, (list, tuple)):
        raise ValueError("native CUDA snapshot requires a segment sequence")
    blocks, other_device_blocks = [], []
    for segment in segments:
        if not isinstance(segment, Mapping):
            raise ValueError("malformed native allocator segment")
        segment_device = _integer(segment.get("device"), name="segment device")
        base = _integer(segment.get("address"), name="segment address", minimum=1)
        total_size = _integer(segment.get("total_size"), name="segment total_size", minimum=1)
        components = segment.get("blocks")
        if not isinstance(components, (list, tuple)):
            raise ValueError("native allocator segment requires explicit blocks")
        cursor = base
        for block in components:
            if not isinstance(block, Mapping):
                raise ValueError("malformed native allocator block")
            size = _integer(block.get("size"), name="block size", minimum=1)
            address = _integer(block.get("address", cursor), name="block address", minimum=1)
            if address < cursor or address + size > base + total_size:
                raise ValueError("allocator block lies outside or overlaps its segment geometry")
            cursor = address + size
            requested = block.get("requested_size", 0)
            requested = _integer(requested, name="block requested_size")
            # PyTorch 2.8 alloc_found_block shrinks a free split remainder
            # without resetting its historical requested_size. That inactive
            # metadata is not an allocation boundary; all physical geometry
            # and every active block/required owner remain strictly validated.
            if block.get("state") != "inactive" and requested > size:
                raise ValueError("allocator requested_size exceeds physical block size")
            record = dict(
                address=address,
                size=size,
                requested_size=requested,
                state=block.get("state"),
                device=segment_device,
                segment_address=base,
                is_expandable=segment.get("is_expandable"),
            )
            (blocks if segment_device == device else other_device_blocks).append(record)
    blocks.sort(key=lambda block: block["address"])
    if any(a["address"] + a["size"] > b["address"] for a, b in pairwise(blocks)):
        raise ValueError("overlapping allocator blocks on the requested device")
    starts = [block["address"] for block in blocks]
    selected, owner_ranges = {}, []
    owner_set = set()
    for owner in owners:
        if not isinstance(owner, (tuple, list)) or len(owner) != 2:
            raise ValueError("owners must be (storage pointer, backing nbytes) pairs")
        pointer = _integer(owner[0], name="owner pointer")
        size = _integer(owner[1], name="owner backing nbytes")
        if size == 0:
            continue
        if pointer == 0:
            raise ValueError("nonempty owner has a null storage pointer")
        index = bisect_right(starts, pointer) - 1
        block = blocks[index] if index >= 0 else None
        if block is None or pointer >= block["address"] + block["size"]:
            if any(b["address"] <= pointer < b["address"] + b["size"] for b in other_device_blocks):
                raise ValueError("owner resolves only to a cross-device allocator block")
            raise ValueError(f"missing allocator block for owner pointer {pointer}")
        if block["state"] != "active_allocated":
            raise ValueError(f"owner block is not active_allocated: {block['state']}")
        if pointer + size > block["address"] + block["requested_size"]:
            raise ValueError("owner range exceeds the allocator's requested-size boundary")
        selected[block["address"]] = block
        owner_ranges.append((pointer, pointer + size))
        owner_set.add((pointer, size))
    selected_blocks = [selected[address] for address in sorted(selected)]
    return dict(
        current_tensor_backing_bytes=_union_bytes(owner_ranges),
        current_allocator_block_bytes=sum(block["size"] for block in selected_blocks),
        current_allocator_requested_bytes=sum(block["requested_size"] for block in selected_blocks),
        unique_tensor_owners=len(owner_set),
        unique_allocator_blocks=len(selected_blocks),
        all_required_owners_resolved=True,
        blocks=selected_blocks,
        owners=[list(owner) for owner in sorted(owner_set)],
    )


def tensor_owners(tensors: Iterable[torch.Tensor], *, device: int) -> tuple[tuple[int, int], ...]:
    """Read strong CUDA storage owners; view aliases share their full backing."""
    result = set()
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("nonempty ownership lists require real CUDA tensor owners")
        storage = tensor.untyped_storage()
        size = storage.nbytes()
        if not size:
            continue
        if tensor.device.type != "cuda":
            raise ValueError("nonempty ownership lists require real CUDA tensor owners")
        if tensor.device.index != device:
            raise ValueError("cross-device tensor owner is outside the declared boundary")
        result.add((storage.data_ptr(), size))
    return tuple(sorted(result))


def verify_ownership_deltas(
    deltas: Sequence[Mapping[str, Any]], *, expected_summary: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """CPU-replay all compact owner/block events, ranges, totals and hashes.

    Initial additions contain the complete resolved state. Later changes use
    explicit actual block records and owner ranges, not rounded-byte formulas.
    Replay verifies only the declared decode-owned boundary, not whole-request
    coverage or observations that were never made by the producer.
    """
    if not isinstance(deltas, (list, tuple)) or not deltas:
        raise ValueError("a complete nonempty ownership-delta sequence is required")
    blocks, owners = {}, set()
    initial = final = device = None
    peak_tensor = peak_allocator = 0
    previous_time = -1
    digest = hashlib.sha256()
    cache_replay = None
    cache_proof_policy = None
    fields = (
        "current_tensor_backing_bytes",
        "current_allocator_block_bytes",
        "current_allocator_requested_bytes",
        "unique_tensor_owners",
        "unique_allocator_blocks",
        "all_required_owners_resolved",
    )
    for index, delta in enumerate(deltas):
        if not isinstance(delta, Mapping) or delta.get("index") != index:
            raise ValueError("ownership-delta indices must be complete and contiguous from zero")
        event_device = _integer(delta.get("device"), name="delta device")
        device = event_device if device is None else device
        if event_device != device:
            raise ValueError("ownership-delta device changed within one declared boundary")
        timestamp = _integer(delta.get("monotonic_ns"), name="delta monotonic_ns")
        if timestamp < previous_time:
            raise ValueError("ownership-delta timestamps must be monotonic")
        previous_time = timestamp
        phase = delta.get("phase")
        if not isinstance(phase, str) or not phase or len(phase) > 256:
            raise ValueError("ownership-delta phase must be bounded and nonempty")
        for name in (
            "removed_blocks",
            "changed_blocks",
            "added_blocks",
            "removed_owners",
            "added_owners",
        ):
            if not isinstance(delta.get(name), list):
                raise ValueError(f"ownership delta requires explicit {name}")
        for address in delta["removed_blocks"]:
            address = _integer(address, name="removed block address", minimum=1)
            if address not in blocks:
                raise ValueError("ownership delta removed an unknown allocator block")
            del blocks[address]
        for operation in ("changed_blocks", "added_blocks"):
            for block in delta[operation]:
                if not isinstance(block, Mapping):
                    raise ValueError("ownership delta contains a malformed block record")
                address = _integer(block.get("address"), name="delta block address", minimum=1)
                if (address in blocks) != (operation == "changed_blocks"):
                    raise ValueError(
                        "ownership delta changed an unknown or added a duplicate block"
                    )
                if block.get("device") != device:
                    raise ValueError("ownership delta contains a cross-device block")
                blocks[address] = dict(block)
        for operation in ("removed_owners", "added_owners"):
            for owner in delta[operation]:
                if not isinstance(owner, list) or len(owner) != 2:
                    raise ValueError("delta owners must be [storage pointer, backing nbytes]")
                pair = (
                    _integer(owner[0], name="delta owner pointer", minimum=1),
                    _integer(owner[1], name="delta owner nbytes", minimum=1),
                )
                if (pair in owners) != (operation == "removed_owners"):
                    raise ValueError(
                        "ownership delta removed an unknown or added a duplicate owner"
                    )
                (owners.remove if operation == "removed_owners" else owners.add)(pair)
        segments = {}
        for block in blocks.values():
            base = _integer(block.get("segment_address"), name="delta segment address", minimum=1)
            address = _integer(block.get("address"), name="delta address", minimum=1)
            size = _integer(block.get("size"), name="delta block size", minimum=1)
            segment = segments.setdefault(
                base,
                dict(
                    device=device,
                    address=base,
                    total_size=0,
                    blocks=[],
                    is_expandable=block.get("is_expandable"),
                ),
            )
            if segment["is_expandable"] != block.get("is_expandable"):
                raise ValueError("ownership delta disagrees on segment expansion metadata")
            segment["total_size"] = max(segment["total_size"], address + size - base)
            segment["blocks"].append(dict(block))
        for segment in segments.values():
            segment["blocks"].sort(key=lambda block: block["address"])
        resolved = resolve_snapshot(list(segments.values()), sorted(owners), device=device)
        has_cache_proof = "snapshot_cache" in delta
        if cache_proof_policy is None:
            cache_proof_policy = has_cache_proof
            if (
                expected_summary is not None
                and expected_summary.get("constructor_snapshot_reuse") is True
                and not has_cache_proof
            ):
                raise ValueError("declared constructor reuse requires every resolution proof")
        elif cache_proof_policy != has_cache_proof:
            raise ValueError("allocator resolution proofs cannot disappear within one ledger")
        if has_cache_proof:
            if cache_replay is None:
                from mosaickv.evaluation.snapshot_reuse_replay import SnapshotReuseReplay

                cache_replay = SnapshotReuseReplay()
            cache_replay.observe(
                delta["snapshot_cache"], index=index, phase=phase, resolved=resolved
            )
        if resolved["blocks"] != [blocks[address] for address in sorted(blocks)]:
            raise ValueError("ownership delta retains an unowned or inconsistent allocator block")
        if any(delta.get(field) != resolved[field] for field in fields):
            raise ValueError(
                "ownership-delta recorded totals disagree with independently resolved bytes"
            )
        observation = dict(index=index, phase=phase, device=device, **resolved)
        encoded = json.dumps(
            observation, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        if hashlib.sha256(encoded).hexdigest() != delta.get("owner_block_mapping_sha256"):
            raise ValueError("ownership-delta mapping hash mismatch")
        digest.update(encoded + b"\n")
        initial = resolved if initial is None else initial
        final = resolved
        peak_tensor = max(peak_tensor, resolved["current_tensor_backing_bytes"])
        peak_allocator = max(peak_allocator, resolved["current_allocator_block_bytes"])
    verified = dict(
        scope="declared_decode_owned_KV_explicit_ownership_observations",
        whole_request_peak_certified=False,
        initial_tensor_backing_bytes=initial["current_tensor_backing_bytes"],
        final_tensor_backing_bytes=final["current_tensor_backing_bytes"],
        peak_tensor_backing_bytes=peak_tensor,
        initial_allocator_block_bytes=initial["current_allocator_block_bytes"],
        final_allocator_block_bytes=final["current_allocator_block_bytes"],
        peak_allocator_block_bytes=peak_allocator,
        observations=len(deltas),
        all_required_owners_resolved=True,
        observations_sha256=digest.hexdigest(),
        ownership_deltas_verified=True,
    )
    if cache_replay is not None:
        verified.update(constructor_snapshot_reuse=True, snapshot_cache=cache_replay.summary())
    if expected_summary is not None:
        for field in verified.keys() - {"ownership_deltas_verified"}:
            if expected_summary.get(field) != verified[field]:
                raise ValueError(
                    f"ownership-delta replay disagrees with expected summary field {field}"
                )
    return verified


class AllocatorLedger:
    """Actual native CUDA blocks at every explicit ownership observation."""

    def __init__(
        self, *, device: Any, max_records: int = 64, constructor_snapshot_reuse: bool = False
    ):
        target = torch.device(f"cuda:{device}" if type(device) is int else device)
        if target.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("allocator ledger requires an allocated CUDA device")
        self.device = torch.cuda.current_device() if target.index is None else target.index
        self.backend = torch.cuda.memory.get_allocator_backend()
        if self.backend != "native":
            raise NotImplementedError("exact ledger requires the pinned native CUDA allocator")
        _integer(max_records, name="max_records", minimum=1)
        self.environment = {
            name: os.environ.get(name)
            for name in (
                "PYTORCH_ALLOC_CONF",
                "PYTORCH_CUDA_ALLOC_CONF",
            )
        }
        for value in self.environment.values():
            if (
                value is None
                or "backend:native" not in value.replace(" ", "")
                or "expandable_segments:True" not in value.replace(" ", "")
            ):
                raise ValueError(
                    "both allocator environments must pin native expandable segments "
                    "before Python starts"
                )
        self.events: deque[dict[str, Any]] = deque(maxlen=max_records)
        self.observations = 0
        self._hash = hashlib.sha256()
        self._initial = self._final = None
        self._peak_tensor = self._peak_allocator = 0
        self._snapshot_seconds = 0.0
        self._failed = False
        self._previous_blocks = {}
        self._previous_owners = set()
        self.ownership_deltas = []
        if type(constructor_snapshot_reuse) is not bool:
            raise ValueError("explicit constructor snapshot-reuse boolean required")
        self.snapshot_resolver = self.snapshot_replay = None
        if constructor_snapshot_reuse:
            from mosaickv.evaluation.live_owner_snapshot_cache import LiveOwnerSnapshotCache
            from mosaickv.evaluation.snapshot_reuse_replay import SnapshotReuseReplay

            self.snapshot_resolver = LiveOwnerSnapshotCache(device=self.device, enabled=True)
            self.snapshot_replay = SnapshotReuseReplay()

    def observe(
        self, tensors: Iterable[torch.Tensor], *, phase: str = "decode_owned"
    ) -> dict[str, Any]:
        if not isinstance(phase, str) or not phase or len(phase) > 256:
            raise ValueError("bounded nonempty ownership phase label required")
        try:
            start = time.perf_counter()
            proof = None
            if self.snapshot_resolver is None:
                owners = tensor_owners(tensors, device=self.device)
                snapshot = torch.cuda.memory_snapshot()
                resolved = resolve_snapshot(snapshot, owners, device=self.device)
                del snapshot
            else:
                resolved = self.snapshot_resolver.resolve(tensors, phase)
                proof = resolved.pop("snapshot_cache")
                self.snapshot_replay.observe(
                    proof, index=self.observations, phase=phase, resolved=resolved
                )
            snapshot_seconds = time.perf_counter() - start
        except Exception:
            self._failed = True
            raise
        observation = dict(index=self.observations, phase=phase, device=self.device, **resolved)
        encoded = json.dumps(
            observation, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        self._hash.update(encoded + b"\n")
        record = {
            key: value for key, value in observation.items() if key not in ("blocks", "owners")
        }
        record["owner_block_mapping_sha256"] = hashlib.sha256(encoded).hexdigest()
        record["expandable_owner_blocks"] = sum(
            block["is_expandable"] is True for block in resolved["blocks"]
        )
        current_blocks = {block["address"]: block for block in resolved["blocks"]}
        current_owners = {tuple(owner) for owner in resolved["owners"]}
        delta = dict(
            **record,
            monotonic_ns=time.monotonic_ns(),
            added_blocks=[
                current_blocks[address]
                for address in sorted(current_blocks.keys() - self._previous_blocks.keys())
            ],
            removed_blocks=sorted(self._previous_blocks.keys() - current_blocks.keys()),
            changed_blocks=[
                current_blocks[address]
                for address in sorted(current_blocks.keys() & self._previous_blocks.keys())
                if current_blocks[address] != self._previous_blocks[address]
            ],
            added_owners=[list(owner) for owner in sorted(current_owners - self._previous_owners)],
            removed_owners=[
                list(owner) for owner in sorted(self._previous_owners - current_owners)
            ],
        )
        if proof is not None:
            delta["snapshot_cache"] = proof
        self.ownership_deltas.append(delta)
        self._previous_blocks, self._previous_owners = current_blocks, current_owners
        self.events.append(record)
        self._initial = record if self._initial is None else self._initial
        self._final = record
        self._peak_tensor = max(self._peak_tensor, record["current_tensor_backing_bytes"])
        self._peak_allocator = max(self._peak_allocator, record["current_allocator_block_bytes"])
        self.observations += 1
        self._snapshot_seconds += snapshot_seconds
        return record

    def summary(self) -> dict[str, Any]:
        if self._failed:
            raise RuntimeError("allocator ledger encountered an unresolved ownership observation")
        if self._initial is None:
            raise ValueError(
                "a decode-owned ledger cannot be summarized before its first observation"
            )
        result = dict(
            scope="declared_decode_owned_KV_explicit_ownership_observations",
            whole_request_peak_certified=False,
            construction_peak_certified=False,
            exact_allocator_mapping_at_every_recorded_observation=True,
            initial_tensor_backing_bytes=self._initial["current_tensor_backing_bytes"],
            final_tensor_backing_bytes=self._final["current_tensor_backing_bytes"],
            peak_tensor_backing_bytes=self._peak_tensor,
            initial_allocator_block_bytes=self._initial["current_allocator_block_bytes"],
            final_allocator_block_bytes=self._final["current_allocator_block_bytes"],
            peak_allocator_block_bytes=self._peak_allocator,
            observations=self.observations,
            all_required_owners_resolved=True,
            observations_sha256=self._hash.hexdigest(),
            backend=self.backend,
            allocator_environment=dict(self.environment),
            device=self.device,
            snapshot_seconds=self._snapshot_seconds,
            mean_snapshot_seconds=self._snapshot_seconds / self.observations,
            retained_events=list(self.events),
            retained_events_policy="bounded_tail; complete ordered mappings hashed incrementally",
            first_event=self._initial,
            final_event=self._final,
            ownership_deltas=list(self.ownership_deltas),
            ownership_delta_schema="initial_complete_state_then_explicit_owner_block_deltas_v1",
        )
        if self.snapshot_replay is not None:
            result.update(
                constructor_snapshot_reuse=True, snapshot_cache=self.snapshot_replay.summary()
            )
        return result


def selftest(*, device: Any = "cuda", observations: int = 100) -> dict[str, Any]:
    """No-weight GPU block/alias test and 1,000-owner snapshot-overhead diagnostic."""
    _integer(observations, name="observations", minimum=1)
    ledger = AllocatorLedger(device=device)
    target = torch.device("cuda", ledger.device)
    # Synthetic non-KV pressure exercises snapshot metadata volume only. It
    # must never be counted as cache ownership or added to a compared method.
    non_kv_noise = [torch.empty(2, dtype=torch.uint8, device=target) for _ in range(7000)]
    small = [
        torch.empty(2 * (1 + index % 63), dtype=torch.uint8, device=target) for index in range(1000)
    ]
    sources = [
        torch.empty((length, 4096), dtype=torch.bfloat16, device=target) for length in (600, 604)
    ]
    metadata = torch.arange(1000, device=target)
    owners = (*small, *sources, metadata)
    alias_owners = (*owners, *(tensor[:] for tensor in small[::10]), sources[0].view(-1))
    torch.cuda.synchronize(target)
    started = time.perf_counter()
    for index in range(observations):
        record = ledger.observe(alias_owners, phase=f"synthetic_joint_head_source_{index}")
    wall = time.perf_counter() - started
    snapshot = torch.cuda.memory_snapshot()
    resolved = resolve_snapshot(
        snapshot, tensor_owners(owners, device=ledger.device), device=ledger.device
    )
    alias_resolved = resolve_snapshot(
        snapshot, tensor_owners(alias_owners, device=ledger.device), device=ledger.device
    )
    all_expandable = all(block["is_expandable"] is True for block in resolved["blocks"])
    alias_parity = resolved == alias_resolved
    mean_wall = wall / observations
    extrapolated = mean_wall * 4322 * 3
    ledger_summary = ledger.summary()
    replay = verify_ownership_deltas(
        ledger_summary["ownership_deltas"], expected_summary=ledger_summary
    )
    return dict(
        measurement_type="synthetic_cuda_allocator_ownership_diagnostic",
        torch_version=torch.__version__,
        no_model_weights_loaded=True,
        declared_tensor_owners=len(owners),
        synthetic_nonKV_snapshot_pressure_blocks=len(non_kv_noise),
        synthetic_nonKV_pressure_excluded_from_KV_totals=True,
        repeated_observations=observations,
        final_consistent_with_last_observation=(
            record["current_allocator_block_bytes"] == resolved["current_allocator_block_bytes"]
            and record["current_tensor_backing_bytes"] == resolved["current_tensor_backing_bytes"]
        ),
        alias_mapping_identical=alias_parity,
        all_owned_segments_actually_expandable=all_expandable,
        measured_rounding_bytes=resolved["current_allocator_block_bytes"]
        - resolved["current_tensor_backing_bytes"],
        measured_block_count=resolved["unique_allocator_blocks"],
        snapshot_loop_wall_seconds=wall,
        mean_observation_wall_seconds=mean_wall,
        extrapolated_4322_observations_times_3_arms_seconds=extrapolated,
        registered_overhead_budget_seconds=360.0,
        overhead_budget_passed=extrapolated <= 360.0,
        snapshot_consistency_passed=all_expandable
        and alias_parity
        and ledger.observations == observations
        and replay["ownership_deltas_verified"],
        ownership_delta_replay=replay,
        ledger=ledger_summary,
    )
