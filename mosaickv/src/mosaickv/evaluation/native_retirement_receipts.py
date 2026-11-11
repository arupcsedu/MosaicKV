"""Generation-bound native CUDA allocation/free-completed receipts.

The pinned PyTorch 2.8 history API is used only for the bounded streaming
prefill.  Weak-reference death is not a free receipt.  Existing tracing is
never replaced, and a full/changed trace ring fails closed.  This module does
not classify every request allocation or itself certify a whole-request peak.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import torch

from mosaickv.evaluation.kv_allocator_ledger import resolve_snapshot, tensor_owners

_TORCH_VERSION = "2.8.0+cu128"
_TORCH_GIT = "a1cb3cc05d46d198467bebbb6e8fba50a325d4e7"
_ACTIONS = {
    "alloc",
    "free_requested",
    "free_completed",
    "snapshot",
    "segment_alloc",
    "segment_free",
    "segment_map",
    "segment_unmap",
    "oom",
}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _hash_field(value, name):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"{name} requires an explicit SHA-256")
    return value


def validate_trace_prefix(previous, current, *, capacity):
    """Require an append-only, untruncated actual-device trace, not a ring tail."""
    _integer(capacity, "trace capacity", 1)
    if not isinstance(previous, (list, tuple)) or not isinstance(current, (list, tuple)):
        raise ValueError("explicit complete trace sequences required")
    if len(current) >= capacity:
        raise ValueError("allocator trace reached its ring capacity; completeness is unproven")
    if len(current) < len(previous) or list(current[: len(previous)]) != list(previous):
        raise ValueError("allocator trace prefix changed or was truncated")
    for event in current:
        if not isinstance(event, Mapping) or event.get("action") not in _ACTIONS:
            raise ValueError("unknown/malformed native allocator history action")
        if event["action"] == "oom":
            raise ValueError("native allocator OOM invalidates retirement verification")
        if event["action"] in {"alloc", "free_requested", "free_completed"}:
            _integer(event.get("addr"), "native allocation address", 1)
            _integer(event.get("size"), "native allocation requested size", 1)
            _integer(event.get("stream"), "native allocation stream")
    return _sha(list(current))


def allocation_generations(events):
    """Reduce observed allocations; unbound baseline frees remain unclassified.

    The trace starts after model weights exist.  Only a generation whose ALLOC
    was actually observed can be bound or certified.  Unbound baseline frees
    cannot create or satisfy a receipt.
    """
    active, records, counts = {}, {}, {}
    for index, event in enumerate(events):
        action = event["action"]
        if action not in {"alloc", "free_requested", "free_completed"}:
            continue
        address, size = event["addr"], event["size"]
        if action == "alloc":
            if address in active:
                raise ValueError("native address reused before its free_completed event")
            generation = counts.get(address, 0)
            counts[address] = generation + 1
            record = dict(
                address=address,
                requested_bytes=size,
                alloc_index=index,
                generation=generation,
                free_requested_index=None,
                free_completed_index=None,
                allocation_stream=event["stream"],
            )
            records[index] = active[address] = record
        elif address in active:
            record = active[address]
            if size != record["requested_bytes"]:
                raise ValueError("native free size differs from its actual allocation generation")
            if action == "free_requested":
                if record["free_requested_index"] is not None:
                    raise ValueError("duplicate free_requested for one native generation")
                record["free_requested_index"] = index
            else:
                if record["free_requested_index"] is None:
                    raise ValueError("native free_completed lacks its free_requested event")
                record["free_completed_index"] = index
                del active[address]
    return records, active


def bind_generations(events, snapshot, owners, *, device, label):
    """Bind each real CUDA backing to its observed, currently active generation."""
    if not isinstance(label, str) or not label or len(label) > 128:
        raise ValueError("bounded nonempty retirement label required")
    _integer(device, "native generation device")
    validate_trace_prefix([], events, capacity=len(events) + 1)
    resolved = resolve_snapshot(snapshot, owners, device=device)
    records, active = allocation_generations(events)
    blocks = {block["address"]: block for block in resolved["blocks"]}
    allocations = []
    for address, block in sorted(blocks.items()):
        record = active.get(address)
        if record is None:
            raise ValueError("required native owner has no observed active ALLOC generation")
        if record["free_requested_index"] is not None:
            raise ValueError("a pending-free generation cannot become a live ownership token")
        if record["requested_bytes"] != block["requested_size"]:
            raise ValueError("native history ALLOC size disagrees with the active snapshot request")
        allocations.append(
            dict(
                record,
                block_size=block["size"],
                owner_ranges=[
                    list(owner)
                    for owner in resolved["owners"]
                    if address <= owner[0] < address + block["size"]
                ],
            )
        )
    if not allocations:
        raise ValueError("nonempty original native K/V owners required")
    del records
    return dict(
        schema="native_cuda_generation_retirement_token_v1",
        label=label,
        device=device,
        captured_trace_count=len(events),
        captured_trace_sha256=_sha(list(events)),
        captured_snapshot_sha256=_sha(snapshot),
        allocations=allocations,
        unique_backing_bytes=resolved["current_tensor_backing_bytes"],
        physical_block_bytes=resolved["current_allocator_block_bytes"],
    )


def verify_retired(events, token):
    """Replay one token against an intact trace, accepting only FREE_COMPLETED."""
    if (
        not isinstance(token, Mapping)
        or token.get("schema") != "native_cuda_generation_retirement_token_v1"
    ):
        raise ValueError("generation-bound original owner token required")
    _integer(token.get("device"), "native generation device")
    label = token.get("label")
    if not isinstance(label, str) or not label or len(label) > 128:
        raise ValueError("bounded nonempty retirement label required")
    _hash_field(token.get("captured_snapshot_sha256"), "capture snapshot hash")
    _hash_field(token.get("captured_trace_sha256"), "capture trace hash")
    validate_trace_prefix([], events, capacity=len(events) + 1)
    count = _integer(token.get("captured_trace_count"), "captured trace count", 1)
    if len(events) < count or _sha(list(events[:count])) != token.get("captured_trace_sha256"):
        raise ValueError("retirement trace no longer contains its immutable capture prefix")
    allocations = token.get("allocations")
    if not isinstance(allocations, list) or not allocations:
        raise ValueError("complete nonempty allocation-generation bindings required")
    records, _active = allocation_generations(events)
    receipts, seen, owners, physical_bytes = [], set(), [], 0
    for allocation in allocations:
        if not isinstance(allocation, Mapping):
            raise ValueError("explicit native allocation-generation records required")
        alloc_index = _integer(allocation.get("alloc_index"), "allocation index")
        if alloc_index in seen:
            raise ValueError("duplicate allocation-generation binding")
        seen.add(alloc_index)
        for name in ("address", "requested_bytes", "generation", "allocation_stream"):
            _integer(allocation.get(name), name, 1 if name in {"address", "requested_bytes"} else 0)
        if (
            allocation.get("free_requested_index") is not None
            or allocation.get("free_completed_index") is not None
        ):
            raise ValueError("ownership capture must predate both native free events")
        current = records.get(alloc_index)
        if current is None or any(
            current.get(field) != allocation.get(field)
            for field in ("address", "requested_bytes", "generation", "allocation_stream")
        ):
            raise ValueError("retirement token does not identify the observed ALLOC generation")
        if alloc_index >= count:
            raise ValueError("native ALLOC generation must predate its captured ownership token")
        block_size = _integer(allocation.get("block_size"), "physical allocation block size", 1)
        if block_size < current["requested_bytes"]:
            raise ValueError("physical allocation cannot be smaller than its native request")
        physical_bytes += block_size
        ranges = allocation.get("owner_ranges")
        if not isinstance(ranges, list) or not ranges:
            raise ValueError("complete nonempty backing ranges required for the native generation")
        for backing in ranges:
            if not isinstance(backing, list) or len(backing) != 2:
                raise ValueError("native owner ranges must be pointer/backing-size pairs")
            pointer = _integer(backing[0], "native owner pointer", 1)
            size = _integer(backing[1], "native owner backing size", 1)
            if (
                not current["address"]
                <= pointer
                < pointer + size
                <= current["address"] + current["requested_bytes"]
            ):
                raise ValueError("native owner backing exceeds its generation's request boundary")
            owners.append((pointer, pointer + size))
        requested, completed = current["free_requested_index"], current["free_completed_index"]
        if requested is None or completed is None or requested < count or completed <= requested:
            raise ValueError(
                "original native generation has no post-capture FREE_COMPLETED receipt"
            )
        receipts.append(
            dict(
                address=current["address"],
                generation=current["generation"],
                alloc_index=alloc_index,
                free_requested_index=requested,
                free_completed_index=completed,
                requested_bytes=current["requested_bytes"],
                block_size=allocation["block_size"],
                owner_ranges=allocation["owner_ranges"],
            )
        )
    union_bytes, previous_end = 0, -1
    for start, end in sorted(owners):
        union_bytes += max(0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    if (
        _integer(token.get("unique_backing_bytes"), "unique native backing bytes", 1) != union_bytes
        or _integer(token.get("physical_block_bytes"), "physical native block bytes", 1)
        != physical_bytes
    ):
        raise ValueError("native capture's backing or physical byte totals disagree")
    return dict(
        schema="native_cuda_generation_retirement_receipt_v1",
        label=token["label"],
        device=token["device"],
        actual_free_completed_verified=True,
        weakref_death_is_not_free_evidence=True,
        captured_trace_count=count,
        verified_trace_count=len(events),
        trace_sha256=_sha(list(events)),
        token_sha256=_sha(dict(token)),
        allocations=receipts,
        source_token=json.loads(json.dumps(dict(token), allow_nan=False)),
        trace_events=list(events),
        whole_request_KV_peak_certified=False,
    )


def verify_receipts(receipts, *, expected_layers=32):
    """Independently replay every layer's immutable native generation receipt."""
    _integer(expected_layers, "expected layer count", 1)
    if not isinstance(receipts, (list, tuple)) or len(receipts) != expected_layers:
        raise ValueError("one complete native retirement receipt per expected layer required")
    device, seen_generations, previous_trace = None, set(), []
    for layer, receipt in enumerate(receipts):
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("label") != f"layer_{layer}"
            or receipt.get("actual_free_completed_verified") is not True
            or receipt.get("weakref_death_is_not_free_evidence") is not True
        ):
            raise ValueError("ordered layer labels and actual free-completed evidence required")
        trace = receipt.get("trace_events")
        if not isinstance(trace, list):
            raise ValueError("complete immutable native trace events required for CPU replay")
        validate_trace_prefix([], trace, capacity=len(trace) + 1)
        validate_trace_prefix(previous_trace, trace, capacity=len(trace) + 1)
        previous_trace = trace
        expected = verify_retired(trace, receipt.get("source_token"))
        if _sha(dict(receipt)) != _sha(expected):
            raise ValueError("native retirement receipt differs from independent generation replay")
        actual_device = _integer(receipt.get("device"), "receipt device")
        device = actual_device if device is None else device
        if device != actual_device:
            raise ValueError("native receipt device changed within one request")
        for allocation in receipt["allocations"]:
            key = device, allocation["alloc_index"]
            if key in seen_generations:
                raise ValueError("a native generation was claimed by more than one layer")
            seen_generations.add(key)
    return dict(
        complete=True,
        expected_layers=expected_layers,
        verified_layers=len(receipts),
        actual_free_completed_verified=True,
        generations=len(seen_generations),
        receipts_sha256=_sha(list(receipts)),
        whole_request_KV_peak_certified=False,
    )


class CUDAFreeCompletedTracker:
    """A strictly owned, finite PyTorch native-memory-history session.

    Start before the prompt forward and stop immediately after its final layer
    is retired.  Do not wrap 128-token decoding in this bounded trace session.
    """

    _active_session = None

    def __init__(self, *, device, max_entries=200_000):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("actual CUDA retirement receipts require an allocated CUDA device")
        self.index = torch.cuda.current_device() if self.device.index is None else self.device.index
        self.capacity = _integer(max_entries, "bounded history capacity", 64)
        if self.capacity > 1_000_000:
            raise ValueError("bounded history capacity cannot exceed one million events")
        self.events, self.tokens, self.receipts = [], [], []
        self._token_hashes = {}
        self.started = self.closed = self.failed = False
        self.original_history_enabled = None

    def _read(self):
        if not self.started or self.closed or not torch._C._cuda_isHistoryEnabled():
            raise RuntimeError("the owned bounded native history must still be active")
        snapshot = torch.cuda.memory._snapshot()
        traces = snapshot.get("device_traces")
        if not isinstance(traces, list) or self.index >= len(traces):
            raise RuntimeError("native snapshot did not return the declared device trace")
        current = traces[self.index]
        validate_trace_prefix(self.events, current, capacity=self.capacity)
        self.events = [dict(event) for event in current]
        return snapshot

    def __enter__(self):
        if self.started or self.closed or CUDAFreeCompletedTracker._active_session is not None:
            raise RuntimeError("one non-reentrant native retirement session required")
        if torch.__version__ != _TORCH_VERSION or torch.version.git_version != _TORCH_GIT:
            raise RuntimeError("retirement source/API review is pinned to Torch 2.8.0+cu128")
        if torch.cuda.memory.get_allocator_backend() != "native":
            raise RuntimeError("native CUDA caching allocator required")
        self.original_history_enabled = bool(torch._C._cuda_isHistoryEnabled())
        if self.original_history_enabled:
            raise RuntimeError("existing native memory tracing is preserved; cannot take ownership")
        before = torch.cuda.memory._snapshot()
        if any(before.get("device_traces", ())):
            raise RuntimeError("preexisting inactive history is preserved; cannot clear its trace")
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context=None,
            stacks="python",
            max_entries=self.capacity,
            clear_history=True,
        )
        self.started = True
        CUDAFreeCompletedTracker._active_session = self
        try:
            self._read()
        except BaseException:
            self.__exit__(*__import__("sys").exc_info())
            raise
        return self

    def capture(self, tensors, *, label):
        if any(token["label"] == label for token in self.tokens):
            raise ValueError("one unique layer label per native capture required")
        snapshot = self._read()
        token = bind_generations(
            self.events,
            snapshot,
            tensor_owners(tensors, device=self.index),
            device=self.index,
            label=label,
        )
        self.tokens.append(token)
        self._token_hashes[id(token)] = _sha(token)
        return token

    def verify(self, token, *, synchronize=True):
        if synchronize is not True:
            raise ValueError("retirement requires an explicit device synchronization boundary")
        if not any(token is registered for registered in self.tokens):
            raise ValueError("retirement token was not captured by this owned session")
        if _sha(token) != self._token_hashes.get(id(token)):
            raise ValueError("registered native ownership token was mutated")
        torch.cuda.synchronize(self.index)
        # synchronize completes CUDA work; the allocator harvests recorded
        # stream events on its next allocation.  This is a verification-only
        # one-byte allocation, not hidden K/V or compared cache padding.
        trigger = torch.empty(1, dtype=torch.uint8, device=self.device)
        del trigger
        self._read()
        receipt = verify_retired(self.events, token)
        if any(receipt["token_sha256"] == prior["token_sha256"] for prior in self.receipts):
            raise ValueError("one retirement verification per original allocation token required")
        self.receipts.append(receipt)
        return receipt

    def summary(self):
        if self.failed or not self.closed:
            raise RuntimeError("only a successful, closed bounded history can be summarized")
        complete = len(self.tokens) == len(self.receipts) and bool(self.tokens)
        return dict(
            schema="native_cuda_generation_retirement_summary_v1",
            complete=complete,
            original_history_enabled=self.original_history_enabled,
            existing_history_preserved=True,
            device=self.index,
            events=list(self.events),
            trace_count=len(self.events),
            capacity=self.capacity,
            trace_sha256=_sha(self.events),
            trace_prefix_complete=True,
            no_ring_truncation=True,
            tokens=list(self.tokens),
            receipts=list(self.receipts),
            all_registered_generations_free_completed=complete,
            whole_request_KV_peak_certified=False,
        )

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is not None:
                self.failed = True
            elif self.started and not self.closed:
                self._read()
                if len(self.tokens) != len(self.receipts):
                    self.failed = True
                    raise RuntimeError("not every registered original native owner was retired")
        finally:
            if self.started and not self.closed:
                # Only our own (previously absent) recording is stopped/cleared.
                torch.cuda.memory._record_memory_history(
                    enabled=None, context=None, clear_history=True
                )
            self.closed = True
            if CUDAFreeCompletedTracker._active_session is self:
                CUDAFreeCompletedTracker._active_session = None
        return False


def selftest(*, device="cuda"):
    """Small real-CUDA aliases, delayed stream frees and generation diagnostic."""
    device = torch.device(device)
    with CUDAFreeCompletedTracker(device=device, max_entries=20_000) as tracker:
        original = torch.empty((2, 17, 128), dtype=torch.bfloat16, device=device)
        alias = original[0]
        token = tracker.capture((original, alias), label="layer_0")
        del original
        alias_rejected = False
        try:
            tracker.verify(token)
        except ValueError:
            alias_rejected = True
        del alias
        first_receipt = tracker.verify(token)
        later = torch.empty((2, 17, 128), dtype=torch.bfloat16, device=device)
        later_address = later.untyped_storage().data_ptr()
        second = tracker.capture((later,), label="layer_1")
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(2_000_000)
            later.fill_(1)
        later.record_stream(stream)
        del later
        second_receipt = tracker.verify(second)
    result = tracker.summary()
    replay = verify_receipts(result["receipts"], expected_layers=2)
    original_address = token["allocations"][0]["address"]
    return dict(
        all_passed=alias_rejected
        and replay["complete"]
        and first_receipt["actual_free_completed_verified"]
        and second_receipt["actual_free_completed_verified"]
        and not torch._C._cuda_isHistoryEnabled(),
        actual_retirement_receipts=len(result["receipts"]),
        actual_free_completed_generations=replay["generations"],
        alias_retention_correctly_rejected=alias_rejected,
        delayed_stream_free_completed_verified=second_receipt["actual_free_completed_verified"],
        native_address_reuse_observed=later_address == original_address,
        same_address_would_require_distinct_alloc_index=True,
        history_restored_disabled=not torch._C._cuda_isHistoryEnabled(),
        replay=replay,
        history=result,
        no_model_weights_loaded=True,
        whole_request_KV_peak_certified=False,
    )
