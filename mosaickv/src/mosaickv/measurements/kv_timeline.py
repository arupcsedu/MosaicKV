"""Time-aligned allocator ownership, including cross-thread KV workspaces.

Use actual alloc/free_completed ordering, never sums of phase maxima. Address
reuse starts a new generation. Unknown allocations are NOT assumed non-KV.
The event reducer is CPU-testable; CUDA capture uses the installed PyTorch
memory.py _record_memory_history/_snapshot APIs (a pinned internal API).
"""

from __future__ import annotations

import threading
import time
from bisect import bisect_right
from contextlib import contextmanager
from itertools import pairwise

KV_ROLES = frozenset(("kv_persistent", "kv_workspace", "kv_shared_workspace"))
ROLES = KV_ROLES | {"model", "non_kv", "unknown"}


class AllocationTimeline:
    def __init__(self):
        self.lock = threading.RLock()
        self.active, self.allocations, self.events, self.errors = {}, {}, [], []
        self.generations = {}
        self.stacks, self.stack_indices = [], {}
        self.error_details = []
        self.by_address = {}
        self.timestamp_index = {}

    def seed(self, address, size, role="unknown"):
        with self.lock:
            self._allocate(address, size, role, -1, True)

    def _allocate(self, address, size, role, sequence, baseline=False):
        if role not in ROLES or type(size) is not int or size < 0:
            raise ValueError("explicit allocation role and nonnegative backing size required")
        if address in self.active:
            self.errors.append("overlapping_address_without_free_completed")
            return
        generation = self.generations.get(address, -1) + 1
        self.generations[address] = generation
        key = (address, generation)
        self.active[address] = key
        self.by_address.setdefault(address, []).append(key)
        self.timestamp_index.pop(address, None)
        self.allocations[key] = dict(
            address=address,
            generation=generation,
            bytes=size,
            role=role,
            start=sequence,
            end=None,
            baseline=baseline,
            evidence=[],
        )

    def ingest(self, traces):
        """Trace size is allocator backing bytes; preserve pending-free owners."""
        with self.lock:
            for trace in traces:
                seq = len(self.events)
                action, addr = trace["action"], trace.get("addr")
                self.events.append(dict(sequence=seq, **trace))
                if action == "alloc":
                    self._allocate(addr, trace["size"], "unknown", seq)
                    key = self.active.get(addr)
                    if key is not None:
                        frames = tuple(
                            (f.get("filename", ""), f.get("name", ""), f.get("line", -1))
                            for f in trace.get("frames", ())
                        )
                        if frames not in self.stack_indices:
                            self.stack_indices[frames] = len(self.stacks)
                            self.stacks.append(frames)
                        self.allocations[key].update(
                            stack_id=self.stack_indices[frames],
                            stream=trace.get("stream"),
                            time_us=trace.get("time_us"),
                        )
                elif action in ("free_requested", "free_completed"):
                    key = self.active.get(addr)
                    if key is None:
                        self.errors.append("free_without_observed_allocation")
                        self.error_details.append(dict(sequence=seq, **trace))
                    elif action == "free_completed":
                        self.allocations[key]["end"] = seq
                        self.allocations[key]["end_time_us"] = trace.get("time_us")
                        del self.active[addr]
                    elif key is not None:
                        self.allocations[key]["free_requested_at"] = seq
                        self.allocations[key]["free_requested_time_us"] = trace.get("time_us")
                elif action in ("segment_free", "segment_unmap"):
                    limit = addr + trace["size"]
                    for address, key in tuple(self.active.items()):
                        row = self.allocations[key]
                        if addr <= address < limit:
                            if address + row["bytes"] > limit:
                                self.errors.append("partial_physical_segment_retirement")
                                continue
                            row.update(end=seq, end_time_us=trace.get("time_us"), retirement=action)
                            del self.active[address]
                elif action == "oom":
                    self.errors.append("CUDA_OOM_in_request")

    def classify(self, address, role, evidence):
        with self.lock:
            if role not in ROLES or role == "unknown" or not evidence:
                raise ValueError("explicit reviewed role and attribution evidence required")
            if address not in self.active:
                raise ValueError("classification requires a currently live backing allocation")
            row = self.allocations[self.active[address]]
            old = row["role"]
            if old != "unknown" and old != role:
                # A retained tensor is also observed during its construction.
                # Both roles are KV; count its backing ONCE for its full life.
                if {old, role} <= KV_ROLES:
                    role = (
                        "kv_shared_workspace"
                        if "kv_shared_workspace" in (old, role)
                        else "kv_persistent"
                    )
                else:
                    raise ValueError("conflicting KV/non-KV attribution")
            row["role"] = role
            row["evidence"].append(evidence)

    def classify_range(self, start, stop, role, evidence):
        """Attribute a reviewed semantic allocation scope, including dead temps.

        Caller must ensure no unrelated thread allocates in this interval. For
        concurrent work, classify per-allocation from its recorded stack instead.
        """
        if role not in ROLES or role == "unknown" or not evidence:
            raise ValueError("explicit semantic allocation scope required")
        with self.lock:
            for row in self.allocations.values():
                if start <= row["start"] < stop:
                    if row["role"] not in ("unknown", role) and not {row["role"], role} <= KV_ROLES:
                        raise ValueError("conflicting semantic scope")
                    if row["role"] == "unknown":
                        row["role"] = role
                    row["evidence"].append(evidence)

    def classify_stacks(self, rules):
        """Rules are exact (filename suffix, function, role); unknown stays unknown.

        Nearest matching allocation frame wins, not an outer model caller. The
        same allocator sequence includes all worker threads and CUDA streams.
        """
        by_sequence = {event["sequence"]: event for event in self.events}
        cached = {}
        for row in self.allocations.values():
            if row["role"] != "unknown" or row["baseline"]:
                continue
            frames = by_sequence[row["start"]].get("frames", ())
            found = None
            for frame in frames:
                cache_key = frame.get("filename", ""), frame.get("name"), frame.get("line")
                if cache_key not in cached:
                    cached[cache_key] = {
                        (rule[0], rule[1], rule[2])
                        for rule in rules
                        if cache_key[0].endswith(rule[0])
                        and cache_key[1] == rule[1]
                        and (len(rule) == 3 or cache_key[2] == rule[3])
                    }
                matches = cached[cache_key]
                if len({role for _suffix, _name, role in matches}) > 1:
                    raise ValueError("ambiguous allocation classification rules")
                if matches:
                    found = sorted(matches)[0]
                    break
            if found is not None:
                suffix, name, role = found
                if role not in ROLES - {"unknown"}:
                    raise ValueError("invalid stack classification role")
                row["role"] = role
                row["evidence"].append(f"reviewed_stack:{suffix}:{name}")

    def classify_at(self, address, role, evidence, begin_us, end_us):
        """Deferred live-tensor observation matched to allocator wall timestamps.

        Match the entire observation interval, never guess the latest address
        generation. The clock domain must be checked by the CUDA session.
        """
        if role not in ROLES - {"unknown"} or not evidence or begin_us > end_us:
            raise ValueError("explicit valid timestamped attribution required")
        if address not in self.timestamp_index:
            keys = self.by_address.get(address, ())
            starts = [
                float("-inf")
                if self.allocations[k]["baseline"]
                else self.allocations[k].get("time_us")
                for k in keys
            ]
            if any(v is None for v in starts) or any(a > b for a, b in pairwise(starts)):
                self.errors.append("missing_or_nonmonotonic_allocator_timestamp")
                return
            self.timestamp_index[address] = starts, keys
        starts, keys = self.timestamp_index[address]
        position = bisect_right(starts, begin_us) - 1
        possible = []
        for index in (position - 1, position):
            if index < 0:
                continue
            key = keys[index]
            row = self.allocations[key]
            death = row.get("end_time_us")
            if starts[index] <= begin_us and (
                row["end"] is None or (death is not None and death >= end_us)
            ):
                possible.append(key)
        if len(possible) != 1:
            self.errors.append("ambiguous_deferred_owner_timestamp")
            return
        row = self.allocations[possible[0]]
        old = row["role"]
        if old not in ("unknown", role) and not {old, role} <= KV_ROLES:
            self.errors.append("conflicting_deferred_owner_role")
            return
        row["role"] = role if old == "unknown" else old
        row["evidence"].append(evidence)
        row.setdefault("attribution_intervals_us", []).append((begin_us, end_us, role, evidence))
        return possible[0]

    def report(
        self, *, complete, history_complete, opaque_workspace_classified, scope="whole_request"
    ):
        if scope not in ("whole_request", "bounded_operator"):
            raise ValueError("explicit audited scope required")
        rows = list(self.allocations.values())
        delta, upper_delta, peak, upper_peak, live, upper = {}, {}, 0, 0, 0, 0
        for row in rows:
            start, end, size = row["start"], row["end"], row["bytes"]
            if row["role"] in KV_ROLES:
                delta[start] = delta.get(start, 0) + size
                if end is not None:
                    delta[end] = delta.get(end, 0) - size
            if row["role"] in KV_ROLES | {"unknown"}:
                upper_delta[start] = upper_delta.get(start, 0) + size
                if end is not None:
                    upper_delta[end] = upper_delta.get(end, 0) - size
        for seq in sorted(set(delta) | set(upper_delta)):
            live += delta.get(seq, 0)
            upper += upper_delta.get(seq, 0)
            peak, upper_peak = max(peak, live), max(upper_peak, upper)
        unknown = [r for r in rows if r["role"] == "unknown"]
        blockers = list(dict.fromkeys(self.errors))
        if not rows or not self.events:
            blockers.append("empty_allocator_capture")
        if not complete:
            blockers.append("request_incomplete")
        if not history_complete:
            blockers.append("allocator_history_incomplete_or_overwritten")
        if unknown:
            blockers.append("unclassified_allocation_generations")
        if not opaque_workspace_classified:
            blockers.append("opaque_allocator_workspace_not_classified")
        certified = not blockers
        return dict(
            schema_version=1,
            boundary="time_aligned_PyTorch_CUDA_KV_backing_including_declared_KV_workspace",
            exact_whole_request_KV_peak_certified=certified and scope == "whole_request",
            exact_scoped_KV_peak_certified=certified,
            scope=scope,
            exact_KV_peak_bytes=peak if certified else None,
            known_KV_peak_lower_bound_bytes=peak,
            KV_peak_upper_bound_including_unknown_bytes=upper_peak,
            blocks_exact_certification=blockers,
            error_details=self.error_details[:50],
            allocation_generations=len(rows),
            unknown_allocation_generations=len(unknown),
            events=len(self.events),
            records=rows,
            stack_table=self.stacks,
            allocator_event_counts={
                action: sum(e["action"] == action for e in self.events)
                for action in (
                    "alloc",
                    "free_requested",
                    "free_completed",
                    "segment_free",
                    "segment_unmap",
                    "snapshot",
                )
            },
            time_alignment="single_device_allocator_sequence_across_all_threads_and_streams",
            includes_free_requested_until_free_completed=True,
            includes_cross_thread_allocator_events=True,
            phase_peaks_not_added=True,
            excludes="CPU_residuals_and_vendor_allocations_outside_PyTorch_allocator",
        )


class CudaOwnershipSession:
    """Audit-only capture. No GPU tensors/weights are retained by this object."""

    def __init__(self, *, max_entries=200000, deferred_marks=False, clear_library_workspace=False):
        self.max_entries = max_entries
        self.timeline = AllocationTimeline()
        self.cursor = 0
        self.baseline_trace_count = 0
        self.complete = False
        self.finished = False
        self.snapshot = None
        self.deferred_marks, self.clear_library_workspace = deferred_marks, clear_library_workspace
        self.marks, self.clock_checks = [], []
        self.beacons = []

    def __enter__(self):
        import torch

        torch.cuda.synchronize()
        if self.clear_library_workspace:
            # Real cache release, not subtraction of an arbitrary baseline.
            # This is excluded instrumentation applied to BOTH audit arms.
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.synchronize()
        torch.cuda.memory._record_memory_history(
            "all", context="all", stacks="python", max_entries=self.max_entries, clear_history=True
        )
        begin = time.time_ns() // 1000
        snapshot = torch.cuda.memory._snapshot()
        end = time.time_ns() // 1000
        self.device = torch.cuda.current_device()
        for segment in snapshot["segments"]:
            if segment["device"] != self.device:
                continue
            for block in segment["blocks"]:
                if block["state"] in ("active_allocated", "active_awaiting_free"):
                    self.timeline.seed(block["address"], block["size"])
        self.baseline_trace_count = len(snapshot["device_traces"][self.device])
        self.cursor = self.baseline_trace_count
        self._check_clock(snapshot, begin, end)
        if self.deferred_marks:
            self._beacon()
        return self

    def _beacon(self):
        import torch

        begin = time.time_ns() // 1000
        tensor = torch.empty(1, device=self.device, dtype=torch.uint8)
        address = tensor.untyped_storage().data_ptr()
        end = time.time_ns() // 1000
        self.beacons.append((address, begin, end))
        # This audit allocation has no inference role and does NOT remain live.
        del tensor

    def _check_clock(self, snapshot, begin, end):
        markers = [e for e in snapshot["device_traces"][self.device] if e["action"] == "snapshot"]
        stamp = markers[-1].get("time_us") if markers else None
        self.clock_checks.append(stamp is not None and begin <= stamp <= end)

    def poll(self):
        import torch

        begin = time.time_ns() // 1000
        snapshot = torch.cuda.memory._snapshot()
        end = time.time_ns() // 1000
        self._check_clock(snapshot, begin, end)
        traces = snapshot["device_traces"][self.device]
        if len(traces) < self.cursor or len(traces) >= self.max_entries:
            raise RuntimeError("allocator history overwritten: certification refused")
        self.timeline.ingest(traces[self.cursor :])
        self.cursor = len(traces)
        self.snapshot = snapshot
        return len(self.timeline.events)

    def mark(self, tensors, role, evidence):
        with self.timeline.lock:
            if self.deferred_marks:
                before = time.time_ns() // 1000
                pointers = {
                    t.untyped_storage().data_ptr() for t in tensors if t.is_cuda and t.numel()
                }
                after = time.time_ns() // 1000
                self.marks.extend((address, role, evidence, before, after) for address in pointers)
                return
            self.poll()
            for tensor in tensors:
                if tensor.is_cuda and tensor.numel():
                    self.timeline.classify(tensor.untyped_storage().data_ptr(), role, evidence)

    @contextmanager
    def serialized_region(self, role, evidence):
        """Only for a bounded allocation-isolated operator diagnostic."""
        with self.timeline.lock:
            begin = self.poll()
            try:
                yield
            finally:
                end = self.poll()
                self.timeline.classify_range(begin, end, role, evidence)

    def __exit__(self, kind, value, trace):
        import torch

        try:
            torch.cuda.synchronize()
            if self.deferred_marks:
                self._beacon()
            self.poll()
            if self.clear_library_workspace:
                visible = {
                    block["address"]
                    for segment in self.snapshot["segments"]
                    if segment["device"] == self.device
                    for block in segment["blocks"]
                    if block["state"] in ("active_allocated", "active_awaiting_free")
                }
                before = {a: key for a, key in self.timeline.active.items() if a in visible}
                torch._C._cuda_clearCublasWorkspaces()
                torch.cuda.synchronize()
                self.poll()
                for address in before.keys() - self.timeline.active.keys():
                    row = self.timeline.allocations[before[address]]
                    if row["role"] == "model":
                        raise RuntimeError(
                            "library workspace release illegally retired a model owner"
                        )
                    row["pre_workspace_classification_role"] = row["role"]
                    row["role"] = "kv_shared_workspace"
                    row["evidence"].append(
                        "actual_immutable_model_preserving_cuda_clearCublasWorkspaces_retirement"
                    )
            if self.deferred_marks:
                offsets = []
                beacon_rows = []
                for ordinal, (address, begin, end) in enumerate(self.beacons):
                    candidates = [
                        self.timeline.allocations[key]
                        for key in self.timeline.by_address.get(address, ())
                        if not self.timeline.allocations[key]["baseline"]
                        and self.timeline.allocations[key]["bytes"] == 512
                    ]
                    exact = [
                        row
                        for row in candidates
                        if any(
                            filename.endswith("/mosaickv/measurements/kv_timeline.py")
                            and name == "_beacon"
                            for filename, name, _line in self.timeline.stacks[row["stack_id"]]
                        )
                    ]
                    if not exact:
                        continue
                    row = exact[0] if ordinal == 0 else exact[-1]
                    offsets.append((begin - row["time_us"], end - row["time_us"]))
                    beacon_rows.append(row)
                lower = max((a for a, _b in offsets), default=1)
                upper = min((b for _a, b in offsets), default=0)
                if len(offsets) != 2 or lower > upper:
                    self.timeline.errors.append("allocator_and_owner_clock_domains_not_verified")
                else:
                    self.clock_translation = dict(
                        minimum_offset_us=lower, maximum_offset_us=upper, beacons=offsets
                    )
                    for row in beacon_rows:
                        row["role"] = "non_kv"
                        row["evidence"].append("explicit_ephemeral_allocator_clock_beacon")
                    for address, role, evidence, before, after in self.marks:
                        self.timeline.classify_at(
                            address, role, evidence, before - upper, after - lower
                        )
            actual = {
                block["address"]: block["size"]
                for segment in self.snapshot["segments"]
                if segment["device"] == self.device
                for block in segment["blocks"]
                if block["state"] in ("active_allocated", "active_awaiting_free")
            }
            observed = {
                address: self.timeline.allocations[key]["bytes"]
                for address, key in self.timeline.active.items()
            }
            if actual != observed:
                self.timeline.errors.append("final_live_backing_snapshot_mismatch")
                self.timeline.error_details.append(
                    dict(
                        kind="snapshot_mismatch",
                        missing=list(actual.keys() - observed.keys())[:20],
                        unexpected=list(observed.keys() - actual.keys())[:20],
                        sizes=[
                            (a, actual[a], observed[a])
                            for a in actual.keys() & observed.keys()
                            if actual[a] != observed[a]
                        ][:20],
                    )
                )
            self.complete = kind is None
            self.finished = True
        finally:
            torch.cuda.memory._record_memory_history(None)
        return False

    def report(self, *, opaque_workspace_classified=False, scope="whole_request"):
        report = self.timeline.report(
            complete=self.finished and self.complete,
            history_complete=self.cursor < self.max_entries,
            opaque_workspace_classified=opaque_workspace_classified,
            scope=scope,
        )
        report["clock_calibration"] = getattr(self, "clock_translation", None)
        return report
