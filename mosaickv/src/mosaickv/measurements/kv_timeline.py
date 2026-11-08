"""Time-aligned allocator ownership, including cross-thread KV workspaces.

Use actual alloc/free_completed ordering, never sums of phase maxima. Address
reuse starts a new generation. Unknown allocations are NOT assumed non-KV.
The event reducer is CPU-testable; CUDA capture uses the installed PyTorch
memory.py _record_memory_history/_snapshot APIs (a pinned internal API).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

KV_ROLES = frozenset(("kv_persistent", "kv_workspace", "kv_shared_workspace"))
ROLES = KV_ROLES | {"model", "non_kv", "unknown"}


class AllocationTimeline:
    def __init__(self):
        self.lock = threading.RLock()
        self.active, self.allocations, self.events, self.errors = {}, {}, [], []
        self.generations = {}

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
                elif action in ("free_requested", "free_completed"):
                    key = self.active.get(addr)
                    if key is None:
                        self.errors.append("free_without_observed_allocation")
                    elif action == "free_completed":
                        self.allocations[key]["end"] = seq
                        del self.active[addr]
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
        for row in self.allocations.values():
            if row["role"] != "unknown" or row["baseline"]:
                continue
            frames = by_sequence[row["start"]].get("frames", ())
            found = None
            for frame in frames:
                matches = {
                    (suffix, name, role)
                    for suffix, name, role in rules
                    if frame.get("filename", "").endswith(suffix) and frame.get("name") == name
                }
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
            allocation_generations=len(rows),
            unknown_allocation_generations=len(unknown),
            events=len(self.events),
            records=rows,
            includes_free_requested_until_free_completed=True,
            includes_cross_thread_allocator_events=True,
            phase_peaks_not_added=True,
            excludes="CPU_residuals_and_vendor_allocations_outside_PyTorch_allocator",
        )


class CudaOwnershipSession:
    """Audit-only capture. No GPU tensors/weights are retained by this object."""

    def __init__(self, *, max_entries=200000):
        self.max_entries = max_entries
        self.timeline = AllocationTimeline()
        self.cursor = 0
        self.baseline_trace_count = 0
        self.complete = False
        self.finished = False
        self.snapshot = None

    def __enter__(self):
        import torch

        torch.cuda.synchronize()
        torch.cuda.memory._record_memory_history(
            "all", context="all", stacks="python", max_entries=self.max_entries, clear_history=True
        )
        snapshot = torch.cuda.memory._snapshot()
        self.device = torch.cuda.current_device()
        for segment in snapshot["segments"]:
            if segment["device"] != self.device:
                continue
            for block in segment["blocks"]:
                if block["state"] in ("active_allocated", "active_awaiting_free"):
                    self.timeline.seed(block["address"], block["size"])
        self.baseline_trace_count = len(snapshot["device_traces"][self.device])
        self.cursor = self.baseline_trace_count
        return self

    def poll(self):
        import torch

        snapshot = torch.cuda.memory._snapshot()
        traces = snapshot["device_traces"][self.device]
        if len(traces) < self.cursor or len(traces) >= self.max_entries:
            raise RuntimeError("allocator history overwritten: certification refused")
        self.timeline.ingest(traces[self.cursor :])
        self.cursor = len(traces)
        self.snapshot = snapshot
        return len(self.timeline.events)

    def mark(self, tensors, role, evidence):
        with self.timeline.lock:
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
            self.poll()
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
            self.complete = kind is None
            self.finished = True
        finally:
            torch.cuda.memory._record_memory_history(None)
        return False

    def report(self, *, opaque_workspace_classified=False, scope="whole_request"):
        return self.timeline.report(
            complete=self.finished and self.complete,
            history_complete=self.cursor < self.max_entries,
            opaque_workspace_classified=opaque_workspace_classified,
            scope=scope,
        )
