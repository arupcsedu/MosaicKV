"""Bounded request-private scheduling of unchanged native repair construction.

Only finalized, sealed prompt packets are eligible. One worker/stream/handle,
at most two submitted layers, no weights or source mutations, no cache across
requests. All numerical correction and optional stride checks still execute.
"""

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from functools import wraps

import torch

_scope = ContextVar("evicastkv_prefill_native_overlap", default=None)


class OrderedQueue:
    """Bounded ordered executor; injectable task permits CPU isolation tests."""

    def __init__(self, task, close, *, maximum_pending=2):
        if type(maximum_pending) is not int or maximum_pending != 2:
            raise ValueError("exactly two bounded pending layers required")
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evicastkv-fit")
        self.task, self.close_task = task, close
        self.pending, self.results = deque(), {}
        self.submitted = self.taken = self.maximum_observed = 0
        self.closed = False

    def drain_one(self):
        index, future = self.pending.popleft()
        self.results[index] = future.result()

    def submit(self, index, item):
        if self.closed or type(index) is not int or index != self.submitted:
            raise RuntimeError("one ordered private sealed layer required")
        if len(self.pending) == 2:
            self.drain_one()
        self.pending.append((index, self.executor.submit(self.task, index, item)))
        self.submitted += 1
        self.maximum_observed = max(self.maximum_observed, len(self.pending))

    def take(self, index):
        if self.closed or index != self.taken or index >= self.submitted:
            raise RuntimeError("exactly one ordered native-conversion consumer required")
        while index not in self.results:
            self.drain_one()
        self.taken += 1
        return self.results.pop(index)

    def close(self, *, complete):
        if self.closed:
            return
        try:
            while self.pending:
                self.drain_one()
            if complete and (self.submitted != self.taken or self.results):
                raise RuntimeError("all scheduled native layers must be consumed")
        finally:
            # Cleanup runs on the handle's owner thread even after task failure.
            try:
                self.executor.submit(self.close_task).result()
            finally:
                self.executor.shutdown(wait=True, cancel_futures=True)
                self.pending.clear()
                self.results.clear()
                self.closed = True


class Pipeline:
    def __init__(self, *, audit, frozen, source_rows=128):
        self.audit, self.frozen = audit, frozen
        self.source_rows = source_rows
        self.stream = self.engine = None
        self.queue = OrderedQueue(self.fit, self.retire)
        self.engines = self.layers = self.audit_operations = self.temporary_peak = 0
        self.fit_wall = self.join_wall = 0.0
        self.retained_repairs = []
        self.completed_repair_bytes = 0
        self.complete = False

    @torch.inference_mode()
    def fit(self, index, item):
        from mosaickv.evaluation.joint_incoming_projection import _scope as joint_scope
        from mosaickv.evaluation.joint_incoming_projection import engine_factory
        from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
        from mosaickv.evaluation.lt_source_format import fit, verify_strides
        from mosaickv.evaluation.projection_cache import ScratchAudit, identity

        base, attention, ready = item
        started = time.monotonic()
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=base.cos.device)
        token = joint_scope.set(
            dict(
                policy="kv",
                frozen_plan=self.frozen,
                frozen_plan_engines=0,
                source_rows=self.source_rows,
            )
        )
        try:
            with torch.cuda.device(base.cos.device), torch.cuda.stream(self.stream):
                self.stream.wait_event(ready)
                if self.engine is None:
                    self.engine = engine_factory(base)
                    self.engines += 1
                excluded = {identity(t) for t in (*base.tensors(), *attention.parameters())}

                def observe(tensors):
                    extra = (
                        *tensors,
                        *self.retained_repairs,
                        self.engine.source,
                        self.engine.output,
                        *self.engine.extra_tensors,
                    )
                    self.temporary_peak = max(self.temporary_peak, backing_bytes(extra))

                mode = ScratchAudit(excluded=excluded, observe=observe) if self.audit else None
                from contextlib import nullcontext

                with mode if mode is not None else nullcontext():
                    candidate = fit(base, attention, self.engine, direct_validation=True)
                    if self.audit:
                        verify_strides(candidate, attention, self.engine)
                self.audit_operations += 0 if mode is None else mode.operations
                self.retained_repairs.extend(t for p in candidate.repairs for t in p.tensors)
                self.completed_repair_bytes = backing_bytes(tuple(self.retained_repairs))
                self.layers += 1
                self.fit_wall += time.monotonic() - started
                return base, candidate
        finally:
            joint_scope.reset(token)

    def submit(self, index, packet, attention):
        from mosaickv.evaluation.lt_source_format import map_sources

        bases = []

        def collect(base):
            bases.append(base)
            return base

        map_sources(packet, collect)
        if len(bases) != 1 or bases[0].cos.device.type != "cuda":
            raise NotImplementedError("exactly one finalized CUDA source per sealed layer required")
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(bases[0].cos.device))
        self.queue.submit(index, (bases[0], attention, ready))

    @torch.inference_mode()
    def retire(self):
        if self.engine is not None:
            # This is called only after main-thread prefill/consumer completion,
            # never between submitted layers. Handles remain thread-private.
            self.engine.close()
            self.engine = None
        if self.stream is not None:
            self.stream.synchronize()
            self.stream = None
        self.retained_repairs.clear()

    def convert(self, packets, attentions, *, policy, audit):
        from mosaickv.evaluation.joint_incoming_projection import _scope as joint_scope
        from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
        from mosaickv.evaluation.lt_source_format import map_sources

        if policy != "direct" or audit is not self.audit or len(packets) != len(attentions):
            raise ValueError("unchanged complete direct-fit controls required")
        started, records, result = time.monotonic(), [], []
        for index, packet in enumerate(packets):
            original, candidate = self.queue.take(index)

            def restore(base, original=original, candidate=candidate):
                if base is not original:
                    raise RuntimeError("a scheduled sealed source was replaced before conversion")
                return candidate

            result.append(map_sources(packet, restore))
            records.append(
                dict(
                    layer=index,
                    added_patch_bytes=backing_bytes(
                        tuple(t for p in candidate.repairs for t in p.tensors)
                    ),
                )
            )
        self.queue.close(complete=True)
        self.join_wall = time.monotonic() - started
        if self.layers != len(packets) or self.engines != 1:
            raise RuntimeError("exactly one private fit engine and every sealed source required")
        if sum(r["added_patch_bytes"] for r in records) > 2**20:
            raise RuntimeError("request numerical repair storage exceeds original one-MiB cap")
        if self.frozen:
            joint_scope.get()["frozen_plan_engines"] += self.engines
        self.complete = True
        return tuple(result), dict(
            converted_layers=records,
            eligible_layers=len(records),
            added_patch_bytes=sum(r["added_patch_bytes"] for r in records),
            parent_KV_bitwise=True,
            original_headers_codes_weights_and_positions_unchanged=True,
            no_labels_or_answers_used=True,
            constructor_policy="direct",
            full_parent_KV_word_validation=True,
            exhaustive_stride_audit=audit,
            overlapping_native_construction=True,
        )

    def record(self):
        return dict(
            completed=self.complete,
            submitted=self.queue.submitted,
            consumed=self.queue.taken,
            engine_count=self.engines,
            maximum_pending_layers=self.queue.maximum_observed,
            thread_retired=self.queue.closed,
            explicit_temporary_peak=self.temporary_peak,
            completed_repair_bytes=self.completed_repair_bytes,
            audited_operations=self.audit_operations,
            overlapped_fit_thread_wall_seconds=self.fit_wall,
            final_conversion_join_wall_seconds=self.join_wall,
            worker_time_NOT_additive_to_request_wall=True,
            whole_request_GPU_peak_certified=False,
        )


def sealed(index, packet, attention):
    state = _scope.get()
    if state is not None:
        state.submit(index, packet, attention)


def current():
    return _scope.get()


def managed(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        enabled = kwargs.get("overlap_native_fit", False)
        if type(enabled) is not bool or _scope.get() is not None:
            raise ValueError("explicit nonnested request-private native overlap required")
        active = enabled and args[3] == "mosaickv_exact"
        state = None
        if active:
            if (
                kwargs.get("lt_source_reader") is not True
                or kwargs.get("lt_fit_policy") != "direct"
                or kwargs.get("joint_incoming_projection") != "kv"
                or kwargs.get("static_transport_patches", False)
                or kwargs.get("source_geometry") is not None
                or kwargs.get("full_prompt_native") is not None
                or kwargs.get("hybrid_reader_policy") is not None
                or kwargs.get("text_overlay_policy") is not None
            ):
                raise NotImplementedError(
                    "only unchanged finalized direct Source packets are eligible"
                )
            state = Pipeline(
                audit=kwargs.get("audit_decode_storage", False),
                frozen=kwargs.get("frozen_joint_plan", False),
                source_rows=kwargs.get("source_tile_rows", 128),
            )
        token = _scope.set(state)
        try:
            values, packets, details = function(*args, **kwargs)
            details["overlapping_native_construction"] = dict(
                requested=enabled, eligible=active, report=None if state is None else state.record()
            )
            if state is not None and not state.complete:
                raise RuntimeError("overlapped construction did not reach its retirement barrier")
            return values, packets, details
        finally:
            try:
                if state is not None:
                    state.queue.close(complete=False)
            finally:
                _scope.reset(token)

    return wrapped
