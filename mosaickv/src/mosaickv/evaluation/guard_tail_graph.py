"""Request-local replay of unchanged native readout and FP64 guard operations.

The graph owns activations, not K/V. Its explicit buffers and observed allocator
increment are reported separately, never called zero-cost KV workspace.
"""

import torch

from mosaickv.evaluation.live_anchor_guard import decision_tensors
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


def native_tail(norm, head, parent, active, *, include_readout=False):
    """No fused arithmetic, approximate reduction, changed threshold or labels."""
    parent_norm, active_norm = norm(parent), norm(active)
    parent_native, active_native = head(parent_norm), head(active_norm)
    parent_logits = parent_native.float().reshape(-1)
    active_logits = active_native.float().reshape(-1)
    p, c = parent_logits.double(), active_logits.double()
    finite = p.isfinite().all() & c.isfinite().all()
    passed, record = decision_tensors(p, c, finite)
    result = (
        torch.where(passed, active, parent),
        parent_logits,
        torch.cat((record, finite.double().reshape(1))),
    )
    if include_readout:
        result += (
            torch.where(passed, active_norm, parent_norm),
            torch.where(passed, active_native, parent_native),
        )
    return result


class GuardTail:
    def __init__(
        self,
        norm,
        head,
        *,
        capture_mode="context_manager",
        reuse_readout=False,
        device_records=False,
        maximum_updates=127,
    ):
        if capture_mode not in ("context_manager", "manual_shared_stream"):
            raise ValueError("explicit registered graph-capture implementation required")
        self.capture_mode = capture_mode
        if type(reuse_readout) is not bool:
            raise ValueError("explicit native-readout reuse boolean required")
        self.reuse_readout = reuse_readout
        if type(device_records) is not bool:
            raise ValueError("explicit request-level record transport required")
        self.device_records, self.maximum_updates = device_records, maximum_updates
        self.records = None
        self.norm, self.head = norm, head
        self.inputs = self.outputs = self.graph = None
        self.calls = self.replays = self.captures = 0
        self.explicit_bytes = self.observed_allocator_increment = 0
        self.closed = False

    def _validate(self, parent, active):
        if (
            self.closed
            or (self.device_records and self.calls >= self.maximum_updates)
            or (self.records is not None and self.records.failed)
            or parent.shape != (1, 1, 4096)
            or active.shape != parent.shape
            or parent.dtype != torch.bfloat16
            or active.dtype != parent.dtype
            or not parent.is_cuda
            or parent.device != active.device
        ):
            raise ValueError("open, isolated actual BF16 LLaVA final hidden states required")

    def _capture(self, parent, active):
        torch.cuda.synchronize(parent.device)
        before = torch.cuda.memory_allocated(parent.device)
        if self.device_records:
            from mosaickv.evaluation.device_guard_records import DeviceRecords

            self.records = DeviceRecords(self.maximum_updates, parent.device)
        self.inputs = (parent.clone(), active.clone())
        current = torch.cuda.current_stream(parent.device)
        if self.capture_mode == "manual_shared_stream":
            from mosaickv.evaluation.post_attention_segment import capture_stream

            stream = capture_stream(parent.device)
        else:
            stream = torch.cuda.Stream(device=parent.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                warm = native_tail(
                    self.norm, self.head, *self.inputs, include_readout=self.reuse_readout
                )
            del warm
        current.wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        if self.capture_mode == "manual_shared_stream":
            # torch.cuda.graph.__enter__ additionally collects all Python
            # garbage and empties the allocator cache. The enclosing request
            # already manages those boundaries. Use the same explicit-capture
            # protocol as the validated common epilogue, without changing any
            # captured operator or allowing overlapping captures.
            try:
                with torch.cuda.stream(stream):
                    self.graph.capture_begin(capture_error_mode="global")
                    try:
                        self.outputs = native_tail(
                            self.norm, self.head, *self.inputs, include_readout=self.reuse_readout
                        )
                    finally:
                        self.graph.capture_end()
            except BaseException:
                self.graph.reset()
                raise
        else:
            with torch.cuda.graph(self.graph, stream=stream):
                self.outputs = native_tail(
                    self.norm, self.head, *self.inputs, include_readout=self.reuse_readout
                )
        current.wait_stream(stream)
        torch.cuda.synchronize(parent.device)
        self.explicit_bytes = backing_bytes(
            (
                *self.inputs,
                *self.outputs,
                *((self.records.data,) if self.records is not None else ()),
            )
        )
        self.observed_allocator_increment = max(
            0, torch.cuda.memory_allocated(parent.device) - before
        )
        self.captures += 1

    def run(self, parent, active):
        self._validate(parent, active)
        if self.graph is None:
            self._capture(parent, active)
        for target, source in zip(self.inputs, (parent, active), strict=True):
            target.copy_(source)
        self.graph.replay()
        self.calls += 1
        self.replays += 1
        output, logits, stats = self.outputs[:3]
        if self.records is not None:
            self.records.append(stats)
            return output, logits, None
        cpu = stats.detach().cpu()
        if not bool(cpu[4]):
            raise ValueError("nonfinite logits are an execution failure, not a successful veto")
        # Borrowed output/logits last until the next call. The record is CPU-owned.
        return output, logits, cpu[:4].clone()

    def recorded_statistics(self):
        if self.records is None:
            raise ValueError("explicit device record transport required")
        return self.records.snapshot()

    def selected_readout(self):
        if not self.reuse_readout or self.closed or self.outputs is None:
            raise RuntimeError("live selected graph readout required")
        return self.outputs[3:]

    def report(self):
        return dict(
            calls=self.calls,
            capture_mode=self.capture_mode,
            selected_native_readout_in_graph=self.reuse_readout,
            device_buffered_records=self.device_records,
            record_buffer_activation_bytes=self.records.bytes if self.records is not None else 0,
            recorded_steps=self.records.used if self.records is not None else self.calls,
            records_validated=self.records.checked if self.records is not None else True,
            record_validation_failed=self.records.failed if self.records is not None else False,
            invalid_record_steps=self.records.invalid_steps if self.records is not None else [],
            replays=self.replays,
            captures=self.captures,
            explicit_activation_backing_bytes=self.explicit_bytes,
            observed_graph_allocator_increment_bytes=self.observed_allocator_increment,
            additional_persistent_KV_bytes=0,
            native_FP64_decision_operations=True,
            capture_and_retirement_in_request_time=True,
            whole_request_GPU_peak_certified=False,
            closed=self.closed,
        )

    def close(self):
        try:
            if self.records is not None and not self.records.checked:
                self.records.snapshot()
        finally:
            try:
                if self.graph is not None:
                    self.graph.reset()
            finally:
                self.inputs = self.outputs = self.graph = None
                self.norm = self.head = None
                if self.records is not None:
                    self.records.release()
                self.closed = True


def retire_after_final_update(tail, completed, expected):
    """Retire before the outer prompt pool audits shared RNG metadata owners.

    The returned hidden tensor remains caller-owned until the ordinary final
    readout completes. Graph lifetime must not extend across the enclosing
    pool's strict two-scalar retirement check.
    """
    if type(completed) is not int or type(expected) is not int or not 1 <= completed <= expected:
        raise ValueError("bounded actual guard-update count required")
    if completed == expected:
        tail.close()
