"""Request-local replay of unchanged native readout and FP64 guard operations.

The graph owns activations, not K/V. Its explicit buffers and observed allocator
increment are reported separately, never called zero-cost KV workspace.
"""

import torch

from mosaickv.evaluation.live_anchor_guard import decision_tensors
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


def native_tail(norm, head, parent, active):
    """No fused arithmetic, approximate reduction, changed threshold or labels."""
    parent_logits = head(norm(parent)).float().reshape(-1)
    active_logits = head(norm(active)).float().reshape(-1)
    p, c = parent_logits.double(), active_logits.double()
    finite = p.isfinite().all() & c.isfinite().all()
    passed, record = decision_tensors(p, c, finite)
    return (
        torch.where(passed, active, parent),
        parent_logits,
        torch.cat((record, finite.double().reshape(1))),
    )


class GuardTail:
    def __init__(self, norm, head):
        self.norm, self.head = norm, head
        self.inputs = self.outputs = self.graph = None
        self.calls = self.replays = 0
        self.explicit_bytes = self.observed_allocator_increment = 0
        self.closed = False

    def _validate(self, parent, active):
        if (
            self.closed
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
        self.inputs = (parent.clone(), active.clone())
        current = torch.cuda.current_stream(parent.device)
        stream = torch.cuda.Stream(device=parent.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(2):
                warm = native_tail(self.norm, self.head, *self.inputs)
            del warm
        current.wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = native_tail(self.norm, self.head, *self.inputs)
        current.wait_stream(stream)
        torch.cuda.synchronize(parent.device)
        self.explicit_bytes = backing_bytes((*self.inputs, *self.outputs))
        self.observed_allocator_increment = max(
            0, torch.cuda.memory_allocated(parent.device) - before
        )

    def run(self, parent, active):
        self._validate(parent, active)
        if self.graph is None:
            self._capture(parent, active)
        for target, source in zip(self.inputs, (parent, active), strict=True):
            target.copy_(source)
        self.graph.replay()
        self.calls += 1
        self.replays += 1
        output, logits, stats = self.outputs
        cpu = stats.detach().cpu()
        if not bool(cpu[4]):
            raise ValueError("nonfinite logits are an execution failure, not a successful veto")
        # Borrowed output/logits last until the next call. The record is CPU-owned.
        return output, logits, cpu[:4].clone()

    def report(self):
        return dict(
            calls=self.calls,
            replays=self.replays,
            captures=int(self.graph is not None),
            explicit_activation_backing_bytes=self.explicit_bytes,
            observed_graph_allocator_increment_bytes=self.observed_allocator_increment,
            additional_persistent_KV_bytes=0,
            native_FP64_decision_operations=True,
            capture_and_retirement_in_request_time=True,
            whole_request_GPU_peak_certified=False,
            closed=self.closed,
        )

    def close(self):
        if self.graph is not None:
            self.graph.reset()
        self.inputs = self.outputs = self.graph = None
        self.norm = self.head = None
        self.closed = True
