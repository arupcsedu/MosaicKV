"""Opt-in shared reconstruction workspace; original HF attention is unchanged.

Graphs capture only immutable prompt/sealed-history reconstruction. Mutable
tails copy outside the graph. Workspace is charged once, and released before
source-seal construction and at request completion. No cross-request reuse.
"""

from typing import Any

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.projection_cache import ScratchAudit, identity
from mosaickv.evaluation.source_history import SourceHistoryCache, SourceHistoryLayer


class SharedSourceCache(SourceHistoryCache):
    shared_tensors: tuple[Any, ...]

    def __init__(self, *, audit_workspace: bool, constructor: str, graphs: bool) -> None:
        super().__init__(audit_workspace=audit_workspace, constructor=constructor)
        self.graphs_enabled = graphs
        self.workspace: tuple[Any, ...] = ()
        self.workspace_allocations = 0
        self.graph_captures = 0
        self.graph_replays = 0
        self.graph_private_allocation_bytes = 0
        self.workspace_capacity = 0

    def acquire(self, layer: Any) -> tuple[Any, ...]:
        if any(other.leased for other in self.layers):
            raise RuntimeError("shared workspace cannot have overlapping layer leases")
        if not self.workspace:
            heads = {x.prompt_shape[1] for x in self.layers}
            devices = {x.representation.private()[0].device for x in self.layers}
            if len(heads) != 1 or len(devices) != 1:
                raise ValueError("one device and common MHA geometry required")
            capacity = max(x.prompt_length for x in self.layers) + layer.maximum_decode_updates
            shape = (1, layer.prompt_shape[1], capacity, layer.prompt_shape[-1])
            values = tuple(
                torch.empty(shape, dtype=torch.bfloat16, device=layer.history[0].device)
                for _ in range(2)
            )
            self.observe(values)
            self.workspace = values
            self.shared_tensors = (*self.shared_tensors, *values)
            self.shared_bytes = backing_bytes(self.shared_tensors)
            self.workspace_allocations += 1
            self.workspace_capacity = capacity
            self.observe()
        return tuple(t[..., : layer.get_seq_length(), :] for t in self.workspace)

    def release_workspace(self) -> None:
        if any(layer.leased for layer in self.layers):
            raise RuntimeError("cannot release an active shared workspace lease")
        for layer in self.layers:
            layer.reconstruction_graph = None
        tensors = self.workspace
        self.shared_tensors = tuple(
            t for t in self.shared_tensors if all(t is not w for w in tensors)
        )
        self.workspace = ()
        self.shared_bytes = backing_bytes(self.shared_tensors)
        self.observe()

    def storage_record(self) -> dict[str, Any]:
        record = super().storage_record()
        record.update(
            shared_reconstruction_workspace=True,
            workspace_allocations=self.workspace_allocations,
            maximum_workspace_capacity=self.workspace_capacity,
            graph_captures=self.graph_captures,
            graph_replays=self.graph_replays,
            graph_private_allocation_bytes=self.graph_private_allocation_bytes,
            graphs_enabled=self.graphs_enabled,
            active_workspace_bytes=backing_bytes(self.workspace),
            whole_request_peak_certified=False,
        )
        return record


class SharedSourceLayer(SourceHistoryLayer):
    source_pending: bool
    leased: tuple[Any, ...]
    history: tuple[Any, ...]
    history_capacity: int

    def __init__(self, *args: Any, cache: SharedSourceCache, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.cache = cache
        self.reconstruction_graph: Any = None

    def update(self, key_states: Any, value_states: Any, cache_kwargs: Any = None) -> Any:
        del cache_kwargs
        expected = (*self.prompt_shape[:2], 1, self.prompt_shape[-1])
        if (
            not self.source_pending
            or self.leased
            or self.tail_used >= 16
            or self.history_length >= self.maximum_decode_updates
            or key_states.shape != expected
            or value_states.shape != expected
            or any(
                t.dtype != torch.bfloat16 or t.device != self.history[0].device
                for t in (key_states, value_states)
            )
        ):
            raise ValueError(
                "coordinated source capture, released lease and bounded BF16 token required"
            )
        if not self.history_capacity:
            shape = (*expected[:2], 16, expected[-1])
            tails = tuple(
                torch.empty(shape, dtype=t.dtype, device=t.device)
                for t in (key_states, value_states)
            )
            self.observe(tails)
            self.history, self.history_capacity = tails, 16
            self.history_bytes = backing_bytes(tails)
            self.resident_bytes += self.history_bytes
        for target, source in zip(self.history, (key_states, value_states), strict=True):
            target[..., self.tail_used : self.tail_used + 1, :].copy_(source)
        self.tail_used += 1
        self.history_length += 1
        self.source_pending = False
        self.leased = self.cache.acquire(self)
        self.lease_bytes = 0  # the unique shared owner is charged by the cache
        self.observe()
        self.materialize()
        return self.leased

    def fill_immutable(self) -> None:
        self.representation.fill(self.attention, self.leased, direct=True)
        offset = self.prompt_length
        for format in self.source_sealed:
            end = offset + format.shape[-2]
            targets = tuple(t[..., offset:end, :] for t in self.leased)
            format.fill(self.attention, targets, direct=True)
            offset = end
        if offset + self.tail_used != self.get_seq_length():
            raise RuntimeError("shared reconstruction lost or duplicated original history")

    def materialize(self) -> None:
        if self.cache.graphs_enabled and self.history[0].device.type != "cuda":
            raise NotImplementedError("CUDA graph replay requires a CUDA device")
        if self.reconstruction_graph is None:
            excluded = self.weights | {identity(t) for t in self.owned_tensors()}
            excluded.update(identity(t) for t in self.cache.shared_tensors)
            mode = ScratchAudit(excluded=excluded, observe=self.observe) if self.audit else None
            from contextlib import nullcontext

            # Warm/compile the exact original kernels and audit allocations.
            with mode if mode is not None else nullcontext():
                self.fill_immutable()
            if mode is not None:
                self.workspace_operations += mode.operations
            if self.cache.graphs_enabled:
                graph = torch.cuda.CUDAGraph()
                before = torch.cuda.memory_allocated()
                with torch.cuda.graph(graph):
                    self.fill_immutable()
                additional = max(0, torch.cuda.memory_allocated() - before)
                self.cache.graph_private_allocation_bytes += additional
                if additional:
                    raise RuntimeError(
                        "unowned graph-private tensor allocation blocks storage certification"
                    )
                self.reconstruction_graph = graph
                self.cache.graph_captures += 1
        elif not self.cache.graphs_enabled:
            self.fill_immutable()
        if self.reconstruction_graph is not None:
            self.reconstruction_graph.replay()
            self.cache.graph_replays += 1
        offset = self.prompt_length + 16 * len(self.source_sealed)
        for tail, target in zip(self.history, self.leased, strict=True):
            target[..., offset : offset + self.tail_used, :].copy_(tail[..., : self.tail_used, :])
