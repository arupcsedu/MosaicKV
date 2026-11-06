"""Default-off request-private split eager replay; adaptive layouts pass through.

This is an engineering diagnostic, not a new selector or benchmark-quality
claim. K and V still overwrite the same charged lease. No model weights,
logical positions, masks, parent numerical repairs or cache values change.
"""

from contextvars import ContextVar
from functools import wraps

import torch

from mosaickv.evaluation.eager_core_operator import Core, opportunities
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.uniform_eager_core import strategy

_ABSENT = object()
_scope = ContextVar("evicastkv_uniform_core_transport", default=_ABSENT)


class Binding:
    def __init__(self, cache):
        self.cache = cache
        self.lengths = tuple(layer.prompt_length for layer in cache.layers)
        updates = max(layer.maximum_decode_updates for layer in cache.layers) - 1
        self.strategy = strategy(self.lengths, updates)
        self.frequencies = opportunities(self.lengths, updates)
        self.audit = cache.audit_workspace
        self.core = None
        self.closed = False
        self.calls = self.workspace = self.warm_library_bytes = 0
        self.capture_checks = 0
        self._before_warm = self._after_warm = None
        self.result = None

    @property
    def eligible(self):
        return self.strategy == "uniform_split_graph"

    def observe_capture(self, phase):
        from mosaickv.evaluation.prompt_graph_reader import allocator_blocks

        torch.cuda.synchronize()
        actual = allocator_blocks()
        if phase == "before_warm":
            self._before_warm = actual
        elif phase == "after_warm":
            self.warm_library_bytes += sum(
                size[1] for ptr, size in actual.items() if ptr not in self._before_warm
            )
            self._after_warm = actual
        elif phase == "after_capture":
            if actual != self._after_warm:
                raise RuntimeError("attention graph captured uncharged private tensor owners")
            reserved = sum(
                segment["total_size"]
                for segment in torch.cuda.memory_snapshot()
                if tuple(segment.get("segment_pool_id", (0, 0))) == self.core.pool
            )
            if reserved:
                raise RuntimeError("explicit-cast attention graph retained a private pool")
            self.capture_checks += 1
        else:
            raise ValueError("unknown graph observation phase")

    def scores(self, query, key, mask, scaling):
        if self.closed or not self.eligible or scaling != 128**-0.5:
            raise RuntimeError("active eligible original eager scaling required")
        lease = self.cache.prompt_graph_pool.buffer
        if (
            lease is None
            or query.dtype != torch.bfloat16
            or tuple(query.shape) != (1, 32, 1, 128)
            or query.device.type != "cuda"
            or key.data_ptr() != lease.data_ptr()
            or key.stride() != lease.stride()
        ):
            raise NotImplementedError("original uniform BF16 fixed CUDA lease required")
        if self.core is None:
            self.core = Core(
                lease,
                mask_dtype=None if mask is None else mask.dtype,
                graph=True,
                frequencies=self.frequencies,
                reuse_views=True,
                explicit_cast=True,
                capture_observer=self.observe_capture if self.audit else None,
            )
            self.workspace = backing_bytes(self.core.tensors())
            if self.workspace > 384 * 1024:
                raise RuntimeError("attention workspace exceeds registered bound")
            self.cache.shared_tensors = (*self.cache.shared_tensors, *self.core.tensors())
            self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)
            self.cache.observe()
        self.calls += 1
        return self.core.scores_for(query, mask, key.shape[-2])

    def values(self, length):
        return self.core.values_for(length)

    def close(self, *, failed=False):
        if self.closed:
            return
        core = self.core
        counts = dict(captures=0, replays=0, warm_calls=0, view_builds=0, view_reuses=0)
        if core is not None:
            counts = dict(
                captures=core.captures,
                replays=core.replays,
                warm_calls=core.warm_calls,
                view_builds=core.view_builds,
                view_reuses=core.view_reuses,
            )
            owners = core.tensors()
            owner_ids = set(map(id, owners))
            self.cache.shared_tensors = tuple(
                t for t in self.cache.shared_tensors if id(t) not in owner_ids
            )
            if failed:
                core.abort()
            else:
                core.close()
            del owners
            self.cache.shared_bytes = backing_bytes(self.cache.shared_tensors)
            self.cache.observe()
        self.result = dict(
            requested=True,
            strategy=self.strategy,
            eligible=self.eligible,
            calls=self.calls,
            workspace_bytes=self.workspace,
            extra_KV_buffer_bytes=0,
            capture_owner_checks=self.capture_checks,
            backend_warm_library_bytes_NOT_KV_certified=self.warm_library_bytes,
            graphs_retired=True,
            retained_tensor_owners=0,
            aborted=failed,
            cold_capture_and_retirement_inside_request=True,
            **counts,
        )
        self.core = self.cache = self._before_warm = self._after_warm = None
        self.closed = True


def bind(cache):
    state = _scope.get()
    if state is _ABSENT or state is None:
        return None
    if state[0] is not None:
        raise RuntimeError("only one cache transaction per private request")
    state[0] = Binding(cache)
    return state[0]


def managed(function):
    @wraps(function)
    def call(*args, **kwargs):
        enabled = kwargs.pop("uniform_attention_core", False)
        if type(enabled) is not bool or _scope.get() is not _ABSENT:
            raise ValueError("explicit nonnested uniform attention scope required")
        if enabled and not kwargs.get("prompt_graph_reader", False):
            raise NotImplementedError("fixed prompt graph lease required")
        state = [None] if enabled else None
        token = _scope.set(state)
        try:
            values, packets, details = function(*args, **kwargs)
            if enabled:
                if state[0] is None or not state[0].closed:
                    raise RuntimeError("attention transaction was not bound and retired")
                report = state[0].result
            else:
                report = dict(requested=False)
            return values, packets, dict(details, uniform_attention_core=report)
        finally:
            if state is not None and state[0] is not None and not state[0].closed:
                state[0].close(failed=True)
            _scope.reset(token)

    return call
