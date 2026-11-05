"""Common lossless native generated-history execution variant, default off.

Original normalized sources, logical phases and all K/V words are preserved.
Construct actual counted repairs against raw history and rebase them whenever
the consolidated GEMM row count changes. Never join incompatible corrections.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from functools import wraps

import torch

from mosaickv.evaluation.encoded_projection import EncodedFormat
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.native_history_feasibility import Engine, validate_rows

_scope = ContextVar("evicastkv_common_native_history", default=None)
MAX_REPAIR_BYTES = 32768


def current():
    return _scope.get()


def phase_guard(cos, sin, n):
    if any(
        t.shape != (1, n, 128)
        or t.dtype != torch.bfloat16
        or not torch.equal(t[..., :64].view(torch.int16), t[..., 64:].view(torch.int16))
        for t in (cos, sin)
    ):
        raise ValueError("original duplicate-half Llama rotary phases required")


def phase_token(tensor):
    # Inference tensors deliberately have no counter. Do not raise/catch a
    # RuntimeError on every K/V reconstruction merely to discover that fact.
    version = (
        "inference_tensor_no_version_counter" if torch.is_inference(tensor) else tensor._version
    )
    return (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), version)


@dataclass(frozen=True)
class NativeHistoryFormat(EncodedFormat):
    phase_versions: tuple = ()

    def fill(self, attention, destinations, *, direct):
        if (
            direct is not True
            or len(destinations) != 2
            or current() is None
            or self.phase_versions != (phase_token(self.archive.cos), phase_token(self.archive.sin))
        ):
            raise RuntimeError("private validated immutable native history required")
        for c, destination in enumerate(destinations):
            self.fill_component(attention, c, destination)

    def fill_component(self, attention, component, destination):
        state = current()
        if state is None or state.closed:
            raise RuntimeError("native history cannot escape its private request")
        if self.phase_versions != (phase_token(self.archive.cos), phase_token(self.archive.sin)):
            raise RuntimeError("native history phases changed after validation")
        n = self.shape[-2]
        state.engine.project(
            self,
            attention,
            component,
            destination[..., :n, :],
            self.patches if self.patches else None,
        )
        state.projections += 1


class State:
    def __init__(self):
        self.engine = self.pool = None
        self.owners = ()
        self.closed = False
        self.projections = self.built_chunks = self.consolidations = 0
        self.extra_workspace = self.maximum_repairs = self.borrowed_workspace = 0

    def bind(self, layer, device=None):
        pool = getattr(layer, "prompt_graph_pool", None)
        if self.closed or pool is None or pool.closed:
            raise RuntimeError("live exclusive common prompt pool required")
        if self.pool is not None:
            if self.pool is not pool:
                raise RuntimeError("cross-request history engine forbidden")
            return
        self.pool = pool
        borrowed = (
            None if pool.lt_engine is None else (pool.lt_engine.source, pool.lt_engine.output)
        )
        if device is None:
            raise RuntimeError("native workspace must be attached before history audit begins")
        self.engine = Engine(device, frozen=True, buffers=borrowed)
        self.owners = () if borrowed is not None else (self.engine.source, self.engine.output)
        self.extra_workspace = backing_bytes(self.owners)
        self.borrowed_workspace = 0 if borrowed is None else backing_bytes(borrowed)
        pool.cache.observe(self.owners)
        pool.cache.shared_tensors = (*pool.cache.shared_tensors, *self.owners)
        pool.cache.shared_bytes = backing_bytes(pool.cache.shared_tensors)
        pool.cache.observe()

    def retire(self, pool=None):
        if self.closed:
            return
        if pool is not None and self.pool is not pool:
            raise RuntimeError("foreign prompt pool cannot retire native history")
        if self.engine is not None:
            self.engine.close()
            self.engine.source = self.engine.output = None
            self.engine = None
        if self.pool is not None:
            cache = self.pool.cache
            cache.shared_tensors = tuple(
                t for t in cache.shared_tensors if all(t is not owner for owner in self.owners)
            )
            cache.shared_bytes = backing_bytes(cache.shared_tensors)
            cache.observe()
        self.pool, self.owners, self.closed = None, (), True

    def record(self):
        return dict(
            retired=self.closed,
            projections=self.projections,
            built_chunks=self.built_chunks,
            consolidated_layers=self.consolidations,
            additional_owned_workspace_bytes=self.extra_workspace,
            reused_existing_prompt_workspace_bytes=self.borrowed_workspace,
            maximum_per_layer_repair_bytes=self.maximum_repairs,
            original_sources_phases_and_KV_preserved=True,
            whole_request_GPU_peak_certified=False,
        )


def retire_pool(pool):
    state = current()
    if state is not None and state.pool is not None:
        state.retire(pool)


@contextmanager
def transport(adapter, cache):
    state = current()
    if state is None:
        yield None
        return
    try:
        # Attach before history's per-operation audit captures its exclusion
        # set: otherwise shared buffers would be counted twice as scratch.
        state.bind(cache.layers[0], adapter.device)
        yield state
    finally:
        state.retire()


def new_format(source, union, selected, cos, sin, shape, slots):
    from mosaickv.evaluation.encoded_projection_kernel import checkpoints
    from mosaickv.evaluation.projection_archive import ProjectionArchive

    n = shape[-2]
    validate_rows(n)
    phase_guard(cos, sin, n)
    archive = ProjectionArchive(source, union, selected, n, cos, sin)
    return NativeHistoryFormat(
        "native_lossless_history_execution",
        shape,
        archive=archive,
        output_slots=slots,
        literal_checkpoints=checkpoints(source),
        phase_versions=(phase_token(cos), phase_token(sin)),
    )


def repair(candidate, attention, reference, targets=None):
    from mosaickv.evaluation.exact_storage import SparseBitPatch

    if targets is None:
        targets = candidate.materialize(attention)
    else:
        candidate.fill(attention, targets, direct=True)
    patches = tuple(SparseBitPatch.build(a, b) for a, b in zip(reference, targets, strict=True))
    cost = backing_bytes(tuple(t for p in patches for t in p.tensors))
    if cost > MAX_REPAIR_BYTES:
        raise RuntimeError("native history exceeds unchanged 32-KiB physical repair cap")
    current().maximum_repairs = max(current().maximum_repairs, cost)
    return replace(candidate, patches=patches)


@torch.inference_mode()
def build_chunk(layer, cos, sin, *, bit_flags=None):
    from mosaickv.evaluation.dual_projection import output_slots
    from mosaickv.evaluation.exact_storage import encode

    state = current()
    state.bind(layer)
    n = 16
    union = torch.arange(n, device=cos.device, dtype=torch.int32)
    selected = union[None].expand(layer.prompt_shape[1], -1)
    source = encode(layer.normalized_tail[:, None].contiguous())
    candidate = new_format(
        source,
        union,
        selected,
        cos,
        sin,
        tuple(layer.history[0].shape),
        output_slots(union, selected),
    )
    candidate = repair(candidate, layer.attention, layer.history)
    for _ in range(3):
        actual = candidate.materialize(layer.attention)
        flags = [
            (a.view(torch.int16) == b.view(torch.int16)).all()
            for a, b in zip(layer.history, actual, strict=True)
        ]
        if bit_flags is not None:
            bit_flags.extend(flags)
        elif not bool(torch.stack(flags).all()):
            raise RuntimeError("native chunk changed original history K/V bits")
    state.built_chunks += 1
    return candidate


@torch.inference_mode()
def consolidate(layer, cos, sin, *, bit_flags=None):
    from mosaickv.evaluation.consolidated_history import merge_sources
    from mosaickv.evaluation.history_verification_workspace import verify

    state = current()
    state.bind(layer)
    formats = tuple(layer.source_sealed)
    if not formats or any(not isinstance(f, NativeHistoryFormat) for f in formats):
        raise RuntimeError("cannot mix native and old history corrections")
    n = sum(f.shape[-2] for f in formats)
    heads, _, dim = layer.prompt_shape[1:]
    union = torch.arange(n, device=cos.device, dtype=torch.int32)
    selected = union[None].expand(heads, -1)
    source = merge_sources(tuple(f.archive.source for f in formats))
    shape = (1, heads, n, dim)
    slots = torch.arange(n, device=cos.device, dtype=torch.int16)[None].expand(heads, -1)
    candidate = new_format(source, union, selected, cos, sin, shape, slots)
    with state.pool.borrow_history(shape) as scratch:
        reference = (
            tuple(torch.empty(shape, device=cos.device, dtype=torch.bfloat16) for _ in range(2))
            if scratch is None
            else scratch[0]
        )
        offset = 0
        for packet in formats:
            end = offset + packet.shape[-2]
            packet.fill(
                layer.attention, tuple(t[..., offset:end, :] for t in reference), direct=True
            )
            offset = end
        candidate = repair(
            candidate, layer.attention, reference, None if scratch is None else scratch[1]
        )
        del reference
        # Keep all original three K and three V checks, including sentinel
        # clearing and verification that persistent packets never alias scratch.
        verify(layer, candidate, formats, shape, bit_flags, scratch)
    state.consolidations += 1
    return candidate


def managed(function):
    @wraps(function)
    def call(*args, **kwargs):
        enabled = kwargs.get("native_history_reader", False)
        if type(enabled) is not bool or _scope.get() is not None:
            raise ValueError("explicit nonnested common native-history policy required")
        if enabled and kwargs.get("prompt_graph_reader") is not True:
            raise NotImplementedError(
                "native history requires the common exclusive prompt workspace"
            )
        state = State() if enabled else None
        token = _scope.set(state)
        try:
            values, packets, details = function(*args, **kwargs)
            if state is not None and (not state.closed or state.engine is not None):
                raise RuntimeError("native history reader failed to retire with its cache")
            details["common_native_history"] = dict(
                enabled=enabled, report=None if state is None else state.record()
            )
            return values, packets, details
        finally:
            try:
                if state is not None:
                    state.retire()
            finally:
                _scope.reset(token)

    return call
