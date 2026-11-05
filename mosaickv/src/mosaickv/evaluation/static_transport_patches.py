"""Bounded constructor-only sparse compaction; canonical packets unchanged.

Use installed torch.nonzero_static to avoid one host-sized allocation/sync per
component. Masked temporary readers preserve every correction during prefill.
One count barrier checks overflow, then emits the ORIGINAL compact patch type
before validation commit, payload hashing, budgeting, or decoding. Default off.
"""

import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace

import torch

_INACTIVE = object()
_state = ContextVar("evicastkv_static_transport_patches", default=_INACTIVE)
CAPACITY = 4096


@dataclass(frozen=True)
class TemporaryPatch:
    indices: object
    values: object
    count: object

    @property
    def tensors(self):
        return self.indices, self.values, self.count

    def apply_strided(self, target, *, prompt_length):
        if target.dtype != torch.bfloat16 or target.stride(-1) != 1 or target.stride(-2) != 128:
            raise ValueError("original BF16 D=128 strided constructor output required")
        if target.device.type == "cuda":
            from mosaickv.evaluation.static_transport_patch_kernel import apply

            apply(self, target, prompt_length=prompt_length)
        else:
            count = int(self.count.item())
            if count > self.indices.numel():
                raise RuntimeError("constructor patch capacity exceeded; no truncation permitted")
            ids = self.indices[:count].long()
            target[0, ids // (prompt_length * 128), ids // 128 % prompt_length, ids % 128] = (
                self.values[:count]
            )


class Construction:
    def __init__(self, capacity):
        if type(capacity) is not int or not 1 <= capacity <= CAPACITY:
            raise ValueError("bounded positive static transport capacity required")
        self.capacity, self.patches, self.closed = capacity, [], False
        self.staging_bytes = self.count_transfer_bytes = self.canonical_bytes = 0
        self.completed_patches = 0
        self.retired_refs = []

    def build(self, expected, mask):
        if (
            self.closed
            or len(self.patches) >= 64
            or expected.dtype != torch.bfloat16
            or mask.dtype != torch.bool
            or mask.numel() != expected.numel()
            or mask.device != expected.device
            or expected.numel() >= 2**31
        ):
            raise ValueError("bounded unchanged BF16 correction mask on one device required")
        ids = torch.nonzero_static(mask.reshape(-1), size=self.capacity, fill_value=-1)
        ids = ids.reshape(-1).to(torch.int32)
        count = mask.sum(dtype=torch.int64).reshape(1)
        values = expected.contiguous().reshape(-1).index_select(0, ids.clamp_min(0).long()).clone()
        patch = TemporaryPatch(ids, values, count)
        self.patches.append(patch)
        return patch

    def finalize(self, packets):
        if self.closed:
            raise RuntimeError("static constructor finalizes exactly once")
        from mosaickv.evaluation.exact_storage import SparseBitPatch
        from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
        from mosaickv.evaluation.lt_source_format import map_sources

        sizes = (
            torch.cat(tuple(p.count for p in self.patches)).cpu().tolist() if self.patches else []
        )
        if any(type(n) is not int or not 0 <= n <= self.capacity for n in sizes):
            raise RuntimeError("static constructor overflow: reject, never truncate a correction")
        canonical = {
            id(p): SparseBitPatch(p.indices[:n].clone(), p.values[:n].clone())
            for p, n in zip(self.patches, sizes, strict=True)
        }
        visited = set()

        def finish(packet):
            patches = []
            for patch in packet.transport_patches:
                if isinstance(patch, TemporaryPatch):
                    if id(patch) not in canonical:
                        raise RuntimeError("foreign constructor patch is not a canonical packet")
                    visited.add(id(patch))
                    patch = canonical[id(patch)]
                patches.append(patch)
            return replace(packet, transport_patches=tuple(patches))

        result = tuple(map_sources(packet, finish) for packet in packets)
        if visited != set(canonical):
            raise RuntimeError("every temporary correction must reach the committed packet")
        self.staging_bytes = backing_bytes(tuple(t for p in self.patches for t in p.tensors))
        self.canonical_bytes = backing_bytes(
            tuple(t for p in canonical.values() for t in p.tensors)
        )
        self.completed_patches = len(self.patches)
        self.count_transfer_bytes = 8 * self.completed_patches
        self.retired_refs = [weakref.ref(t) for p in self.patches for t in p.tensors]
        self.closed = True
        self.patches.clear()
        return result

    def check_retired(self):
        if not self.closed or any(ref() is not None for ref in self.retired_refs):
            raise RuntimeError("constructor temporary patch owners must retire before decode")

    def report(self):
        return dict(
            enabled=True,
            committed=self.closed,
            constructor_patches=self.completed_patches,
            dynamic_transport_nonzero_calls_avoided=self.completed_patches,
            fixed_capacity_per_component=self.capacity,
            temporary_patch_staging_bytes=self.staging_bytes,
            count_barrier_CPU_bytes=self.count_transfer_bytes,
            canonical_patch_bytes=self.canonical_bytes,
            canonical_packets_have_no_temporary_patches=True,
            constructor_temporary_objects_alive=sum(ref() is not None for ref in self.retired_refs),
            canonical_SparseBitPatch_type_preserved=True,
            constructor_peak_not_certified=True,
        )


@contextmanager
def use(enabled, *, capacity=CAPACITY):
    if type(enabled) is not bool or _state.get() is not _INACTIVE:
        raise ValueError("explicit nonnested static constructor policy required")
    state = Construction(capacity) if enabled else None
    token = _state.set(state)
    try:
        yield state
        if state is not None and not state.closed:
            raise RuntimeError("all static patches must commit before decode")
    finally:
        _state.reset(token)
        if state is not None:
            state.patches.clear()


def enabled():
    return isinstance(_state.get(), Construction)


def build(expected, mask):
    state = _state.get()
    if not isinstance(state, Construction):
        raise RuntimeError("temporary patches require an explicit constructor transaction")
    return state.build(expected, mask)


def finalize(packets, state):
    return packets if state is None else state.finalize(packets)


@torch.inference_mode()
def gpu_controls():
    from mosaickv.evaluation.exact_storage import SparseBitPatch
    from mosaickv.evaluation.lossless_bf16 import PackedBF16
    from mosaickv.evaluation.precision_tier import pack
    from mosaickv.evaluation.shared_source_format import (
        SharedSourceFormat,
        apply_transport_patch,
        transport_corrections,
    )

    h, n, d = 2, 17, 128
    expected = torch.zeros(1, h, n, d, device="cuda", dtype=torch.bfloat16)
    locations = torch.tensor([0, 127, 128, 16 * 128, h * n * d - 1], device="cuda")
    expected.reshape(-1)[locations] = 2.0
    original = torch.zeros_like(expected)
    protected = torch.tensor([2], device="cuda", dtype=torch.int32)
    ordinary = transport_corrections((expected,), (original,), protected)[0]
    source = pack(torch.zeros(1, 1, n, h * d, device="cuda", dtype=torch.bfloat16), bits=3, axis=2)
    exact_body = torch.zeros(1, 1, 1, d, device="cuda", dtype=torch.bfloat16)
    records = []
    for padding in (0, 1, 127):
        with use(True, capacity=8) as state:
            temporary = transport_corrections((expected,), (original,), protected)[0]
            destination = torch.full(
                (1, h, n + padding, d), -71.0, device="cuda", dtype=torch.bfloat16
            )
            destination[..., :n, :].copy_(original)
            apply_transport_patch(temporary, destination, prompt_length=n)
            if not torch.equal(
                destination[..., :n, :].view(torch.int16), expected.view(torch.int16)
            ):
                raise RuntimeError("fixed-capacity constructor dropped or changed a correction")
            if not bool((destination[..., n:, :] == -71).all()):
                raise RuntimeError("fixed-capacity constructor changed a padding sentinel")
            packet = SharedSourceFormat(
                source,
                tuple(expected.shape),
                protected,
                (PackedBF16(tuple(exact_body.shape), True, (exact_body,)),) * 2,
                torch.ones(1, n, d, device="cuda", dtype=torch.bfloat16),
                torch.zeros(1, n, d, device="cuda", dtype=torch.bfloat16),
                torch.arange(n, device="cuda", dtype=torch.int32),
                torch.arange(n, device="cuda", dtype=torch.int16)[None].expand(h, n),
                transport_patches=(temporary,),
            )
            canonical = finalize((packet,), state)[0].transport_patches[0]
        if type(canonical) is not SparseBitPatch or not torch.equal(
            canonical.indices, ordinary.indices
        ):
            raise RuntimeError("canonical sparse index order differs from original nonzero")
        if not torch.equal(canonical.values.view(torch.int16), ordinary.values.view(torch.int16)):
            raise RuntimeError("canonical sparse values differ from original constructor")
        records.append(state.report())
    try:
        with use(True, capacity=1) as state:
            temporary = build(expected, expected != 0)
            finalize((replace(packet, transport_patches=(temporary,)),), state)
    except RuntimeError as error:
        if "overflow" not in str(error):
            raise
    else:
        raise RuntimeError("capacity overflow must fail closed")
    return dict(
        status="completed",
        actual_CUDA_masked_writes=True,
        overflow_rejected=True,
        canonical_indices_and_value_words_unchanged=True,
        sentinel_padding_0_1_127_preserved=True,
        records=records,
        synthetic_control_only=True,
        independent_quality_N=0,
    )
