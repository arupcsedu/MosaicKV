"""Default-off single projection of mixed older/recent source history.

Preserve the registered INT8/exact precision split. Recent archives remain
available for aging, but are decoded into one existing workspace. Exact counted
corrections preserve ALL words of the unfused reader, including older history.
No extra persistent source copy or enlarged numerical tolerance is permitted.
"""

import weakref
from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


@dataclass(frozen=True)
class MixedHistory:
    older: object
    recent: tuple
    cos: object
    sin: object
    repairs: tuple
    owner: object
    shape: tuple
    phase_versions: tuple
    kind: str = "same_precision_single_projection_mixed_history"

    @property
    def archive(self):
        # The shared verification helper needs only archive.cos.device.
        from types import SimpleNamespace

        return SimpleNamespace(cos=self.cos)

    @property
    def formats(self):
        return (*(() if self.older is None else (self.older,)), *self.recent)

    def private(self):
        return (
            *(t for f in self.formats for t in f.private()),
            *(t for patch in self.repairs for t in patch.tensors),
        )

    def shared(self):
        return self.cos, self.sin, *(t for f in self.formats for t in f.shared())

    def tensors(self):
        return (*self.private(), *self.shared())

    def materialize(self, attention):
        pair = tuple(
            torch.empty(self.shape, device=self.cos.device, dtype=torch.bfloat16) for _ in range(2)
        )
        self.fill(attention, pair, direct=True)
        return pair

    def fill(self, attention, pair, *, direct):
        if direct is not True:
            raise ValueError("original direct native history required")
        for component, target in enumerate(pair):
            self.fill_component(attention, component, target)

    def fill_component(self, attention, component, target):
        from mosaickv.evaluation import native_history_reader as native
        from mosaickv.evaluation.history_source_precision import unpack_into as unpack_int8
        from mosaickv.evaluation.lossless_bf16 import unpack_into as unpack_exact
        from mosaickv.evaluation.shared_source_kernel import rotate_native

        state = native.current()
        if state is None or self.owner() is not state or state.closed or state.graphs is not None:
            raise RuntimeError("private original uncaptured native history owner required")
        if self.phase_versions != (native.phase_token(self.cos), native.phase_token(self.sin)):
            raise RuntimeError("mixed history phases changed")
        n = self.shape[-2]
        if component not in (0, 1) or tuple(target.shape) != self.shape:
            raise ValueError("complete mixed-history component shape required")
        workspace = state.engine.source.view(1, 1, -1, 4096)
        offset = 0
        if self.older is not None:
            offset = self.older.shape[-2]
            unpack_int8(self.older.body, workspace[:, :, :offset])
        for packet in self.recent:
            end = offset + packet.shape[-2]
            unpack_exact(packet.archive.source, workspace[:, :, offset:end])
            offset = end
        if offset != n:
            raise RuntimeError("mixed reconstruction lost a logical source position")
        state.engine.plans.execute(
            state.engine.source,
            (attention.k_proj, attention.v_proj)[component].weight,
            state.engine.output,
            n,
        )
        target[0].copy_(state.engine.output[:n].view(n, 32, 128).transpose(0, 1))
        if component == 0:
            rotate_native(target, self.cos, self.sin, 0, n)
        if self.repairs:
            if target.is_cuda:
                from mosaickv.evaluation.dual_projection_kernel import apply_patch

                apply_patch(self.repairs[component], target)
            else:
                patch = self.repairs[component]
                ids = patch.indices.long()
                head, within = ids // (n * 128), ids % (n * 128)
                target.view(torch.int16)[0, head, within // 128, within % 128] = patch.values.view(
                    torch.int16
                )
        state.projections += 1


@torch.inference_mode()
def construct(layer, cos, sin):
    from mosaickv.evaluation import native_history_reader as native
    from mosaickv.evaluation.history_source_precision import HistoryFormat
    from mosaickv.evaluation.history_verification_workspace import verify

    formats = tuple(layer.source_sealed)
    if not formats:
        raise ValueError("complete original history formats required")
    older = formats[0] if isinstance(formats[0], HistoryFormat) else None
    recent = formats[1:] if older is not None else formats
    if not recent or any(not isinstance(f, native.NativeHistoryFormat) for f in recent):
        raise ValueError("one optional older INT8 body plus original exact recent packets required")
    state = native.current()
    if state is None or state.graphs is not None or state.closed:
        raise RuntimeError("live uncaptured native history state required")
    n = sum(f.shape[-2] for f in formats)
    native.validate_rows(n)
    candidate = MixedHistory(
        older,
        recent,
        cos,
        sin,
        (),
        weakref.ref(state),
        (1, 32, n, 128),
        (native.phase_token(cos), native.phase_token(sin)),
    )
    with state.pool.borrow_history(candidate.shape) as scratch:
        reference = (
            tuple(
                torch.empty(candidate.shape, dtype=torch.bfloat16, device=cos.device)
                for _ in range(2)
            )
            if scratch is None
            else scratch[0]
        )
        offset = 0
        for f in formats:
            end = offset + f.shape[-2]
            f.fill(layer.attention, tuple(t[..., offset:end, :] for t in reference), direct=True)
            offset = end
        actual = candidate.materialize(layer.attention) if scratch is None else scratch[1]
        if scratch is not None:
            candidate.fill(layer.attention, actual, direct=True)
        repairs = tuple(SparseBitPatch.build(a, b) for a, b in zip(reference, actual, strict=True))
        if backing_bytes(tuple(t for p in repairs for t in p.tensors)) > native.MAX_REPAIR_BYTES:
            raise RuntimeError("mixed history exceeded unchanged 32-KiB correction cap")
        candidate = replace(candidate, repairs=repairs)
        del reference, actual
        # Three original K/V and scratch-sentinel checks, never skipped.
        verify(layer, candidate, formats, candidate.shape, None, scratch)
    return candidate


@contextmanager
def use():
    from unittest.mock import patch

    from mosaickv.evaluation import consolidated_history as combined
    from mosaickv.evaluation import native_history_reader as native
    from mosaickv.evaluation.history_source_precision import HistoryFormat

    original = combined.consolidate
    report = dict(
        variant="single_projection_same_precision_history",
        mixed_constructions=0,
        correction_bytes=0,
        no_extra_persistent_source_copy=True,
        original_unfused_KV_words_verified=True,
    )

    def consolidate(cache):
        from mosaickv.evaluation.projection_cache import identity

        retiring = {identity(t) for pair in cache.history_rotary_chunks for t in pair}
        retiring.update(identity(t) for pair in cache.consolidated_phase_versions for t in pair)
        retiring.update(
            identity(t) for layer in cache.layers for f in layer.source_sealed for t in f.shared()
        )
        for layer in cache.layers:
            if layer.source_sealed and isinstance(layer.source_sealed[0], MixedHistory):
                previous = layer.source_sealed[0]
                old = tuple(t for f in layer.source_sealed for t in f.private())
                layer.source_sealed = [*previous.formats, *layer.source_sealed[1:]]
                now = tuple(t for f in layer.source_sealed for t in f.private())
                layer.resident_bytes += backing_bytes(now) - backing_bytes(old)
                cache.observe(old)
                del previous, old, now
        changed = original(cache)
        active = [layer for layer in cache.layers if len(layer.source_sealed) >= 2]
        if not active:
            return changed
        n = sum(f.shape[-2] for f in active[0].source_sealed)
        if len(cache.history_rotary_chunks) == 1:
            cos, sin = cache.history_rotary_chunks[0]
        else:
            cos, sin = (
                torch.cat(tuple(pair[j] for pair in cache.history_rotary_chunks), 1) for j in (0, 1)
            )
        if cos.shape != (1, n, 128) or sin.shape != cos.shape:
            raise RuntimeError("shared mixed phases must cover every original history position")
        cache.shared_tensors = (*cache.shared_tensors, cos, sin)
        cache.shared_bytes = backing_bytes(cache.shared_tensors)
        cache.observe()
        for layer in cache.layers:
            formats = tuple(layer.source_sealed)
            if len(formats) < 2:
                continue
            if any(not isinstance(f, (native.NativeHistoryFormat, HistoryFormat)) for f in formats):
                raise ValueError("mixed history cannot replace an unknown format")
            mixed = construct(layer, cos, sin)
            old = tuple(t for f in formats for t in f.private())
            layer.source_sealed = [mixed]
            layer.resident_bytes += backing_bytes(mixed.private()) - backing_bytes(old)
            cache.observe(old)
            report["mixed_constructions"] += 1
            report["correction_bytes"] += backing_bytes(
                tuple(t for p in mixed.repairs for t in p.tensors)
            )
            del old, formats, mixed
        used = {
            identity(t) for layer in cache.layers for f in layer.source_sealed for t in f.shared()
        }
        cache.shared_tensors = tuple(
            t for t in cache.shared_tensors if identity(t) not in retiring or identity(t) in used
        )
        cache.shared_bytes = backing_bytes(cache.shared_tensors)
        cache.history_rotary_chunks = [(cos, sin)]
        cache.consolidated_phase_versions = [(cos, sin)]
        cache.observe()
        return True

    with patch.object(combined, "consolidate", consolidate):
        yield report


@torch.inference_mode()
def gpu_controls():
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import patch

    from mosaickv.evaluation import native_history_reader as native
    from mosaickv.evaluation.exact_storage import encode
    from mosaickv.evaluation.history_source_precision import HistoryFormat, pack_source
    from mosaickv.evaluation.native_history_feasibility import Engine

    class State:
        closed = False
        graphs = None
        projections = 0

    state = State()
    state.engine = Engine("cuda", frozen=True)
    state.pool = SimpleNamespace(borrow_history=lambda _: nullcontext(None))
    weight = torch.eye(4096, device="cuda", dtype=torch.bfloat16)
    attention = SimpleNamespace(
        k_proj=SimpleNamespace(weight=weight), v_proj=SimpleNamespace(weight=weight * 2)
    )
    records = []
    try:
        with patch.object(native, "current", lambda: state):
            for older in (False, True):
                cos = torch.ones(1, 48, 128, device="cuda", dtype=torch.bfloat16)
                sin = torch.zeros_like(cos)
                formats = []
                for i in range(3):
                    x = torch.randn(1, 1, 16, 4096, device="cuda").bfloat16()
                    c, s = cos[:, i * 16 : (i + 1) * 16], sin[:, i * 16 : (i + 1) * 16]
                    if i == 0 and older:
                        formats.append(
                            HistoryFormat(
                                pack_source(x, group=32),
                                c,
                                s,
                                (1, 32, 16, 128),
                                weakref.ref(state),
                                (native.phase_token(c), native.phase_token(s)),
                            )
                        )
                    else:
                        ids = torch.arange(16, device="cuda", dtype=torch.int32)
                        formats.append(
                            native.new_format(
                                encode(x),
                                ids,
                                ids[None].expand(32, -1),
                                c,
                                s,
                                (1, 32, 16, 128),
                                ids.short()[None].expand(32, -1),
                            )
                        )
                layer = SimpleNamespace(source_sealed=formats, attention=attention)
                mixed = construct(layer, cos, sin)
                expected = tuple(
                    torch.empty(mixed.shape, device="cuda", dtype=torch.bfloat16) for _ in range(2)
                )
                for i, f in enumerate(formats):
                    f.fill(
                        attention,
                        tuple(t[..., i * 16 : (i + 1) * 16, :] for t in expected),
                        direct=True,
                    )
                actual = torch.full((1, 32, 64, 128), -91, device="cuda", dtype=torch.bfloat16)
                before = state.projections
                mixed.fill_component(attention, 0, actual[..., :48, :])
                if state.projections != before + 1 or not torch.equal(
                    actual[..., :48, :], expected[0]
                ):
                    raise RuntimeError("single mixed GEMM changed original reader words")
                if not bool((actual[..., 48:, :] == -91).all()):
                    raise RuntimeError("mixed projection exceeded the requested component")
                records.append(
                    dict(
                        older_INT8=older,
                        original_reader_KV_bitwise=True,
                        strided_sentinels_intact=True,
                        one_GEMM_per_component=True,
                        correction_bytes=backing_bytes(
                            tuple(t for p in mixed.repairs for t in p.tensors)
                        ),
                    )
                )
    finally:
        state.engine.close()
    return records
