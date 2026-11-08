"""Default-off INT8 older SOURCE history; not a lossless/exact-cache method.

The latest sealed 16-token chunk and mutable tail remain on the original exact
path. Older normalized sources are quantized per 128 features; no token, RoPE
position or model weight changes. There are no original-bit correction patches
for the deliberately approximate older source. Original native chunk checks
remain enabled. A separate complete-horizon model guard controls admission.
"""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from unittest.mock import patch

import torch

from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import pack


def join(bodies):
    if not bodies or any(
        b.bits != 8
        or b.axis != 3
        or b.row_map is not None
        or b.shape[:2] != (1, 1)
        or b.shape[-1] != bodies[0].shape[-1]
        for b in bodies
    ):
        raise ValueError("complete compatible row-wise INT8 source bodies required")
    if len(bodies) == 1:
        return bodies[0]
    return replace(
        bodies[0],
        shape=(1, 1, sum(b.shape[-2] for b in bodies), bodies[0].shape[-1]),
        codes=torch.cat(tuple(b.codes for b in bodies)),
        minimum=torch.cat(tuple(b.minimum for b in bodies), 2),
        scale=torch.cat(tuple(b.scale for b in bodies), 2),
    )


def unpack_into(body, target):
    if (
        body.bits != 8
        or body.axis != 3
        or body.group != 128
        or body.row_map is not None
        or target.dtype != torch.bfloat16
        or not target.is_contiguous()
        or target.numel() < body.codes.numel()
    ):
        raise ValueError("INT8 feature-group source and contiguous BF16 workspace required")
    if body.codes.is_cuda:
        from mosaickv.evaluation.history_source_precision_kernel import unpack

        unpack[((body.codes.numel() + 255) // 256,)](
            body.codes,
            body.minimum,
            body.scale,
            target,
            body.codes.numel(),
            BLOCK=256,
            enable_fp_fusion=False,
        )
    else:
        value = body.codes.float().view(*body.shape[:-1], body.shape[-1] // 128, 128)
        value = value * body.scale[..., None]
        value = value + body.minimum[..., None]
        target.flatten()[: body.codes.numel()].copy_(value.flatten().bfloat16())


@dataclass(frozen=True)
class HistoryFormat:
    body: object
    cos: object
    sin: object
    shape: tuple
    kind: str = "evicastkv_older_INT8_source_history_variant"

    def private(self):
        return self.body.tensors()

    def shared(self):
        return self.cos, self.sin

    def tensors(self):
        return (*self.private(), *self.shared())

    @torch.inference_mode()
    def fill_component(self, attention, component, target):
        from mosaickv.evaluation.native_history_reader import current
        from mosaickv.evaluation.shared_source_kernel import rotate_native

        state = current()
        if state is None or state.closed or state.graphs is not None:
            raise RuntimeError("private original native history engine, graphs disabled, required")
        n = self.shape[-2]
        unpack_into(self.body, state.engine.source)
        state.engine.plans.execute(
            state.engine.source,
            (attention.k_proj, attention.v_proj)[component].weight,
            state.engine.output,
            n,
        )
        target[0].copy_(state.engine.output[:n].view(n, 32, 128).transpose(0, 1))
        if component == 0:
            rotate_native(target, self.cos, self.sin, 0, n)
        state.projections += 1

    def fill(self, attention, destinations, *, direct):
        if direct is not True:
            raise ValueError("direct original-layout history projection required")
        for component, target in enumerate(destinations):
            self.fill_component(attention, component, target[..., : self.shape[-2], :])


@contextmanager
def use():
    from mosaickv.evaluation import consolidated_history as combined
    from mosaickv.evaluation import native_history_reader as native
    from mosaickv.evaluation import source_history
    from mosaickv.evaluation.exact_storage import decoded
    from mosaickv.evaluation.projection_cache import identity

    report = dict(
        variant="older_source_INT8_exact_recent_16",
        conversions=0,
        epochs=0,
        latest_sealed_exact_tokens=16,
        mutable_tail_exact=True,
        benchmark_labels_accessed=False,
        lossy_older_history=True,
        original_KV_bitwise_NOT_claimed=True,
        CPU_residual_bytes=0,
    )
    original_record, original_state_record = (
        source_history.SourceHistoryCache.storage_record,
        native.State.record,
    )

    @torch.inference_mode()
    def consolidate(cache):
        layers = cache.layers
        if (
            not layers
            or any(layer.leased or layer.source_pending for layer in layers)
            or len({layer.history_length for layer in layers}) != 1
        ):
            raise ValueError("released synchronized request-private history required")
        if all(len(layer.source_sealed) < 2 for layer in layers):
            return False
        counts = [sum(f.shape[-2] for f in layer.source_sealed) for layer in layers]
        if len(set(counts)) != 1 or counts[0] < 32:
            raise ValueError("complete equal-length original history required")
        state = native.current()
        if state is None or state.graphs is not None:
            raise NotImplementedError(
                "precision diagnostic requires original history graphs disabled"
            )
        old_phase = {identity(t) for pair in cache.history_rotary_chunks for t in pair}
        old_phase.update(identity(t) for pair in cache.consolidated_phase_versions for t in pair)
        cos, sin = (
            torch.cat(tuple(p[i] for p in cache.history_rotary_chunks), 1) for i in range(2)
        )
        if cos.shape[1] != counts[0]:
            raise ValueError("history phase membership changed")
        cache.shared_tensors = (*cache.shared_tensors, cos, sin)
        cache.shared_bytes = backing_bytes(cache.shared_tensors)
        cache.observe()
        for layer in layers:
            old = tuple(layer.source_sealed)
            if not isinstance(old[-1], native.NativeHistoryFormat) or old[-1].shape[-2] != 16:
                raise ValueError("most recent sealed chunk must retain original exact source")
            bodies = []
            for packet in old[:-1]:
                if isinstance(packet, HistoryFormat):
                    bodies.append(packet.body)
                elif isinstance(packet, native.NativeHistoryFormat):
                    bodies.append(pack(decoded(packet.archive.source), bits=8, axis=3))
                    report["conversions"] += 1
                else:
                    raise TypeError("unsupported history source format")
            body = join(bodies)
            n = body.shape[-2]
            replacement = HistoryFormat(body, cos[:, :n], sin[:, :n], (1, 32, n, 128))
            if n + 16 != counts[0]:
                raise RuntimeError("older/recent split lost a logical history position")
            old_private = tuple(t for packet in old for t in packet.private())
            new_private = (*replacement.private(), *old[-1].private())
            layer.source_sealed = [replacement, old[-1]]
            layer.resident_bytes += backing_bytes(new_private) - backing_bytes(old_private)
            cache.observe(old_private)
            cache.source_consolidation_layers += 1
            del bodies, old_private, new_private, old, replacement, body
            cache.observe()
        used = {
            identity(t)
            for layer in layers
            for packet in layer.source_sealed
            for t in packet.shared()
        }
        cache.shared_tensors = tuple(
            t for t in cache.shared_tensors if identity(t) not in old_phase or identity(t) in used
        )
        cache.shared_bytes = backing_bytes(cache.shared_tensors)
        cache.history_rotary_chunks = [(cos, sin)]
        cache.consolidated_phase_versions = [(cos, sin)]
        cache.source_consolidation_epochs += 1
        report["epochs"] += 1
        cache.observe()
        return True

    def record(cache):
        return dict(
            original_record(cache),
            generated_history_lossless_and_append_only=False,
            generated_history_positions_append_only=True,
            older_source_precision=dict(report),
            original_KV_bitwise=False,
        )

    def state_record(state):
        return dict(
            original_state_record(state),
            original_sources_phases_and_KV_preserved=False,
            original_positions_and_phases_preserved=True,
            older_source_lossy=True,
        )

    with (
        patch.object(combined, "consolidate", consolidate),
        patch.object(source_history.SourceHistoryCache, "storage_record", record),
        patch.object(native.State, "record", state_record),
    ):
        yield report
