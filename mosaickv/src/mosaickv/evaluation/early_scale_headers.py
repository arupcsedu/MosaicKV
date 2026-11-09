"""Default-off header lifetime experiment; no codec or policy approximation.

Only completed async source packets are archived. Native fitting has finished
before publication; neither in-flight inputs nor graph pointers are modified.
Exact original FP32 scales are restored before the original final publication
and all downstream source readers. Late persistent packing remains separate.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from time import perf_counter
from unittest.mock import patch

import torch

from mosaickv.evaluation.exact_scale_codec import encode_cpu, unpack_into
from mosaickv.evaluation.layer_code_delta import map_bodies
from mosaickv.evaluation.precision_tier import QuantizedBody

_active = ContextVar("evicastkv_early_scale_headers", default=False)


@dataclass(frozen=True)
class ArchivedScaleBody(QuantizedBody):
    archived_scale: object = None

    def tensors(self):
        return (*super().tensors(), *self.archived_scale.tensors())


def replace_scale(body, scale, *, archive=None):
    metadata = {f.name: getattr(body, f.name) for f in fields(QuantizedBody)}
    metadata["scale"] = scale
    return (
        QuantizedBody(**metadata)
        if archive is None
        else ArchivedScaleBody(**metadata, archived_scale=archive)
    )


def archive_packets(packets):
    """Shared source metadata is replaced once across prepared/native packets."""
    seen, records = {}, []

    def archive(role, body):
        if not isinstance(body, QuantizedBody):
            raise ValueError("only completed affine source packets can be archived")
        if isinstance(body, ArchivedScaleBody):
            return body
        key = id(body)
        if key not in seen:
            original = body.scale
            packet = encode_cpu(original.detach().to(device="cpu", copy=True))
            if packet.kind == "raw":
                seen[key] = body
            else:
                packet = packet.to(original.device)
                seen[key] = replace_scale(body, original.new_empty(0), archive=packet)
            records.append(
                dict(
                    role=str(role),
                    original_bytes=original.untyped_storage().nbytes(),
                    encoded_bytes=packet.storage_bytes,
                    kind=packet.kind,
                )
            )
        return seen[key]

    return tuple(map_bodies(p, archive) for p in packets), records


def restore_packets(packets):
    seen, count = {}, 0

    def restore(role, body):
        nonlocal count
        if not isinstance(body, ArchivedScaleBody):
            return body
        key = id(body)
        if key not in seen:
            packet = body.archived_scale
            scale = torch.empty(packet.shape, device=packet.payload.device, dtype=torch.float32)
            unpack_into(packet, scale)
            seen[key] = replace_scale(body, scale)
            count += 1
        return seen[key]

    return tuple(map_bodies(p, restore) for p in packets), count


class Placement:
    def __init__(self):
        self.owner = None
        self.packed = set()
        self.report = dict(
            common_to_compatible_methods=True,
            packed_layers=0,
            restored_bodies=0,
            archive_seconds=0.0,
            restore_seconds=0.0,
            original_header_bytes=0,
            archived_header_bytes=0,
            CPU_residual_bytes=0,
            downstream_headers_restored=False,
        )

    def before(self, pipeline, cache, final):
        if not pipeline.deferred_text:
            raise ValueError("completed deferred-text publication required; no unsafe early read")
        if self.owner is None:
            self.owner = pipeline
        if pipeline is not self.owner:
            raise RuntimeError("header publication cannot cross request ownership")
        if not final:
            return
        started = perf_counter()
        for i in sorted(self.packed):
            prepared, native, patch_bytes = pipeline.prepared[i]
            if cache.layers[i].packet is not prepared:
                raise RuntimeError("archived packet was replaced outside its owner")
            (prepared, native), count = restore_packets((prepared, native))
            pipeline.prepared[i] = prepared, native, patch_bytes
            cache.layers[i].packet = prepared
            cache.layer_owners_changed(i)
            self.report["restored_bodies"] += count
        self.packed.clear()
        self.report["restore_seconds"] += perf_counter() - started
        self.report["downstream_headers_restored"] = True

    def after(self, pipeline, cache, final):
        if final:
            return
        started = perf_counter()
        for i in sorted(pipeline.prepared.keys() - self.packed):
            prepared, native, patch_bytes = pipeline.prepared[i]
            if cache.layers[i].packet is not prepared:
                raise RuntimeError("only actually published complete layers may be archived")
            (prepared, native), records = archive_packets((prepared, native))
            pipeline.prepared[i] = prepared, native, patch_bytes
            cache.layers[i].packet = prepared
            cache.layer_owners_changed(i)
            self.packed.add(i)
            self.report["packed_layers"] += 1
            self.report["original_header_bytes"] += sum(r["original_bytes"] for r in records)
            self.report["archived_header_bytes"] += sum(r["encoded_bytes"] for r in records)
        self.report["archive_seconds"] += perf_counter() - started


@contextmanager
def use():
    from mosaickv.evaluation.prefill_native_overlap import Pipeline

    if _active.get():
        raise RuntimeError("nonnested private early-header scope required")
    token = _active.set(True)
    original = Pipeline.publish_ready
    state = Placement()

    def publish(pipeline, cache, *, final):
        state.before(pipeline, cache, final)
        result = original(pipeline, cache, final=final)
        state.after(pipeline, cache, final)
        return result

    try:
        with patch.object(Pipeline, "publish_ready", publish):
            yield state.report
            if state.packed or not state.report["downstream_headers_restored"]:
                raise RuntimeError(
                    "all archived headers must be restored before leaving the request"
                )
    finally:
        _active.reset(token)
