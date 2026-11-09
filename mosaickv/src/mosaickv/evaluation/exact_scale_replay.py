"""Default-off, request-private exact FP32 scale serialization and replay.

All K/V readers retain their existing FP32-scale ABI. One charged workspace per
stream role replaces layer-private headers only when the COMPLETE transaction
saves storage. CPU dictionary construction and transfers are online costs, not
hidden precomputation. This common transport is not a six-mechanism policy.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from math import prod
from time import perf_counter
from unittest.mock import patch

import torch

from mosaickv.evaluation import layer_code_delta as runtime
from mosaickv.evaluation.exact_scale_codec import encode_cpu, unpack_into
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

_active = ContextVar("evicastkv_exact_scale_replay", default=False)


@dataclass(frozen=True)
class ScaleEncodedBody(runtime.EncodedBody):
    scale_packet: object

    def tensors(self):
        return (*super().tensors(), *self.scale_packet.tensors())


def convert(packets):
    """Convert immutable code specifications, never rebound mutable requests."""
    started = perf_counter()
    packets = tuple(packets)
    if not packets or not all(isinstance(p, runtime.DeltaFormat) for p in packets):
        raise ValueError("complete immutable code-encoded sequence required")
    requirements, code_requirements, records, result = {}, {}, [], []
    for index, packet in enumerate(packets):

        def encode(role, body, *, index=index):
            if type(body) is not runtime.EncodedBody:
                raise ValueError("original encoded body required; no repeated scale conversion")
            if not body.stream.raw:
                code_requirements[role] = max(code_requirements.get(role, 0), body.stream.length)
            scale = body.metadata.scale
            archive = encode_cpu(scale.detach().to(device="cpu", copy=True))
            record = dict(
                layer=index,
                role=str(role),
                original_scale_bytes=scale.untyped_storage().nbytes(),
                archive_bytes=archive.storage_bytes,
                kind=archive.kind,
            )
            records.append(record)
            if archive.kind == "raw":
                return body
            archive = archive.to(scale.device)
            requirements[role] = max(requirements.get(role, 0), scale.numel() * 4)
            metadata = replace(scale_body := body.metadata, scale=scale.new_empty(0))
            if metadata.minimum is not scale_body.minimum:
                raise RuntimeError("scale conversion changed unrelated affine headers")
            return ScaleEncodedBody(metadata, body.stream, archive)

        result.append(runtime.DeltaFormat(runtime.map_bodies(packet.template, encode)))
    old = backing_bytes(tuple(t for p in packets for t in p.tensors()))
    new = backing_bytes(tuple(t for p in result for t in p.tensors()))
    workspace = sum(requirements.values())
    code_workspace = sum(code_requirements.values())
    enabled = bool(requirements and new + workspace < old)
    report = dict(
        enabled=enabled,
        original_encoded_storage_bytes=old,
        scale_encoded_storage_bytes=new,
        existing_code_workspace_bytes=code_workspace,
        original_bound_active_bytes=old + code_workspace,
        charged_shared_scale_workspace_bytes=workspace,
        net_active_bytes=(new + workspace if enabled else old) + code_workspace,
        online_CPU_encode_and_transfer_seconds=perf_counter() - started,
        CPU_residual_bytes=0,
        common_to_compatible_methods=True,
        records=records,
    )
    return (tuple(result) if enabled else packets), report


class ScaleReplay(runtime.Replay):
    def __init__(self, packets):
        self.scale_scratch, self.header_bindings = {}, []
        self.warmed = set()
        for packet in packets:

            def collect(role, body):
                if isinstance(body, ScaleEncodedBody):
                    count = prod(body.scale_packet.shape)
                    old = self.scale_scratch.get(role)
                    if old is None or old.numel() < count:
                        self.scale_scratch[role] = torch.empty(
                            count, device=body.scale_packet.payload.device, dtype=torch.float32
                        )
                return body

            runtime.map_bodies(packet.template, collect)
        super().__init__(packets)
        # The existing ownership API inventories Replay.scratch, including these
        # new buffers. No separate, uncharged header cache is permitted.
        self.scratch.update({("exact_scale", role): t for role, t in self.scale_scratch.items()})
        for packet in packets:
            bindings = []

            def collect_bindings(role, body, *, bindings=bindings):
                if isinstance(body, ScaleEncodedBody):
                    bindings.append((body.scale_packet, self.scale_scratch[role]))
                return body

            runtime.map_bodies(packet.template, collect_bindings)
            self.header_bindings.append(tuple(bindings))

    def bind_body(self, role, encoded, codes):
        body = super().bind_body(role, encoded, codes)
        if isinstance(encoded, ScaleEncodedBody):
            shape = encoded.scale_packet.shape
            body = replace(body, scale=self.scale_scratch[role][: prod(shape)].view(shape))
        return body

    def prepare_metadata(self, index, *, launch):
        # First graph warmup needs actual headers too. Subsequent header decoding
        # is inside the KEY graph; V reuses the exact same initialized workspace.
        if launch or index not in self.warmed:
            self.decode_metadata_for_graph(index)
            self.warmed.add(index)

    def decode_metadata_for_graph(self, index):
        for packet, destination in self.header_bindings[index]:
            unpack_into(packet, destination)


def header_only(packets):
    """Keep arbitrary original code bytes, including mixed 3/8-bit baselines.

    No three-bit source predictor is applied to these raw code streams. Raw
    BF16 archives lacking affine headers retain their unchanged original form.
    """
    from mosaickv.evaluation.lossless_code_delta_gpu import DeviceCodes
    from mosaickv.evaluation.precision_tier import QuantizedBody

    packets = tuple(packets)
    if any(isinstance(p, runtime.DeltaFormat) for p in packets):
        if not all(isinstance(p, runtime.DeltaFormat) for p in packets):
            raise ValueError("do not mix already encoded and plain packet transactions")
        return packets, None
    bodies = 0

    def raw(role, body):
        nonlocal bodies
        if not isinstance(body, QuantizedBody) or body.codes.dtype != torch.uint8:
            raise ValueError("original affine headers and serialized uint8 codes required")
        bodies += 1
        return runtime.EncodedBody(
            replace(body, codes=None), DeviceCodes(body.codes.numel(), body.codes)
        )

    encoded = tuple(runtime.DeltaFormat(runtime.map_bodies(p, raw)) for p in packets)
    if not bodies:
        return packets, dict(enabled=False, reason="no compatible affine scale headers")
    compact, report = convert(encoded)
    report.update(header_only=True, original_code_bytes_unchanged=True, source_predictor_used=False)
    return (compact if report["enabled"] else packets), report


@contextmanager
def use():
    """Single-threaded diagnostic scope, identically available to comparators."""
    if _active.get():
        raise RuntimeError("nonnested private exact-scale scope required")
    token = _active.set(True)
    from mosaickv.evaluation import storage_runs

    original, original_builder = runtime.convert, storage_runs.cache_from_formats
    reports = []

    def convert_codes(packets):
        encoded, code_report = original(packets)
        if code_report["enabled"]:
            encoded, scale_report = convert(encoded)
            reports.append(scale_report)
            code_report = dict(
                code_report,
                exact_scale_report=scale_report,
                code_only_net_active_bytes=code_report["net_active_bytes"],
                net_active_bytes=scale_report["net_active_bytes"],
            )
        else:
            reports.append(dict(enabled=False, reason="code codec declined; no scale binding"))
        return encoded, code_report

    def builder(adapter, formats, **kwargs):
        formats, report = header_only(formats)
        if report is not None:
            reports.append(report)
        return original_builder(adapter, formats, **kwargs)

    try:
        with (
            patch.object(runtime, "convert", convert_codes),
            patch.object(runtime, "Replay", ScaleReplay),
            patch.object(storage_runs, "cache_from_formats", builder),
        ):
            yield reports
    finally:
        _active.reset(token)
