"""Request-private replay of losslessly compressed adjacent-layer code bytes.

This is a representation variant, not a new selection or quantization method.
One charged code buffer per stream role replaces all uncompressed code copies.
Unsupported layouts fail explicitly. No mutable replay state is stored in the
immutable packet specifications or shared between requests.
"""

from dataclasses import dataclass, replace

import torch

from mosaickv.evaluation.lossless_code_delta_gpu import decode, encode
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import QuantizedBody


def map_bodies(packet, function):
    from mosaickv.evaluation.exact_storage import ExactFormat
    from mosaickv.evaluation.head_local_precision import HeadLocalFormat
    from mosaickv.evaluation.precision_tier import PrecisionFormat
    from mosaickv.evaluation.shared_source_format import SharedSourceFormat
    from mosaickv.evaluation.text_source_storage import TextSourceFormat

    if isinstance(packet, (HeadLocalFormat, TextSourceFormat)):
        return replace(packet, base=map_bodies(packet.base, function))
    if isinstance(packet, PrecisionFormat):
        return replace(
            packet, pair=tuple(function(role, body) for role, body in enumerate(packet.pair))
        )
    if isinstance(packet, SharedSourceFormat) and packet.projection_backend == "fused":
        return replace(packet, source=function("source", packet.source))
    if isinstance(packet, ExactFormat) and packet.kind == "kv":
        return packet
    raise NotImplementedError("unaudited packet cannot use layer-code deltas")


@dataclass(frozen=True)
class EncodedBody:
    # The metadata contains NO original code tensor, not even an empty view of
    # its backing storage. All original floating-point headers stay unchanged.
    metadata: QuantizedBody
    stream: object

    @property
    def codes(self):
        raise RuntimeError("encoded codes require an isolated, sequential request binding")

    def tensors(self):
        return (*self.stream.tensors(), *self.metadata.tensors()[1:])


@dataclass(frozen=True)
class DeltaFormat:
    template: object
    kind: str = "common_lossless_layer_code_delta_variant"

    @property
    def shape(self):
        return self.template.shape

    def private(self):
        return self.template.private()

    def shared(self):
        return self.template.shared()

    def tensors(self):
        return self.template.tensors()

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())


def convert(packets):
    """Return immutable specifications; decline if scratch removes all savings."""
    packets = tuple(packets)
    previous, requirements, records, converted = {}, {}, [], []
    for index, packet in enumerate(packets):
        current = {}

        def body(role, original, *, current=current, previous=previous, index=index):
            if not isinstance(original, QuantizedBody):
                raise ValueError("original affine body required, never a rebound packet")
            current[role] = original.codes
            stream = encode(previous.get(role), original.codes)
            if not stream.raw:
                requirements[role] = max(requirements.get(role, 0), stream.length)
                if not torch.equal(decode(previous[role], stream), original.codes):
                    raise RuntimeError("GPU code serialization changed an original bit")
            records.append(
                dict(
                    layer=index,
                    role=str(role),
                    original_bytes=original.codes.numel(),
                    serialized_bytes=stream.serialized_bytes,
                    raw_fallback=stream.raw,
                )
            )
            return EncodedBody(replace(original, codes=None), stream)

        converted.append(DeltaFormat(map_bodies(packet, body)))
        previous = current
    old = backing_bytes(tuple(t for p in packets for t in p.tensors()))
    new = backing_bytes(tuple(t for p in converted for t in p.tensors()))
    workspace = sum(requirements.values())
    accepted = bool(workspace and new + workspace < old)
    report = dict(
        original_persistent_bytes=old,
        encoded_persistent_bytes=new,
        required_charged_workspace_bytes=workspace,
        enabled=accepted,
        net_active_bytes=(new + workspace) if accepted else old,
        records=records,
    )
    return (tuple(converted) if accepted else packets), report


class Replay:
    def __init__(self, packets):
        if not packets or not all(isinstance(p, DeltaFormat) for p in packets):
            raise ValueError("complete immutable delta specification sequence required")
        self.packets, self.previous, self.scratch = tuple(packets), {}, {}
        self.last, self.current = -1, None
        for packet in packets:

            def collect(role, body):
                if not isinstance(body, EncodedBody):
                    raise ValueError("explicit immutable encoded bodies required")
                if not body.stream.raw:
                    old = self.scratch.get(role)
                    if old is None or old.numel() < body.stream.length:
                        self.scratch[role] = torch.empty(
                            body.stream.length, dtype=torch.uint8, device=body.stream.payload.device
                        )
                return body

            map_bodies(packet.template, collect)

        from mosaickv.evaluation.replay_descriptor_policy import current

        self.descriptor_metrics = current()
        self.prepared_views, self.prepared_bindings = None, None
        if self.descriptor_metrics is not None:
            views, bindings = [], []
            for packet in self.packets:
                prepared = []

                def prepare(role, encoded, *, prepared=prepared):
                    if not isinstance(encoded, EncodedBody):
                        raise ValueError("immutable encoded metadata required")
                    stream = encoded.stream
                    codes = stream.payload if stream.raw else self.scratch[role][: stream.length]
                    prepared.append((role, encoded, codes))
                    return replace(encoded.metadata, codes=codes)

                views.append(map_bodies(packet.template, prepare))
                bindings.append(tuple(prepared))
            self.prepared_views, self.prepared_bindings = tuple(views), tuple(bindings)
            self.descriptor_metrics["prepared_formats"] += len(views)

    def view(self, index):
        if type(index) is not int or not 0 <= index < len(self.packets):
            raise ValueError("valid private replay layer index required")
        if index == self.last:
            return self.current
        if index != (self.last + 1) % len(self.packets):
            raise RuntimeError("ordered layer replay required; no skipped or shared requests")
        if index == 0:
            self.previous = {}
        current = {}

        if self.prepared_views is None:

            def body(role, encoded):
                stream = encoded.stream
                codes = (
                    stream.payload
                    if stream.raw
                    else decode(
                        self.previous.get(role), stream, self.scratch[role][: stream.length]
                    )
                )
                current[role] = codes
                return replace(encoded.metadata, codes=codes)

            self.current = map_bodies(self.packets[index].template, body)
        else:
            for role, encoded, destination in self.prepared_bindings[index]:
                stream = encoded.stream
                codes = (
                    stream.payload
                    if stream.raw
                    else decode(self.previous.get(role), stream, destination)
                )
                if codes.data_ptr() != destination.data_ptr() or codes.shape != destination.shape:
                    raise RuntimeError("codec returned a foreign replay destination")
                current[role] = codes
            self.current = self.prepared_views[index]
            self.descriptor_metrics["reused_formats"] += 1
        self.previous, self.last = current, index
        return self.current


@dataclass(frozen=True)
class BoundDeltaFormat:
    specification: DeltaFormat
    replay: Replay
    index: int
    kind: str = "common_lossless_layer_code_delta_variant"

    @property
    def shape(self):
        return self.specification.shape

    def private(self):
        return self.specification.private()

    def shared(self):
        # One request owns these buffers, not each layer separately. They must
        # appear in shared_bytes as well as the independent backing inventory.
        return (*self.specification.shared(), *self.replay.scratch.values())

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    def fill(self, attention, targets, *, direct):
        self.replay.view(self.index).fill(attention, targets, direct=direct)

    def fill_component(self, attention, component, destination):
        from mosaickv.evaluation.sequential_kv_lease import fill_component

        fill_component(self.replay.view(self.index), attention, component, destination)


def bind(packets):
    packets = tuple(packets)
    if any(isinstance(p, BoundDeltaFormat) for p in packets):
        raise ValueError("mutable decode bindings cannot be reused across requests")
    if not any(isinstance(p, DeltaFormat) for p in packets):
        return packets
    replay = Replay(packets)
    return tuple(BoundDeltaFormat(p, replay, i) for i, p in enumerate(packets))
