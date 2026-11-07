"""Integer-only native source codec, explicitly scoped and default off."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from unittest.mock import patch

import torch

_scope = ContextVar("evicastkv_all_head_source_codec", default=None)


@dataclass(frozen=True)
class DeviceSymbols:
    length: int
    payload: object
    offsets: object
    has_previous: bool
    raw: bool = False

    def tensors(self):
        return self.payload, self.offsets

    @property
    def serialized_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors())


@dataclass(frozen=True)
class CodecChoice:
    packet: object
    selector: object

    def __getattr__(self, name):
        return getattr(self.packet, name)

    def tensors(self):
        return (*self.packet.tensors(), self.selector)

    @property
    def serialized_bytes(self):
        return self.packet.serialized_bytes + self.selector.numel()


def encode(previous, current):
    from mosaickv.evaluation.all_head_source_kernel import count, pack
    from mosaickv.evaluation.lossless_code_delta_gpu import validate_bytes

    validate_bytes(current)
    if current.numel() % 192:
        raise ValueError("complete original three-bit source symbol groups required")
    if previous is not None:
        validate_bytes(previous)
        if previous.shape != current.shape or previous.device != current.device:
            raise ValueError("unchanged complete predecessor layout required")
    groups = current.numel() // 192
    sizes = torch.empty(groups, dtype=torch.int32, device=current.device)
    modes = torch.empty_like(sizes)
    parent = previous if previous is not None else current
    count[(groups,)](parent, current, sizes, modes, HAS_PREVIOUS=previous is not None)
    ends = sizes.cumsum(0, dtype=torch.int32)
    length = int(ends[-1].item())
    if length + 4 * (groups + 1) >= current.numel():
        return None
    payload = torch.empty(length, dtype=torch.uint8, device=current.device)
    offsets = torch.empty(groups + 1, dtype=torch.int32, device=current.device)
    pack[(groups,)](
        parent,
        current,
        sizes,
        modes,
        ends,
        payload,
        offsets,
        GROUPS=groups,
        HAS_PREVIOUS=previous is not None,
    )
    return DeviceSymbols(current.numel(), payload, offsets, previous is not None)


def decode(previous, packet, destination=None):
    from mosaickv.evaluation.all_head_source_kernel import unpack
    from mosaickv.evaluation.lossless_code_delta_gpu import validate_bytes

    if not isinstance(packet, DeviceSymbols):
        raise ValueError("native source symbol packet required")
    if (
        type(packet.length) is not int
        or packet.length <= 0
        or packet.length % 384
        or type(packet.has_previous) is not bool
        or packet.raw is not False
        or packet.payload.device.type != "cuda"
        or packet.payload.dtype != torch.uint8
        or packet.payload.ndim != 1
        or not packet.payload.is_contiguous()
        or not packet.payload.numel()
        or packet.offsets.dtype != torch.int32
        or packet.offsets.shape != (packet.length // 192 + 1,)
        or packet.offsets.device != packet.payload.device
        or not packet.offsets.is_contiguous()
    ):
        raise ValueError("complete encoder-produced CUDA source descriptor required")
    if packet.has_previous:
        validate_bytes(previous)
        if previous.numel() != packet.length or previous.device != packet.payload.device:
            raise ValueError("original same-device predecessor required")
    output = (
        torch.empty(packet.length, dtype=torch.uint8, device=packet.payload.device)
        if (destination is None)
        else validate_bytes(destination)
    )
    if output.numel() != packet.length or output.device != packet.payload.device:
        raise ValueError("original charged complete source destination required")
    if any(
        output.untyped_storage().data_ptr() == t.untyped_storage().data_ptr()
        for t in packet.tensors()
    ):
        raise ValueError("reconstruction cannot alias immutable source archive")
    unpack[(packet.length // 192,)](
        previous if packet.has_previous else output,
        packet.payload,
        packet.offsets,
        output,
        HAS_PREVIOUS=packet.has_previous,
    )
    return output


class CodecState:
    def __init__(self):
        self.records, self.references = [], []
        self.coverage = []

    def report(self):
        return dict(
            variant="all_head_lossless_source_codec_v1",
            default_off=True,
            records=self.records,
            compressed_replacements=sum(r["symbol_codec_selected"] for r in self.records),
            original_owners_retired=sum(ref() is None for ref in self.references),
            benchmark_labels_accessed=False,
            all_shared_source_consumers_covered=bool(self.coverage)
            and all(x["all_shared_consumers_covered"] for x in self.coverage),
            coverage=self.coverage,
            quality_improvement_established=False,
            whole_request_peak_certified=False,
        )


@contextmanager
def use(enabled=True):
    """Scope new packet types/decoder to one private runtime, never globally on."""
    import weakref

    from mosaickv.evaluation import layer_code_delta as runtime
    from mosaickv.evaluation.all_head_source_codec import SourceCoverage

    state = CodecState()
    if not enabled:
        yield state
        return
    token = _scope.set(state)
    original_encode, original_decode, original_convert = (
        runtime.encode,
        runtime.decode,
        runtime.convert,
    )

    def encoder(previous, current):
        active = _scope.get()
        if active is None:
            return original_encode(previous, current)
        legacy = original_encode(previous, current)
        alternate = encode(previous, current)
        selected = alternate is not None and alternate.serialized_bytes < legacy.serialized_bytes
        packet = alternate if selected else legacy
        selector = torch.tensor([int(selected)], dtype=torch.uint8, device=current.device)
        active.records.append(
            dict(
                source_code_bytes=current.numel(),
                legacy_serialized_bytes=legacy.serialized_bytes,
                selected_serialized_bytes=packet.serialized_bytes + 1,
                symbol_codec_selected=selected,
            )
        )
        if selected:
            active.references.append(weakref.ref(current))
        return CodecChoice(packet, selector)

    def decoder(previous, packet, destination=None):
        if isinstance(packet, CodecChoice):
            packet = packet.packet
        return (
            decode(previous, packet, destination)
            if isinstance(packet, DeviceSymbols)
            else (original_decode(previous, packet, destination))
        )

    def convert(packets):
        packets = tuple(packets)
        for layer, packet in enumerate(packets):
            h, n, dim = packet.shape[1:]

            def validate(role, body, *, h=h, n=n, dim=dim, layer=layer):
                if (
                    role != "source"
                    or body.bits != 3
                    or body.axis != 2
                    or body.shape != (1, 1, n, h * dim)
                ):
                    raise ValueError("source-only original full-MHA layout required")
                proof = SourceCoverage(
                    layer, tuple(range(h)), ("key", "value"), tuple(range(n)), ((0, h * dim),)
                ).verify(kv_heads=h, head_dim=dim, length=n)
                state.coverage.append(dict(layer=layer, **proof))
                return body

            runtime.map_bodies(packet, validate)
        return original_convert(packets)

    try:
        with (
            patch.object(runtime, "encode", encoder),
            patch.object(runtime, "decode", decoder),
            patch.object(runtime, "convert", convert),
        ):
            yield state
    finally:
        _scope.reset(token)
