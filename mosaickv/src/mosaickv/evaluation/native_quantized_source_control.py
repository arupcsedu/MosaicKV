"""Native-projection control for an UNCHANGED lossy three-bit source packet.

This diagnostic separates fused projection/RoPE roundoff from quantized-input
loss. It does not restore the original FullKV source, select a better cache, or
claim latency/quality superiority. All original packet buffers, including its
now-unused fused-transport patches, remain owned and charged. The only changed
operation is how that same source is read into an existing transient K/V lease.

The context is deliberately process-exclusive and request-private. Register
the final ``SharedSourceFormat`` objects used by the raw-history runtime, not
an ancestor packet subsequently replaced by another constructor. Calls from
another request/thread fail closed while active, and methods are restored even
if decoding or an observer raises. No model hooks or weights are changed.
"""

from __future__ import annotations

from threading import Lock, get_ident

import torch

from mosaickv.evaluation.exact_storage import decoded
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.lossless_source_control import (
    native_component,
    owner,
    projection_signature,
)
from mosaickv.evaluation.precision_tier import QuantizedBody, unpack_cpu
from mosaickv.evaluation.shared_source_format import SharedSourceFormat

KIND = "native_quantized_source_projection_diagnostic"
_scope_lock = Lock()


def _tensor_signature(tensor):
    try:
        version = tensor._version
    except RuntimeError:
        version = "untracked_inference_tensor"
    return owner(tensor), tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype), version


def _packet_signature(packet):
    return (
        id(packet),
        tuple(packet.shape),
        id(packet.source),
        tuple(packet.source.shape),
        packet.source.bits,
        packet.source.axis,
        packet.source.group,
        tuple(_tensor_signature(t) for t in packet.tensors()),
    )


def _validate(packet, attention):
    if type(packet) is not SharedSourceFormat or type(packet.source) is not QuantizedBody:
        raise ValueError("Exact SharedSourceFormat with decoded affine source metadata required")
    if (
        packet.kind != "evicastkv_full_prompt_shared_source_precision_variant"
        or type(packet.shape) is not tuple
        or len(packet.shape) != 4
        or packet.shape[0] != 1
        or packet.shape[-1] != 128
        or not all(type(v) is int and v > 0 for v in packet.shape)
        or packet.shape[2] >= 32768
        or packet.source.bits != 3
        or packet.source.axis != 2
        or packet.source.group != 128
        or packet.source.codebook is not None
        or packet.input_overlay is not None
        or packet.channel_exceptions is not None
        or len(packet.exact) != 2
    ):
        raise ValueError(
            "Original full-MHA three-bit source without optional input overlays required"
        )
    _, heads, length, dim = packet.shape
    width, device = heads * dim, packet.source.codes.device
    if tuple(packet.source.shape) != (1, 1, length, width):
        raise ValueError("Complete original normalized projection source required")
    signature = projection_signature(attention)
    if any(
        projection.weight.shape != (width, width)
        or projection.weight.dtype != torch.bfloat16
        or projection.weight.device != device
        or not projection.weight.is_contiguous()
        for projection in (attention.k_proj, attention.v_proj)
    ):
        raise ValueError("Original same-device full-width BF16 native projections required")
    if (
        packet.row_positions.ndim != 1
        or packet.row_positions.dtype != torch.int32
        or any(t.device != device for t in packet.tensors())
        or any(
            phase.shape != (1, length, dim) or phase.dtype != torch.bfloat16
            for phase in (packet.cos, packet.sin)
        )
        or packet.union.shape != (length,)
        or packet.union.dtype != torch.int32
        or packet.slots.shape != (heads, length)
        or packet.slots.dtype != torch.int16
        or any(
            tuple(body.shape) != (1, 1, packet.row_positions.numel(), dim) for body in packet.exact
        )
    ):
        raise ValueError("Original phases, protected rows and complete identity positions required")
    positions = packet.row_positions
    if (
        (positions < 0).any().item()
        or (positions >= heads * length).any().item()
        or (positions[1:] <= positions[:-1]).any().item()
        or not torch.equal(packet.union, torch.arange(length, dtype=torch.int32, device=device))
        or not torch.equal(packet.slots, packet.union.to(torch.int16)[None].expand(heads, length))
        or any(not torch.equal(p[..., :64], p[..., 64:]) for p in (packet.cos, packet.sin))
        or not all(torch.isfinite(p).all().item() for p in (packet.cos, packet.sin))
    ):
        raise ValueError("Monotonic protected positions and captured original RoPE phases required")
    if {owner(p.weight) for p in (attention.k_proj, attention.v_proj)}.intersection(
        owner(t) for t in packet.tensors()
    ):
        raise ValueError("Packet storage cannot alias model weights")
    return signature


def workspace_upper_bound(packet):
    """Conservative explicit tensor-workspace bound, excluding caller K/V leases.

    Includes the decoded source, original full-width projection, separately
    rounded RoPE intermediates, exact overlay decode and int64 scatter indices.
    BLAS implementation scratch is not certified by this tensor-only bound.
    """
    _, heads, length, dim = packet.shape
    source_bytes = heads * length * dim * 2
    rows = packet.row_positions.numel()
    return int(8 * source_bytes + rows * dim * 2 + rows * 3 * 8)


def _targets(packet, attention, targets):
    _, heads, length, dim = packet.shape
    if len(targets) != 2 or any(
        t.ndim != 4
        or t.shape[:2] != (1, heads)
        or t.shape[-2] < length
        or t.shape[-1] != dim
        or t.dtype != torch.bfloat16
        or t.device != packet.source.codes.device
        or t.stride(-1) != 1
        or t.stride(-2) != dim
        or t.stride(1) < t.shape[-2] * dim
        for t in targets
    ):
        raise ValueError("Two nonoverlapping complete BF16 strided K/V leases required")
    protected = {owner(t) for t in packet.tensors()}
    protected.update(owner(p.weight) for p in (attention.k_proj, attention.v_proj))
    if owner(targets[0]) == owner(targets[1]) or any(owner(t) in protected for t in targets):
        raise ValueError("K/V destinations must not alias source, weights or one another")


@torch.inference_mode()
def _fill(packet, attention, targets, *, direct, workspace_observer=None):
    if type(direct) is not bool:
        raise ValueError("Explicit direct-lease policy required")
    if torch.is_autocast_enabled(packet.source.codes.device.type):
        raise ValueError("Native source control requires autocast disabled")
    _targets(packet, attention, targets)
    if packet.source.codes.device.type == "cuda":
        from mosaickv.evaluation.shared_source_kernel import decode_source_for_construction

        normalized = decode_source_for_construction(packet)
    else:
        normalized = unpack_cpu(packet.source).squeeze(1)

    scatter_workspace = ()

    def observe(tensors):
        if workspace_observer is not None:
            workspace_observer((normalized, *scatter_workspace, *tensors))

    observe(())
    if normalized.shape != (1, packet.shape[2], packet.shape[1] * packet.shape[3]):
        raise ValueError("Decoded source geometry differs from registered native source")
    length = packet.shape[2]
    indices = packet.row_positions.long()
    heads, positions = indices // length, indices % length
    scatter_workspace = indices, heads, positions
    observe((indices, heads, positions))
    for component, (target, body) in enumerate(zip(targets, packet.exact, strict=True)):
        value = native_component(
            normalized, attention, component, packet.cos, packet.sin, observer=observe
        )
        target[..., :length, :].copy_(value)
        exact = decoded(body)
        observe((value, indices, heads, positions, exact))
        target[0, heads, positions] = exact[0, 0]
        del value, exact
    # Deliberately no apply_transport_patch: those patches target the fused
    # reader, not this exact native projection of the same quantized source.


def native_quantized_fill(packet, attention, targets, *, direct, workspace_observer=None):
    """Validated standalone native source control; no persistent representation change."""
    _validate(packet, attention)
    _fill(packet, attention, targets, direct=direct, workspace_observer=workspace_observer)


class NativeQuantizedSourceControl:
    """Scope native reads to exact packet+attention identities in one request.

    Registering stores only references and scalar signatures, never duplicate
    tensor owners. Packet bytes stay identical, including unused transport
    patches. This is not a low-latency implementation or a FullKV fidelity fix.
    """

    def __init__(self, packets, attentions):
        packets, attentions = tuple(packets), tuple(attentions)
        if (
            not packets
            or len(packets) != len(attentions)
            or len({id(p) for p in packets}) != len(packets)
        ):
            raise ValueError("Unique request-private packet per original attention required")
        self._registered = {}
        for packet, attention in zip(packets, attentions, strict=True):
            signature = _validate(packet, attention)
            self._registered[id(packet)] = (packet, attention, _packet_signature(packet), signature)
        self._entered = False
        self.report = dict(
            kind=KIND,
            same_lossy_three_bit_source=True,
            original_fullkv_source_losslessness_claim=False,
            persistent_tensor_bytes_added=0,
            stored_transport_patches_ignored_but_charged=True,
            packet_bytes=backing_bytes(tuple(t for p in packets for t in p.tensors())),
            tensor_workspace_upper_bound=max(workspace_upper_bound(p) for p in packets),
            native_fill_calls=0,
            reference="native_quantized_source_with_original_protected_rows",
        )

    def _check(self, packet, attention):
        if get_ident() != self._thread:
            raise RuntimeError("Native source control cannot cross request/thread boundaries")
        entry = self._registered.get(id(packet))
        if entry is None or entry[0] is not packet or entry[1] is not attention:
            raise ValueError("Foreign packet or attention is not registered in this native control")
        if _packet_signature(packet) != entry[2] or projection_signature(attention) != entry[3]:
            raise RuntimeError("Registered packet or native projection signature changed")

    def __enter__(self):
        if self._entered or not _scope_lock.acquire(blocking=False):
            raise RuntimeError("Native source control is process-exclusive and cannot be nested")
        self._entered, self._thread = True, get_ident()
        try:
            for packet, attention, _, _ in self._registered.values():
                self._check(packet, attention)
        except BaseException:
            self._entered = False
            _scope_lock.release()
            raise
        self._original_fill, self._original_materialize = (
            SharedSourceFormat.fill,
            SharedSourceFormat.materialize,
        )
        control = self

        def fill(packet, attention, targets, *, direct, workspace_observer=None):
            control._check(packet, attention)
            _fill(packet, attention, targets, direct=direct, workspace_observer=workspace_observer)
            control.report["native_fill_calls"] += 1

        def materialize(packet, attention):
            control._check(packet, attention)
            targets = tuple(
                torch.empty(packet.shape, device=packet.source.codes.device, dtype=torch.bfloat16)
                for _ in range(2)
            )
            fill(packet, attention, targets, direct=True)
            return targets

        SharedSourceFormat.fill, SharedSourceFormat.materialize = fill, materialize
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            SharedSourceFormat.fill, SharedSourceFormat.materialize = (
                self._original_fill,
                self._original_materialize,
            )
            if exc_type is None:
                for packet, attention, _, _ in self._registered.values():
                    self._check(packet, attention)
        finally:
            self._entered = False
            _scope_lock.release()
        return False
