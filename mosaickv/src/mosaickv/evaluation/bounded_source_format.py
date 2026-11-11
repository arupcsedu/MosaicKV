"""Exact original-source reconstruction into bounded, per-head K/V readers.

This is a lossless BF16 source control, not the lossy three-bit format or the
six-mechanism algorithm.  Projection is one native ``mm`` per head.  Its exact
rounding differences from ORIGINAL prefill K/V are stored as sparse BF16 word
patches.  Reading never reconstructs a complete layer's K/V.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.lossless_source_control import owner, projection_signature
from mosaickv.evaluation.shared_source_format import apply_transport_patch

KIND = "original_bf16_source_bounded_head_reader"


def _signature(tensor):
    try:
        version = tensor._version
    except RuntimeError:
        version = "untracked_inference_tensor"
    return owner(tensor), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, version


def _notify(observer, tensors):
    if observer is not None:
        observer(tuple(tensors))


def _heads(heads, count):
    if (
        type(heads) is not tuple
        or any(type(h) is not int or not 0 <= h < count for h in heads)
        or tuple(sorted(set(heads))) != heads
        or len(heads) == count
    ):
        raise ValueError("Sorted unique cached heads must leave at least one source head")
    return heads


def _validate_inputs(normalized, original_pair, attention, phases):
    signature = projection_signature(attention)
    if type(original_pair) is not tuple or len(original_pair) != 2:
        raise ValueError("Complete original K/V pair required")
    key, value = original_pair
    if (
        key.shape != value.shape
        or key.ndim != 4
        or key.shape[0] != 1
        or any(n < 1 for n in key.shape)
        or key.shape[-1] % 2
        or key.numel() >= 2**31
        or any(t.dtype != torch.bfloat16 or t.device != key.device for t in original_pair)
    ):
        raise ValueError("Positive same-device batch-one BF16 full-MHA K/V required")
    _, heads, length, dim = key.shape
    width = heads * dim
    if (
        normalized.shape != (1, length, width)
        or normalized.dtype != torch.bfloat16
        or normalized.device != key.device
        or not normalized.is_contiguous()
        or any(
            p.weight.shape != (width, width)
            or p.weight.dtype != torch.bfloat16
            or p.weight.device != key.device
            or not p.weight.is_contiguous()
            for p in (attention.k_proj, attention.v_proj)
        )
    ):
        raise ValueError("Complete contiguous original BF16 normalized source required")
    if (
        type(phases) is not tuple
        or len(phases) != 2
        or any(
            t.shape != (1, length, dim)
            or t.dtype != torch.bfloat16
            or t.device != key.device
            or not t.is_contiguous()
            for t in phases
        )
    ):
        raise ValueError("Shared original contiguous same-device BF16 phases required")
    if not all(torch.isfinite(t).all().item() for t in (normalized, *original_pair, *phases)):
        raise ValueError("Source, original K/V and phases must be finite")
    if owner(normalized) in {owner(p.weight) for p in (attention.k_proj, attention.v_proj)}:
        raise ValueError("Source cannot alias model weights")
    return signature


@torch.no_grad()
def _project_head(normalized, attention, component, head, target, cos, sin, scratch):
    """Native BF16 output boundaries, including separate RoPE mul/mul/add."""
    dim = target.shape[-1]
    weight = (attention.k_proj, attention.v_proj)[component].weight
    torch.mm(normalized[0], weight[head * dim : (head + 1) * dim].T, out=target)
    if component == 0:
        half = dim // 2
        torch.neg(target[:, half:], out=scratch[:, :half])
        scratch[:, half:].copy_(target[:, :half])
        target.mul_(cos[0])
        scratch.mul_(sin[0])
        target.add_(scratch)


@dataclass(frozen=True)
class OriginalSourceHeadPacket:
    normalized: object
    cos: object
    sin: object
    shape: tuple
    patches: tuple
    cached: tuple
    frozen_projections: tuple
    frozen_inputs: tuple
    source_ownership: str
    kind: str = KIND

    def __post_init__(self):
        if (
            self.kind != KIND
            or type(self.shape) is not tuple
            or len(self.shape) != 4
            or any(type(n) is not int or n < 1 for n in self.shape)
            or self.shape[0] != 1
            or self.shape[-1] % 2
        ):
            raise ValueError("Positive batch-one even-dimension original-source geometry required")
        _, heads, length, dim = self.shape
        if (
            self.normalized.shape != (1, length, heads * dim)
            or self.normalized.dtype != torch.bfloat16
            or not self.normalized.is_contiguous()
            or any(
                t.shape != (1, length, dim)
                or t.dtype != torch.bfloat16
                or t.device != self.normalized.device
                or not t.is_contiguous()
                for t in (self.cos, self.sin)
            )
            or self.source_ownership not in {"private_clone", "caller_transferred_private_capture"}
            or type(self.cached) is not tuple
            or type(self.patches) is not tuple
            or len(self.cached) != heads
            or len(self.patches) != heads
            or all(pair is not None for pair in self.cached)
        ):
            raise ValueError(
                "Complete original source, shared phases and per-head ownership required"
            )
        for pair, patches in zip(self.cached, self.patches, strict=True):
            if pair is not None:
                if (
                    patches is not None
                    or type(pair) is not tuple
                    or len(pair) != 2
                    or any(
                        t.shape != (length, dim)
                        or t.dtype != torch.bfloat16
                        or t.device != self.normalized.device
                        or not t.is_contiguous()
                        for t in pair
                    )
                ):
                    raise ValueError("Functional raw head must retire its patches")
                continue
            if type(patches) is not tuple or len(patches) != 2:
                raise ValueError("Every projected head needs explicit K/V patches")
            for patch in patches:
                if (
                    not isinstance(patch, SparseBitPatch)
                    or patch.indices.dtype != torch.int32
                    or patch.values.dtype != torch.bfloat16
                    or patch.indices.ndim != 1
                    or patch.values.shape != patch.indices.shape
                    or any(
                        t.device != self.normalized.device or not t.is_contiguous()
                        for t in patch.tensors
                    )
                    or (patch.indices < 0).any()
                    or (patch.indices >= length * dim).any()
                    or (patch.indices[1:] <= patch.indices[:-1]).any()
                ):
                    raise ValueError(
                        "Sorted unique in-range per-head original-word patches required"
                    )

    def private(self):
        return (
            self.normalized,
            *(t for pair in self.cached if pair is not None for t in pair),
            *(t for pair in self.patches if pair is not None for p in pair for t in p.tensors),
        )

    def shared(self):
        return self.cos, self.sin

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    @property
    def cached_heads(self):
        return tuple(h for h, pair in enumerate(self.cached) if pair is not None)

    @property
    def raw_head_bytes(self):
        return 4 * self.shape[-2] * self.shape[-1]

    @property
    def patch_bytes_by_head(self):
        return tuple(
            0 if pair is None else backing_bytes(tuple(t for p in pair for t in p.tensors))
            for pair in self.patches
        )

    @property
    def raw_increment_costs(self):
        return tuple(
            0 if pair is not None else self.raw_head_bytes - patch_bytes
            for pair, patch_bytes in zip(self.cached, self.patch_bytes_by_head, strict=True)
        )

    @property
    def diagnostics(self):
        return dict(
            kind=self.kind,
            persistent_bytes=self.component_bytes,
            source_bytes=backing_bytes((self.normalized,)),
            phase_bytes=backing_bytes(self.shared()),
            patch_bytes_by_head=list(self.patch_bytes_by_head),
            raw_increment_costs=list(self.raw_increment_costs),
            raw_head_bytes=self.raw_head_bytes,
            cached_heads=list(self.cached_heads),
            source_heads=self.shape[1] - len(self.cached_heads),
            patch_scalars=sum(p.indices.numel() for pair in self.patches if pair for p in pair),
            all_prompt_positions_retained=True,
            logical_positions="implicit_original_contiguous_prompt_order",
            source_ownership=self.source_ownership,
            phase_backing_shared_not_duplicated=True,
            storage_accounting="unique_tensor_backing_including_patch_indices_and_values",
            workspace_charged_separately=True,
            full_layer_reconstruction=False,
            whole_request_peak_certified=False,
            six_mechanism_performance_claim=False,
        )

    def _check_attention(self, attention):
        if projection_signature(attention) != self.frozen_projections:
            raise ValueError("Frozen projection identity or version changed")
        if (
            tuple(_signature(t) for t in (self.normalized, self.cos, self.sin))
            != self.frozen_inputs
        ):
            raise ValueError("Frozen source or shared phase identity or version changed")

    @torch.no_grad()
    def fill_group(
        self,
        attention,
        component,
        head_start,
        head_end,
        target,
        *,
        rope_scratch=None,
        workspace_observer=None,
    ):
        self._check_attention(attention)
        if (
            type(component) is not int
            or component not in (0, 1)
            or type(head_start) is not int
            or type(head_end) is not int
            or not 0 <= head_start < head_end <= self.shape[1]
            or target.ndim != 4
            or target.shape[:2] != (1, head_end - head_start)
            or target.shape[-2] < self.shape[-2]
            or target.shape[-1] != self.shape[-1]
            or target.dtype != torch.bfloat16
            or target.device != self.normalized.device
            or target.stride(-1) != 1
            or target.stride(-2) != self.shape[-1]
        ):
            raise ValueError(
                "Explicit component/head interval and contiguous per-head BF16 target required"
            )
        forbidden = {owner(t) for t in self.tensors()}
        forbidden.update(owner(p.weight) for p in (attention.k_proj, attention.v_proj))
        if owner(target) in forbidden:
            raise ValueError(
                "Target cannot alias persistent source, cache, phases or model weights"
            )
        if component == 0 and any(self.cached[h] is None for h in range(head_start, head_end)):
            if rope_scratch is None:
                rope_scratch = torch.empty_like(target, memory_format=torch.contiguous_format)
            if (
                rope_scratch.shape != target.shape
                or rope_scratch.dtype != target.dtype
                or rope_scratch.device != target.device
                or rope_scratch.stride(-1) != 1
                or rope_scratch.stride(-2) != self.shape[-1]
                or owner(rope_scratch) == owner(target)
                or owner(rope_scratch) in forbidden
            ):
                raise ValueError("Private disjoint compatible RoPE scratch required")
            _notify(workspace_observer, (target, rope_scratch))
        else:
            _notify(workspace_observer, (target,))
        length = self.shape[-2]
        for local, head in enumerate(range(head_start, head_end)):
            destination = target[0, local, :length, :]
            if self.cached[head] is not None:
                destination.copy_(self.cached[head][component])
            else:
                scratch = rope_scratch[0, local, :length, :] if component == 0 else None
                _project_head(
                    self.normalized,
                    attention,
                    component,
                    head,
                    destination,
                    self.cos,
                    self.sin,
                    scratch,
                )
                apply_transport_patch(
                    self.patches[head][component],
                    destination.view(1, 1, length, self.shape[-1]),
                    prompt_length=length,
                )
        self._check_attention(attention)


@torch.no_grad()
def construct(
    normalized,
    original_pair,
    attention,
    phases,
    *,
    cached_heads=(),
    take_source_ownership=False,
    workspace_observer=None,
):
    signature = _validate_inputs(normalized, original_pair, attention, phases)
    if type(take_source_ownership) is not bool:
        raise ValueError("Explicit source-ownership transfer flag required")
    shape = tuple(original_pair[0].shape)
    _, heads, length, dim = shape
    _heads(cached_heads, heads)
    source = normalized if take_source_ownership else normalized.detach().clone().contiguous()
    source_ownership = (
        "caller_transferred_private_capture" if take_source_ownership else "private_clone"
    )
    temporary = tuple(
        torch.empty((length, dim), dtype=torch.bfloat16, device=source.device) for _ in range(2)
    )
    patches, cached = [], []
    _notify(workspace_observer, (source, *phases, *temporary))
    for head in range(heads):
        if head in cached_heads:
            cached.append(tuple(t[0, head].detach().clone().contiguous() for t in original_pair))
            patches.append(None)
            continue
        cached.append(None)
        pair = []
        for component in (0, 1):
            _project_head(source, attention, component, head, temporary[0], *phases, temporary[1])
            reference = original_pair[component][0, head]
            patch = SparseBitPatch.build(reference, temporary[0])
            apply_transport_patch(patch, temporary[0].view(1, 1, length, dim), prompt_length=length)
            if not torch.equal(
                reference.contiguous().view(torch.int16), temporary[0].view(torch.int16)
            ):
                raise RuntimeError("Original-source head repair did not preserve every K/V word")
            pair.append(patch)
            _notify(
                workspace_observer,
                (source, *phases, *temporary, *(t for p in pair for t in p.tensors)),
            )
        patches.append(tuple(pair))
    return OriginalSourceHeadPacket(
        source,
        *phases,
        shape,
        tuple(patches),
        tuple(cached),
        signature,
        tuple(_signature(t) for t in (source, *phases)),
        source_ownership,
    )


@torch.no_grad()
def reconfigure_cached_heads(packet, original_pair, selected_heads, *, attention=None):
    """Replace patches with FUNCTIONAL raw heads; no unused alignment payload."""
    selected = _heads(selected_heads, packet.shape[1])
    if any(h in packet.cached_heads and h not in selected for h in range(packet.shape[1])):
        if attention is None:
            raise ValueError(
                "Removing a functional raw head requires its frozen attention projection"
            )
        return construct(
            packet.normalized,
            original_pair,
            attention,
            packet.shared(),
            cached_heads=selected,
            take_source_ownership=True,
        )
    if (
        type(original_pair) is not tuple
        or len(original_pair) != 2
        or any(
            t.shape != packet.shape
            or t.dtype != torch.bfloat16
            or t.device != packet.normalized.device
            for t in original_pair
        )
    ):
        raise ValueError("Same-shape original BF16 K/V pair required")
    cached, patches = list(packet.cached), list(packet.patches)
    for head in selected:
        if cached[head] is None:
            cached[head] = tuple(t[0, head].detach().clone().contiguous() for t in original_pair)
            patches[head] = None
    return replace(packet, cached=tuple(cached), patches=tuple(patches))
