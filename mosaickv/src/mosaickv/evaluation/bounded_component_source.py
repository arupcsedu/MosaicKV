"""Original BF16 source with one full-width native component projection.

The serving buffer contains K OR V, never a simultaneously materialized layer
pair.  Its sequence-major physical layout is the original Linear output layout.
Per-head RoPE uses one reusable head scratch, with separate BF16 mul/mul/add.
Sparse patches target original prefill words, not the lossy three-bit decoder.
This numerical/storage diagnostic does not establish the old 49 MiB claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mosaickv.evaluation.bounded_source_format import _notify, _signature, _validate_inputs
from mosaickv.evaluation.exact_storage import SparseBitPatch
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.lossless_source_control import (
    FeasibilityDeclined,
    owner,
    projection_signature,
)
from mosaickv.evaluation.shared_source_format import apply_transport_patch

KIND = "original_bf16_full_width_component_source_control"


def _projection_view(target, *, heads, length, dim):
    """A view, never a layout-changing copy or hidden full-component tensor."""
    if (
        target.ndim != 4
        or target.shape[:2] != (1, heads)
        or target.shape[-2] < length
        or target.shape[-1] != dim
        or target.stride(-1) != 1
        or target.stride(-3) != dim
        or target.stride(-2) != heads * dim
    ):
        raise ValueError(
            "Projection-layout BHND target with head stride D and token stride H*D required"
        )
    result = target.transpose(1, 2)[0, :length].view(length, heads * dim)
    if not result.is_contiguous() or owner(result) != owner(target):
        raise RuntimeError("Native projection output must be an existing contiguous buffer view")
    return result


def _scratch_view(scratch, *, length, dim, target, forbidden):
    if (
        scratch.ndim != 2
        or scratch.shape[0] < length
        or scratch.shape[1] != dim
        or scratch.dtype != torch.bfloat16
        or scratch.device != target.device
        or not scratch.is_contiguous()
        or owner(scratch) == owner(target)
        or owner(scratch) in forbidden
    ):
        raise ValueError("Private disjoint contiguous [capacity,D] BF16 RoPE scratch required")
    return scratch[:length]


@torch.no_grad()
def _project_component(normalized, attention, component, target, cos, sin, scratch):
    _, heads, length, dim = 1, target.shape[1], normalized.shape[1], target.shape[-1]
    output = _projection_view(target, heads=heads, length=length, dim=dim)
    weight = (attention.k_proj, attention.v_proj)[component].weight
    # Full M, N and K are identical to the original native Linear projection.
    # In particular, do not replace this with one head-width GEMM per head.
    torch.mm(normalized[0], weight.T, out=output)
    if component == 0:
        half = dim // 2
        for head in range(heads):
            value = target[0, head, :length]
            torch.neg(value[:, half:], out=scratch[:, :half])
            scratch[:, half:].copy_(value[:, :half])
            value.mul_(cos[0])
            scratch.mul_(sin[0])
            value.add_(scratch)


@torch.no_grad()
def _apply_patch(patch, target, shape):
    _, heads, length, dim = shape
    flat = _projection_view(target, heads=heads, length=length, dim=dim)
    # Patch indices are position/head/width-major. This reshaping lets the
    # existing int32 CUDA patch kernel operate without an int64 index copy.
    apply_transport_patch(patch, flat.view(1, 1, length * heads, dim), prompt_length=length * heads)


@dataclass(frozen=True)
class OriginalComponentSourcePacket:
    normalized: object
    cos: object
    sin: object
    patches: tuple
    shape: tuple
    raw_kv_bytes: int
    frozen_projections: tuple
    frozen_inputs: tuple
    source_ownership: str
    maximum_persistent_bytes: int | None = None
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
            raise ValueError("Positive batch-one original-source geometry required")
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
            or type(self.patches) is not tuple
            or len(self.patches) != 2
            or type(self.raw_kv_bytes) is not int
            or self.raw_kv_bytes < 1
        ):
            raise ValueError(
                "Complete original BF16 source, shared phases and two patches required"
            )
        size = heads * length * dim
        for patch in self.patches:
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
                or (patch.indices >= size).any()
                or (patch.indices[1:] <= patch.indices[:-1]).any()
            ):
                raise ValueError("Sorted unique sequence-major original-word patches required")
        if self.maximum_persistent_bytes is not None:
            if type(self.maximum_persistent_bytes) is not int or self.maximum_persistent_bytes < 1:
                raise ValueError("Positive exact persistent-byte cap required")
            if self.component_bytes > self.maximum_persistent_bytes:
                raise FeasibilityDeclined(dict(self.diagnostics, feasible=False))

    def private(self):
        return (self.normalized, *(t for patch in self.patches for t in patch.tensors))

    def shared(self):
        return self.cos, self.sin

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    @property
    def diagnostics(self):
        changed = [int(p.indices.numel()) for p in self.patches]
        scalars = 2 * self.shape[1] * self.shape[2] * self.shape[3]
        return dict(
            kind=self.kind,
            persistent_bytes=self.component_bytes,
            source_bytes=backing_bytes((self.normalized,)),
            phase_bytes=backing_bytes(self.shared()),
            patch_bytes=backing_bytes(tuple(t for patch in self.patches for t in patch.tensors)),
            patch_scalars=changed,
            total_kv_scalars=scalars,
            patch_density=sum(changed) / scalars,
            raw_kv_bytes=self.raw_kv_bytes,
            strictly_smaller_than_original_raw_kv=self.component_bytes < self.raw_kv_bytes,
            source_ownership=self.source_ownership,
            all_original_prompt_positions_retained=True,
            logical_positions="implicit_original_contiguous_prompt_order",
            patch_index_layout="original_position_head_width",
            projection="original_full_M_full_width_native_mm",
            rope="separate_BF16_mul_mul_add_one_reusable_head_scratch",
            raw_head_caches=False,
            maximum_persistent_bytes=self.maximum_persistent_bytes,
            workspace_charged_separately=True,
            simultaneous_full_layer_KV_pair=False,
            whole_request_peak_certified=False,
            old_49_MiB_request_claim=False,
            six_mechanism_performance_claim=False,
        )

    def _check_attention(self, attention):
        if projection_signature(attention) != self.frozen_projections:
            raise ValueError("Frozen projection identity or version changed")
        actual = tuple(_signature(t) for t in (self.normalized, self.cos, self.sin))
        if actual != self.frozen_inputs:
            raise ValueError("Frozen source or shared phase identity or version changed")

    @torch.no_grad()
    def fill_component(
        self, attention, component, target, *, rope_scratch=None, workspace_observer=None
    ):
        self._check_attention(attention)
        if (
            type(component) is not int
            or component not in (0, 1)
            or target.dtype != torch.bfloat16
            or target.device != self.normalized.device
        ):
            raise ValueError(
                "Explicit K/V component and same-device BF16 projection target required"
            )
        _, heads, length, dim = self.shape
        _projection_view(target, heads=heads, length=length, dim=dim)
        forbidden = {owner(t) for t in self.tensors()}
        forbidden.update(owner(p.weight) for p in (attention.k_proj, attention.v_proj))
        if owner(target) in forbidden:
            raise ValueError(
                "Projection target cannot alias source, phases, patches or model weights"
            )
        if component == 0:
            if rope_scratch is None:
                rope_scratch = torch.empty(
                    (length, dim), dtype=torch.bfloat16, device=target.device
                )
            scratch = _scratch_view(
                rope_scratch, length=length, dim=dim, target=target, forbidden=forbidden
            )
            _notify(workspace_observer, (target, rope_scratch))
        else:
            scratch = None
            _notify(workspace_observer, (target,))
        _project_component(
            self.normalized, attention, component, target, self.cos, self.sin, scratch
        )
        _apply_patch(self.patches[component], target, self.shape)
        self._check_attention(attention)


@torch.no_grad()
def construct(
    normalized,
    original_pair,
    attention,
    phases,
    *,
    take_source_ownership=False,
    maximum_persistent_bytes=None,
    workspace_observer=None,
    workspace=None,
    exact_only=False,
):
    signature = _validate_inputs(normalized, original_pair, attention, phases)
    if type(take_source_ownership) is not bool:
        raise ValueError("Explicit source-ownership transfer flag required")
    if type(exact_only) is not bool:
        raise ValueError("Explicit exact-only construction flag required")
    if exact_only and workspace is None:
        raise ValueError("Exact-only streaming construction requires the charged workspace")
    if maximum_persistent_bytes is not None and (
        type(maximum_persistent_bytes) is not int or maximum_persistent_bytes < 1
    ):
        raise ValueError("Positive exact persistent-byte cap required")
    shape = tuple(original_pair[0].shape)
    _, heads, length, dim = shape
    source = normalized if take_source_ownership else normalized.detach().clone().contiguous()
    if workspace is None:
        projection = torch.empty(
            (1, length, heads, dim), dtype=torch.bfloat16, device=source.device
        )
        target = projection.transpose(1, 2)
        scratch = torch.empty((length, dim), dtype=torch.bfloat16, device=source.device)
        workspace_owners = (projection, scratch)
    else:
        from mosaickv.evaluation.bounded_component_attention import ComponentWorkspace

        if (
            not isinstance(workspace, ComponentWorkspace)
            or workspace.heads != heads
            or workspace.head_dim != dim
            or workspace.component.shape != (workspace.slots, heads * dim)
            or workspace.head.shape != (workspace.slots, dim)
            or workspace.component.device != source.device
            or workspace.head.device != source.device
            or any(
                t.dtype != torch.bfloat16 or not t.is_contiguous()
                for t in workspace.owned_tensors()
            )
            or owner(workspace.component) == owner(workspace.head)
            or any(
                owner(t) in {owner(x) for x in (source, normalized, *original_pair, *phases)}
                for t in workspace.owned_tensors()
            )
            or any(
                owner(t) in {owner(p.weight) for p in (attention.k_proj, attention.v_proj)}
                for t in workspace.owned_tensors()
            )
        ):
            raise ValueError("private compatible same-device component/head workspace required")
        target = workspace.component_view(length)
        projection = target.transpose(1, 2)
        scratch = workspace.head[:length]
        workspace_owners = workspace.owned_tensors()
    _notify(workspace_observer, (source, *phases, *workspace_owners))
    patches = []
    for component in (0, 1):
        _project_component(source, attention, component, target, *phases, scratch)
        original = original_pair[component].transpose(1, 2)
        if exact_only:
            from mosaickv.evaluation.bounded_word_equality import equal_words

            if not equal_words(original_pair[component], target, scratch=workspace.head):
                raise RuntimeError("Exact-only streaming source changed an original prefill word")
            patch = SparseBitPatch(
                torch.empty(0, dtype=torch.int32, device=source.device),
                torch.empty(0, dtype=torch.bfloat16, device=source.device),
            )
        else:
            patch = SparseBitPatch.build(original, projection)
            _apply_patch(patch, target, shape)
            if not torch.equal(
                original.contiguous().view(torch.int16), projection.view(torch.int16)
            ):
                raise RuntimeError(
                    "Native component source repair changed an original prefill word"
                )
        patches.append(patch)
        _notify(
            workspace_observer,
            (source, *phases, *workspace_owners, *(t for p in patches for t in p.tensors)),
        )
    return OriginalComponentSourcePacket(
        source,
        *phases,
        tuple(patches),
        shape,
        backing_bytes(original_pair),
        signature,
        tuple(_signature(t) for t in (source, *phases)),
        "caller_transferred_private_capture" if take_source_ownership else "private_clone",
        maximum_persistent_bytes,
    )
