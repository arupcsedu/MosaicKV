"""Lossless BF16 normalized-source fidelity control, not the six-part method.

Store a complete projection input once, recompute native K/V, and repair every
different BF16 word against the ORIGINAL prefill cache. This is not the earlier
lossless execution replacement for an already lossy three-bit packet decoder.
No labels, future tokens, selection, fitting of weights or approximate guard.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

KIND = "lossless_bf16_normalized_source_fidelity_control"


def owner(tensor):
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage._cdata


def projection_signature(attention):
    result = []
    for projection in (attention.k_proj, attention.v_proj):
        if type(projection) is not torch.nn.Linear or projection.bias is not None:
            raise ValueError("Native bias-free nn.Linear projections required")
        if projection.training or projection._forward_hooks or projection._forward_pre_hooks:
            raise ValueError("Frozen eval projections without active hooks required")
        weight = projection.weight
        try:
            version = weight._version
        except RuntimeError:
            version = "untracked_inference_tensor"
        result.append((owner(weight), tuple(weight.shape), str(weight.dtype), version))
    if getattr(attention, "training", False) or getattr(attention, "num_key_value_groups", 1) != 1:
        raise ValueError("Eval full-MHA attention required; GQA is not supported")
    return tuple(result)


def notify(observer, tensors):
    if observer is not None:
        observer(tuple(tensors))


@torch.no_grad()
def native_component(normalized, attention, component, cos, sin, *, observer=None):
    """Use the original full-width projection, not a head-BMM or token tile."""
    if type(component) is not int or component not in (0, 1):
        raise ValueError("Explicit K or V component required")
    heads = attention.k_proj.out_features // cos.shape[-1]
    length, dim = normalized.shape[1], cos.shape[-1]
    projected = (attention.k_proj, attention.v_proj)[component](normalized)
    notify(observer, (projected,))
    value = projected.view(1, length, heads, dim).transpose(1, 2)
    if component == 0:
        # Preserve eager Llama's multiply / rotate-half / multiply / add order.
        left = value * cos[:, None]
        negative = -value[..., dim // 2 :]
        rotated = torch.cat((negative, value[..., : dim // 2]), dim=-1)
        right = rotated * sin[:, None]
        notify(observer, (projected, left, negative, rotated, right))
        result = left + right
        notify(observer, (projected, left, negative, rotated, right, result))
        return result
    return value


def validate_inputs(normalized, original_pair, attention, phases, logical_positions):
    signature = projection_signature(attention)
    if len(original_pair) != 2:
        raise ValueError("Complete original K/V pair required")
    key, value = original_pair
    if (
        key.shape != value.shape
        or key.ndim != 4
        or key.shape[0] != 1
        or key.dtype != torch.bfloat16
        or value.dtype != key.dtype
        or key.device != value.device
        or key.numel() >= 2**31
        or not key.numel()
    ):
        raise ValueError("Finite batch-one int32-addressable BF16 original cache required")
    _, heads, length, dim = key.shape
    width = heads * dim
    if (
        dim % 2
        or normalized.shape != (1, length, width)
        or normalized.dtype != key.dtype
        or normalized.device != key.device
        or any(
            p.weight.shape != (width, width)
            or p.weight.dtype != key.dtype
            or p.weight.device != key.device
            or not p.weight.is_contiguous()
            for p in (attention.k_proj, attention.v_proj)
        )
    ):
        raise ValueError("Complete same-device full-MHA BF16 projection input required")
    if len(phases) != 2:
        raise ValueError("Captured original cosine/sine required")
    if any(
        p.shape != (1, length, dim) or p.dtype != key.dtype or p.device != key.device
        for p in phases
    ):
        raise ValueError("Captured same-device BF16 phases required")
    if (
        logical_positions.shape != (length,)
        or logical_positions.dtype not in (torch.int32, torch.int64)
        or logical_positions.device != key.device
        or (logical_positions < 0).any()
        or (logical_positions[1:] <= logical_positions[:-1]).any()
    ):
        raise ValueError("All original positions must be nonnegative and strictly monotonic")
    if not all(torch.isfinite(t).all().item() for t in (normalized, key, value, *phases)):
        raise ValueError("All original source, K/V and phases must be finite")
    weights = {owner(p.weight) for p in (attention.k_proj, attention.v_proj)}
    if weights.intersection(owner(t) for t in original_pair):
        raise ValueError("Original cache cannot be a model-weight view")
    return signature


class FeasibilityDeclined(RuntimeError):
    def __init__(self, report):
        super().__init__("Lossless source is not strictly smaller within the registered cap")
        self.report = report


@dataclass(frozen=True)
class LosslessSourceFormat:
    normalized: object
    cos: object
    sin: object
    logical_positions: object
    patches: tuple
    shape: tuple
    raw_kv_bytes: int
    frozen_projections: tuple
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
            raise ValueError("Registered full-prompt source geometry required")
        _, heads, length, dim = self.shape
        if (
            self.normalized.shape != (1, length, heads * dim)
            or self.normalized.dtype != torch.bfloat16
            or not self.normalized.is_contiguous()
            or any(
                p.shape != (1, length, dim)
                or p.dtype != torch.bfloat16
                or p.device != self.normalized.device
                or not p.is_contiguous()
                for p in (self.cos, self.sin)
            )
        ):
            raise ValueError("Owned complete BF16 source and phases required")
        if (
            self.logical_positions.shape != (length,)
            or self.logical_positions.dtype not in (torch.int32, torch.int64)
            or self.logical_positions.device != self.normalized.device
            or not self.logical_positions.is_contiguous()
            or (self.logical_positions < 0).any()
            or (self.logical_positions[1:] <= self.logical_positions[:-1]).any()
        ):
            raise ValueError("Complete monotonic original position map required")
        if type(self.patches) is not tuple or len(self.patches) != 2:
            raise ValueError("Two explicit original-K/V bitpatches required")
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
                raise ValueError("Sorted unique in-range same-device exact bitpatches required")
        if type(self.raw_kv_bytes) is not int or self.raw_kv_bytes < 1:
            raise ValueError("Actual positive original-cache backing cost required")
        if self.maximum_persistent_bytes is not None and (
            type(self.maximum_persistent_bytes) is not int or self.maximum_persistent_bytes < 1
        ):
            raise ValueError("Positive explicit persistent-byte cap required")
        if self.component_bytes >= self.raw_kv_bytes or (
            self.maximum_persistent_bytes is not None
            and self.component_bytes > self.maximum_persistent_bytes
        ):
            raise FeasibilityDeclined(dict(self.diagnostics, feasible=False))

    def private(self):
        return (self.normalized, *(t for patch in self.patches for t in patch.tensors))

    def shared(self):
        return self.cos, self.sin, self.logical_positions

    def tensors(self):
        return (*self.private(), *self.shared())

    @property
    def component_bytes(self):
        return backing_bytes(self.tensors())

    @property
    def diagnostics(self):
        scalars = 2 * self.shape[1] * self.shape[2] * self.shape[3]
        changed = [int(p.indices.numel()) for p in self.patches]
        return {
            "kind": self.kind,
            "raw_kv_bytes": self.raw_kv_bytes,
            "persistent_bytes": self.component_bytes,
            "source_bytes": backing_bytes((self.normalized,)),
            "phase_and_map_bytes": backing_bytes(self.shared()),
            "patch_bytes": backing_bytes(tuple(t for p in self.patches for t in p.tensors)),
            "patch_scalars": changed,
            "total_kv_scalars": scalars,
            "patch_density": sum(changed) / scalars,
            "maximum_persistent_bytes": self.maximum_persistent_bytes,
            "all_prompt_positions_retained": True,
            "mandatory_tokens_preserved_by_exact_all_row_reconstruction": True,
            "reference": "original_prefill_KV_not_lossy_parent_decoder",
            "model_weights_modified": False,
            "benchmark_labels_used": False,
            "projection": "original_full_prompt_nn_Linear",
            "workspace_charged_separately": True,
            "whole_request_peak_certified": False,
            "latency_claim": False,
            "six_mechanism_performance_claim": False,
        }

    def _check_attention(self, attention):
        if projection_signature(attention) != self.frozen_projections:
            raise ValueError("Frozen projection identity or version changed")

    @torch.no_grad()
    def _reconstruct(self, attention, component, *, workspace_observer=None):
        self._check_attention(attention)
        candidate = native_component(
            self.normalized,
            attention,
            component,
            self.cos,
            self.sin,
            observer=workspace_observer,
        )
        # Native Linear returns sequence-major storage viewed as [B,H,N,D].
        # The existing exact CUDA patch kernel requires sequence stride D.
        # Keep both allocations observable while the head-major copy is live.
        native = candidate
        candidate = native.contiguous()
        notify(workspace_observer, (native, candidate))
        del native
        if not torch.isfinite(candidate).all():
            raise ValueError("Nonfinite native reconstruction is not admissible")
        from mosaickv.evaluation.shared_source_format import apply_transport_patch

        apply_transport_patch(self.patches[component], candidate, prompt_length=self.shape[-2])
        notify(workspace_observer, (candidate,))
        self._check_attention(attention)
        return candidate

    @torch.no_grad()
    def fill_component(self, attention, component, destination, *, workspace_observer=None):
        if type(component) is not int or component not in (0, 1):
            raise ValueError("Explicit K or V component required")
        if (
            destination.ndim != 4
            or destination.shape[:2] != self.shape[:2]
            or destination.shape[-2] < self.shape[-2]
            or destination.shape[-1] != self.shape[-1]
            or destination.dtype != torch.bfloat16
            or destination.device != self.normalized.device
            or destination.stride(-1) != 1
        ):
            raise ValueError("Compatible same-device BF16 layer destination required")
        forbidden = {owner(t) for t in self.tensors()}
        forbidden.update(owner(p.weight) for p in (attention.k_proj, attention.v_proj))
        if owner(destination) in forbidden:
            raise ValueError("Destination must not alias packet storage or model weights")
        candidate = self._reconstruct(attention, component, workspace_observer=workspace_observer)
        destination[..., : self.shape[-2], :].copy_(candidate)

    @torch.no_grad()
    def fill(self, attention, targets, *, direct, workspace_observer=None):
        if direct is not True or len(targets) != 2 or owner(targets[0]) == owner(targets[1]):
            raise ValueError(
                "Two private distinct K/V destinations and direct construction required"
            )
        for component, target in enumerate(targets):
            self.fill_component(attention, component, target, workspace_observer=workspace_observer)

    @torch.no_grad()
    def materialize(self, attention, *, workspace_observer=None):
        targets = tuple(
            torch.empty(self.shape, device=self.normalized.device, dtype=torch.bfloat16)
            for _ in range(2)
        )
        notify(workspace_observer, targets)
        self.fill(attention, targets, direct=True, workspace_observer=workspace_observer)
        return targets


@torch.no_grad()
def construct(
    normalized,
    original_pair,
    attention,
    phases,
    logical_positions,
    *,
    maximum_persistent_bytes=None,
    workspace_observer=None,
):
    """Exact original-K/V target; a declined control never falls back to raw KV.

    Observers receive explicit native tensor temporaries; a surrounding
    ScratchAudit/ownership timeline must also account for patch-builder
    intermediates and opaque vendor workspace. No whole-request peak claim.
    """
    if maximum_persistent_bytes is not None and (
        type(maximum_persistent_bytes) is not int or maximum_persistent_bytes < 1
    ):
        raise ValueError("Positive explicit persistent-byte cap required")
    signature = validate_inputs(normalized, original_pair, attention, phases, logical_positions)
    source = normalized.detach().clone().contiguous()
    cos, sin = (p.detach().clone().contiguous() for p in phases)
    positions = logical_positions.detach().clone().contiguous()
    persistent = source, cos, sin, positions
    notify(workspace_observer, (*original_pair, normalized, *persistent))
    patches = []
    for component, reference in enumerate(original_pair):
        actual = native_component(
            source, attention, component, cos, sin, observer=workspace_observer
        )
        if not torch.isfinite(actual).all():
            raise ValueError("Nonfinite native reconstruction is not admissible")
        patch = SparseBitPatch.build(reference, actual)
        patches.append(patch)
        notify(
            workspace_observer,
            (*original_pair, *persistent, actual, *(t for p in patches for t in p.tensors)),
        )
        del actual
    packet = LosslessSourceFormat(
        source,
        cos,
        sin,
        positions,
        tuple(patches),
        tuple(original_pair[0].shape),
        backing_bytes(tuple(original_pair)),
        signature,
        maximum_persistent_bytes,
    )
    forbidden = {owner(t) for t in (*original_pair, normalized, *phases, logical_positions)}
    forbidden.update(owner(p.weight) for p in (attention.k_proj, attention.v_proj))
    if any(owner(t) in forbidden for t in packet.tensors() if t.numel()):
        raise RuntimeError("Persistent source control aliases an original owner or model weight")
    if packet.component_bytes >= packet.raw_kv_bytes or (
        maximum_persistent_bytes is not None and packet.component_bytes > maximum_persistent_bytes
    ):
        raise FeasibilityDeclined(dict(packet.diagnostics, feasible=False))
    # Verify every original word, not a tolerance, sampled row, or task output.
    actual = packet.materialize(attention, workspace_observer=workspace_observer)
    if not all(
        torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))
        for a, b in zip(original_pair, actual, strict=True)
    ):
        raise RuntimeError("Exact normalized-source control changed an original K/V bit")
    packet._check_attention(attention)
    return packet
