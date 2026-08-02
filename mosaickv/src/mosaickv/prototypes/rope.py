"""Audited RoPE phase transforms for safe prototype-key construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mosaickv.adapters.huggingface.types import CachedKeyState
from mosaickv.cache_state import KVBlockDescriptor


class PrototypeKeyTransformError(RuntimeError):
    """Raised when cached keys cannot be mapped through an exact phase transform."""


@dataclass(frozen=True, slots=True)
class PrototypeKeyMerge:
    """Pre-phase pooled members, their mean, and one anchor-phase cache key."""

    pooled_keys_fp32: tuple[Any, ...]
    average_key_fp32: Any
    prototype_key: Any


@dataclass(frozen=True, slots=True)
class RotaryPrototypeKeyTransform:
    """Invert source RoPE, merge in phase-neutral space, and rotate at an anchor.

    ``cos`` and ``sin`` contain the exact model-produced phase after any
    model-specific section selection (for Qwen, temporal/height/width M-RoPE),
    with shape ``[batch, 1, sequence, head_dim]``.  The inverse divides by
    ``cos**2 + sin**2`` so non-unit attention scaling is handled rather than
    assumed away.
    """

    cos: Any
    sin: Any
    provenance: str
    source_key_state: CachedKeyState = CachedKeyState.POST_ROPE

    def __post_init__(self) -> None:
        if not self.provenance.strip():
            raise ValueError("prototype key-transform provenance must be non-empty")
        if self.source_key_state is not CachedKeyState.POST_ROPE:
            raise ValueError("rotary prototype transforms require post-RoPE source keys")
        if not self.cos.__class__.__module__.startswith("torch"):
            raise TypeError("rotary prototype transforms currently require torch tensors")
        if tuple(self.cos.shape) != tuple(self.sin.shape) or self.cos.ndim != 4:
            raise ValueError("prototype RoPE cos/sin must share rank-four shapes")
        if int(self.cos.shape[0]) != 1 or int(self.cos.shape[1]) != 1:
            raise ValueError("prototype RoPE cos/sin require batch=head broadcast dimensions one")
        if int(self.cos.shape[-2]) < 1 or int(self.cos.shape[-1]) < 2:
            raise ValueError("prototype RoPE cos/sin have invalid sequence/head dimensions")
        if int(self.cos.shape[-1]) % 2:
            raise ValueError("prototype RoPE head dimension must be even")

    @staticmethod
    def _rotate_half(value: Any) -> Any:
        import torch

        half = int(value.shape[-1]) // 2
        return torch.cat((-value[..., half:], value[..., :half]), dim=-1)

    def _phase(self, logical_positions: tuple[int, ...], reference: Any) -> tuple[Any, Any]:
        import torch

        if not logical_positions:
            raise PrototypeKeyTransformError("prototype member has no logical positions")
        if min(logical_positions) < 0 or max(logical_positions) >= int(self.cos.shape[-2]):
            raise PrototypeKeyTransformError(
                "prototype logical position lies outside the captured prefill RoPE phase"
            )
        indices = torch.tensor(logical_positions, dtype=torch.long, device=self.cos.device)
        cos = torch.index_select(self.cos, -2, indices).to(
            device=reference.device, dtype=torch.float32
        )
        sin = torch.index_select(self.sin, -2, indices).to(
            device=reference.device, dtype=torch.float32
        )
        return cos, sin

    def phase_neutral_pool(
        self,
        key: Any,
        block: KVBlockDescriptor,
        *,
        sequence_dimension: int,
    ) -> Any:
        """Invert source RoPE and pool one source block in FP32."""

        normalized_axis = sequence_dimension % key.ndim
        if normalized_axis != key.ndim - 2:
            raise PrototypeKeyTransformError(
                "rotary prototype transform requires the cache sequence axis at -2"
            )
        if int(key.shape[-1]) != int(self.cos.shape[-1]):
            raise PrototypeKeyTransformError(
                "cached key head dimension differs from the captured RoPE phase"
            )
        if int(key.shape[-2]) != len(block.original_logical_positions):
            raise PrototypeKeyTransformError(
                "cached key sequence length differs from its block descriptor"
            )
        key_fp32 = key.detach().float()
        cos, sin = self._phase(block.original_logical_positions, key_fp32)
        denominator = (cos * cos + sin * sin).clamp_min(1e-12)
        phase_neutral = (key_fp32 * cos - self._rotate_half(key_fp32) * sin) / denominator
        return phase_neutral.mean(dim=normalized_axis, keepdim=True)

    def remove_anchor_phase(self, prototype_key: Any, anchor_logical_position: int) -> Any:
        """Return the FP32 phase-neutral mean stored by one prototype key."""

        key_fp32 = prototype_key.detach().float()
        if int(key_fp32.shape[-2]) != 1:
            raise PrototypeKeyTransformError("a prototype key must contain exactly one slot")
        cos, sin = self._phase((anchor_logical_position,), key_fp32)
        denominator = (cos * cos + sin * sin).clamp_min(1e-12)
        return (key_fp32 * cos - self._rotate_half(key_fp32) * sin) / denominator

    def apply_anchor_phase(
        self,
        phase_neutral_key: Any,
        anchor_logical_position: int,
        reference: Any,
    ) -> Any:
        """Rotate one FP32 phase-neutral mean into its cache anchor phase."""

        anchor_cos, anchor_sin = self._phase((anchor_logical_position,), phase_neutral_key)
        rotated = phase_neutral_key * anchor_cos + self._rotate_half(phase_neutral_key) * anchor_sin
        return rotated.to(dtype=reference.dtype, device=reference.device)

    def merge(
        self,
        key_blocks: tuple[Any, ...],
        blocks: tuple[KVBlockDescriptor, ...],
        weights: tuple[float, ...],
        *,
        sequence_dimension: int,
        anchor_logical_position: int,
    ) -> PrototypeKeyMerge:
        """Return one cache-ready key without averaging incompatible phases."""

        if not key_blocks or len(key_blocks) != len(blocks) or len(blocks) != len(weights):
            raise PrototypeKeyTransformError(
                "prototype key tensors, descriptors, and weights must align"
            )
        if any(weight <= 0 for weight in weights):
            raise PrototypeKeyTransformError("prototype key weights must be positive")
        pooled: list[Any] = []
        for key, block in zip(key_blocks, blocks, strict=True):
            pooled.append(
                self.phase_neutral_pool(
                    key,
                    block,
                    sequence_dimension=sequence_dimension,
                )
            )
        total = float(sum(weights))
        normalized = tuple(weight / total for weight in weights)
        average = pooled[0] * normalized[0]
        for value, weight in zip(pooled[1:], normalized[1:], strict=True):
            average = average + value * weight
        prototype = self.apply_anchor_phase(
            average,
            anchor_logical_position,
            key_blocks[0],
        )
        return PrototypeKeyMerge(tuple(pooled), average, prototype)


__all__ = [
    "PrototypeKeyMerge",
    "PrototypeKeyTransformError",
    "RotaryPrototypeKeyTransform",
]
