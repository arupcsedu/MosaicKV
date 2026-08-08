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
        position_weights: tuple[float, ...] | None = None,
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
        if position_weights is None:
            return phase_neutral.mean(dim=normalized_axis, keepdim=True)
        if len(position_weights) != int(key.shape[-2]):
            raise PrototypeKeyTransformError(
                "prototype key position weights do not match the cache block width"
            )
        if any(weight <= 0 for weight in position_weights):
            raise PrototypeKeyTransformError("prototype key position weights must be positive")
        import torch

        total = float(sum(position_weights))
        shape = [1] * int(phase_neutral.ndim)
        shape[normalized_axis] = len(position_weights)
        weight_tensor = torch.as_tensor(
            tuple(weight / total for weight in position_weights),
            dtype=phase_neutral.dtype,
            device=phase_neutral.device,
        ).reshape(shape)
        return torch.sum(
            phase_neutral * weight_tensor,
            dim=normalized_axis,
            keepdim=True,
        )

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

    def apply_anchor_phase_batch(
        self,
        phase_neutral_keys: Any,
        anchor_logical_positions: tuple[int, ...],
        reference: Any,
    ) -> Any:
        """Rotate a batch of phase-neutral means with one device launch.

        The leading dimension indexes prototypes; every remaining dimension is
        identical to one cache-key payload.  This is numerically the same
        operation as :meth:`apply_anchor_phase`, but avoids one small RoPE
        kernel per prototype on long multimodal prompts.
        """

        import torch

        if phase_neutral_keys.ndim != reference.ndim + 1:
            raise PrototypeKeyTransformError(
                "batched prototype keys require one leading prototype dimension"
            )
        if int(phase_neutral_keys.shape[0]) != len(anchor_logical_positions):
            raise PrototypeKeyTransformError(
                "batched prototype keys and anchor positions must align"
            )
        if int(phase_neutral_keys.shape[-2]) != 1:
            raise PrototypeKeyTransformError(
                "each batched prototype key must contain exactly one slot"
            )
        if not anchor_logical_positions:
            raise PrototypeKeyTransformError("batched prototype anchors cannot be empty")
        if min(anchor_logical_positions) < 0 or max(anchor_logical_positions) >= int(
            self.cos.shape[-2]
        ):
            raise PrototypeKeyTransformError(
                "prototype logical position lies outside the captured prefill RoPE phase"
            )
        indices = torch.as_tensor(
            anchor_logical_positions,
            dtype=torch.long,
            device=self.cos.device,
        )
        cos = (
            self.cos[0, 0]
            .index_select(0, indices)
            .to(
                device=phase_neutral_keys.device,
                dtype=torch.float32,
            )
        )
        sin = (
            self.sin[0, 0]
            .index_select(0, indices)
            .to(
                device=phase_neutral_keys.device,
                dtype=torch.float32,
            )
        )
        phase_shape = (
            (len(anchor_logical_positions),)
            + (1,) * (phase_neutral_keys.ndim - 2)
            + (int(cos.shape[-1]),)
        )
        cos = cos.reshape(phase_shape)
        sin = sin.reshape(phase_shape)
        rotated = phase_neutral_keys * cos + self._rotate_half(phase_neutral_keys) * sin
        return rotated.to(dtype=reference.dtype, device=reference.device)

    def merge(
        self,
        key_blocks: tuple[Any, ...],
        blocks: tuple[KVBlockDescriptor, ...],
        weights: tuple[float, ...],
        *,
        sequence_dimension: int,
        anchor_logical_position: int,
        position_weights: tuple[tuple[float, ...], ...] | None = None,
    ) -> PrototypeKeyMerge:
        """Return one cache-ready key without averaging incompatible phases."""

        if not key_blocks or len(key_blocks) != len(blocks) or len(blocks) != len(weights):
            raise PrototypeKeyTransformError(
                "prototype key tensors, descriptors, and weights must align"
            )
        if any(weight <= 0 for weight in weights):
            raise PrototypeKeyTransformError("prototype key weights must be positive")
        if position_weights is not None and len(position_weights) != len(key_blocks):
            raise PrototypeKeyTransformError(
                "prototype key position weights must align with source blocks"
            )
        import torch

        pooled: list[Any | None] = [None] * len(key_blocks)
        shape_groups: dict[tuple[int, ...], list[int]] = {}
        for index, key in enumerate(key_blocks):
            normalized_axis = sequence_dimension % key.ndim
            if normalized_axis != key.ndim - 2:
                raise PrototypeKeyTransformError(
                    "rotary prototype transform requires the cache sequence axis at -2"
                )
            if int(key.shape[-1]) != int(self.cos.shape[-1]):
                raise PrototypeKeyTransformError(
                    "cached key head dimension differs from the captured RoPE phase"
                )
            if int(key.shape[-2]) != len(blocks[index].original_logical_positions):
                raise PrototypeKeyTransformError(
                    "cached key sequence length differs from its block descriptor"
                )
            shape_groups.setdefault(tuple(int(size) for size in key.shape), []).append(index)
        for indices in shape_groups.values():
            references = tuple(key_blocks[index] for index in indices)
            stacked_keys = torch.stack(
                tuple(reference.detach().float() for reference in references), dim=0
            )
            positions = torch.as_tensor(
                tuple(blocks[index].original_logical_positions for index in indices),
                dtype=torch.long,
                device=self.cos.device,
            )
            if int(positions.min()) < 0 or int(positions.max()) >= int(self.cos.shape[-2]):
                raise PrototypeKeyTransformError(
                    "prototype logical position lies outside the captured prefill RoPE phase"
                )
            cos = self.cos[0, 0][positions].to(device=stacked_keys.device, dtype=torch.float32)
            sin = self.sin[0, 0][positions].to(device=stacked_keys.device, dtype=torch.float32)
            phase_shape = (len(indices),) + (1,) * (stacked_keys.ndim - 3) + tuple(cos.shape[-2:])
            cos = cos.reshape(phase_shape)
            sin = sin.reshape(phase_shape)
            denominator = (cos * cos + sin * sin).clamp_min(1e-12)
            phase_neutral = (
                stacked_keys * cos - self._rotate_half(stacked_keys) * sin
            ) / denominator
            if position_weights is None:
                pooled_batch = phase_neutral.mean(dim=-2, keepdim=True)
            else:
                local_weights = tuple(position_weights[index] for index in indices)
                width = int(stacked_keys.shape[-2])
                if any(len(row) != width for row in local_weights):
                    raise PrototypeKeyTransformError(
                        "prototype key position weights do not match a source block width"
                    )
                if any(weight <= 0 for row in local_weights for weight in row):
                    raise PrototypeKeyTransformError(
                        "prototype key position weights must be positive"
                    )
                normalized_rows = tuple(
                    tuple(weight / sum(row) for weight in row) for row in local_weights
                )
                weight_tensor = torch.as_tensor(
                    normalized_rows,
                    dtype=phase_neutral.dtype,
                    device=phase_neutral.device,
                ).reshape((len(indices),) + (1,) * (phase_neutral.ndim - 3) + (width, 1))
                pooled_batch = torch.sum(
                    phase_neutral * weight_tensor,
                    dim=-2,
                    keepdim=True,
                )
            for batch_index, source_index in enumerate(indices):
                pooled[source_index] = pooled_batch[batch_index]
        if any(value is None for value in pooled):  # pragma: no cover - grouping invariant
            raise RuntimeError("batched RoPE pooling left an uninitialized member")
        pooled_values = tuple(value for value in pooled if value is not None)
        total = float(sum(weights))
        normalized = tuple(weight / total for weight in weights)
        stacked = torch.stack(pooled_values, dim=0)
        weight_tensor = torch.as_tensor(
            normalized,
            dtype=stacked.dtype,
            device=stacked.device,
        ).reshape((-1,) + (1,) * (stacked.ndim - 1))
        average = torch.sum(stacked * weight_tensor, dim=0)
        prototype = self.apply_anchor_phase(
            average,
            anchor_logical_position,
            key_blocks[0],
        )
        return PrototypeKeyMerge(pooled_values, average, prototype)


__all__ = [
    "PrototypeKeyMerge",
    "PrototypeKeyTransformError",
    "RotaryPrototypeKeyTransform",
]
