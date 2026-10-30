"""Lossless numerical exceptions for compact BF16 projection replay.

Fixed two-bit signed-ULP codes (0, -1, +1, literal), never value quantization.
All bit patterns, including signed zeros and NaN payloads, are preserved.
"""

from dataclasses import dataclass
from typing import Any

import torch

BLOCK = 1024


@dataclass(frozen=True)
class ProjectionDelta:
    shape: tuple[int, ...]
    kind: str
    buffers: tuple[Any, ...]
    changed: int
    escaped: int

    @property
    def storage_bytes(self) -> int:
        from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

        return backing_bytes(self.buffers)

    @classmethod
    def build(cls, reference: Any, candidate: Any) -> "ProjectionDelta":
        if (
            reference.dtype != torch.bfloat16
            or candidate.dtype != reference.dtype
            or reference.shape != candidate.shape
            or reference.device != candidate.device
            or reference.numel() == 0
        ):
            raise ValueError("matching nonempty BF16 tensors required")
        original = reference.contiguous().view(torch.int16).flatten()
        other = candidate.contiguous().view(torch.int16).flatten()
        delta = original.to(torch.int32) - other.to(torch.int32)
        count, changed = original.numel(), int((delta != 0).sum())
        if not changed:
            return cls(tuple(reference.shape), "zero", (), 0, 0)
        escaped = (delta < -1) | (delta > 1)
        codes = torch.where(
            escaped, 3, torch.where(delta == -1, 1, torch.where(delta == 1, 2, 0))
        ).to(torch.int32)
        blocks = (count + BLOCK - 1) // BLOCK
        padded = torch.nn.functional.pad(codes, (0, blocks * BLOCK - count)).reshape(-1, 4)
        packed = (padded << (2 * torch.arange(4, device=codes.device))).sum(-1).to(torch.uint8)
        counts = torch.nn.functional.pad(escaped, (0, blocks * BLOCK - count))
        counts = counts.reshape(blocks, BLOCK).sum(-1, dtype=torch.int32)
        offsets = torch.nn.functional.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
        literals = original[escaped].clone()
        result = cls(
            tuple(reference.shape),
            "two_bit",
            (packed, offsets, literals),
            changed,
            literals.numel(),
        )
        if result.storage_bytes >= reference.numel() * reference.element_size():
            return cls(
                tuple(reference.shape),
                "raw",
                (reference.contiguous().clone(),),
                changed,
                literals.numel(),
            )
        return result

    def apply(self, destination: Any) -> None:
        if (
            tuple(destination.shape) != self.shape
            or destination.dtype != torch.bfloat16
            or destination.ndim != 4
            or self.shape[0] != 1
            or destination.stride(-1) != 1
            or destination.stride(-2) != self.shape[-1]
            or destination.stride(-3) < self.shape[-2] * self.shape[-1]
            or destination.stride(-3) % self.shape[-1]
            or any(t.device != destination.device for t in self.buffers)
        ):
            raise ValueError("compatible nonoverlapping BF16 BHSD correction destination required")
        with torch.inference_mode():
            if self.kind == "zero":
                return
            if self.kind == "raw":
                destination.copy_(self.buffers[0])
            elif destination.device.type == "cuda":
                from mosaickv.evaluation.projection_delta_kernel import apply

                apply(self, destination)
            else:
                packed, offsets, literals = self.buffers
                shifts = 2 * torch.arange(4, dtype=torch.int32)
                codes = ((packed.to(torch.int32)[:, None] >> shifts) & 3).flatten()
                codes = codes[: destination.numel()]
                escaped = codes == 3
                if int(offsets[-1]) != literals.numel() or int(escaped.sum()) != literals.numel():
                    raise ValueError("invalid correction literal accounting")
                words = destination.contiguous().view(torch.int16).flatten().to(torch.int32)
                words += torch.where(codes == 1, -1, torch.where(codes == 2, 1, 0))
                words[escaped] = literals.to(torch.int32)
                destination.copy_(words.to(torch.int16).view(torch.bfloat16).reshape(self.shape))


def gpu_controls() -> list[dict[str, Any]]:
    """All BF16 patterns, mixed ULP/escaped differences and strided sentinel writes."""
    words = torch.arange(-32768, 32768, device="cuda", dtype=torch.int32).to(torch.int16)
    original = words.view(torch.bfloat16).reshape(1, 4, 1024, 16)
    other_words = words.to(torch.int32) + (torch.arange(words.numel(), device="cuda") % 7) - 3
    candidate = other_words.to(torch.int16).view(torch.bfloat16).reshape_as(original)
    result = []
    for target in (original, candidate):
        patch = ProjectionDelta.build(original, target)
        backing = torch.full((1, 4, 1030, 16), -71, device="cuda", dtype=torch.bfloat16)
        window = backing[..., 3:1027, :]
        window.copy_(target)
        patch.apply(window)
        same = torch.equal(window.view(torch.int16), original.view(torch.int16))
        sentinel = bool(
            (backing[..., :3, :] == -71).all() and (backing[..., 1027:, :] == -71).all()
        )
        if not same or not sentinel:
            raise RuntimeError("projection-delta CUDA bit/sentinel failure")
        result.append(
            dict(
                kind=patch.kind,
                bitwise_equal=same,
                sentinels_unchanged=sentinel,
                actual_storage_bytes=patch.storage_bytes,
            )
        )
    return result
