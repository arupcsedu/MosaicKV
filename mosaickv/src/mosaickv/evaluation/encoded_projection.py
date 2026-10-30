"""Experimental exact projection without a full decoded normalized-source buffer."""

from dataclasses import dataclass
from typing import Any

import torch

from mosaickv.evaluation.compact_storage import CompactFormat
from mosaickv.evaluation.dual_projection import DualFormat
from mosaickv.evaluation.exact_storage import SparseBitPatch


def checkpoint_bytes_bound(shape: tuple[int, ...], raw: bool) -> int:
    if (
        len(shape) != 4
        or shape[:2] != (1, 1)
        or any(type(d) is not int or d < 1 for d in shape)
        or shape[-1] % 128
        or type(raw) is not bool
    ):
        raise ValueError("positive MHA normalized-source shape and explicit raw flag required")
    return 0 if raw else ((shape[-2] * shape[-1] + 1023) // 1024) * 16


@dataclass(frozen=True)
class EncodedFormat(DualFormat):
    literal_checkpoints: Any = None

    def private(self) -> tuple[Any, ...]:
        return (*super().private(), self.literal_checkpoints)

    def fill(self, attention: Any, destinations: Any, *, direct: bool) -> None:
        from mosaickv.evaluation.dual_projection_kernel import apply_patch
        from mosaickv.evaluation.encoded_projection_kernel import project

        if self.archive is None or self.literal_checkpoints is None:
            raise ValueError("complete encoded source/checkpoints required")
        targets = tuple(t[..., : self.shape[-2], :] for t in destinations)
        with torch.inference_mode():
            project(
                self.archive.source,
                self.literal_checkpoints,
                attention,
                self.archive,
                self.output_slots,
                targets,
            )
            CompactFormat.overlay(self, targets)
            if self.patches:
                for patch, target in zip(self.patches, targets, strict=True):
                    apply_patch(patch, target)


def build(base: DualFormat, attention: Any) -> tuple[EncodedFormat, dict[str, Any]]:
    from mosaickv.evaluation.encoded_projection_kernel import checkpoints

    if base.archive is None:
        raise ValueError("original normalized source required")
    check = checkpoints(base.archive.source)
    result = EncodedFormat(
        "encoded_projection",
        base.shape,
        base.pair,
        base.archive,
        base.slots,
        (),
        base.output_slots,
        check,
    )
    original = base.materialize(attention)
    candidate = result.materialize(attention)
    patches = tuple(SparseBitPatch.build(a, b) for a, b in zip(original, candidate, strict=True))
    result = EncodedFormat(
        result.kind,
        result.shape,
        result.pair,
        result.archive,
        result.slots,
        patches,
        result.output_slots,
        check,
    )
    for _ in range(3):
        actual = result.materialize(attention)
        if not all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(original, actual, strict=True)
        ):
            raise RuntimeError("encoded projection changed original K/V bits")
    return result, dict(
        checkpoint_bytes=check.numel() * 2,
        correction_bytes=sum(t.numel() * t.element_size() for p in patches for t in p.tensors),
        total_component_bytes=result.component_bytes,
        original_KV_bitwise=True,
        repetitions=3,
        no_decoded_normalized_source_tensor=True,
    )
