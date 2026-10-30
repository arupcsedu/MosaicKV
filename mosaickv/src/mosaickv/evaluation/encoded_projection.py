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
    packed_literal_counts: bool = False

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
                packed_counts=self.packed_literal_counts,
            )
            CompactFormat.overlay(self, targets)
            if self.patches:
                for patch, target in zip(self.patches, targets, strict=True):
                    apply_patch(patch, target)


def build(
    base: DualFormat, attention: Any, *, packed_literal_counts: bool = False
) -> tuple[EncodedFormat, dict[str, Any]]:
    from mosaickv.evaluation.encoded_projection_kernel import checkpoints

    if base.archive is None:
        raise ValueError("original normalized source required")
    check = checkpoints(base.archive.source, packed_counts=packed_literal_counts)
    result = EncodedFormat(
        "encoded_projection",
        base.shape,
        base.pair,
        base.archive,
        base.slots,
        (),
        base.output_slots,
        check,
        packed_literal_counts,
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
        packed_literal_counts,
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
        packed_literal_counts=packed_literal_counts,
    )


def checkpoint_words_reference(half_counts: tuple[int, ...]) -> tuple[int, ...]:
    """CPU specification for the unchanged sixteen-byte checkpoint boundary."""
    if len(half_counts) != 16 or any(type(c) is not int or not 0 <= c <= 64 for c in half_counts):
        raise ValueError("sixteen literal counts in [0,64] required")
    flags = sum(int(half_counts[2 * group] == 64) << group for group in range(8))
    return tuple(
        (flags if group == 0 else sum(half_counts[: 2 * group]))
        | ((half_counts[2 * group] & 63) << 10)
        for group in range(8)
    )


def checkpoint_offsets_reference(words: tuple[int, ...]) -> tuple[int, ...]:
    """Literal offsets at each sixty-four-value boundary inside a block."""
    if len(words) != 8 or any(type(w) is not int or not 0 <= w <= 65535 for w in words):
        raise ValueError("eight unsigned sixteen-bit checkpoint words required")
    flags = words[0] & 255
    offsets: list[int] = []
    for group, word in enumerate(words):
        prefix = 0 if group == 0 else word & 1023
        first_count = ((word >> 10) & 63) + (((flags >> group) & 1) << 6)
        offsets.extend((prefix, prefix + first_count))
    return tuple(offsets)
