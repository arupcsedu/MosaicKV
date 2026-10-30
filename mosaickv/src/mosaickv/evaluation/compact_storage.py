"""Experimental bit-exact compact replay; same original source, positions and weights."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast

import torch

from mosaickv.evaluation.exact_storage import ExactFormat, decoded
from mosaickv.evaluation.projection_delta import ProjectionDelta


@contextmanager
def projection_policy(reduced_precision: bool) -> Iterator[None]:
    """Isolated replay-only flag; original prefill and model decoding are untouched."""
    original = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced_precision
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original


def compact_replay(format: ExactFormat, attention: Any, reduced_precision: bool) -> tuple[Any, ...]:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    archive = format.archive
    if archive is None or archive.union.numel() == 0:
        raise ValueError("nonempty original source archive required")
    heads, kept = archive.selected.shape
    dimension = attention.head_dim
    if (
        archive.source.shape[-1] != heads * dimension
        or attention.k_proj.out_features != heads * dimension
        or attention.v_proj.out_features != heads * dimension
    ):
        raise NotImplementedError("compact replay requires full MHA")
    with torch.inference_mode(), projection_policy(reduced_precision):
        source = decoded(archive.source).squeeze(1)
        shape = (1, archive.union.numel(), heads, dimension)
        keys = attention.k_proj(source).view(shape).transpose(1, 2)
        values = attention.v_proj(source).view(shape).transpose(1, 2)
        cos = archive.cos.index_select(1, archive.union)
        sin = archive.sin.index_select(1, archive.union)
        unused, keys = cast(Any, apply_rotary_pos_emb)(keys[:, :1], keys, cos, sin)
        del unused, source, cos, sin
        local = torch.searchsorted(archive.union, archive.selected.contiguous())
        local.clamp_(max=archive.union.numel() - 1)
        present = archive.union[local] == archive.selected
        index = local[None, :, :, None].expand(1, heads, kept, dimension)
        pair = tuple(
            t.gather(2, index).masked_fill(~present[None, :, :, None], 0) for t in (keys, values)
        )
        # Low-degree head-token pairs retain their original K/V, not zero rows.
        format.overlay(pair)
        return pair


@dataclass(frozen=True)
class CompactFormat(ExactFormat):
    corrections: tuple[ProjectionDelta, ...] = ()
    reduced_precision: bool = True
    source_map: Any = None

    def private(self) -> tuple[Any, ...]:
        return (
            *super().private(),
            *(t for p in self.corrections for t in p.buffers),
            *((self.source_map,) if self.source_map is not None else ()),
        )

    def overlay(self, outputs: tuple[Any, ...]) -> None:
        if self.slots is None or self.slots.numel() == 0:
            return
        if self.slots.ndim != 2 or self.slots.shape[0] != 2:
            raise ValueError("compact format requires int32 head/position coordinates")
        for body, target in zip(self.pair, outputs, strict=True):
            if target.device.type == "cuda":
                from mosaickv.evaluation.compact_direct_kernel import scatter

                scatter(body, self.slots, target)
            else:
                with torch.inference_mode():
                    target[0, self.slots[0], self.slots[1], :] = decoded(body).reshape(
                        -1, self.shape[-1]
                    )

    def materialize(self, attention: Any) -> tuple[Any, ...]:
        outputs = compact_replay(self, attention, self.reduced_precision)
        for patch, value in zip(self.corrections, outputs, strict=True):
            patch.apply(value)
        return outputs

    def fill(self, attention: Any, destinations: tuple[Any, ...], *, direct: bool) -> None:
        if direct and destinations[0].device.type == "cuda":
            self.fill_direct(attention, destinations)
            return
        outputs = self.materialize(attention)
        for value, target in zip(outputs, destinations, strict=True):
            target[..., : self.shape[-2], :].copy_(value)

    def fill_direct(self, attention: Any, destinations: tuple[Any, ...]) -> None:
        from mosaickv.evaluation.compact_direct_kernel import gather

        archive = self.archive
        if archive is None or self.source_map is None:
            raise ValueError("compact direct fill requires original source map")
        heads, kept = archive.selected.shape
        targets = tuple(t[..., :kept, :] for t in destinations)
        with torch.inference_mode(), projection_policy(self.reduced_precision):
            source = decoded(archive.source).squeeze(1)
            shape = (1, archive.union.numel(), heads, attention.head_dim)
            key = attention.k_proj(source).view(shape).transpose(1, 2)
            gather(key, archive, self.source_map, targets[0], key=True)
            del key
            value = attention.v_proj(source).view(shape).transpose(1, 2)
            gather(value, archive, self.source_map, targets[1], key=False)
            del value, source
            self.overlay(targets)
            for patch, target in zip(self.corrections, targets, strict=True):
                patch.apply(target)


def build_compact(
    format: ExactFormat,
    attention: Any,
    references: tuple[Any, ...],
    *,
    reduced_precision: bool,
    repeats: int = 3,
) -> tuple[CompactFormat, dict[str, Any]]:
    if type(repeats) is not int or repeats < 3 or format.archive is None:
        raise ValueError(
            "source representation and three independent reconstruction checks required"
        )
    candidate = compact_replay(format, attention, reduced_precision)
    patches = tuple(ProjectionDelta.build(a, b) for a, b in zip(references, candidate, strict=True))
    kind = "compact_delta_token" if format.slots is not None else "compact_delta_source"
    coordinates = (
        torch.stack((format.slots // format.shape[-2], format.slots % format.shape[-2])).to(
            torch.int32
        )
        if format.slots is not None
        else None
    )
    mapping = torch.full(
        (format.archive.original_length,), -1, device=format.archive.union.device, dtype=torch.int32
    )
    mapping[format.archive.union] = torch.arange(
        format.archive.union.numel(), device=mapping.device, dtype=torch.int32
    )
    result = CompactFormat(
        kind,
        format.shape,
        format.pair,
        format.archive,
        coordinates,
        (),
        patches,
        reduced_precision,
        mapping,
    )
    for _ in range(repeats):
        actual = result.materialize(attention)
        if not all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(references, actual, strict=True)
        ):
            raise RuntimeError("fixed compact deltas failed original-KV bit equality")
    proof = dict(
        kind=kind,
        reduced_precision_projection_only=reduced_precision,
        changed_elements=[p.changed for p in patches],
        literal_elements=[p.escaped for p in patches],
        correction_kinds=[p.kind for p in patches],
        correction_storage_bytes=sum(p.storage_bytes for p in patches),
        total_component_bytes=result.component_bytes,
        source_tokens=format.archive.union.numel(),
        original_tokens=format.archive.original_length,
        repeated_original_KV_bitwise=True,
        repetitions=repeats,
    )
    return result, proof
