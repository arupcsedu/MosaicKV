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

    def private(self) -> tuple[Any, ...]:
        return (*super().private(), *(t for p in self.corrections for t in p.buffers))

    def materialize(self, attention: Any) -> tuple[Any, ...]:
        outputs = compact_replay(self, attention, self.reduced_precision)
        for patch, value in zip(self.corrections, outputs, strict=True):
            patch.apply(value)
        return outputs

    def fill(self, attention: Any, destinations: tuple[Any, ...], *, direct: bool) -> None:
        # A direct fused variant may be considered only after this reference passes.
        del direct
        outputs = self.materialize(attention)
        for value, target in zip(outputs, destinations, strict=True):
            target[..., : self.shape[-2], :].copy_(value)


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
    result = CompactFormat(
        kind,
        format.shape,
        format.pair,
        format.archive,
        format.slots,
        (),
        patches,
        reduced_precision,
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
