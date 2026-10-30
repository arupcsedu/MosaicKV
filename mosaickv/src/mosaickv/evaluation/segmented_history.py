"""Exact history batching by immutable pointer tables, without source recopying."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch

from mosaickv.evaluation.consolidated_history import join_patches
from mosaickv.evaluation.encoded_projection import EncodedFormat
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.projection_cache import ScratchAudit, identity
from mosaickv.evaluation.source_history import SourceHistoryLayer


def validate_formats(formats: tuple[Any, ...], *, heads: int) -> None:
    if len(formats) < 2 or any(
        not isinstance(f, EncodedFormat)
        or f.archive is None
        or f.shape != (1, heads, 16, 128)
        or f.archive.source.shape != (1, 1, 16, heads * 128)
        or f.archive.source.shared_literal_window
        or f.packed_literal_counts
        or f.slots is not None
        or len(f.patches) != 2
        or f.archive.cos.shape != (1, 16, 128)
        or f.archive.sin.shape != (1, 16, 128)
        or any(
            t.dtype != torch.bfloat16 or not t.is_contiguous()
            for t in (f.archive.cos, f.archive.sin)
        )
        for f in formats
    ):
        raise ValueError("independent original 16-token MHA encoded-source packets required")
    for f in formats:
        expected = torch.arange(16, device=f.archive.union.device, dtype=torch.int32)
        if not torch.equal(f.archive.union, expected) or not torch.equal(
            f.output_slots, expected[None].expand(heads, -1)
        ):
            raise ValueError("complete original chronological storage maps required")


@dataclass(frozen=True)
class SegmentTables:
    count: int
    pointers: Any
    raw: Any
    patches: tuple[Any, ...]

    def tensors(self) -> tuple[Any, ...]:
        return self.pointers, self.raw, *(t for p in self.patches for t in p.tensors)


def make_tables(formats: tuple[Any, ...], *, heads: int) -> SegmentTables:
    validate_formats(formats, heads=heads)
    descriptors = []
    for f in formats:
        a, b = f.archive, f.archive.source
        sources = (b.buffers[0],) * 5 if b.raw else b.buffers
        descriptors.append([t.data_ptr() for t in (*sources, f.literal_checkpoints, a.cos, a.sin)])
    device = formats[0].archive.source.buffers[0].device
    return SegmentTables(
        len(formats),
        torch.tensor(descriptors, device=device, dtype=torch.int64),
        torch.tensor([f.archive.source.raw for f in formats], device=device, dtype=torch.int32),
        tuple(
            join_patches(
                tuple(f.patches[i] for f in formats),
                (16,) * len(formats),
                heads=heads,
                dimension=128,
            )
            for i in range(2)
        ),
    )


class SegmentedSourceLayer(SourceHistoryLayer):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.segment_tables: SegmentTables | None = None

    def owned_tensors(self) -> tuple[Any, ...]:
        return (
            *super().owned_tensors(),
            *(self.segment_tables.tensors() if self.segment_tables else ()),
        )

    def fill_history(self) -> None:
        if len(self.source_sealed) < 2:
            return super().fill_history()
        from mosaickv.evaluation.segmented_history_kernel import fill

        table = self.segment_tables
        if table is None or table.count != len(self.source_sealed):
            raise RuntimeError("history pointer table is stale or missing")
        count = table.count * 16
        targets = tuple(
            t[..., self.prompt_length : self.prompt_length + count, :] for t in self.leased
        )
        fill(table, self.attention, targets)
        for tail, target in zip(self.history, self.leased, strict=True):
            target[
                ..., self.prompt_length + count : self.prompt_length + count + self.tail_used, :
            ].copy_(tail[..., : self.tail_used, :])
        if count + self.tail_used != self.history_length:
            raise RuntimeError("segmented history lost or duplicated original positions")


def prepare_tables(cache: Any) -> None:
    if any(
        not isinstance(layer, SegmentedSourceLayer) or layer.leased or layer.source_pending
        for layer in cache.layers
    ):
        raise ValueError("released synchronized segmented source-history layers required")
    if len({len(layer.source_sealed) for layer in cache.layers}) != 1:
        raise ValueError("source memberships differ between layers")
    if len(cache.layers[0].source_sealed) < 2:
        return
    for layer in cache.layers:
        previous = layer.segment_tables
        excluded = {identity(t) for item in cache.layers for t in item.owned_tensors()}
        excluded.update(identity(t) for t in cache.shared_tensors)
        excluded.update(owner for item in cache.layers for owner in item.weights)
        mode = (
            ScratchAudit(excluded=excluded, observe=cache.observe)
            if cache.audit_workspace
            else None
        )
        with mode if mode is not None else nullcontext():
            newer = make_tables(tuple(layer.source_sealed), heads=layer.prompt_shape[1])
        if mode is not None:
            layer.workspace_operations += mode.operations
        # Original archives remain owned; descriptor tables never own/replace source data.
        owners = {identity(t) for t in layer.owned_tensors() if t.numel()}
        if any(t.numel() and identity(t) in owners for t in newer.tensors()):
            raise RuntimeError("new descriptor metadata aliases an existing mutable owner")
        cache.observe(newer.tensors())  # old/new table overlap, before commit
        before = backing_bytes(previous.tensors()) if previous else 0
        layer.segment_tables = newer
        layer.resident_bytes += backing_bytes(newer.tensors()) - before
        cache.observe(previous.tensors() if previous else ())
        cache.segment_table_builds = getattr(cache, "segment_table_builds", 0) + 1
        del previous
        cache.observe()
