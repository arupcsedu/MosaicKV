"""Typed CPU residual payload metadata and original-position index."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, overload

from mosaickv.cache_state import ResidualTier
from mosaickv.types import ResidualStorageDType


@dataclass(frozen=True, slots=True)
class ResidualPayloadMetadata:
    """Encoding and placement metadata for one source-block payload pair."""

    payload_index: int
    source_node_id: int
    prototype_id: int
    storage_dtype: ResidualStorageDType
    source_key_dtype: str
    source_value_dtype: str
    key_scale: float | None
    value_scale: float | None
    key_pinned: bool
    value_pinned: bool

    def __post_init__(self) -> None:
        if self.payload_index < 0 or self.source_node_id < 0 or self.prototype_id < 0:
            raise ValueError("residual payload identifiers must be nonnegative")
        if not self.source_key_dtype or not self.source_value_dtype:
            raise ValueError("residual source dtype labels must be non-empty")
        if self.storage_dtype is ResidualStorageDType.INT8:
            if self.key_scale is None or self.value_scale is None:
                raise ValueError("INT8 residual payloads require K and V scales")
        elif self.key_scale is not None or self.value_scale is not None:
            raise ValueError("only INT8 residual payloads use quantization scales")
        for scale in (self.key_scale, self.value_scale):
            if scale is not None and (not math.isfinite(scale) or scale <= 0):
                raise ValueError("residual quantization scales must be finite and positive")


@dataclass(frozen=True, slots=True)
class ResidualIndexEntry:
    """Lookup from original cache identity to encoded block payload offset."""

    layer: int
    kv_head: int
    prototype_id: int
    original_position: int
    physical_position: int
    payload_index: int
    block_offset: int

    def __post_init__(self) -> None:
        if any(
            value < 0
            for value in (
                self.layer,
                self.kv_head,
                self.prototype_id,
                self.original_position,
                self.physical_position,
                self.payload_index,
                self.block_offset,
            )
        ):
            raise ValueError("residual index values must be nonnegative")

    @property
    def identity(self) -> tuple[int, int, int, int]:
        """Layer, head, prototype, and original-position lookup key."""

        return (self.layer, self.kv_head, self.prototype_id, self.original_position)


@dataclass(frozen=True, slots=True)
class _ResidualIndexSpan:
    """Compact index payload for all positions in one residual source block."""

    layer: int
    kv_head: int
    prototype_id: int
    original_positions: tuple[int, ...]
    physical_positions: tuple[int, ...]
    payload_index: int

    @property
    def identity(self) -> tuple[int, int, int, int]:
        return (self.layer, self.kv_head, self.prototype_id, self.original_positions[0])


@dataclass(frozen=True, slots=True)
class ResidualPositionIndex(Sequence[ResidualIndexEntry]):
    """Lazy position index backed by one compact span per residual block.

    Long visual prompts contain hundreds of thousands of residual positions.
    Materializing a Python dataclass for every position added seconds to the
    online compression path and hundreds of MiB of transient host memory.  A
    block already stores two monotonic position tuples, so the index can retain
    those tuples and instantiate an entry only when a caller iterates or looks
    up a position.  The public sequence/lookup semantics remain unchanged.
    """

    spans: tuple[_ResidualIndexSpan, ...] = ()
    cumulative_lengths: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if len(self.cumulative_lengths) != len(self.spans) + 1:
            raise ValueError("residual compact-index offsets must align with spans")
        if not self.cumulative_lengths or self.cumulative_lengths[0] != 0:
            raise ValueError("residual compact-index offsets must start at zero")
        if any(
            left >= right
            for left, right in zip(
                self.cumulative_lengths,
                self.cumulative_lengths[1:],
                strict=False,
            )
        ):
            raise ValueError("residual compact-index offsets must be strictly increasing")
        identities = tuple(span.identity for span in self.spans)
        if identities != tuple(sorted(identities)) or len(set(identities)) != len(identities):
            raise ValueError("residual compact-index spans must be sorted and unique")
        for span in self.spans:
            if not span.original_positions or len(span.original_positions) != len(
                span.physical_positions
            ):
                raise ValueError("residual compact-index positions must be nonempty and aligned")
            if any(
                left >= right
                for left, right in zip(
                    span.original_positions,
                    span.original_positions[1:],
                    strict=False,
                )
            ):
                raise ValueError("residual compact-index logical positions must be monotonic")

    @classmethod
    def from_tier_metadata(
        cls,
        tier: ResidualTier,
        payloads: Sequence[ResidualPayloadMetadata],
    ) -> ResidualPositionIndex:
        if len(tier.source_blocks) != len(payloads):
            raise ValueError("residual tier and payload metadata must align")
        spans = tuple(
            sorted(
                (
                    _ResidualIndexSpan(
                        block.layer,
                        block.kv_head,
                        payload.prototype_id,
                        block.original_logical_positions,
                        block.physical_cache_indices,
                        payload.payload_index,
                    )
                    for block, payload in zip(
                        tier.source_blocks,
                        payloads,
                        strict=True,
                    )
                ),
                key=lambda span: span.identity,
            )
        )
        cumulative = [0]
        for span in spans:
            cumulative.append(cumulative[-1] + len(span.original_positions))
        return cls(spans, tuple(cumulative))

    def __len__(self) -> int:
        return self.cumulative_lengths[-1]

    @staticmethod
    def _entry(span: _ResidualIndexSpan, block_offset: int) -> ResidualIndexEntry:
        return ResidualIndexEntry(
            layer=span.layer,
            kv_head=span.kv_head,
            prototype_id=span.prototype_id,
            original_position=span.original_positions[block_offset],
            physical_position=span.physical_positions[block_offset],
            payload_index=span.payload_index,
            block_offset=block_offset,
        )

    def __iter__(self) -> Iterator[ResidualIndexEntry]:
        for span in self.spans:
            for offset in range(len(span.original_positions)):
                yield self._entry(span, offset)

    @overload
    def __getitem__(self, index: int, /) -> ResidualIndexEntry: ...

    @overload
    def __getitem__(self, index: slice, /) -> tuple[ResidualIndexEntry, ...]: ...

    def __getitem__(
        self, index: int | slice, /
    ) -> ResidualIndexEntry | tuple[ResidualIndexEntry, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        normalized = index + len(self) if index < 0 else index
        if normalized < 0 or normalized >= len(self):
            raise IndexError("residual position index is out of range")
        span_index = bisect_right(self.cumulative_lengths, normalized) - 1
        return self._entry(
            self.spans[span_index],
            normalized - self.cumulative_lengths[span_index],
        )

    def lookup(
        self,
        layer: int,
        kv_head: int,
        prototype_id: int,
        original_position: int,
    ) -> ResidualIndexEntry:
        prefix = (layer, kv_head, prototype_id)
        target = (*prefix, original_position)
        identities = tuple(span.identity for span in self.spans)
        offset = bisect_right(identities, target) - 1
        if offset < 0:
            raise KeyError(f"residual position does not exist: {target}")
        span = self.spans[offset]
        if (span.layer, span.kv_head, span.prototype_id) != prefix:
            raise KeyError(f"residual position does not exist: {target}")
        block_offset = bisect_left(span.original_positions, original_position)
        if (
            block_offset >= len(span.original_positions)
            or span.original_positions[block_offset] != original_position
        ):
            raise KeyError(f"residual position does not exist: {target}")
        return self._entry(span, block_offset)


@dataclass(frozen=True, slots=True)
class ResidualStorageReport:
    """Pinned CPU tier plus its payload and original-position indices."""

    tier: ResidualTier
    payloads: tuple[ResidualPayloadMetadata, ...]
    index: Sequence[ResidualIndexEntry]
    cpu_bytes: int

    def __post_init__(self) -> None:
        if len(self.tier.source_blocks) != len(self.payloads):
            raise ValueError("residual tier and payload metadata must align")
        if tuple(item.payload_index for item in self.payloads) != tuple(range(len(self.payloads))):
            raise ValueError("residual payload indices must be contiguous")
        if len({item.source_node_id for item in self.payloads}) != len(self.payloads):
            raise ValueError("residual source node IDs must be unique")
        if self.cpu_bytes != self.tier.active_bytes:
            raise ValueError("residual CPU bytes do not match underlying payload storage")
        expected_entries = sum(block.position_count for block in self.tier.source_blocks)
        if len(self.index) != expected_entries:
            raise ValueError("residual index must cover every original source position")
        if isinstance(self.index, ResidualPositionIndex):
            if len(self.index.spans) != len(self.payloads):
                raise ValueError("residual compact index must contain one span per payload")
            for span in self.index.spans:
                if span.payload_index >= len(self.payloads):
                    raise ValueError("residual index references an unknown payload")
                payload = self.payloads[span.payload_index]
                block = self.tier.source_blocks[span.payload_index]
                if payload.prototype_id != span.prototype_id:
                    raise ValueError("residual index prototype differs from payload metadata")
                if block.layer != span.layer or block.kv_head != span.kv_head:
                    raise ValueError("residual index layer/head differs from its source block")
                if block.original_logical_positions != span.original_positions:
                    raise ValueError("residual compact-index logical positions differ")
                if block.physical_cache_indices != span.physical_positions:
                    raise ValueError("residual compact-index physical positions differ")
            return
        identities = {entry.identity for entry in self.index}
        if len(identities) != len(self.index):
            raise ValueError("residual original-position index contains duplicates")
        if tuple(entry.identity for entry in self.index) != tuple(sorted(identities)):
            raise ValueError("residual original-position index must be sorted")
        for entry in self.index:
            if entry.payload_index >= len(self.payloads):
                raise ValueError("residual index references an unknown payload")
            payload = self.payloads[entry.payload_index]
            block = self.tier.source_blocks[entry.payload_index]
            if payload.prototype_id != entry.prototype_id:
                raise ValueError("residual index prototype differs from payload metadata")
            if block.layer != entry.layer or block.kv_head != entry.kv_head:
                raise ValueError("residual index layer/head differs from its source block")
            if entry.block_offset >= block.position_count:
                raise ValueError("residual block offset lies outside its source block")
            if block.original_logical_positions[entry.block_offset] != entry.original_position:
                raise ValueError("residual logical position differs from its source block")
            if block.physical_cache_indices[entry.block_offset] != entry.physical_position:
                raise ValueError("residual physical position differs from its source block")

    @property
    def all_payloads_pinned(self) -> bool:
        return all(item.key_pinned and item.value_pinned for item in self.payloads)

    def lookup(
        self,
        layer: int,
        kv_head: int,
        prototype_id: int,
        original_position: int,
    ) -> ResidualIndexEntry:
        """Find one residual position by its complete cache identity in O(log n)."""

        identity = (layer, kv_head, prototype_id, original_position)
        if isinstance(self.index, ResidualPositionIndex):
            return self.index.lookup(layer, kv_head, prototype_id, original_position)
        offset = bisect_left(self.index, identity, key=lambda entry: entry.identity)
        if offset >= len(self.index) or self.index[offset].identity != identity:
            raise KeyError(f"residual position does not exist: {identity}")
        return self.index[offset]


@dataclass(frozen=True, slots=True)
class ResidualTransferBatch:
    """Restored source payloads and measured host-to-device transfer metadata."""

    payload_indices: tuple[int, ...]
    key_blocks: tuple[Any, ...]
    value_blocks: tuple[Any, ...]
    transfer_time_ms: float
    asynchronous: bool

    def __post_init__(self) -> None:
        if self.payload_indices != tuple(sorted(set(self.payload_indices))):
            raise ValueError("residual transfer payload indices must be sorted and unique")
        if not (len(self.payload_indices) == len(self.key_blocks) == len(self.value_blocks)):
            raise ValueError("residual transfer payloads must align")
        if not math.isfinite(self.transfer_time_ms) or self.transfer_time_ms < 0:
            raise ValueError("residual transfer time must be finite and nonnegative")


__all__ = [
    "ResidualIndexEntry",
    "ResidualPayloadMetadata",
    "ResidualPositionIndex",
    "ResidualStorageReport",
    "ResidualTransferBatch",
]
