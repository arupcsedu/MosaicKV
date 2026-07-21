"""Typed CPU residual payload metadata and original-position index."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any

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
class ResidualStorageReport:
    """Pinned CPU tier plus its payload and original-position indices."""

    tier: ResidualTier
    payloads: tuple[ResidualPayloadMetadata, ...]
    index: tuple[ResidualIndexEntry, ...]
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
    "ResidualStorageReport",
    "ResidualTransferBatch",
]
