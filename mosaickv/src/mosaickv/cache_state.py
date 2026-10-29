"""Backend-independent MosaicKV cache state, blockization, and invariants.

This module intentionally uses a small tensor protocol instead of importing
PyTorch.  NumPy arrays support CPU property tests, while torch tensors support
the same indexing, cloning, assignment, shape, dtype, and device operations in
the Hugging Face integration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Any, Self

from mosaickv.execution_audit import audit_operation


class Modality(StrEnum):
    """Logical token modalities represented in the language sequence."""

    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"


@dataclass(frozen=True, slots=True)
class ModalitySpan:
    """Half-open logical range with optional multimodal source metadata."""

    start: int
    end: int
    modality: Modality
    image_index: int | None = None
    frame_index: int | None = None
    page_index: int | None = None
    region: tuple[float, float, float, float] | None = None
    grid_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError(f"invalid modality span [{self.start}, {self.end})")
        for name in ("image_index", "frame_index", "page_index"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be nonnegative when provided")
        if self.region is not None:
            if len(self.region) != 4 or any(not math.isfinite(value) for value in self.region):
                raise ValueError("region must contain four finite coordinates")
            left, top, right, bottom = self.region
            if right < left or bottom < top:
                raise ValueError("region right/bottom cannot precede left/top")
        if self.grid_shape is not None:
            if len(self.grid_shape) != 2 or any(value <= 0 for value in self.grid_shape):
                raise ValueError("grid_shape must contain positive (rows, columns)")
            rows, columns = self.grid_shape
            if self.end - self.start != rows * columns:
                raise ValueError(
                    "a grid-backed modality span must contain exactly rows*columns tokens"
                )

    def contains(self, logical_position: int) -> bool:
        return self.start <= logical_position < self.end


@dataclass(frozen=True, slots=True)
class MediaMetadata:
    """Media provenance attached to one logical cache position."""

    logical_position: int
    image_index: int | None = None
    frame_index: int | None = None
    page_index: int | None = None
    region: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.logical_position < 0:
            raise ValueError("media metadata logical_position must be nonnegative")
        # Reuse span validation for index and region checks.
        ModalitySpan(
            self.logical_position,
            self.logical_position + 1,
            Modality.TEXT,
            self.image_index,
            self.frame_index,
            self.page_index,
            self.region,
            None,
        )


@dataclass(frozen=True, slots=True)
class LogicalPositionMap:
    """Map active physical cache slots back to original logical positions."""

    physical_to_logical: tuple[int, ...]
    original_logical_sequence_length: int
    next_decode_position: int

    def __post_init__(self) -> None:
        if not self.physical_to_logical:
            raise ValueError("logical position map cannot be empty")
        if self.original_logical_sequence_length <= 0:
            raise ValueError("original_logical_sequence_length must be positive")
        if self.next_decode_position < self.original_logical_sequence_length:
            raise ValueError("next_decode_position cannot precede the original sequence")
        if any(position < 0 for position in self.physical_to_logical):
            raise ValueError("logical positions must be nonnegative")
        if any(
            current >= following
            for current, following in zip(
                self.physical_to_logical, self.physical_to_logical[1:], strict=False
            )
        ):
            raise ValueError("physical-to-logical positions must be strictly increasing")
        if self.physical_to_logical[-1] >= self.original_logical_sequence_length:
            raise ValueError("mapped logical position lies outside the original sequence")

    @classmethod
    def identity(cls, sequence_length: int, *, next_decode_position: int | None = None) -> Self:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        return cls(
            tuple(range(sequence_length)),
            sequence_length,
            sequence_length if next_decode_position is None else next_decode_position,
        )

    @property
    def active_sequence_length(self) -> int:
        return len(self.physical_to_logical)

    def logical_for_physical(self, physical_index: int) -> int:
        try:
            return self.physical_to_logical[physical_index]
        except IndexError as error:
            raise IndexError(f"physical cache index out of range: {physical_index}") from error

    def gather(self, physical_indices: Sequence[int]) -> tuple[int, ...]:
        positions = tuple(int(index) for index in physical_indices)
        if any(index < 0 or index >= self.active_sequence_length for index in positions):
            raise IndexError("selected physical cache position is out of range")
        if len(set(positions)) != len(positions):
            raise ValueError("selected physical cache positions must be unique")
        return tuple(self.physical_to_logical[index] for index in positions)


@dataclass(frozen=True, slots=True)
class KVBlockDescriptor:
    """Immutable source membership and provenance for one KV-head block."""

    layer: int
    kv_head: int
    modality: Modality
    physical_cache_indices: tuple[int, ...]
    original_logical_positions: tuple[int, ...]
    token_ids: tuple[int, ...] | None
    media_metadata: tuple[MediaMetadata, ...]
    key_dtype: str
    value_dtype: str
    key_device: str
    value_device: str
    byte_size: int
    mandatory: bool

    def __post_init__(self) -> None:
        if self.layer < 0 or self.kv_head < 0:
            raise ValueError("block layer and KV head must be nonnegative")
        if not self.physical_cache_indices:
            raise ValueError("block must contain at least one cache position")
        count = len(self.physical_cache_indices)
        if len(self.original_logical_positions) != count:
            raise ValueError("physical and logical block positions must have equal lengths")
        if self.token_ids is not None and len(self.token_ids) != count:
            raise ValueError("block token_ids must align with cache positions")
        if len(self.media_metadata) != count:
            raise ValueError("block media_metadata must align with cache positions")
        if any(
            current >= following
            for current, following in zip(
                self.physical_cache_indices, self.physical_cache_indices[1:], strict=False
            )
        ):
            raise ValueError("physical cache positions must be strictly increasing within a block")
        if any(
            current >= following
            for current, following in zip(
                self.original_logical_positions,
                self.original_logical_positions[1:],
                strict=False,
            )
        ):
            raise ValueError("logical positions must remain strictly monotonic within a block")
        if tuple(item.logical_position for item in self.media_metadata) != (
            self.original_logical_positions
        ):
            raise ValueError("media metadata logical positions do not align with the block")
        metadata_sources = {
            (item.image_index, item.frame_index, item.page_index, item.region)
            for item in self.media_metadata
        }
        if len(metadata_sources) != 1:
            raise ValueError("a block cannot combine different media sources or regions")
        for name in ("key_dtype", "value_dtype", "key_device", "value_device"):
            if not getattr(self, name).strip():
                raise ValueError(f"block {name} must be non-empty")
        if self.byte_size <= 0:
            raise ValueError("block byte_size must be positive")

    @classmethod
    def _from_validated(
        cls,
        *,
        layer: int,
        kv_head: int,
        modality: Modality,
        physical_cache_indices: tuple[int, ...],
        original_logical_positions: tuple[int, ...],
        token_ids: tuple[int, ...] | None,
        media_metadata: tuple[MediaMetadata, ...],
        key_dtype: str,
        value_dtype: str,
        key_device: str,
        value_device: str,
        byte_size: int,
        mandatory: bool,
    ) -> Self:
        """Build one block from the already-validated blockization template."""

        instance = object.__new__(cls)
        object.__setattr__(instance, "layer", layer)
        object.__setattr__(instance, "kv_head", kv_head)
        object.__setattr__(instance, "modality", modality)
        object.__setattr__(instance, "physical_cache_indices", physical_cache_indices)
        object.__setattr__(instance, "original_logical_positions", original_logical_positions)
        object.__setattr__(instance, "token_ids", token_ids)
        object.__setattr__(instance, "media_metadata", media_metadata)
        object.__setattr__(instance, "key_dtype", key_dtype)
        object.__setattr__(instance, "value_dtype", value_dtype)
        object.__setattr__(instance, "key_device", key_device)
        object.__setattr__(instance, "value_device", value_device)
        object.__setattr__(instance, "byte_size", byte_size)
        object.__setattr__(instance, "mandatory", mandatory)
        return instance

    @property
    def position_count(self) -> int:
        return len(self.physical_cache_indices)

    @property
    def source_memberships(self) -> frozenset[tuple[int, int, int]]:
        return frozenset(
            (self.layer, self.kv_head, position) for position in self.physical_cache_indices
        )

    @property
    def image_index(self) -> int | None:
        """Shared image index for this block, when one is present."""

        return self.media_metadata[0].image_index

    @property
    def frame_index(self) -> int | None:
        """Shared video-frame index for this block, when one is present."""

        return self.media_metadata[0].frame_index

    @property
    def page_index(self) -> int | None:
        """Shared document-page index for this block, when one is present."""

        return self.media_metadata[0].page_index

    @property
    def region(self) -> tuple[float, float, float, float] | None:
        """Shared source-region coordinates for this block, when present."""

        return self.media_metadata[0].region

    @property
    def non_compressible(self) -> bool:
        """Alias spelling out the meaning of the mandatory flag."""

        return self.mandatory


def _shape(tensor: Any) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise TypeError(f"object has no tensor shape: {type(tensor).__qualname__}")
    return tuple(int(value) for value in shape)


def _ndim(tensor: Any) -> int:
    return len(_shape(tensor))


def _normalize_axis(axis: int, rank: int, name: str) -> int:
    normalized = axis if axis >= 0 else rank + axis
    if normalized < 0 or normalized >= rank:
        raise ValueError(f"{name} axis {axis} is invalid for rank {rank}")
    return normalized


def _tensor_numel(tensor: Any) -> int:
    numel = getattr(tensor, "numel", None)
    if numel is not None:
        return int(numel())
    size = getattr(tensor, "size", None)
    if isinstance(size, int):
        return size
    result = 1
    for dimension in _shape(tensor):
        result *= dimension
    return result


def _tensor_element_size(tensor: Any) -> int:
    element_size = getattr(tensor, "element_size", None)
    if element_size is not None:
        return int(element_size())
    dtype = getattr(tensor, "dtype", None)
    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is None:
        raise TypeError(f"cannot determine tensor element size: {type(tensor).__qualname__}")
    return int(itemsize)


def tensor_storage_bytes(tensor: Any) -> int:
    """Return the tensor payload boundary used by MosaicKV accounting."""

    value = _tensor_numel(tensor) * _tensor_element_size(tensor)
    if value < 0:
        raise RuntimeError("tensor storage byte count cannot be negative")
    return value


def _dtype_name(tensor: Any) -> str:
    return str(getattr(tensor, "dtype", "unknown"))


def _device_name(tensor: Any) -> str:
    return str(getattr(tensor, "device", "cpu"))


def _clone_tensor(tensor: Any) -> Any:
    detached = getattr(tensor, "detach", None)
    if detached is not None:
        tensor = detached()
    clone = getattr(tensor, "clone", None)
    if clone is not None:
        return clone()
    copy = getattr(tensor, "copy", None)
    if copy is not None:
        return copy()
    raise TypeError(f"tensor cannot be cloned: {type(tensor).__qualname__}")


def _empty_like(tensor: Any) -> Any:
    creator = getattr(tensor, "new_empty", None)
    if creator is not None:
        return creator(_shape(tensor))
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - NumPy is a core dependency
        raise RuntimeError("NumPy is required for non-torch cache reinjection") from error
    return np.empty_like(tensor)


def _tensor_equal(first: Any, second: Any) -> bool:
    if _shape(first) != _shape(second) or _dtype_name(first) != _dtype_name(second):
        return False
    tensor_equal = getattr(first, "equal", None)
    if tensor_equal is not None:
        return bool(tensor_equal(second))
    try:
        import numpy as np

        first_array = first.detach().cpu().numpy() if hasattr(first, "detach") else first
        second_array = second.detach().cpu().numpy() if hasattr(second, "detach") else second
        return bool(np.array_equal(first_array, second_array, equal_nan=True))
    except (AttributeError, TypeError, ValueError):
        equal = first == second
        all_method = getattr(equal, "all", None)
        return bool(all_method() if all_method is not None else equal)


def _tensor_index(
    tensor: Any,
    *,
    head_axis: int,
    head: int,
    sequence_axis: int,
    positions: Sequence[int],
) -> tuple[Any, ...]:
    index: list[Any] = [slice(None)] * _ndim(tensor)
    index[head_axis] = slice(head, head + 1)
    index[sequence_axis] = list(positions)
    return tuple(index)


def _gather_tensor(
    tensor: Any,
    *,
    head_axis: int,
    head: int,
    sequence_axis: int,
    positions: Sequence[int],
) -> Any:
    index = _tensor_index(
        tensor,
        head_axis=head_axis,
        head=head,
        sequence_axis=sequence_axis,
        positions=positions,
    )
    return _clone_tensor(tensor[index])


def _gather_torch_heads_by_count(
    tensor: Any,
    *,
    head_axis: int,
    sequence_axis: int,
    positions_by_head: Mapping[int, Sequence[int]],
) -> dict[int, Any] | None:
    """Gather equal-length KV-head rows together without changing their order.

    ``torch.gather`` can copy many heads in one device launch. Heads with
    different selected lengths are grouped separately so the result contains
    no padding and retains the exact per-head storage/accounting contract.
    Non-torch tensors return ``None`` and use the portable reference path.
    """

    if not tensor.__class__.__module__.startswith("torch"):
        return None
    import torch

    if not isinstance(tensor, torch.Tensor):
        return None
    rank = int(tensor.ndim)
    normalized_head = _normalize_axis(head_axis, rank, "head")
    normalized_sequence = _normalize_axis(sequence_axis, rank, "sequence")
    moved = tensor.movedim((normalized_head, normalized_sequence), (0, 1))
    grouped_heads: dict[int, list[int]] = {}
    for head, positions in positions_by_head.items():
        if not positions:
            raise ValueError("batched head gather requires at least one position")
        grouped_heads.setdefault(len(positions), []).append(head)

    gathered_by_head: dict[int, Any] = {}
    for position_count, raw_heads in sorted(grouped_heads.items()):
        heads = tuple(sorted(raw_heads))
        head_indices = torch.tensor(heads, dtype=torch.long, device=tensor.device)
        selected_heads = moved.index_select(0, head_indices)
        position_rows = torch.tensor(
            tuple(tuple(int(position) for position in positions_by_head[head]) for head in heads),
            dtype=torch.long,
            device=tensor.device,
        )
        expanded_positions = position_rows.reshape(
            len(heads), position_count, *([1] * (rank - 2))
        ).expand(len(heads), position_count, *selected_heads.shape[2:])
        gathered = torch.gather(selected_heads, 1, expanded_positions)
        for local_index, head in enumerate(heads):
            canonical = gathered[local_index : local_index + 1]
            gathered_by_head[head] = canonical.movedim(
                (0, 1),
                (normalized_head, normalized_sequence),
            )
    return gathered_by_head


@dataclass(frozen=True, slots=True)
class KVLayerStorage:
    """One layer's K/V tensors plus explicit head and sequence axes."""

    key: Any
    value: Any
    key_head_dimension: int
    value_head_dimension: int
    key_sequence_dimension: int
    value_sequence_dimension: int

    def __post_init__(self) -> None:
        key_shape = _shape(self.key)
        value_shape = _shape(self.value)
        key_head = _normalize_axis(self.key_head_dimension, len(key_shape), "key head")
        value_head = _normalize_axis(self.value_head_dimension, len(value_shape), "value head")
        key_sequence = _normalize_axis(self.key_sequence_dimension, len(key_shape), "key sequence")
        value_sequence = _normalize_axis(
            self.value_sequence_dimension, len(value_shape), "value sequence"
        )
        if key_head == key_sequence or value_head == value_sequence:
            raise ValueError("head and sequence axes must be different")
        if key_shape[key_head] != value_shape[value_head]:
            raise ValueError("K and V tensors must have the same KV-head count")
        if key_shape[key_sequence] != value_shape[value_sequence]:
            raise ValueError("K and V tensors must have the same sequence length")

    @property
    def kv_heads(self) -> int:
        return _shape(self.key)[self.key_head_dimension]

    @property
    def sequence_length(self) -> int:
        return _shape(self.key)[self.key_sequence_dimension]

    @property
    def byte_size(self) -> int:
        return tensor_storage_bytes(self.key) + tensor_storage_bytes(self.value)


def _memberships(blocks: Sequence[KVBlockDescriptor]) -> frozenset[tuple[int, int, int]]:
    memberships: set[tuple[int, int, int]] = set()
    for block in blocks:
        for position in block.physical_cache_indices:
            member = (block.layer, block.kv_head, position)
            if member in memberships:
                raise ValueError(f"duplicate source membership within tier: {member}")
            memberships.add(member)
    return frozenset(memberships)


@dataclass(frozen=True, slots=True)
class _MembershipIndex:
    """Exact run-length index for block source memberships.

    Normal cache blocks contain consecutive positions.  Keeping those positions
    as merged inclusive intervals avoids allocating millions of redundant
    ``(layer, head, position)`` tuples during every tier invariant check.  A
    sparse descriptor is represented by multiple one-position runs, so this is
    an exact representation rather than an interval approximation.
    """

    ranges_by_identity: Mapping[tuple[int, int], tuple[tuple[int, int], ...]]
    count: int

    @classmethod
    def from_blocks(cls, blocks: Sequence[KVBlockDescriptor]) -> Self:
        pending: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for block in blocks:
            identity = (block.layer, block.kv_head)
            runs = pending.setdefault(identity, [])
            start = previous = block.physical_cache_indices[0]
            for position in block.physical_cache_indices[1:]:
                if position == previous + 1:
                    previous = position
                    continue
                runs.append((start, previous))
                start = previous = position
            runs.append((start, previous))

        normalized: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
        count = 0
        for identity, raw_ranges in pending.items():
            merged: list[tuple[int, int]] = []
            for start, end in sorted(raw_ranges):
                if merged and start <= merged[-1][1]:
                    overlap = (identity[0], identity[1], start)
                    raise ValueError(f"duplicate source membership within tier: {overlap}")
                if merged and start == merged[-1][1] + 1:
                    merged[-1] = (merged[-1][0], end)
                else:
                    merged.append((start, end))
                count += end - start + 1
            normalized[identity] = tuple(merged)
        return cls(normalized, count)

    def memberships(self) -> frozenset[tuple[int, int, int]]:
        return frozenset(
            (layer, head, position)
            for (layer, head), ranges in self.ranges_by_identity.items()
            for start, end in ranges
            for position in range(start, end + 1)
        )

    def is_subset_of(self, other: _MembershipIndex) -> bool:
        for identity, ranges in self.ranges_by_identity.items():
            containers = other.ranges_by_identity.get(identity, ())
            container_index = 0
            for start, end in ranges:
                while (
                    container_index < len(containers)
                    and containers[container_index][1] < start
                ):
                    container_index += 1
                if container_index >= len(containers):
                    return False
                container_start, container_end = containers[container_index]
                if start < container_start or end > container_end:
                    return False
        return True

    def first_overlap(self, other: _MembershipIndex) -> tuple[int, int, int] | None:
        for identity in self.ranges_by_identity.keys() & other.ranges_by_identity.keys():
            left = self.ranges_by_identity[identity]
            right = other.ranges_by_identity[identity]
            left_index = right_index = 0
            while left_index < len(left) and right_index < len(right):
                left_start, left_end = left[left_index]
                right_start, right_end = right[right_index]
                start = max(left_start, right_start)
                if start <= min(left_end, right_end):
                    return identity[0], identity[1], start
                if left_end < right_end:
                    left_index += 1
                else:
                    right_index += 1
        return None

    def same_memberships(self, other: _MembershipIndex) -> bool:
        return self.ranges_by_identity == other.ranges_by_identity


def _ensure_unique_tensors(tensors: Sequence[Any], name: str) -> None:
    identities = [id(tensor) for tensor in tensors]
    if len(identities) != len(set(identities)):
        raise ValueError(f"{name} cannot count the same tensor object more than once")


@dataclass(frozen=True, slots=True)
class ExactTier:
    """Lossless selected K/V blocks copied from the source cache."""

    blocks: tuple[KVBlockDescriptor, ...] = ()
    key_blocks: tuple[Any, ...] = ()
    value_blocks: tuple[Any, ...] = ()
    _membership_index: _MembershipIndex = field(init=False, repr=False, compare=False)
    _source_memberships: frozenset[tuple[int, int, int]] | None = field(
        init=False, repr=False, compare=False, default=None
    )

    @audit_operation("exact.tier_invariants")
    def __post_init__(self) -> None:
        if len(self.blocks) != len(self.key_blocks) or len(self.blocks) != len(self.value_blocks):
            raise ValueError("exact tier blocks and K/V payloads must align")
        object.__setattr__(self, "_membership_index", _MembershipIndex.from_blocks(self.blocks))
        _ensure_unique_tensors((*self.key_blocks, *self.value_blocks), "exact tier")
        for index, (block, key, value) in enumerate(
            zip(self.blocks, self.key_blocks, self.value_blocks, strict=True)
        ):
            payload_bytes = tensor_storage_bytes(key) + tensor_storage_bytes(value)
            if payload_bytes != block.byte_size:
                raise ValueError(
                    f"exact block {index} byte accounting mismatch: "
                    f"descriptor={block.byte_size}, tensors={payload_bytes}"
                )
            observed = (
                _dtype_name(key),
                _dtype_name(value),
                _device_name(key),
                _device_name(value),
            )
            declared = (
                block.key_dtype,
                block.value_dtype,
                block.key_device,
                block.value_device,
            )
            if observed != declared:
                raise ValueError(f"exact block {index} dtype/device metadata mismatch")

    @property
    def source_memberships(self) -> frozenset[tuple[int, int, int]]:
        cached = self._source_memberships
        if cached is None:
            cached = self._membership_index.memberships()
            object.__setattr__(self, "_source_memberships", cached)
        return cached

    @property
    def source_membership_count(self) -> int:
        return self._membership_index.count

    @property
    def active_bytes(self) -> int:
        return sum(tensor_storage_bytes(item) for item in (*self.key_blocks, *self.value_blocks))

    def selected_positions(self, layer: int, kv_head: int) -> tuple[int, ...]:
        return tuple(
            position
            for block in self.blocks
            if block.layer == layer and block.kv_head == kv_head
            for position in block.physical_cache_indices
        )

    def selected_logical_positions(self, layer: int, kv_head: int) -> tuple[int, ...]:
        return tuple(
            position
            for block in self.blocks
            if block.layer == layer and block.kv_head == kv_head
            for position in block.original_logical_positions
        )


@dataclass(frozen=True, slots=True)
class PrototypeTier:
    """Active prototype payloads and the source blocks represented by them."""

    source_blocks: tuple[KVBlockDescriptor, ...] = ()
    prototype_keys: tuple[Any, ...] = ()
    prototype_values: tuple[Any, ...] = ()
    assignments: tuple[int, ...] = ()
    _membership_index: _MembershipIndex = field(init=False, repr=False, compare=False)
    _source_memberships: frozenset[tuple[int, int, int]] | None = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_membership_index",
            _MembershipIndex.from_blocks(self.source_blocks),
        )
        if len(self.prototype_keys) != len(self.prototype_values):
            raise ValueError("prototype K/V payload counts must match")
        if len(self.assignments) != len(self.source_blocks):
            raise ValueError("prototype assignments must align with source blocks")
        if self.source_blocks and not self.prototype_keys:
            raise ValueError("prototype source blocks require at least one prototype payload")
        if self.prototype_keys and not self.source_blocks:
            raise ValueError("prototype payloads require at least one source block")
        if any(index < 0 or index >= len(self.prototype_keys) for index in self.assignments):
            raise ValueError("prototype assignment index is out of range")
        if self.prototype_keys and set(self.assignments) != set(range(len(self.prototype_keys))):
            raise ValueError("every prototype payload must have at least one source assignment")
        _ensure_unique_tensors((*self.prototype_keys, *self.prototype_values), "prototype tier")

    @property
    def source_memberships(self) -> frozenset[tuple[int, int, int]]:
        cached = self._source_memberships
        if cached is None:
            cached = self._membership_index.memberships()
            object.__setattr__(self, "_source_memberships", cached)
        return cached

    @property
    def source_membership_count(self) -> int:
        return self._membership_index.count

    @property
    def active_bytes(self) -> int:
        return sum(
            tensor_storage_bytes(item) for item in (*self.prototype_keys, *self.prototype_values)
        )


@dataclass(frozen=True, slots=True)
class ResidualTier:
    """CPU residual payloads indexed by their original source blocks."""

    source_blocks: tuple[KVBlockDescriptor, ...] = ()
    key_residuals: tuple[Any, ...] = ()
    value_residuals: tuple[Any, ...] = ()
    _membership_index: _MembershipIndex = field(init=False, repr=False, compare=False)
    _source_memberships: frozenset[tuple[int, int, int]] | None = field(
        init=False, repr=False, compare=False, default=None
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_membership_index",
            _MembershipIndex.from_blocks(self.source_blocks),
        )
        if len(self.source_blocks) != len(self.key_residuals) or len(self.source_blocks) != len(
            self.value_residuals
        ):
            raise ValueError("residual source blocks and K/V payloads must align")
        _ensure_unique_tensors((*self.key_residuals, *self.value_residuals), "residual tier")

    @property
    def source_memberships(self) -> frozenset[tuple[int, int, int]]:
        cached = self._source_memberships
        if cached is None:
            cached = self._membership_index.memberships()
            object.__setattr__(self, "_source_memberships", cached)
        return cached

    @property
    def source_membership_count(self) -> int:
        return self._membership_index.count

    def source_is_subset_of(self, prototypes: PrototypeTier) -> bool:
        return self._membership_index.is_subset_of(prototypes._membership_index)

    @property
    def active_bytes(self) -> int:
        return sum(
            tensor_storage_bytes(item) for item in (*self.key_residuals, *self.value_residuals)
        )


@dataclass(frozen=True, slots=True)
class CompressionStatistics:
    """Exact byte and source-membership accounting for one cache state."""

    source_kv_bytes: int
    exact_kv_bytes: int
    prototype_kv_bytes: int
    residual_kv_bytes: int
    active_kv_bytes: int
    source_blocks: int
    exact_blocks: int
    prototype_source_blocks: int
    residual_source_blocks: int
    source_memberships: int
    active_source_memberships: int
    byte_retention_ratio: float

    def __post_init__(self) -> None:
        integer_fields = (
            "source_kv_bytes",
            "exact_kv_bytes",
            "prototype_kv_bytes",
            "residual_kv_bytes",
            "active_kv_bytes",
            "source_blocks",
            "exact_blocks",
            "prototype_source_blocks",
            "residual_source_blocks",
            "source_memberships",
            "active_source_memberships",
        )
        if any(getattr(self, name) < 0 for name in integer_fields):
            raise ValueError("compression statistics cannot be negative")
        expected = self.exact_kv_bytes + self.prototype_kv_bytes
        if self.active_kv_bytes != expected:
            raise ValueError("active KV bytes must equal exact + prototype bytes")
        if self.source_kv_bytes <= 0:
            raise ValueError("source KV bytes must be positive")
        expected_ratio = self.active_kv_bytes / self.source_kv_bytes
        if not math.isclose(self.byte_retention_ratio, expected_ratio, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("byte_retention_ratio does not match byte accounting")

    @property
    def total_stored_bytes(self) -> int:
        """Active device KV plus separately stored CPU residual payloads."""

        return self.active_kv_bytes + self.residual_kv_bytes


@dataclass(frozen=True, slots=True)
class FullKVState:
    """Uncompressed source tensors, block partition, and logical-position state."""

    layers: tuple[KVLayerStorage, ...]
    blocks: tuple[KVBlockDescriptor, ...]
    modality_spans: tuple[ModalitySpan, ...]
    logical_positions: LogicalPositionMap
    token_ids: tuple[int, ...] | None
    mandatory_logical_positions: frozenset[int]
    block_size: int
    source_class: type[Any]
    source_kind: str
    cached_key_state: Any
    _membership_index: _MembershipIndex = field(init=False, repr=False, compare=False)
    _source_memberships: frozenset[tuple[int, int, int]] | None = field(
        init=False, repr=False, compare=False, default=None
    )
    _source_block_ids: frozenset[int] = field(init=False, repr=False, compare=False)
    _source_blocks_by_membership: Mapping[
        tuple[int, int, tuple[int, ...]], KVBlockDescriptor
    ] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("FullKV state must contain at least one layer")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        active_length = self.logical_positions.active_sequence_length
        if any(layer.sequence_length != active_length for layer in self.layers):
            raise ValueError("all source layers must match the logical map's active length")
        if self.token_ids is not None and len(self.token_ids) != active_length:
            raise ValueError("token_ids must align with active physical cache positions")
        if not self.source_kind.strip():
            raise ValueError("source_kind must be non-empty")
        self._validate_spans()
        self._validate_block_partition()
        object.__setattr__(self, "_source_block_ids", frozenset(id(block) for block in self.blocks))
        object.__setattr__(
            self,
            "_source_blocks_by_membership",
            {
                (block.layer, block.kv_head, block.physical_cache_indices): block
                for block in self.blocks
            },
        )

    @property
    def original_logical_sequence_length(self) -> int:
        return self.logical_positions.original_logical_sequence_length

    @property
    def next_decode_position(self) -> int:
        return self.logical_positions.next_decode_position

    @property
    def active_sequence_length(self) -> int:
        return self.logical_positions.active_sequence_length

    @property
    def active_bytes(self) -> int:
        return sum(layer.byte_size for layer in self.layers)

    @property
    def source_memberships(self) -> frozenset[tuple[int, int, int]]:
        cached = self._source_memberships
        if cached is None:
            cached = self._membership_index.memberships()
            object.__setattr__(self, "_source_memberships", cached)
        return cached

    @property
    def source_membership_count(self) -> int:
        return self._membership_index.count

    def _validate_spans(self) -> None:
        ordered = sorted(self.modality_spans, key=lambda span: (span.start, span.end))
        for previous, following in pairwise(ordered):
            if previous.end > following.start:
                raise ValueError("modality spans cannot overlap")
        for logical_position in self.logical_positions.physical_to_logical:
            matches = [span for span in self.modality_spans if span.contains(logical_position)]
            if len(matches) != 1:
                raise ValueError(
                    f"logical position {logical_position} must belong to exactly one modality span"
                )

    def _validate_block_partition(self) -> None:
        observed = _MembershipIndex.from_blocks(self.blocks)
        expected_identities = {
            (layer_index, head)
            for layer_index, layer in enumerate(self.layers)
            for head in range(layer.kv_heads)
        }
        observed_identities = set(observed.ranges_by_identity)
        complete = observed_identities == expected_identities and all(
            observed.ranges_by_identity[identity]
            == ((0, self.layers[identity[0]].sequence_length - 1),)
            for identity in expected_identities
        )
        if not complete:
            expected_count = sum(layer.kv_heads * layer.sequence_length for layer in self.layers)
            raise ValueError(
                "every source cache position must belong to exactly one block; "
                f"expected={expected_count}, observed={observed.count}"
            )
        object.__setattr__(self, "_membership_index", observed)
        expected_bytes = self.active_bytes
        descriptor_bytes = sum(block.byte_size for block in self.blocks)
        if descriptor_bytes != expected_bytes:
            raise ValueError(
                f"source block byte accounting mismatch: blocks={descriptor_bytes}, "
                f"tensors={expected_bytes}"
            )
        for block in self.blocks:
            if block.position_count > self.block_size:
                raise ValueError("source block exceeds configured block_size")
            expected_mandatory = any(
                position in self.mandatory_logical_positions
                for position in block.original_logical_positions
            )
            if block.mandatory != expected_mandatory:
                raise ValueError("block mandatory flag does not match mandatory token positions")

    @staticmethod
    def _token_tuple(token_ids: Any | None, expected_length: int) -> tuple[int, ...] | None:
        if token_ids is None:
            return None
        value = token_ids
        detach = getattr(value, "detach", None)
        if detach is not None:
            value = detach().cpu().reshape(-1).tolist()
        elif hasattr(value, "reshape") and hasattr(value, "tolist"):
            value = value.reshape(-1).tolist()
        result = tuple(int(item) for item in value)
        if len(result) != expected_length:
            raise ValueError("token_ids do not match source cache sequence length")
        return result

    @classmethod
    def from_tensors(
        cls,
        layers: Sequence[tuple[Any, Any]],
        *,
        modality_spans: Sequence[ModalitySpan] | None = None,
        token_ids: Any | None = None,
        block_size: int,
        sequence_dimension: int = -2,
        head_dimension: int = -3,
        logical_positions: Sequence[int] | None = None,
        original_logical_sequence_length: int | None = None,
        next_decode_position: int | None = None,
        mandatory_logical_positions: Sequence[int] = (),
        source_class: type[Any] = tuple,
        source_kind: str = "tuple",
        cached_key_state: Any = "not_applicable",
    ) -> Self:
        if not layers:
            raise ValueError("FullKV state requires at least one K/V layer")
        storages: list[KVLayerStorage] = []
        sequence_length: int | None = None
        for layer_index, (key, value) in enumerate(layers):
            key_rank = _ndim(key)
            value_rank = _ndim(value)
            storage = KVLayerStorage(
                key,
                value,
                _normalize_axis(head_dimension, key_rank, "key head"),
                _normalize_axis(head_dimension, value_rank, "value head"),
                _normalize_axis(sequence_dimension, key_rank, "key sequence"),
                _normalize_axis(sequence_dimension, value_rank, "value sequence"),
            )
            if sequence_length is None:
                sequence_length = storage.sequence_length
            elif storage.sequence_length != sequence_length:
                raise ValueError(
                    f"cache layer {layer_index} has a different source sequence length"
                )
            storages.append(storage)
        if sequence_length is None or sequence_length <= 0:
            raise ValueError("source sequence length must be positive")
        mapped = (
            tuple(range(sequence_length))
            if logical_positions is None
            else tuple(int(position) for position in logical_positions)
        )
        original_length = (
            sequence_length
            if original_logical_sequence_length is None
            else original_logical_sequence_length
        )
        position_map = LogicalPositionMap(
            mapped,
            original_length,
            original_length if next_decode_position is None else next_decode_position,
        )
        spans = tuple(modality_spans or (ModalitySpan(0, original_length, Modality.TEXT),))
        normalized_tokens = cls._token_tuple(token_ids, sequence_length)
        mandatory = frozenset(int(position) for position in mandatory_logical_positions)
        if any(position < 0 or position >= original_length for position in mandatory):
            raise ValueError("mandatory logical position lies outside the original sequence")
        blocks = cls._blockize(
            tuple(storages),
            spans,
            position_map,
            normalized_tokens,
            mandatory,
            block_size,
        )
        return cls(
            tuple(storages),
            blocks,
            spans,
            position_map,
            normalized_tokens,
            mandatory,
            block_size,
            source_class,
            source_kind,
            cached_key_state,
        )

    @classmethod
    def from_cache_snapshot(
        cls,
        snapshot: Any,
        *,
        modality_spans: Sequence[ModalitySpan],
        token_ids: Any | None,
        block_size: int,
        head_dimension: int = -3,
        logical_positions: Sequence[int] | None = None,
        original_logical_sequence_length: int | None = None,
        next_decode_position: int | None = None,
        mandatory_logical_positions: Sequence[int] = (),
    ) -> Self:
        layers = tuple((layer.key, layer.value) for layer in snapshot.layers)
        sequence_dimensions = {int(layer.sequence_dimension) for layer in snapshot.layers}
        if len(sequence_dimensions) != 1:
            raise ValueError("cache snapshot layers must share one sequence dimension")
        return cls.from_tensors(
            layers,
            modality_spans=modality_spans,
            token_ids=token_ids,
            block_size=block_size,
            sequence_dimension=sequence_dimensions.pop(),
            head_dimension=head_dimension,
            logical_positions=logical_positions,
            original_logical_sequence_length=(
                int(snapshot.active_sequence_length)
                if original_logical_sequence_length is None
                else original_logical_sequence_length
            ),
            next_decode_position=next_decode_position,
            mandatory_logical_positions=mandatory_logical_positions,
            source_class=snapshot.source_class,
            source_kind=str(snapshot.source_kind),
            cached_key_state=snapshot.cached_key_state,
        )

    @staticmethod
    def _span_for(spans: Sequence[ModalitySpan], logical_position: int) -> ModalitySpan:
        matches = [span for span in spans if span.contains(logical_position)]
        if len(matches) != 1:
            raise ValueError(
                f"logical position {logical_position} must map to exactly one modality span"
            )
        return matches[0]

    @staticmethod
    def _span_signature(
        span: ModalitySpan,
    ) -> tuple[
        Modality,
        int | None,
        int | None,
        int | None,
        tuple[float, float, float, float] | None,
        tuple[int, int] | None,
    ]:
        return (
            span.modality,
            span.image_index,
            span.frame_index,
            span.page_index,
            span.region,
            span.grid_shape,
        )

    @staticmethod
    def _block_region(
        span: ModalitySpan,
        logical_positions: Sequence[int],
    ) -> tuple[float, float, float, float] | None:
        """Return one normalized raster box shared by a fixed-size KV block.

        Qwen expands every visual grid cell into one language-sequence token.
        Blocks may cross a raster row, so their graph node is represented by
        the smallest normalized bounding box containing every member cell.
        Explicit dataset/processor regions remain authoritative.
        """

        if span.region is not None:
            return span.region
        if span.grid_shape is None:
            return None
        rows, columns = span.grid_shape
        offsets = tuple(position - span.start for position in logical_positions)
        if any(offset < 0 or offset >= rows * columns for offset in offsets):
            raise ValueError("block logical position lies outside its visual grid span")
        row_ids = tuple(offset // columns for offset in offsets)
        column_ids = tuple(offset % columns for offset in offsets)
        return (
            min(column_ids) / columns,
            min(row_ids) / rows,
            (max(column_ids) + 1) / columns,
            (max(row_ids) + 1) / rows,
        )

    @classmethod
    def _block_metadata(
        cls,
        span: ModalitySpan,
        logical_positions: Sequence[int],
    ) -> tuple[MediaMetadata, ...]:
        region = cls._block_region(span, logical_positions)
        return tuple(
            MediaMetadata(
                logical_position,
                span.image_index,
                span.frame_index,
                span.page_index,
                region,
            )
            for logical_position in logical_positions
        )

    @classmethod
    def _blockize(
        cls,
        layers: tuple[KVLayerStorage, ...],
        spans: tuple[ModalitySpan, ...],
        positions: LogicalPositionMap,
        token_ids: tuple[int, ...] | None,
        mandatory: frozenset[int],
        block_size: int,
    ) -> tuple[KVBlockDescriptor, ...]:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if not layers:
            return ()

        # Physical/logical membership, modality boundaries, token IDs, media
        # metadata, and the mandatory flag are identical for every layer/head.
        # Build that sequence template once.  The previous implementation
        # repeated span lookup and constructed/validated a MediaMetadata object
        # for every source position in every layer/head (more than one million
        # objects for a typical LLaVA document prompt) despite the immutable
        # values being identical.
        source_length = layers[0].sequence_length
        if any(layer.sequence_length != source_length for layer in layers):
            raise ValueError("all cache layers must share one sequence length for blockization")
        ordered_spans = tuple(sorted(spans, key=lambda span: (span.start, span.end)))
        mapped = positions.physical_to_logical
        span_by_physical: list[ModalitySpan] = []
        span_index = 0
        for logical_position in mapped:
            while (
                span_index < len(ordered_spans)
                and logical_position >= ordered_spans[span_index].end
            ):
                span_index += 1
            if span_index >= len(ordered_spans) or not ordered_spans[span_index].contains(
                logical_position
            ):
                raise ValueError(
                    f"logical position {logical_position} must map to exactly one modality span"
                )
            span_by_physical.append(ordered_spans[span_index])

        templates: list[
            tuple[
                tuple[int, ...],
                tuple[int, ...],
                ModalitySpan,
                tuple[int, ...] | None,
                tuple[MediaMetadata, ...],
                bool,
            ]
        ] = []
        physical = 0
        while physical < source_length:
            first_span = span_by_physical[physical]
            first_signature = cls._span_signature(first_span)
            end = physical + 1
            while end < source_length and end - physical < block_size:
                if cls._span_signature(span_by_physical[end]) != first_signature:
                    break
                end += 1
            selected_physical = tuple(range(physical, end))
            selected_logical = mapped[physical:end]
            templates.append(
                (
                    selected_physical,
                    selected_logical,
                    first_span,
                    None if token_ids is None else token_ids[physical:end],
                    cls._block_metadata(first_span, selected_logical),
                    any(logical in mandatory for logical in selected_logical),
                )
            )
            physical = end

        blocks: list[KVBlockDescriptor] = []
        for layer_index, layer in enumerate(layers):
            key_bytes_per_position = cls._position_bytes(
                layer.key,
                layer.key_head_dimension,
                layer.key_sequence_dimension,
                1,
            )
            value_bytes_per_position = cls._position_bytes(
                layer.value,
                layer.value_head_dimension,
                layer.value_sequence_dimension,
                1,
            )
            key_dtype = _dtype_name(layer.key)
            value_dtype = _dtype_name(layer.value)
            key_device = _device_name(layer.key)
            value_device = _device_name(layer.value)
            for head in range(layer.kv_heads):
                for (
                    selected_physical,
                    selected_logical,
                    first_span,
                    selected_tokens,
                    metadata,
                    is_mandatory,
                ) in templates:
                    position_count = len(selected_physical)
                    blocks.append(
                        KVBlockDescriptor._from_validated(
                            layer=layer_index,
                            kv_head=head,
                            modality=first_span.modality,
                            physical_cache_indices=selected_physical,
                            original_logical_positions=selected_logical,
                            token_ids=selected_tokens,
                            media_metadata=metadata,
                            key_dtype=key_dtype,
                            value_dtype=value_dtype,
                            key_device=key_device,
                            value_device=value_device,
                            byte_size=(
                                (key_bytes_per_position + value_bytes_per_position)
                                * position_count
                            ),
                            mandatory=is_mandatory,
                        )
                    )
        return tuple(blocks)

    @staticmethod
    def _position_bytes(
        tensor: Any, head_axis: int, sequence_axis: int, position_count: int
    ) -> int:
        shape = _shape(tensor)
        elements_per_head_position = _tensor_numel(tensor) // (
            shape[head_axis] * shape[sequence_axis]
        )
        return elements_per_head_position * position_count * _tensor_element_size(tensor)

    def _validate_descriptor(self, block: KVBlockDescriptor) -> None:
        if id(block) in self._source_block_ids:
            return
        try:
            layer = self.layers[block.layer]
        except IndexError as error:
            raise ValueError(f"block layer is outside the source cache: {block.layer}") from error
        if block.kv_head >= layer.kv_heads:
            raise ValueError(f"block KV head is outside the source cache: {block.kv_head}")
        if any(position >= layer.sequence_length for position in block.physical_cache_indices):
            raise ValueError("block physical position lies outside the source cache")
        expected_logical = self.logical_positions.gather(block.physical_cache_indices)
        if expected_logical != block.original_logical_positions:
            raise ValueError("block logical positions differ from the source position map")
        expected_tokens = (
            None
            if self.token_ids is None
            else tuple(self.token_ids[index] for index in block.physical_cache_indices)
        )
        if expected_tokens != block.token_ids:
            raise ValueError("block token IDs differ from the source cache")
        expected_spans = tuple(
            self._span_for(self.modality_spans, logical)
            for logical in block.original_logical_positions
        )
        if any(span.modality is not block.modality for span in expected_spans):
            raise ValueError("block modality differs from the source modality map")
        expected_metadata = self._block_metadata(
            expected_spans[0], block.original_logical_positions
        )
        if expected_metadata != block.media_metadata:
            raise ValueError("block media metadata differs from the source modality map")
        expected_bytes = self._position_bytes(
            layer.key,
            layer.key_head_dimension,
            layer.key_sequence_dimension,
            block.position_count,
        ) + self._position_bytes(
            layer.value,
            layer.value_head_dimension,
            layer.value_sequence_dimension,
            block.position_count,
        )
        if block.byte_size != expected_bytes:
            raise ValueError("block byte size differs from its source cache positions")
        observed_storage = (
            _dtype_name(layer.key),
            _dtype_name(layer.value),
            _device_name(layer.key),
            _device_name(layer.value),
        )
        declared_storage = (
            block.key_dtype,
            block.value_dtype,
            block.key_device,
            block.value_device,
        )
        if observed_storage != declared_storage:
            raise ValueError("block dtype/device metadata differs from the source cache")
        expected_mandatory = any(
            logical in self.mandatory_logical_positions
            for logical in block.original_logical_positions
        )
        if block.mandatory != expected_mandatory:
            raise ValueError("block mandatory flag differs from the source cache")

    @audit_operation("exact.descriptor_gather_and_views")
    def _gather_descriptors(
        self,
        descriptors: Sequence[KVBlockDescriptor],
        *,
        gather_mode: str,
    ) -> ExactTier:
        if gather_mode not in {"per_head_reference", "batched_layer", "source_views"}:
            raise ValueError(
                "gather_mode must be 'per_head_reference', 'batched_layer', or 'source_views'"
            )
        selected = tuple(descriptors)
        if gather_mode == "source_views":
            aliased_key_blocks: list[Any] = []
            aliased_value_blocks: list[Any] = []
            for block in selected:
                self._validate_descriptor(block)
                positions = block.physical_cache_indices
                if positions != tuple(range(positions[0], positions[-1] + 1)):
                    raise ValueError(
                        "source-view exact gathering requires contiguous source blocks"
                    )
                layer = self.layers[block.layer]

                def source_view(
                    tensor: Any,
                    head_axis: int,
                    sequence_axis: int,
                    *,
                    head: int,
                    start: int,
                    end: int,
                ) -> Any:
                    index: list[Any] = [slice(None)] * _ndim(tensor)
                    index[head_axis] = slice(head, head + 1)
                    index[sequence_axis] = slice(start, end)
                    return tensor[tuple(index)]

                aliased_key_blocks.append(
                    source_view(
                        layer.key,
                        layer.key_head_dimension,
                        layer.key_sequence_dimension,
                        head=block.kv_head,
                        start=positions[0],
                        end=positions[-1] + 1,
                    )
                )
                aliased_value_blocks.append(
                    source_view(
                        layer.value,
                        layer.value_head_dimension,
                        layer.value_sequence_dimension,
                        head=block.kv_head,
                        start=positions[0],
                        end=positions[-1] + 1,
                    )
                )
            return ExactTier(
                selected,
                tuple(aliased_key_blocks),
                tuple(aliased_value_blocks),
            )
        grouped: dict[tuple[int, int], list[tuple[int, KVBlockDescriptor]]] = {}
        for block_index, block in enumerate(selected):
            self._validate_descriptor(block)
            grouped.setdefault((block.layer, block.kv_head), []).append((block_index, block))
        key_blocks: list[Any | None] = [None] * len(selected)
        value_blocks: list[Any | None] = [None] * len(selected)
        gathered_pairs: dict[tuple[int, int], tuple[Any, Any]] = {}
        if gather_mode == "batched_layer":
            identities_by_layer: dict[int, list[int]] = {}
            for layer_index, kv_head in grouped:
                identities_by_layer.setdefault(layer_index, []).append(kv_head)
            for layer_index, raw_heads in identities_by_layer.items():
                layer = self.layers[layer_index]
                heads = tuple(sorted(raw_heads))
                positions_by_head = {
                    head: tuple(
                        position
                        for _block_index, block in grouped[(layer_index, head)]
                        for position in block.physical_cache_indices
                    )
                    for head in heads
                }
                gathered_keys = _gather_torch_heads_by_count(
                    layer.key,
                    head_axis=layer.key_head_dimension,
                    sequence_axis=layer.key_sequence_dimension,
                    positions_by_head=positions_by_head,
                )
                gathered_values = _gather_torch_heads_by_count(
                    layer.value,
                    head_axis=layer.value_head_dimension,
                    sequence_axis=layer.value_sequence_dimension,
                    positions_by_head=positions_by_head,
                )
                if gathered_keys is None or gathered_values is None:
                    continue
                for head in heads:
                    gathered_pairs[(layer_index, head)] = (
                        gathered_keys[head],
                        gathered_values[head],
                    )
        # Gather all selected positions for a layer/head once, then expose
        # block-sized views.  Launching one advanced-index operation per block
        # dominated LLaVA construction because it has 32 KV heads per layer.
        for (layer_index, kv_head), indexed_blocks in grouped.items():
            layer = self.layers[layer_index]
            positions = tuple(
                position
                for _block_index, block in indexed_blocks
                for position in block.physical_cache_indices
            )
            pair = gathered_pairs.get((layer_index, kv_head))
            if pair is None:
                gathered_key = _gather_tensor(
                    layer.key,
                    head_axis=layer.key_head_dimension,
                    head=kv_head,
                    sequence_axis=layer.key_sequence_dimension,
                    positions=positions,
                )
                gathered_value = _gather_tensor(
                    layer.value,
                    head_axis=layer.value_head_dimension,
                    head=kv_head,
                    sequence_axis=layer.value_sequence_dimension,
                    positions=positions,
                )
            else:
                gathered_key, gathered_value = pair
            offset = 0
            for block_index, block in indexed_blocks:
                end = offset + block.position_count
                key_index: list[Any] = [slice(None)] * _ndim(gathered_key)
                value_index: list[Any] = [slice(None)] * _ndim(gathered_value)
                key_index[layer.key_sequence_dimension] = slice(offset, end)
                value_index[layer.value_sequence_dimension] = slice(offset, end)
                key_blocks[block_index] = gathered_key[tuple(key_index)]
                value_blocks[block_index] = gathered_value[tuple(value_index)]
                offset = end
        if any(block is None for block in key_blocks + value_blocks):
            raise RuntimeError("exact-tier grouped gather left an uninitialized block")
        return ExactTier(selected, tuple(key_blocks), tuple(value_blocks))

    @audit_operation("exact.gather_source_membership_and_descriptors")
    def gather_exact_blocks(
        self,
        blocks: Sequence[KVBlockDescriptor],
        *,
        gather_mode: str = "batched_layer",
    ) -> ExactTier:
        """Copy complete selected source blocks into the lossless exact tier."""

        selected = tuple(blocks)
        if any(
            self._source_blocks_by_membership.get(
                (block.layer, block.kv_head, block.physical_cache_indices)
            )
            != block
            for block in selected
        ):
            raise ValueError("exact block selection contains a non-source descriptor")
        return self._gather_descriptors(selected, gather_mode=gather_mode)

    def gather_selected_positions(
        self,
        selected: Mapping[tuple[int, int], Sequence[int]],
    ) -> ExactTier:
        """Gather arbitrary source positions without losing logical provenance."""

        descriptors: list[KVBlockDescriptor] = []
        for (layer_index, head), raw_positions in sorted(selected.items()):
            try:
                layer = self.layers[layer_index]
            except IndexError as error:
                raise ValueError(
                    f"selected layer is outside the source cache: {layer_index}"
                ) from error
            if head < 0 or head >= layer.kv_heads:
                raise ValueError(f"selected KV head is outside the source cache: {head}")
            physical_positions = tuple(sorted(int(position) for position in raw_positions))
            if len(physical_positions) != len(set(physical_positions)):
                raise ValueError("selected positions must be unique per layer and KV head")
            if any(
                position < 0 or position >= layer.sequence_length for position in physical_positions
            ):
                raise ValueError("selected physical position is outside the source cache")
            offset = 0
            while offset < len(physical_positions):
                first_physical = physical_positions[offset]
                first_logical = self.logical_positions.logical_for_physical(first_physical)
                first_span = self._span_for(self.modality_spans, first_logical)
                modality = first_span.modality
                end = offset + 1
                while end < len(physical_positions) and end - offset < self.block_size:
                    logical = self.logical_positions.logical_for_physical(physical_positions[end])
                    following_span = self._span_for(self.modality_spans, logical)
                    if self._span_signature(following_span) != self._span_signature(first_span):
                        break
                    end += 1
                positions = physical_positions[offset:end]
                logical_positions = self.logical_positions.gather(positions)
                metadata = self._block_metadata(first_span, logical_positions)
                key_bytes = self._position_bytes(
                    layer.key,
                    layer.key_head_dimension,
                    layer.key_sequence_dimension,
                    len(positions),
                )
                value_bytes = self._position_bytes(
                    layer.value,
                    layer.value_head_dimension,
                    layer.value_sequence_dimension,
                    len(positions),
                )
                descriptors.append(
                    KVBlockDescriptor(
                        layer_index,
                        head,
                        modality,
                        positions,
                        logical_positions,
                        (
                            None
                            if self.token_ids is None
                            else tuple(self.token_ids[index] for index in positions)
                        ),
                        metadata,
                        _dtype_name(layer.key),
                        _dtype_name(layer.value),
                        _device_name(layer.key),
                        _device_name(layer.value),
                        key_bytes + value_bytes,
                        any(
                            logical in self.mandatory_logical_positions
                            for logical in logical_positions
                        ),
                    )
                )
                offset = end
        return self._gather_descriptors(descriptors, gather_mode="batched_layer")

    def reinject_exact(self, exact: ExactTier) -> FullKVState:
        """Reconstruct the complete source cache from exact-tier payloads only."""

        if not exact._membership_index.same_memberships(self._membership_index):
            raise ValueError("full reinjection requires exact coverage of every source position")
        reconstructed = [
            (_empty_like(layer.key), _empty_like(layer.value)) for layer in self.layers
        ]
        for block, key_block, value_block in zip(
            exact.blocks, exact.key_blocks, exact.value_blocks, strict=True
        ):
            self._validate_descriptor(block)
            layer = self.layers[block.layer]
            key_target, value_target = reconstructed[block.layer]
            key_target[
                _tensor_index(
                    key_target,
                    head_axis=layer.key_head_dimension,
                    head=block.kv_head,
                    sequence_axis=layer.key_sequence_dimension,
                    positions=block.physical_cache_indices,
                )
            ] = key_block
            value_target[
                _tensor_index(
                    value_target,
                    head_axis=layer.value_head_dimension,
                    head=block.kv_head,
                    sequence_axis=layer.value_sequence_dimension,
                    positions=block.physical_cache_indices,
                )
            ] = value_block
        result = FullKVState.from_tensors(
            reconstructed,
            modality_spans=self.modality_spans,
            token_ids=self.token_ids,
            block_size=self.block_size,
            sequence_dimension=self.layers[0].key_sequence_dimension,
            head_dimension=self.layers[0].key_head_dimension,
            logical_positions=self.logical_positions.physical_to_logical,
            original_logical_sequence_length=self.original_logical_sequence_length,
            next_decode_position=self.next_decode_position,
            mandatory_logical_positions=tuple(self.mandatory_logical_positions),
            source_class=self.source_class,
            source_kind=self.source_kind,
            cached_key_state=self.cached_key_state,
        )
        for source_layer, result_layer in zip(self.layers, result.layers, strict=True):
            if not _tensor_equal(source_layer.key, result_layer.key) or not _tensor_equal(
                source_layer.value, result_layer.value
            ):
                raise RuntimeError("retention-1.0 exact reinjection changed source cache values")
        return result

    def to_cache_snapshot(self) -> Any:
        """Convert back to the adapter snapshot type without changing tensor values."""

        from mosaickv.adapters.huggingface.types import CacheLayerSnapshot, CacheSnapshot

        layers = tuple(
            CacheLayerSnapshot(layer.key, layer.value, layer.key_sequence_dimension)
            for layer in self.layers
        )
        return CacheSnapshot(
            layers,
            self.source_class,
            self.source_kind,
            self.active_sequence_length,
            self.cached_key_state,
        )


@dataclass(frozen=True, slots=True)
class MosaicKVState:
    """Three-tier cache payload plus immutable source membership metadata."""

    source_blocks: tuple[KVBlockDescriptor, ...]
    exact: ExactTier
    prototypes: PrototypeTier
    residuals: ResidualTier
    logical_positions: LogicalPositionMap
    statistics: CompressionStatistics
    _source_membership_index: _MembershipIndex = field(
        init=False, repr=False, compare=False
    )

    @audit_operation("state.membership_and_storage_invariants")
    def __post_init__(self) -> None:
        source = _MembershipIndex.from_blocks(self.source_blocks)
        object.__setattr__(self, "_source_membership_index", source)
        exact = self.exact._membership_index
        prototypes = self.prototypes._membership_index
        residuals = self.residuals._membership_index
        if not (
            exact.is_subset_of(source)
            and prototypes.is_subset_of(source)
            and residuals.is_subset_of(source)
        ):
            raise ValueError("tier source membership is not part of the source cache")
        conflict = exact.first_overlap(prototypes) or exact.first_overlap(residuals)
        if conflict is not None:
            raise ValueError(f"exact memberships conflict with a compressed tier: {conflict}")
        _ensure_unique_tensors(
            (
                *self.exact.key_blocks,
                *self.exact.value_blocks,
                *self.prototypes.prototype_keys,
                *self.prototypes.prototype_values,
                *self.residuals.key_residuals,
                *self.residuals.value_residuals,
            ),
            "MosaicKV tiers",
        )
        mandatory = _MembershipIndex.from_blocks(
            tuple(block for block in self.source_blocks if block.mandatory)
        )
        if not mandatory.is_subset_of(exact):
            raise ValueError("mandatory/non-compressible source positions must remain exact")
        observed_active = self.exact.active_bytes + self.prototypes.active_bytes
        if observed_active != self.statistics.active_kv_bytes:
            raise ValueError("active byte accounting differs from underlying tier storage")
        if self.residuals.active_bytes != self.statistics.residual_kv_bytes:
            raise ValueError("residual byte accounting differs from CPU tier storage")
        active_memberships = exact.count + prototypes.count
        if active_memberships != self.statistics.active_source_memberships:
            raise ValueError("active source-membership accounting mismatch")
        if source.count != self.statistics.source_memberships:
            raise ValueError("source-membership accounting mismatch")

    @classmethod
    @audit_operation("state.create_and_account_storage")
    def create(
        cls,
        full: FullKVState,
        *,
        exact: ExactTier | None = None,
        prototypes: PrototypeTier | None = None,
        residuals: ResidualTier | None = None,
    ) -> Self:
        exact_tier = exact or ExactTier()
        prototype_tier = prototypes or PrototypeTier()
        residual_tier = residuals or ResidualTier()
        for descriptor in (
            *exact_tier.blocks,
            *prototype_tier.source_blocks,
            *residual_tier.source_blocks,
        ):
            full._validate_descriptor(descriptor)
        if exact_tier._membership_index.first_overlap(prototype_tier._membership_index):
            raise ValueError("exact memberships conflict with prototype memberships")
        active_memberships = (
            exact_tier.source_membership_count + prototype_tier.source_membership_count
        )
        active_bytes = exact_tier.active_bytes + prototype_tier.active_bytes
        statistics = CompressionStatistics(
            source_kv_bytes=full.active_bytes,
            exact_kv_bytes=exact_tier.active_bytes,
            prototype_kv_bytes=prototype_tier.active_bytes,
            residual_kv_bytes=residual_tier.active_bytes,
            active_kv_bytes=active_bytes,
            source_blocks=len(full.blocks),
            exact_blocks=len(exact_tier.blocks),
            prototype_source_blocks=len(prototype_tier.source_blocks),
            residual_source_blocks=len(residual_tier.source_blocks),
            source_memberships=full.source_membership_count,
            active_source_memberships=active_memberships,
            byte_retention_ratio=active_bytes / full.active_bytes,
        )
        return cls(
            full.blocks,
            exact_tier,
            prototype_tier,
            residual_tier,
            full.logical_positions,
            statistics,
        )

    @classmethod
    def retention_one(cls, full: FullKVState) -> Self:
        """Create and immediately verify the non-lossy 100%-retention state."""

        state = cls.create(full, exact=full.gather_exact_blocks(full.blocks))
        state.reconstruct_full_state(full)
        return state

    @property
    def original_logical_sequence_length(self) -> int:
        return self.logical_positions.original_logical_sequence_length

    @property
    def next_decode_position(self) -> int:
        return self.logical_positions.next_decode_position

    @property
    def is_retention_one(self) -> bool:
        return (
            self.exact._membership_index.same_memberships(self._source_membership_index)
            and not self.prototypes.source_blocks
            and not self.residuals.source_blocks
            and math.isclose(self.statistics.byte_retention_ratio, 1.0, rel_tol=0.0, abs_tol=1e-15)
        )

    def selected_positions(self, layer: int, kv_head: int) -> tuple[int, ...]:
        """Return exact physical positions for one layer/head in source order."""

        return self.exact.selected_positions(layer, kv_head)

    def reconstruct_full_state(self, source: FullKVState) -> FullKVState:
        """Reinject a retention-1.0 exact tier and verify value/position identity."""

        if tuple(source.blocks) != self.source_blocks:
            raise ValueError("reinjection source blocks differ from MosaicKV source metadata")
        if source.logical_positions != self.logical_positions:
            raise ValueError("reinjection logical position map differs from MosaicKV state")
        if not self.is_retention_one:
            raise ValueError("full reconstruction is only valid for a retention-1.0 exact state")
        reconstructed = source.reinject_exact(self.exact)
        if reconstructed.original_logical_sequence_length != self.original_logical_sequence_length:
            raise RuntimeError("reinjection changed original logical sequence length")
        if reconstructed.next_decode_position != self.next_decode_position:
            raise RuntimeError("reinjection changed next decode position")
        return reconstructed


__all__ = [
    "CompressionStatistics",
    "ExactTier",
    "FullKVState",
    "KVBlockDescriptor",
    "KVLayerStorage",
    "LogicalPositionMap",
    "MediaMetadata",
    "Modality",
    "ModalitySpan",
    "MosaicKVState",
    "PrototypeTier",
    "ResidualTier",
    "tensor_storage_bytes",
]
