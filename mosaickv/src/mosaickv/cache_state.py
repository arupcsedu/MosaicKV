"""Backend-independent MosaicKV cache state, blockization, and invariants.

This module intentionally uses a small tensor protocol instead of importing
PyTorch.  NumPy arrays support CPU property tests, while torch tensors support
the same indexing, cloning, assignment, shape, dtype, and device operations in
the Hugging Face integration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any, Self


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
        overlap = memberships.intersection(block.source_memberships)
        if overlap:
            member = min(overlap)
            raise ValueError(f"duplicate source membership within tier: {member}")
        memberships.update(block.source_memberships)
    return frozenset(memberships)


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

    def __post_init__(self) -> None:
        if len(self.blocks) != len(self.key_blocks) or len(self.blocks) != len(self.value_blocks):
            raise ValueError("exact tier blocks and K/V payloads must align")
        _memberships(self.blocks)
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
        return _memberships(self.blocks)

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

    def __post_init__(self) -> None:
        _memberships(self.source_blocks)
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
        return _memberships(self.source_blocks)

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

    def __post_init__(self) -> None:
        _memberships(self.source_blocks)
        if len(self.source_blocks) != len(self.key_residuals) or len(self.source_blocks) != len(
            self.value_residuals
        ):
            raise ValueError("residual source blocks and K/V payloads must align")
        _ensure_unique_tensors((*self.key_residuals, *self.value_residuals), "residual tier")

    @property
    def source_memberships(self) -> frozenset[tuple[int, int, int]]:
        return _memberships(self.source_blocks)

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
        return _memberships(self.blocks)

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
        observed = _memberships(self.blocks)
        expected = frozenset(
            (layer_index, head, position)
            for layer_index, layer in enumerate(self.layers)
            for head in range(layer.kv_heads)
            for position in range(layer.sequence_length)
        )
        if observed != expected:
            missing = expected - observed
            extra = observed - expected
            raise ValueError(
                "every source cache position must belong to exactly one block; "
                f"missing={len(missing)}, extra={len(extra)}"
            )
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
    ]:
        return (
            span.modality,
            span.image_index,
            span.frame_index,
            span.page_index,
            span.region,
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
        blocks: list[KVBlockDescriptor] = []
        for layer_index, layer in enumerate(layers):
            for head in range(layer.kv_heads):
                physical = 0
                while physical < layer.sequence_length:
                    first_logical = positions.logical_for_physical(physical)
                    first_span = cls._span_for(spans, first_logical)
                    end = physical + 1
                    while end < layer.sequence_length and end - physical < block_size:
                        logical = positions.logical_for_physical(end)
                        following_span = cls._span_for(spans, logical)
                        if cls._span_signature(following_span) != cls._span_signature(first_span):
                            break
                        end += 1
                    selected_physical = tuple(range(physical, end))
                    selected_logical = positions.gather(selected_physical)
                    metadata = tuple(
                        cls._media_metadata(cls._span_for(spans, logical), logical)
                        for logical in selected_logical
                    )
                    key_bytes = cls._position_bytes(
                        layer.key,
                        layer.key_head_dimension,
                        layer.key_sequence_dimension,
                        len(selected_physical),
                    )
                    value_bytes = cls._position_bytes(
                        layer.value,
                        layer.value_head_dimension,
                        layer.value_sequence_dimension,
                        len(selected_physical),
                    )
                    blocks.append(
                        KVBlockDescriptor(
                            layer=layer_index,
                            kv_head=head,
                            modality=first_span.modality,
                            physical_cache_indices=selected_physical,
                            original_logical_positions=selected_logical,
                            token_ids=(
                                None
                                if token_ids is None
                                else tuple(token_ids[index] for index in selected_physical)
                            ),
                            media_metadata=metadata,
                            key_dtype=_dtype_name(layer.key),
                            value_dtype=_dtype_name(layer.value),
                            key_device=_device_name(layer.key),
                            value_device=_device_name(layer.value),
                            byte_size=key_bytes + value_bytes,
                            mandatory=any(logical in mandatory for logical in selected_logical),
                        )
                    )
                    physical = end
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

    @staticmethod
    def _media_metadata(span: ModalitySpan, logical_position: int) -> MediaMetadata:
        return MediaMetadata(
            logical_position,
            span.image_index,
            span.frame_index,
            span.page_index,
            span.region,
        )

    def _validate_descriptor(self, block: KVBlockDescriptor) -> None:
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
        expected_metadata = tuple(
            self._media_metadata(span, logical)
            for span, logical in zip(expected_spans, block.original_logical_positions, strict=True)
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

    def _gather_descriptors(self, descriptors: Sequence[KVBlockDescriptor]) -> ExactTier:
        selected = tuple(descriptors)
        key_blocks: list[Any] = []
        value_blocks: list[Any] = []
        for block in selected:
            self._validate_descriptor(block)
            layer = self.layers[block.layer]
            key_blocks.append(
                _gather_tensor(
                    layer.key,
                    head_axis=layer.key_head_dimension,
                    head=block.kv_head,
                    sequence_axis=layer.key_sequence_dimension,
                    positions=block.physical_cache_indices,
                )
            )
            value_blocks.append(
                _gather_tensor(
                    layer.value,
                    head_axis=layer.value_head_dimension,
                    head=block.kv_head,
                    sequence_axis=layer.value_sequence_dimension,
                    positions=block.physical_cache_indices,
                )
            )
        return ExactTier(selected, tuple(key_blocks), tuple(value_blocks))

    def gather_exact_blocks(self, blocks: Sequence[KVBlockDescriptor]) -> ExactTier:
        """Copy complete selected source blocks into the lossless exact tier."""

        source = set(self.blocks)
        selected = tuple(blocks)
        if any(block not in source for block in selected):
            raise ValueError("exact block selection contains a non-source descriptor")
        return self._gather_descriptors(selected)

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
                metadata = tuple(
                    self._media_metadata(self._span_for(self.modality_spans, logical), logical)
                    for logical in logical_positions
                )
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
        return self._gather_descriptors(descriptors)

    def reinject_exact(self, exact: ExactTier) -> FullKVState:
        """Reconstruct the complete source cache from exact-tier payloads only."""

        if exact.source_memberships != self.source_memberships:
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

    def __post_init__(self) -> None:
        source = _memberships(self.source_blocks)
        exact = self.exact.source_memberships
        prototypes = self.prototypes.source_memberships
        residuals = self.residuals.source_memberships
        if not exact <= source or not prototypes <= source or not residuals <= source:
            raise ValueError("tier source membership is not part of the source cache")
        conflicts = (exact & prototypes) | (exact & residuals)
        if conflicts:
            raise ValueError(f"exact memberships conflict with a compressed tier: {min(conflicts)}")
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
        mandatory = frozenset(
            membership
            for block in self.source_blocks
            if block.mandatory
            for membership in block.source_memberships
        )
        if not mandatory <= exact:
            raise ValueError("mandatory/non-compressible source positions must remain exact")
        observed_active = self.exact.active_bytes + self.prototypes.active_bytes
        if observed_active != self.statistics.active_kv_bytes:
            raise ValueError("active byte accounting differs from underlying tier storage")
        if self.residuals.active_bytes != self.statistics.residual_kv_bytes:
            raise ValueError("residual byte accounting differs from CPU tier storage")
        active_memberships = len(exact | prototypes)
        if active_memberships != self.statistics.active_source_memberships:
            raise ValueError("active source-membership accounting mismatch")
        if len(source) != self.statistics.source_memberships:
            raise ValueError("source-membership accounting mismatch")

    @classmethod
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
        active_memberships = exact_tier.source_memberships | prototype_tier.source_memberships
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
            source_memberships=len(full.source_memberships),
            active_source_memberships=len(active_memberships),
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
        source = _memberships(self.source_blocks)
        return (
            self.exact.source_memberships == source
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
