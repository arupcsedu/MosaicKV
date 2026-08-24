"""Backend-neutral ownership guards for disposable decode-cache candidates.

The helpers in this module are intentionally model- and GPU-independent. A
backend adapter supplies cache extraction and reinjection, while this module
owns cloning, alias checks, reference-state checkpoints, candidate adoption,
and disposal. Candidate construction never performs model inference.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from os import PathLike
from types import TracebackType
from typing import Any, Literal, Protocol

from mosaickv.adapters.huggingface.types import (
    CacheLayerSnapshot,
    CacheSnapshot,
    DecodeState,
)


class CacheIsolationError(RuntimeError):
    """Raised when disposable cache ownership cannot be proven."""


class CacheStateAdapter(Protocol):
    """Minimal cache API required by the isolation boundary."""

    def extract_past_key_values(self, value: Any, *, clone: bool = True) -> Any: ...

    def inject_past_key_values(self, snapshot: CacheSnapshot) -> Any: ...


_IMMUTABLE_TYPES = (type(None), bool, int, float, complex, str, bytes, Enum, type, PathLike)


def _clone_value(value: Any, *, path: str) -> Any:
    """Clone nested backend metadata without retaining mutable aliases."""

    detach = getattr(value, "detach", None)
    clone = getattr(value, "clone", None)
    if callable(detach) and callable(clone):
        result = detach().clone()
        if result is value:
            raise CacheIsolationError(f"{path} tensor-like clone retained object identity")
        return result
    if isinstance(value, Mapping):
        return {
            _clone_value(key, path=f"{path}.key"): _clone_value(item, path=f"{path}[{key!r}]")
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(
            _clone_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [_clone_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, frozenset):
        return frozenset(_clone_value(item, path=f"{path}.item") for item in value)
    if isinstance(value, set):
        return {_clone_value(item, path=f"{path}.item") for item in value}
    if isinstance(value, _IMMUTABLE_TYPES):
        if isinstance(value, float) and not math.isfinite(value):
            raise CacheIsolationError(f"{path} contains a non-finite float")
        return value
    try:
        result = copy.deepcopy(value)
    except Exception as error:  # pragma: no cover - defensive adapter boundary
        raise CacheIsolationError(f"{path} cannot be cloned safely") from error
    if result is value and hasattr(value, "__dict__"):
        raise CacheIsolationError(f"{path} retained an unknown mutable object")
    return result


def clone_cache_snapshot(snapshot: CacheSnapshot) -> CacheSnapshot:
    """Clone every K/V tensor in a typed cache snapshot."""

    cloned = CacheSnapshot(
        layers=tuple(
            CacheLayerSnapshot(
                key=_clone_value(layer.key, path=f"cache.layers[{index}].key"),
                value=_clone_value(layer.value, path=f"cache.layers[{index}].value"),
                sequence_dimension=layer.sequence_dimension,
            )
            for index, layer in enumerate(snapshot.layers)
        ),
        source_class=snapshot.source_class,
        source_kind=snapshot.source_kind,
        active_sequence_length=snapshot.active_sequence_length,
        cached_key_state=snapshot.cached_key_state,
    )
    for index, layer in enumerate(cloned.layers):
        key_shape = tuple(int(item) for item in getattr(layer.key, "shape", ()))
        value_shape = tuple(int(item) for item in getattr(layer.value, "shape", ()))
        if not key_shape or not value_shape:
            raise CacheIsolationError(f"cache layer {index} does not contain shaped K/V tensors")
        key_axis = (
            layer.sequence_dimension
            if layer.sequence_dimension >= 0
            else len(key_shape) + layer.sequence_dimension
        )
        value_axis = (
            layer.sequence_dimension
            if layer.sequence_dimension >= 0
            else len(value_shape) + layer.sequence_dimension
        )
        if not 0 <= key_axis < len(key_shape) or not 0 <= value_axis < len(value_shape):
            raise CacheIsolationError(f"cache layer {index} has an invalid sequence axis")
        if (
            key_shape[key_axis] != cloned.active_sequence_length
            or value_shape[value_axis] != cloned.active_sequence_length
        ):
            raise CacheIsolationError(
                f"cache layer {index} length differs from active_sequence_length"
            )
    return cloned


def _value_equal(first: Any, second: Any) -> bool:
    if type(first) is not type(second):
        return False
    equal = getattr(first, "equal", None)
    if callable(equal):
        try:
            return bool(equal(second))
        except (TypeError, ValueError, RuntimeError):
            return False
    if isinstance(first, Mapping):
        return set(first) == set(second) and all(
            _value_equal(first[key], second[key]) for key in first
        )
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(
            _value_equal(left, right) for left, right in zip(first, second, strict=True)
        )
    if isinstance(first, (set, frozenset)):
        return bool(first == second)
    try:
        comparison = first == second
        all_method = getattr(comparison, "all", None)
        return bool(all_method() if callable(all_method) else comparison)
    except (TypeError, ValueError, RuntimeError):
        return False


def _snapshot_equal(first: CacheSnapshot, second: CacheSnapshot) -> bool:
    return bool(
        first.source_class is second.source_class
        and first.source_kind == second.source_kind
        and first.active_sequence_length == second.active_sequence_length
        and first.cached_key_state == second.cached_key_state
        and len(first.layers) == len(second.layers)
        and all(
            left.sequence_dimension == right.sequence_dimension
            and _value_equal(left.key, right.key)
            and _value_equal(left.value, right.value)
            for left, right in zip(first.layers, second.layers, strict=True)
        )
    )


@dataclass(frozen=True, slots=True)
class DecodeStateCheckpoint:
    """Private value snapshot used to detect reference-state mutation."""

    cache_snapshot: CacheSnapshot
    attention_mask: Any
    active_cache_length: int
    logical_sequence_length: int
    next_decode_position: int
    modality_map: tuple[Any, ...]
    model_state: dict[str, Any]


def capture_decode_state(
    adapter: CacheStateAdapter,
    state: DecodeState,
) -> DecodeStateCheckpoint:
    """Capture an independent checkpoint of one live backend decode state."""

    snapshot = adapter.extract_past_key_values(state.past_key_values, clone=True)
    if not isinstance(snapshot, CacheSnapshot):
        raise CacheIsolationError("adapter extraction did not return a CacheSnapshot")
    return DecodeStateCheckpoint(
        cache_snapshot=clone_cache_snapshot(snapshot),
        attention_mask=_clone_value(state.attention_mask, path="attention_mask"),
        active_cache_length=state.active_cache_length,
        logical_sequence_length=state.logical_sequence_length,
        next_decode_position=state.next_decode_position,
        modality_map=state.modality_map,
        model_state=_clone_value(state.model_state, path="model_state"),
    )


def assert_decode_state_unchanged(
    adapter: CacheStateAdapter,
    state: DecodeState,
    checkpoint: DecodeStateCheckpoint,
) -> None:
    """Fail if any cache, position, mask, modality, or backend metadata changed."""

    current = adapter.extract_past_key_values(state.past_key_values, clone=True)
    if not isinstance(current, CacheSnapshot):
        raise CacheIsolationError("adapter extraction did not return a CacheSnapshot")
    unchanged = bool(
        _snapshot_equal(current, checkpoint.cache_snapshot)
        and _value_equal(state.attention_mask, checkpoint.attention_mask)
        and state.active_cache_length == checkpoint.active_cache_length
        and state.logical_sequence_length == checkpoint.logical_sequence_length
        and state.next_decode_position == checkpoint.next_decode_position
        and state.modality_map == checkpoint.modality_map
        and _value_equal(state.model_state, checkpoint.model_state)
    )
    if not unchanged:
        raise CacheIsolationError("candidate activity mutated the reference decode state")


def _tensor_storage_token(value: Any) -> tuple[str, str, int] | None:
    untyped_storage = getattr(value, "untyped_storage", None)
    if callable(untyped_storage):
        storage = untyped_storage()
        data_ptr = getattr(storage, "data_ptr", None)
        if callable(data_ptr):
            return ("tensor", str(getattr(value, "device", "unknown")), int(data_ptr()))
    array_interface = getattr(value, "__array_interface__", None)
    if isinstance(array_interface, Mapping):
        raw_data = array_interface.get("data")
        if isinstance(raw_data, tuple) and raw_data:
            return ("array", "cpu", int(raw_data[0]))
    return None


def _storage_tokens(value: Any, *, seen: set[int] | None = None) -> set[tuple[str, str, int]]:
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return set()
    seen.add(identity)
    token = _tensor_storage_token(value)
    if token is not None:
        return {token}
    if isinstance(value, CacheSnapshot):
        return {
            token
            for layer in value.layers
            for item in (layer.key, layer.value)
            for token in _storage_tokens(item, seen=seen)
        }
    if isinstance(value, Mapping):
        return {
            token
            for key, item in value.items()
            for nested in (key, item)
            for token in _storage_tokens(nested, seen=seen)
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return {token for item in value for token in _storage_tokens(item, seen=seen)}
    return set()


def _live_cache_values(adapter: CacheStateAdapter, cache: Any) -> Any:
    raw = adapter.extract_past_key_values(cache, clone=False)
    if isinstance(raw, CacheSnapshot):
        return raw
    converter = getattr(raw, "to_legacy_cache", None)
    if callable(converter):
        return tuple(converter())
    if isinstance(raw, (tuple, list)):
        return raw
    layers = getattr(raw, "layers", None)
    if layers is not None:
        return tuple((layer.keys, layer.values) for layer in layers)
    raise CacheIsolationError("live cache representation cannot be audited for storage aliases")


def _state_storage_tokens(
    adapter: CacheStateAdapter,
    state: DecodeState,
) -> tuple[set[tuple[str, str, int]], set[tuple[str, str, int]]]:
    cache_tokens = _storage_tokens(_live_cache_values(adapter, state.past_key_values))
    auxiliary_tokens = _storage_tokens((state.attention_mask, state.model_state))
    return cache_tokens, auxiliary_tokens


@dataclass(frozen=True, slots=True)
class CacheIsolationAudit:
    """Storage-ownership evidence captured when a candidate is constructed."""

    candidate_id: str
    reference_cache_storage_count: int
    candidate_cache_storage_count: int
    reference_auxiliary_storage_count: int
    candidate_auxiliary_storage_count: int
    shared_cache_storage_count: int
    shared_auxiliary_storage_count: int
    reference_unchanged_at_construction: bool

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise CacheIsolationError("candidate_id cannot be empty")
        counts = (
            self.reference_cache_storage_count,
            self.candidate_cache_storage_count,
            self.reference_auxiliary_storage_count,
            self.candidate_auxiliary_storage_count,
            self.shared_cache_storage_count,
            self.shared_auxiliary_storage_count,
        )
        if any(count < 0 for count in counts):
            raise CacheIsolationError("isolation audit counts cannot be negative")
        if self.shared_cache_storage_count or self.shared_auxiliary_storage_count:
            raise CacheIsolationError("candidate and reference states share mutable storage")
        if not self.reference_unchanged_at_construction:
            raise CacheIsolationError("candidate construction changed the reference state")


class IsolatedDecodeStateHandle:
    """Owned disposable state plus a guard for its immutable reference state."""

    def __init__(
        self,
        *,
        candidate_id: str,
        adapter: CacheStateAdapter,
        reference_state: DecodeState,
        reference_checkpoint: DecodeStateCheckpoint,
        state: DecodeState,
        audit: CacheIsolationAudit,
    ) -> None:
        self.candidate_id = candidate_id
        self._adapter = adapter
        self._reference_state = reference_state
        self._reference_checkpoint = reference_checkpoint
        self._state: DecodeState | None = state
        self.audit = audit

    @property
    def closed(self) -> bool:
        return self._state is None

    @property
    def state(self) -> DecodeState:
        if self._state is None:
            raise CacheIsolationError("isolated candidate state has been disposed")
        return self._state

    def assert_reference_unchanged(self) -> None:
        assert_decode_state_unchanged(
            self._adapter,
            self._reference_state,
            self._reference_checkpoint,
        )

    def adopt(self, state: DecodeState) -> None:
        """Adopt a backend-produced successor after rechecking ownership."""

        if self.closed:
            raise CacheIsolationError("cannot adopt into a disposed candidate")
        self.assert_reference_unchanged()
        reference_cache, reference_auxiliary = _state_storage_tokens(
            self._adapter, self._reference_state
        )
        candidate_cache, candidate_auxiliary = _state_storage_tokens(self._adapter, state)
        if reference_cache & candidate_cache or reference_auxiliary & candidate_auxiliary:
            raise CacheIsolationError("successor state aliases mutable reference storage")
        self._state = state

    def close(self) -> None:
        """Verify the reference once more and release the candidate ownership."""

        if self.closed:
            return
        try:
            self.assert_reference_unchanged()
        finally:
            self._state = None

    def __enter__(self) -> IsolatedDecodeStateHandle:
        if self.closed:
            raise CacheIsolationError("cannot enter a disposed candidate")
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exception_type, exception, traceback
        self.close()
        return False


def construct_isolated_decode_state(
    adapter: CacheStateAdapter,
    reference_state: DecodeState,
    *,
    candidate_id: str,
    cache_snapshot: CacheSnapshot | None = None,
    attention_mask: Any | None = None,
    active_cache_length: int | None = None,
    model_state: Mapping[str, Any] | None = None,
) -> IsolatedDecodeStateHandle:
    """Construct an audited candidate without mutating or aliasing its reference.

    Logical sequence length, next decode position, and modality spans always
    come from the reference. A supplied packed cache may change only the active
    physical length and accompanying mask/backend metadata.
    """

    if not candidate_id:
        raise CacheIsolationError("candidate_id cannot be empty")
    checkpoint = capture_decode_state(adapter, reference_state)
    source_snapshot = checkpoint.cache_snapshot if cache_snapshot is None else cache_snapshot
    owned_snapshot = clone_cache_snapshot(source_snapshot)
    candidate_active_length = (
        owned_snapshot.active_sequence_length
        if active_cache_length is None
        else active_cache_length
    )
    if candidate_active_length != owned_snapshot.active_sequence_length:
        raise CacheIsolationError("candidate active length differs from its cache snapshot")
    candidate_attention = _clone_value(
        reference_state.attention_mask if attention_mask is None else attention_mask,
        path="candidate.attention_mask",
    )
    attention_shape = tuple(int(item) for item in getattr(candidate_attention, "shape", ()))
    if not attention_shape or attention_shape[-1] != candidate_active_length:
        raise CacheIsolationError(
            "candidate attention-mask length differs from its active cache length"
        )
    candidate_model_state = _clone_value(
        reference_state.model_state if model_state is None else model_state,
        path="candidate.model_state",
    )
    if not isinstance(candidate_model_state, dict):
        raise CacheIsolationError("candidate model_state must clone to a dictionary")
    candidate = DecodeState(
        past_key_values=adapter.inject_past_key_values(owned_snapshot),
        attention_mask=candidate_attention,
        active_cache_length=candidate_active_length,
        logical_sequence_length=reference_state.logical_sequence_length,
        next_decode_position=reference_state.next_decode_position,
        modality_map=reference_state.modality_map,
        model_state=candidate_model_state,
    )
    assert_decode_state_unchanged(adapter, reference_state, checkpoint)
    reference_cache, reference_auxiliary = _state_storage_tokens(adapter, reference_state)
    candidate_cache, candidate_auxiliary = _state_storage_tokens(adapter, candidate)
    audit = CacheIsolationAudit(
        candidate_id=candidate_id,
        reference_cache_storage_count=len(reference_cache),
        candidate_cache_storage_count=len(candidate_cache),
        reference_auxiliary_storage_count=len(reference_auxiliary),
        candidate_auxiliary_storage_count=len(candidate_auxiliary),
        shared_cache_storage_count=len(reference_cache & candidate_cache),
        shared_auxiliary_storage_count=len(reference_auxiliary & candidate_auxiliary),
        reference_unchanged_at_construction=True,
    )
    return IsolatedDecodeStateHandle(
        candidate_id=candidate_id,
        adapter=adapter,
        reference_state=reference_state,
        reference_checkpoint=checkpoint,
        state=candidate,
        audit=audit,
    )


__all__ = [
    "CacheIsolationAudit",
    "CacheIsolationError",
    "CacheStateAdapter",
    "DecodeStateCheckpoint",
    "IsolatedDecodeStateHandle",
    "assert_decode_state_unchanged",
    "capture_decode_state",
    "clone_cache_snapshot",
    "construct_isolated_decode_state",
]
