"""Append-only Hugging Face cache storage without per-token reallocation."""

from __future__ import annotations

from typing import Any, cast


class PreallocatedDynamicLayer:
    """Dynamic-cache-compatible layer backed by one fixed allocation.

    Unlike Transformers' ``DynamicLayer``, ``update`` copies new K/V values
    into reserved storage and returns a view of the occupied prefix.  Logical
    RoPE positions remain model inputs and are deliberately not used as
    physical storage indices.
    """

    is_sliding = False
    is_compileable = False

    def __init__(self, keys: Any, values: Any, additional_positions: int) -> None:
        if additional_positions < 1:
            raise ValueError("additional_positions must be positive")
        if keys.ndim < 3 or values.ndim != keys.ndim:
            raise ValueError("preallocated K/V tensors must have matching rank >= 3")
        if tuple(keys.shape) != tuple(values.shape):
            raise ValueError("preallocated K/V tensors must have matching shapes")
        self._sequence_dimension = keys.ndim - 2
        self._length = int(keys.shape[self._sequence_dimension])
        self._capacity = self._length + additional_positions
        shape = list(keys.shape)
        shape[self._sequence_dimension] = self._capacity
        self._key_storage = keys.new_empty(shape)
        self._value_storage = values.new_empty(shape)
        self._prefix(self._key_storage, self._length).copy_(keys)
        self._prefix(self._value_storage, self._length).copy_(values)
        self.keys = self._prefix(self._key_storage, self._length)
        self.values = self._prefix(self._value_storage, self._length)

    def _prefix(self, tensor: Any, length: int) -> Any:
        return tensor.narrow(self._sequence_dimension, 0, length)

    def update(
        self,
        key_states: Any,
        value_states: Any,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        del cache_kwargs
        count = int(key_states.shape[self._sequence_dimension])
        if tuple(key_states.shape) != tuple(value_states.shape):
            raise ValueError("decode K/V updates must have matching shapes")
        end = self._length + count
        if end > self._capacity:
            raise RuntimeError(
                "preallocated decode cache capacity exceeded: "
                f"required={end}, capacity={self._capacity}"
            )
        self._key_storage.narrow(self._sequence_dimension, self._length, count).copy_(key_states)
        self._value_storage.narrow(self._sequence_dimension, self._length, count).copy_(
            value_states
        )
        self._length = end
        self.keys = self._prefix(self._key_storage, self._length)
        self.values = self._prefix(self._value_storage, self._length)
        return self.keys, self.values

    def get_seq_length(self, cache_position: Any | None = None) -> int:
        del cache_position
        return self._length

    def get_max_cache_shape(self) -> int:
        # Preserve DynamicCache mask semantics: only the occupied prefix is
        # visible to attention even though storage is reserved beyond it.
        return -1

    def get_mask_sizes(self, cache_position: Any) -> tuple[int, int]:
        return self._length + int(cache_position.shape[0]), 0

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self._length - abs(max_length)
        self._length = max(0, min(self._length, max_length))
        self.keys = self._prefix(self._key_storage, self._length)
        self.values = self._prefix(self._value_storage, self._length)

    def reorder_cache(self, beam_idx: Any) -> None:
        if int(beam_idx.numel()) != int(self._key_storage.shape[0]):
            raise NotImplementedError("preallocated cache does not support batch-size changes")
        self._key_storage = self._key_storage.index_select(0, beam_idx.to(self.keys.device))
        self._value_storage = self._value_storage.index_select(0, beam_idx.to(self.values.device))
        self.keys = self._prefix(self._key_storage, self._length)
        self.values = self._prefix(self._value_storage, self._length)


def build_preallocated_dynamic_cache(
    layers: tuple[tuple[Any, Any], ...],
    additional_positions: int,
) -> Any:
    """Return a Transformers ``DynamicCache`` with preallocated layer storage."""

    if not layers:
        raise ValueError("cannot preallocate an empty cache")
    try:
        from transformers import DynamicCache
    except ImportError as error:  # pragma: no cover - optional HF environment
        raise RuntimeError("preallocated cache requires Transformers") from error
    cache = DynamicCache()
    cache.layers = cast(
        "Any",
        [PreallocatedDynamicLayer(keys, values, additional_positions) for keys, values in layers],
    )
    if len({int(key.shape[-2]) for key, _value in layers}) != 1:
        cache._mosaickv_independent_layer_lengths = True
    return cache


__all__ = ["PreallocatedDynamicLayer", "build_preallocated_dynamic_cache"]
