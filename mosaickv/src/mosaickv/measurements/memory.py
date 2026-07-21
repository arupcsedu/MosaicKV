"""Explicit tensor-payload byte accounting for cache and residual state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def tensor_payload_bytes(tensor: Any) -> int:
    """Return ``numel * element_size`` and reject non-tensor-like objects."""

    numel = getattr(tensor, "numel", None)
    element_size = getattr(tensor, "element_size", None)
    if numel is None or element_size is None:
        raise TypeError(f"object is not tensor-like: {type(tensor).__qualname__}")
    value = int(numel()) * int(element_size())
    if value < 0:
        raise RuntimeError("tensor payload byte count cannot be negative")
    return value


def cache_tensors(cache: Any) -> tuple[Any, ...]:
    """Return K/V tensors without cloning or changing the cache object."""

    converter = getattr(cache, "to_legacy_cache", None)
    if converter is not None:
        layers = tuple(converter())
    elif isinstance(cache, (tuple, list)):
        layers = tuple(cache)
    elif getattr(cache, "layers", None) is not None:
        layers = tuple((layer.keys, layer.values) for layer in cache.layers)
    else:
        raise TypeError(f"unsupported cache type for byte accounting: {type(cache).__qualname__}")
    tensors: list[Any] = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, (tuple, list)) or len(layer) != 2:
            raise TypeError(f"cache layer {index} is not a K/V pair")
        tensors.extend((layer[0], layer[1]))
    if not tensors:
        raise ValueError("cache contains no K/V tensors")
    return tuple(tensors)


def active_kv_bytes(cache: Any) -> int:
    """Sum exact logical tensor payload bytes for every active K and V."""

    return sum(tensor_payload_bytes(tensor) for tensor in cache_tensors(cache))


def cpu_residual_bytes(value: Any) -> int:
    """Recursively sum unique CPU tensor payloads in residual state."""

    seen: set[int] = set()

    def visit(item: Any) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        if hasattr(item, "numel") and hasattr(item, "element_size"):
            device = getattr(item, "device", None)
            device_type = getattr(device, "type", str(device))
            return tensor_payload_bytes(item) if device_type == "cpu" else 0
        if isinstance(item, Mapping):
            return sum(visit(key) + visit(child) for key, child in item.items())
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            return sum(visit(child) for child in item)
        return 0

    return visit(value)


__all__ = ["active_kv_bytes", "cache_tensors", "cpu_residual_bytes", "tensor_payload_bytes"]
