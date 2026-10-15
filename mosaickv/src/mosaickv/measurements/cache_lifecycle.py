"""Metadata-only accounting of the live cache, never process GPU memory.

Payload includes physical padded slots. Storage additionally includes unused
capacity behind tensor views, counted once per allocation. Neither includes
inactive source caches, allocator reservations, model weights or planner scratch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CacheLifecycle:
    initial_payload_bytes: int
    initial_storage_bytes: int
    final_payload_bytes: int
    final_storage_bytes: int
    peak_payload_bytes: int
    peak_storage_bytes: int
    observations: int = 1

    @classmethod
    def start(cls, layers: Any) -> CacheLifecycle:
        payload, storage = cls.count(layers)
        return cls(payload, storage, payload, storage, payload, storage)

    @staticmethod
    def count(layers: Any) -> tuple[int, int]:
        payload = 0
        allocations: dict[tuple[str, int], int] = {}
        for key, value in layers:
            for tensor in (key, value):
                payload += int(tensor.numel()) * int(tensor.element_size())
                storage = tensor.untyped_storage()
                identity = (str(tensor.device), int(storage.data_ptr()))
                allocations[identity] = int(storage.nbytes())
        if payload <= 0:
            raise ValueError("live cache must contain nonempty K/V tensors")
        return payload, sum(allocations.values())

    def observe(self, layers: Any) -> None:
        self.final_payload_bytes, self.final_storage_bytes = self.count(layers)
        self.peak_payload_bytes = max(self.peak_payload_bytes, self.final_payload_bytes)
        self.peak_storage_bytes = max(self.peak_storage_bytes, self.final_storage_bytes)
        self.observations += 1

    def record(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "boundary": "live_decode_cache_after_compression",
            "measurement": "tensor_metadata_at_all_decode_transaction_boundaries",
            "initial_active_kv_bytes": self.initial_payload_bytes,
            "final_active_kv_bytes": self.final_payload_bytes,
            "peak_active_kv_bytes": self.peak_payload_bytes,
            "initial_active_kv_storage_bytes": self.initial_storage_bytes,
            "final_active_kv_storage_bytes": self.final_storage_bytes,
            "peak_active_kv_storage_bytes": self.peak_storage_bytes,
            "observations": self.observations,
            "excludes": "weights,inactive_source,planner_scratch,allocator_reserved",
            "not_a_process_peak_memory_measurement": True,
        }
