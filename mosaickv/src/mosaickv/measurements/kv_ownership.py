"""Explicit KV ownership accounting, including transient replacement buffers.

Not a process/GPU-memory profiler. Only supplied K/V owners are counted; the
record states its scope. Keeping weak references must not extend KV lifetime.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any


@dataclass
class KVOwnershipLedger:
    peak_storage_bytes: int = 0
    observations: int = 0
    peak_roles: dict[str, int] = field(default_factory=dict)
    _owners: dict[str, tuple[weakref.ReferenceType[Any], ...]] = field(
        default_factory=dict, repr=False
    )

    def track(self, role: str, layers: Any) -> None:
        self._owners[role] = tuple(weakref.ref(t) for pair in layers for t in pair)

    def observe(self, **temporary: Any) -> int:
        roles = {
            role: tuple(t for ref in refs if (t := ref()) is not None)
            for role, refs in self._owners.items()
        }
        roles.update(
            {role: tuple(t for pair in pairs for t in pair) for role, pairs in temporary.items()}
        )
        allocations = {}
        role_bytes = {}
        for role, tensors in roles.items():
            local = {}
            for tensor in tensors:
                storage = tensor.untyped_storage()
                identity = (str(tensor.device), storage.data_ptr())
                local[identity] = storage.nbytes()
            role_bytes[role] = sum(local.values())
            allocations.update(local)
        total = sum(allocations.values())
        self.observations += 1
        if total > self.peak_storage_bytes:
            self.peak_storage_bytes = total
            self.peak_roles = role_bytes
        return total

    def record(self) -> dict[str, Any]:
        return {
            "boundary": "explicitly_registered_KV_owners_and_reallocation_transactions",
            "peak_storage_bytes": self.peak_storage_bytes,
            "peak_roles_bytes_not_additive_when_aliased": self.peak_roles,
            "observations": self.observations,
            "includes_registered_inactive_source_and_transient_KV": True,
            "not_process_peak_memory": True,
            "not_all_prefill_internal_transients_instrumented": True,
        }
