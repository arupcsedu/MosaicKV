"""Residual CPU storage, indexing, and restoration."""

from mosaickv.cache_state import ResidualTier
from mosaickv.residual.storage import (
    PinnedMemoryUnavailableError,
    ResidualStorageError,
    build_residual_storage,
    discard_residual_payloads,
    empty_residual_storage,
    restore_residual_payload,
    restore_residual_payloads_async,
)
from mosaickv.residual.types import (
    ResidualIndexEntry,
    ResidualPayloadMetadata,
    ResidualStorageReport,
    ResidualTransferBatch,
)

__all__ = [
    "PinnedMemoryUnavailableError",
    "ResidualIndexEntry",
    "ResidualPayloadMetadata",
    "ResidualStorageError",
    "ResidualStorageReport",
    "ResidualTier",
    "ResidualTransferBatch",
    "build_residual_storage",
    "discard_residual_payloads",
    "empty_residual_storage",
    "restore_residual_payload",
    "restore_residual_payloads_async",
]
