"""MosaicKV research infrastructure and implemented core components.

The backend-independent three-tier constructor and residual-repair controller
are implemented conservatively and connected through an eager Hugging Face
runtime. Post-RoPE model families fail closed to exact-only cache selection.
"""

from importlib.metadata import PackageNotFoundError, version

from mosaickv.cache_state import (
    CompressionStatistics,
    ExactTier,
    FullKVState,
    KVBlockDescriptor,
    KVLayerStorage,
    LogicalPositionMap,
    MediaMetadata,
    Modality,
    ModalitySpan,
    MosaicKVState,
    PrototypeTier,
    ResidualTier,
    tensor_storage_bytes,
)
from mosaickv.prototypes import (
    ThreeTierCacheConstruction,
    TierConstructionMode,
    assess_prototype_safety,
    construct_three_tier_cache,
)
from mosaickv.repair import RepairCacheState, RepairEvent, repair_decode_step

try:
    __version__ = version("mosaickv")
except PackageNotFoundError:  # pragma: no cover - source-tree import fallback
    __version__ = "0+unknown"

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
    "RepairCacheState",
    "RepairEvent",
    "ResidualTier",
    "ThreeTierCacheConstruction",
    "TierConstructionMode",
    "__version__",
    "assess_prototype_safety",
    "construct_three_tier_cache",
    "repair_decode_step",
    "tensor_storage_bytes",
]
