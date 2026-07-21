"""Uncertainty-guided decode-time residual repair."""

from mosaickv.repair.core import (
    ReDecodeCallback,
    calculate_repair_signals,
    draft_kl_divergence,
    normalized_next_token_entropy,
    repair_decode_step,
)
from mosaickv.repair.types import (
    RepairCacheState,
    RepairEvent,
    RepairStepResult,
    RepairStepSignals,
    RepairTriggerReason,
)

__all__ = [
    "ReDecodeCallback",
    "RepairCacheState",
    "RepairEvent",
    "RepairStepResult",
    "RepairStepSignals",
    "RepairTriggerReason",
    "calculate_repair_signals",
    "draft_kl_divergence",
    "normalized_next_token_entropy",
    "repair_decode_step",
]
