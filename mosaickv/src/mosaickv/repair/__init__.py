"""Uncertainty-guided decode-time residual repair."""

from mosaickv.repair.calibration import (
    CalibratedRepairPolicy,
    RepairCalibrationObservation,
    calibrate_decision_repair,
)
from mosaickv.repair.core import (
    ReDecodeCallback,
    calculate_repair_signals,
    draft_kl_divergence,
    normalized_next_token_entropy,
    repair_decode_step,
    stage_gpu_rescue_payloads,
)
from mosaickv.repair.oracle import evaluate_repair_decode_step
from mosaickv.repair.types import (
    GPURescuePayload,
    RepairCacheState,
    RepairEvent,
    RepairStepResult,
    RepairStepSignals,
    RepairTriggerReason,
)

__all__ = [
    "CalibratedRepairPolicy",
    "GPURescuePayload",
    "ReDecodeCallback",
    "RepairCacheState",
    "RepairCalibrationObservation",
    "RepairEvent",
    "RepairStepResult",
    "RepairStepSignals",
    "RepairTriggerReason",
    "calculate_repair_signals",
    "calibrate_decision_repair",
    "draft_kl_divergence",
    "evaluate_repair_decode_step",
    "normalized_next_token_entropy",
    "repair_decode_step",
    "stage_gpu_rescue_payloads",
]
