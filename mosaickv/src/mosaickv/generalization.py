"""Scientific controls for bounded hyperparameter-generalization studies."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class GeneralizationProtocolError(ValueError):
    """Raised when calibration, freezing, or held-out evaluation is invalid."""


@dataclass(frozen=True, slots=True)
class CalibrationCandidate:
    """Predeclared candidate and its measured calibration metrics."""

    candidate_id: str
    mean_task_score: float
    mean_token_agreement: float
    mean_selected_block_recall: float
    mean_forecast_regret: float
    mean_end_to_end_seconds: float

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise GeneralizationProtocolError("candidate_id must be non-empty")
        values = (
            self.mean_task_score,
            self.mean_token_agreement,
            self.mean_selected_block_recall,
            self.mean_forecast_regret,
            self.mean_end_to_end_seconds,
        )
        if not all(math.isfinite(value) for value in values):
            raise GeneralizationProtocolError("calibration metrics must be finite")
        if not 0 <= self.mean_token_agreement <= 1:
            raise GeneralizationProtocolError("mean_token_agreement must be in [0, 1]")
        if not 0 <= self.mean_selected_block_recall <= 1:
            raise GeneralizationProtocolError("mean_selected_block_recall must be in [0, 1]")
        if self.mean_forecast_regret < 0 or self.mean_end_to_end_seconds < 0:
            raise GeneralizationProtocolError("regret and latency must be nonnegative")


def select_calibration_winner(
    candidates: tuple[CalibrationCandidate, ...],
) -> CalibrationCandidate:
    """Select using a fully predeclared deterministic lexicographic criterion.

    Task quality is primary. FullKV token agreement, selected-block recall,
    forecast regret, and latency are successive tie-breakers. Candidate ID is
    the final stable tie-breaker and is never outcome-dependent.
    """

    if not candidates:
        raise GeneralizationProtocolError("at least one calibration candidate is required")
    identifiers = [candidate.candidate_id for candidate in candidates]
    if len(identifiers) != len(set(identifiers)):
        raise GeneralizationProtocolError("calibration candidate IDs must be unique")
    return min(
        candidates,
        key=lambda item: (
            -item.mean_task_score,
            -item.mean_token_agreement,
            -item.mean_selected_block_recall,
            item.mean_forecast_regret,
            item.mean_end_to_end_seconds,
            item.candidate_id,
        ),
    )


def assert_disjoint_sample_sets(
    calibration_ids: tuple[str, ...],
    evaluation_ids_by_task: dict[str, tuple[str, ...]],
) -> None:
    """Reject calibration leakage and duplicate identities before evaluation."""

    if not calibration_ids or len(calibration_ids) != len(set(calibration_ids)):
        raise GeneralizationProtocolError("calibration sample IDs must be non-empty and unique")
    calibration = set(calibration_ids)
    for task, identities in evaluation_ids_by_task.items():
        if not task.strip() or not identities or len(identities) != len(set(identities)):
            raise GeneralizationProtocolError(
                f"evaluation sample IDs for {task!r} must be non-empty and unique"
            )
        overlap = calibration.intersection(identities)
        if overlap:
            raise GeneralizationProtocolError(
                f"calibration/evaluation leakage for {task}: {sorted(overlap)}"
            )


def canonical_hyperparameters(value: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the frozen, architecture-neutral parameter set."""

    required = {
        "retention_ratio",
        "block_size",
        "forecast_mode",
        "prompt_window",
        "draft_steps",
        "centroid_count",
        "graph_max_neighbors",
        "minimum_stratum_fraction",
        "prototype_group_size",
        "prototype_max_position_span",
        "repair_policy",
        "repair_entropy_threshold",
        "repair_prototype_risk_threshold",
        "repair_prototype_risk_min_entropy",
        "repair_max_blocks_per_step",
        "repair_candidate_ranking",
        "preserve_partial_prototype",
    }
    unknown = sorted(set(value) - required)
    missing = sorted(required - set(value))
    if missing or unknown:
        raise GeneralizationProtocolError(
            f"frozen hyperparameter fields differ: missing={missing}, unknown={unknown}"
        )
    result = {key: value[key] for key in sorted(required)}
    if not 0 < float(result["retention_ratio"]) <= 1:
        raise GeneralizationProtocolError("retention_ratio must be in (0, 1]")
    if int(result["block_size"]) < 1 or int(result["draft_steps"]) < 0:
        raise GeneralizationProtocolError("block_size must be positive and draft_steps nonnegative")
    if not isinstance(result["preserve_partial_prototype"], bool):
        raise GeneralizationProtocolError("preserve_partial_prototype must be boolean")
    return result


def is_completed_forecast_diagnostic(value: Mapping[str, Any]) -> bool:
    """Recognize the runtime's completed evaluation-only forecast diagnostic.

    Unlike fidelity diagnostics, forecast diagnostics do not currently emit a
    redundant ``status`` field.  Completion is therefore defined by the
    required finite metrics plus the isolation marker that proves the oracle
    output was not supplied to the online compressor.
    """

    required_metrics = (
        "mean_cosine_similarity",
        "mean_attention_rank_correlation",
        "mean_selected_block_recall",
        "mean_forecast_regret",
    )
    try:
        metrics = tuple(float(value[field]) for field in required_metrics)
    except (KeyError, TypeError, ValueError):
        return False
    return (
        all(math.isfinite(metric) for metric in metrics)
        and value.get("online_selection_isolation")
        == "evaluation_only_output_not_passed_to_compressor"
    )


__all__ = [
    "CalibrationCandidate",
    "GeneralizationProtocolError",
    "assert_disjoint_sample_sets",
    "canonical_hyperparameters",
    "is_completed_forecast_diagnostic",
    "select_calibration_winner",
]
