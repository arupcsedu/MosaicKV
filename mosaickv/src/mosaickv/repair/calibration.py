"""Label-free calibration of bounded decision-aware residual repair."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RepairCalibrationObservation:
    """One calibration-only provisional step labeled against FullKV behavior."""

    sample_id: str
    step: int
    normalized_prototype_risk: float
    next_token_probability_margin: float
    oracle_beneficial: bool
    draft_kl_divergence: float | None = None

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or self.step < 0:
            raise ValueError("repair calibration identity is invalid")
        if not 0 <= self.normalized_prototype_risk <= 1:
            raise ValueError("normalized prototype risk must lie in [0, 1]")
        if not 0 <= self.next_token_probability_margin <= 1:
            raise ValueError("next-token probability margin must lie in [0, 1]")
        if self.draft_kl_divergence is not None and (
            not math.isfinite(self.draft_kl_divergence)
            or self.draft_kl_divergence < 0
        ):
            raise ValueError("draft KL divergence must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class CalibratedRepairPolicy:
    """Frozen online thresholds and their FullKV-fidelity calibration evidence."""

    prototype_risk_threshold: float
    decision_margin_threshold: float
    decision_high_risk_threshold: float
    draft_kl_threshold: float | None
    enabled: bool
    target_trigger_rate: float
    observed_trigger_rate: float
    oracle_beneficial_steps: int
    predicted_beneficial_repairs: int
    predicted_triggered_repairs: int
    predicted_precision: float
    predicted_recall: float
    calibration_sample_ids: tuple[str, ...]
    evaluation_sample_ids: tuple[str, ...]
    calibration_rule: str = (
        "fullkv_token_counterfactual_with_prefix_safe_draft_kl_without_task_labels_v2"
    )

    def __post_init__(self) -> None:
        if set(self.calibration_sample_ids) & set(self.evaluation_sample_ids):
            raise ValueError("repair calibration and evaluation ledgers must be disjoint")
        if not self.calibration_sample_ids or not self.evaluation_sample_ids:
            raise ValueError("repair calibration provenance requires both ledgers")
        for name in (
            "prototype_risk_threshold",
            "decision_margin_threshold",
            "decision_high_risk_threshold",
            "target_trigger_rate",
            "observed_trigger_rate",
            "predicted_precision",
            "predicted_recall",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"calibrated repair {name} must lie in [0, 1]")
        if self.draft_kl_threshold is not None and (
            not math.isfinite(self.draft_kl_threshold) or self.draft_kl_threshold < 0
        ):
            raise ValueError("calibrated repair draft_kl_threshold must be finite and >= 0")

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {**asdict(self), "profile_sha256": self.sha256},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


def _quantiles(values: tuple[float, ...]) -> tuple[float, ...]:
    ordered = tuple(sorted(values))
    result = {0.0, 1.0}
    for numerator in range(1, 10):
        offset = numerator * (len(ordered) - 1) / 10
        lower = math.floor(offset)
        upper = math.ceil(offset)
        fraction = offset - lower
        result.add(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)
    return tuple(sorted(result))


def calibrate_decision_repair(
    observations: tuple[RepairCalibrationObservation, ...],
    *,
    evaluation_sample_ids: tuple[str, ...],
    target_trigger_rate: float = 0.1,
    minimum_precision: float = 0.5,
) -> CalibratedRepairPolicy:
    """Choose a high-risk OR low-margin/risk gate under a fixed trigger budget."""

    if not observations:
        raise ValueError("repair calibration requires observations")
    if not 0 < target_trigger_rate <= 1:
        raise ValueError("target_trigger_rate must lie in (0, 1]")
    if not 0 <= minimum_precision <= 1:
        raise ValueError("minimum_precision must lie in [0, 1]")
    calibration_ids = tuple(sorted({row.sample_id for row in observations}))
    if set(calibration_ids) & set(evaluation_sample_ids):
        raise ValueError("repair calibration and evaluation sample IDs must be disjoint")
    risks = tuple(row.normalized_prototype_risk for row in observations)
    margins = tuple(row.next_token_probability_margin for row in observations)
    draft_kls = tuple(
        row.draft_kl_divergence
        for row in observations
        if row.draft_kl_divergence is not None
    )
    kl_thresholds: tuple[float | None, ...] = (
        (None, *_quantiles(draft_kls)) if draft_kls else (None,)
    )
    beneficial = sum(row.oracle_beneficial for row in observations)
    maximum_triggers = max(1, math.floor(len(observations) * target_trigger_rate))
    candidates: list[
        tuple[float, float, int, float, float, float, int, float, int, int]
    ] = []
    for moderate in _quantiles(risks):
        for high in _quantiles(risks):
            if high < moderate:
                continue
            for margin in _quantiles(margins):
                for kl_index, kl_threshold in enumerate(kl_thresholds):
                    triggered = tuple(
                        row
                        for row in observations
                        if row.normalized_prototype_risk >= high
                        or (
                            row.normalized_prototype_risk >= moderate
                            and (
                                row.next_token_probability_margin <= margin
                                or (
                                    kl_threshold is not None
                                    and row.draft_kl_divergence is not None
                                    and row.draft_kl_divergence >= kl_threshold
                                )
                            )
                        )
                    )
                    if not triggered or len(triggered) > maximum_triggers:
                        continue
                    true_positive = sum(row.oracle_beneficial for row in triggered)
                    precision = true_positive / len(triggered)
                    recall = true_positive / beneficial if beneficial else 0.0
                    if beneficial and precision < minimum_precision:
                        continue
                    candidates.append(
                        (
                            -recall,
                            -precision,
                            len(triggered),
                            moderate,
                            margin,
                            high,
                            kl_index,
                            0.0 if kl_threshold is None else kl_threshold,
                            true_positive,
                            len(triggered),
                        )
                    )
    if candidates:
        (
            _negative_recall,
            _negative_precision,
            _trigger_count,
            moderate,
            margin,
            high,
            kl_index,
            kl_value,
            true_positive,
            triggered_count,
        ) = min(candidates)
        kl_threshold = None if kl_index == 0 else kl_value
        enabled = True
    else:
        # A zero-trigger frozen policy is preferable to uncalibrated repair.
        moderate = high = 1.0
        margin = 0.0
        kl_threshold = None
        true_positive = triggered_count = 0
        enabled = False
    precision = true_positive / triggered_count if triggered_count else 0.0
    recall = true_positive / beneficial if beneficial else 0.0
    return CalibratedRepairPolicy(
        prototype_risk_threshold=moderate,
        decision_margin_threshold=margin,
        decision_high_risk_threshold=high,
        draft_kl_threshold=kl_threshold,
        enabled=enabled,
        target_trigger_rate=target_trigger_rate,
        observed_trigger_rate=triggered_count / len(observations),
        oracle_beneficial_steps=beneficial,
        predicted_beneficial_repairs=true_positive,
        predicted_triggered_repairs=triggered_count,
        predicted_precision=precision,
        predicted_recall=recall,
        calibration_sample_ids=calibration_ids,
        evaluation_sample_ids=tuple(sorted(evaluation_sample_ids)),
    )


__all__ = [
    "CalibratedRepairPolicy",
    "RepairCalibrationObservation",
    "calibrate_decision_repair",
]
