from __future__ import annotations

import pytest

from mosaickv.repair import RepairCalibrationObservation, calibrate_decision_repair


def test_repair_calibration_respects_trigger_budget_and_selects_beneficial_steps() -> None:
    observations = tuple(
        RepairCalibrationObservation(
            sample_id=f"cal-{index // 5}",
            step=index,
            normalized_prototype_risk=(0.95 if index in {2, 7} else 0.1),
            next_token_probability_margin=(0.02 if index in {2, 7} else 0.9),
            oracle_beneficial=index in {2, 7},
        )
        for index in range(20)
    )
    profile = calibrate_decision_repair(
        observations,
        evaluation_sample_ids=("eval-1",),
        target_trigger_rate=0.1,
        minimum_precision=1.0,
    )
    assert profile.enabled
    assert profile.predicted_triggered_repairs == 2
    assert profile.predicted_precision == pytest.approx(1.0)
    assert profile.predicted_recall == pytest.approx(1.0)


def test_repair_calibration_rejects_evaluation_leakage() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        calibrate_decision_repair(
            (
                RepairCalibrationObservation("same", 0, 0.5, 0.1, True),
            ),
            evaluation_sample_ids=("same",),
        )
