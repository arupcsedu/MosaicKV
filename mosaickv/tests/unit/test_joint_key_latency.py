import math

import evicastkv_joint_key_latency as control
import pytest


def fixture(candidate_time):
    rows, audits = [], {}
    for method, arm in control.ARMS:
        audits[method, arm] = dict(
            zip(
                control.driver.STORAGE,
                [90, 110, 120] if method == "mosaickv_exact" else [100, 130, 150],
                strict=True,
            )
        )
        for trial in range(-5, 3):
            measured = dict(
                total_wall_seconds=candidate_time if arm == "joint-keys" else 10.0,
                preprocessing_wall_seconds=0.01,
                prefill_construct_retire_wall_seconds=1.0,
                decode_wall_seconds=8.0,
                logits_bitwise=True,
                no_FullKV_fallback=True,
            )
            rows.append(
                dict(method=method, arm=arm, trial=trial, warmup=trial < 0, measurement=measured)
            )
    return rows, audits


@pytest.mark.parametrize("time,expected", [(9.5, True), (9.7, False), (10.1, False)])
def test_screen_does_not_turn_a_point_estimate_into_formal_acceptance(time, expected):
    result = control.summary(*fixture(time))
    assert result["four_percent_POINT_only"] is expected
    assert not result["formal_latency_acceptance"]
    assert result["independent_quality_N"] == 0


def test_higher_peak_cannot_be_hidden_behind_lower_persistent_storage():
    rows, audits = fixture(9.5)
    audits["mosaickv_exact", "joint-keys"][control.driver.STORAGE[-1]] = 151
    result = control.summary(rows, audits)
    assert not result["smaller_at_all_three_decode_owned_boundaries"]
    assert not result["four_percent_POINT_only"]


@pytest.mark.parametrize("kind", ["missing", "duplicate", "nan", "failed", "warmup"])
def test_missing_or_invalid_trials_are_rejected(kind):
    rows, audits = fixture(9.5)
    if kind == "missing":
        rows.pop()
    elif kind == "duplicate":
        rows.append(rows[0])
    elif kind == "nan":
        rows[0]["measurement"]["total_wall_seconds"] = math.nan
    elif kind == "failed":
        rows[0]["measurement"]["logits_bitwise"] = False
    else:
        rows[0]["warmup"] = False
    with pytest.raises(ValueError):
        control.summary(rows, audits)


def test_a_timing_row_does_not_acquire_an_audited_peak_flag():
    from test_joint_key_controller import measured

    data = measured(True)
    data["peak_measurement_admissible"] = False
    control.correctness.validate(data, True, require_audited=False)
    assert not data["peak_measurement_admissible"]
    with pytest.raises(ValueError, match="audited"):
        control.correctness.validate(data, True)
