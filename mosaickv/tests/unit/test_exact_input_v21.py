"""The funded numerical repair cannot be promoted through bookkeeping alone."""

import copy

import evicastkv_exact_input_v21 as study
import pytest


def rows():
    return [
        dict(
            arm=arm,
            measured=dict(
                storage=dict(
                    initial_active_kv_storage_bytes=study.CAP,
                    final_active_kv_storage_bytes=study.CAP + 64 * 2**20,
                    peak_active_kv_storage_bytes=study.CAP + 80 * 2**20,
                    history_updates_per_layer=[127] * 32,
                    outstanding_layer_leases=0,
                ),
                used_fullkv_fallback=False,
            ),
            parent_transport_parity=dict(pass_all128_bitwise=True),
            FullKV_quality=dict(
                registered_guard_32=dict(registered_guard_pass=True),
                prefix_128=dict(registered_guard_pass=True),
            ),
            added_original_input_rows=8 if arm == study.ARMS[2] else 0,
        )
        for arm in study.ARMS
    ]


def test_successful_engineering_diagnostic_is_not_paper_admission():
    result = study.outcome(rows())
    assert result["pass_bounded_numerical_repair"]
    assert not result["benchmark_quality_or_superiority_claim"]
    assert not result["independent_confirmation_or_campaign_authorized"]
    assert not result["whole_request_KV_peak_certified"]


def test_complete_later_prefix_required_not_just_early_guard():
    value = rows()
    value[2]["FullKV_quality"]["prefix_128"]["registered_guard_pass"] = False
    assert not study.outcome(value)["pass_bounded_numerical_repair"]
    value = rows()
    value[1]["parent_transport_parity"]["pass_all128_bitwise"] = False
    assert not study.outcome(value)["pass_bounded_numerical_repair"]


def test_no_cap_relaxation_or_noop_can_pass():
    for modify in ("cap", "noop", "guard"):
        value = copy.deepcopy(rows())
        if modify == "cap":
            value[2]["measured"]["storage"]["initial_active_kv_storage_bytes"] += 1
        elif modify == "noop":
            value[2]["added_original_input_rows"] = 0
        else:
            value[2]["FullKV_quality"]["registered_guard_32"]["registered_guard_pass"] = False
        assert not study.outcome(value)["pass_bounded_numerical_repair"]
    with pytest.raises(ValueError, match="three"):
        study.outcome(rows()[:2])


def test_failed_prepare_always_receives_finite_failure_receipt(tmp_path):
    result = study.analyze(tmp_path)
    assert not result["complete"]
    assert result["failures"]
    assert not result["calibration_or_paper_campaign_authorized"]


@pytest.mark.parametrize("invalid", [0, -1, True, float("nan")])
def test_invalid_measured_storage_cannot_pass(invalid):
    value = rows()
    value[2]["measured"]["storage"]["initial_active_kv_storage_bytes"] = invalid
    assert not study.outcome(value)["pass_bounded_numerical_repair"]
