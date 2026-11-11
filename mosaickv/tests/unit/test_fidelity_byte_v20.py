"""V20 engineering gates must not erase the distinct FullKV quality failure."""

from copy import deepcopy

import evicastkv_fidelity_byte_v20 as study
import pytest
import torch


def measured(initial=100, final=200, peak=220):
    return dict(
        storage=dict(
            initial_active_kv_storage_bytes=initial,
            final_active_kv_storage_bytes=final,
            peak_active_kv_storage_bytes=peak,
            history_updates_per_layer=[127, 127],
            outstanding_layer_leases=0,
        )
    )


def test_gate_requires_realized_final_and_peak_not_configured_initial_only():
    left = measured()
    assert study.storage_pair(left, measured())["pass_constructed_decode_boundary"]
    for kwargs in (dict(final=201), dict(peak=221), dict(initial=101)):
        result = study.storage_pair(left, measured(**kwargs))
        assert not result["pass_constructed_decode_boundary"]
        assert result["whole_request_KV_peak_certified"] is False


def test_leaks_missing_history_and_invalid_accounting_fail_closed():
    for field, value in (
        ("history_updates_per_layer", [126, 127]),
        ("outstanding_layer_leases", 1),
    ):
        bad = deepcopy(measured())
        bad["storage"][field] = value
        assert not study.storage_pair(measured(), bad)["pass_constructed_decode_boundary"]
    for bad in (None, float("nan"), 0, True):
        broken = measured()
        broken["storage"]["peak_active_kv_storage_bytes"] = bad
        with pytest.raises(ValueError, match="positive measured"):
            study.storage_pair(measured(), broken)


def test_native_transport_parity_is_not_a_fullkv_quality_claim():
    original = torch.zeros(128, 5)
    original[:, 0] = 1
    selected = original.roll(1, -1)
    assert study.parity(selected, selected)["pass_all128_bitwise"]
    assert not study.parity(original, selected)["pass_all128_bitwise"]
    assert study.POLICY["FullKV_guard_reference_never_replaced"] is True
    assert study.POLICY["whole_request_KV_peak_certified"] is False
    assert study.POLICY["cap_is_lifetime"] is False


def test_full_length_finite_logits_and_exact_bitwise_tolerance_required():
    x = torch.randn(128, 5)
    y = x.clone()
    y[17, 2] += 1e-6
    assert not study.parity(x, y)["pass_all128_bitwise"]
    with pytest.raises(ValueError, match="128"):
        study.parity(x[:32], y[:32])
    y[17, 2] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        study.parity(x, y)


def test_bitwise_guard_does_not_conflate_signed_zero():
    x = torch.zeros(128, 3)
    y = x.clone()
    y[33, 2] = -0.0
    assert torch.equal(x, y)
    assert not study.parity(x, y)["pass_all128_bitwise"]


def test_missing_prepare_receives_finite_failure_analysis(tmp_path):
    gate = study.analyze(tmp_path)
    assert not gate["complete"]
    assert gate["failures"]
    assert not gate["calibration_or_paper_campaign_authorized"]
    assert (tmp_path / "analysis/GATE_fidelity_byte_v20.json").is_file()
