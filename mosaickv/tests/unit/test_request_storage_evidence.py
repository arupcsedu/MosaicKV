"""Regression for the consumed-input audited/unaudited 286,879-byte difference."""

import pytest

from mosaickv.evaluation.request_storage_evidence import audited, timed


def fixture():
    # Measured accounting-only diagnostic, not synthetic benchmark scores:
    # paired-request-storage-diagnostic-consumed-20261102-v1, input zero.
    frozen = dict(
        initial_active_kv_storage_bytes=36497641,
        final_active_kv_storage_bytes=70308157,
        peak_active_kv_storage_bytes=76550620,
    )
    actual = dict(
        initial_encoded_bytes=36497641,
        final_encoded_bytes=70308157,
        decode_owned_peak_bytes=76550620,
        reported_counter_peak_bytes=76550620,
        tensor_result_auditing=True,
        peak_measurement_admissible=True,
    )
    fast = dict(
        actual,
        tensor_result_auditing=False,
        peak_measurement_admissible=False,
        reported_counter_peak_bytes=76263741,
        decode_owned_peak_bytes=None,
    )
    return frozen, actual, fast


def test_physical_peak_gate_remains_exact_without_using_fast_counter_as_peak():
    frozen, actual, fast = fixture()
    proof = audited(actual, frozen)
    assert proof["decode_owned_peak_bytes"] == frozen["peak_active_kv_storage_bytes"]
    timed(fast, frozen, expected_counter=76263741)
    with pytest.raises(RuntimeError, match="independently observed"):
        audited(fast, frozen)
    for difference in (-286879, -1, 1):
        with pytest.raises(RuntimeError, match="AUDITED physical peak"):
            audited(dict(actual, decode_owned_peak_bytes=76550620 + difference), frozen)


def test_fast_counter_cannot_be_relabelled_as_audited_or_changed_silently():
    frozen, actual, fast = fixture()
    with pytest.raises(RuntimeError, match="masquerade"):
        timed(actual, frozen, expected_counter=76550620)
    with pytest.raises(RuntimeError, match="masquerade"):
        timed(dict(fast, decode_owned_peak_bytes=76263741), frozen, expected_counter=76263741)
    for counter in (None, 76263740, 76263742):
        with pytest.raises(RuntimeError, match="counter"):
            timed(fast, frozen, expected_counter=counter)


def test_endpoint_or_peak_overrun_still_fails_with_valid_provenance_flags():
    frozen, actual, fast = fixture()
    for field in ("initial_encoded_bytes", "final_encoded_bytes"):
        with pytest.raises(RuntimeError, match="endpoint"):
            audited(dict(actual, **{field: actual[field] + 1}), frozen)
        with pytest.raises(RuntimeError, match="endpoint"):
            timed(dict(fast, **{field: fast[field] - 1}), frozen, expected_counter=76263741)
    with pytest.raises(RuntimeError, match="exceeds audited"):
        timed(dict(fast, reported_counter_peak_bytes=76550621), frozen, expected_counter=76550621)
