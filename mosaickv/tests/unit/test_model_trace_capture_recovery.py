"""CPU fixtures for a retrospective metadata diagnosis, not model evidence."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def protocol():
    path = Path(__file__).resolve().parents[2] / "scripts/evicastkv_model_trace_capture_recovery.py"
    spec = importlib.util.spec_from_file_location("model_capture_recovery_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture(protocol):
    previous = [dict(action="alloc", addr=4096, size=900, stream=0, time_us=100)]
    current = deepcopy(previous)
    current[0]["time_us"] = 101
    current.append(dict(action="free_requested", addr=4096, size=900, stream=0, time_us=102))
    return dict(
        accepted=False,
        online_guard_modified=False,
        previous_trace=previous,
        current_trace=current,
        previous_count=1,
        current_count=2,
        trace_capacity=100_000,
        previous_trace_sha256=protocol._sha(previous),
        current_trace_sha256=protocol._sha(current),
    )


def test_one_microsecond_drift_is_not_an_allocation_change(protocol):
    record = capture(protocol)
    before = deepcopy(record)
    result = protocol.compare_capture(record)
    assert record == before
    assert result["actual_model_clock_only_drift_observed"] is True
    assert result["timestamp_deltas_microseconds"] == [1]
    assert result["GPU_inference_repeated"] is False
    assert result["whole_request_peak_certified"] is False


@pytest.mark.parametrize(
    "field,bad", [("addr", 8192), ("stream", 1), ("size", 901), ("compile_context", "new")]
)
def test_structural_changes_cannot_be_waived_as_clock_drift(protocol, field, bad):
    record = capture(protocol)
    record["current_trace"][0][field] = bad
    record["current_trace_sha256"] = protocol._sha(record["current_trace"])
    result = protocol.compare_capture(record)
    assert result["actual_model_clock_only_drift_observed"] is False
    assert result["all_other_event_fields_identical"] is False


@pytest.mark.parametrize(
    "field,bad",
    [
        ("accepted", True),
        ("accepted", 0),
        ("online_guard_modified", True),
        ("trace_capacity", 1),
        ("trace_capacity", True),
        ("previous_count", 2),
        ("current_count", 3),
        ("current_trace_sha256", "f" * 64),
    ],
)
def test_scope_capacity_counts_and_hashes_are_strict(protocol, field, bad):
    record = capture(protocol)
    record[field] = bad
    with pytest.raises(ValueError):
        protocol.compare_capture(record)


def test_unchanged_prefix_is_not_claimed_as_positive_clock_evidence(protocol):
    record = capture(protocol)
    record["current_trace"][0]["time_us"] = 100
    record["current_trace_sha256"] = protocol._sha(record["current_trace"])
    assert protocol.compare_capture(record)["actual_model_clock_only_drift_observed"] is False
