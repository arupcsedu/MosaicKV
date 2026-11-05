import copy

import evicastkv_event_only_reader_profile as controller
import pytest

from mosaickv.evaluation.event_only_reader_profile import summarize


def fixture():
    return (
        [dict(step=i, step_stream_ms=2.0) for i in range(127)],
        [dict(step=i, reader_stream_intervals_ms=[0.01] * 64) for i in (1, 64, 126)],
    )


def test_positive_intervals_are_attribution_not_acceptance():
    proof = summarize(*fixture())
    assert proof["CUPTI_enabled"] is False
    assert proof["CUDA_events_include_host_gaps_NOT_pure_kernel_time"]
    assert proof["instrumentation_NOT_request_latency_acceptance"]
    assert proof["independent_quality_N"] == 0
    assert len(controller.controller.ARMS) == 3


@pytest.mark.parametrize("failure", ["steps", "reader", "windows", "nan", "zero", "boolean"])
def test_missing_or_invalid_window_fails_closed(failure):
    steps, windows = copy.deepcopy(fixture())
    if failure == "steps":
        steps.pop()
    elif failure == "reader":
        windows[0]["reader_stream_intervals_ms"].pop()
    elif failure == "windows":
        windows.pop()
    else:
        steps[0]["step_stream_ms"] = {"nan": float("nan"), "zero": 0.0, "boolean": True}[failure]
    with pytest.raises(RuntimeError):
        summarize(steps, windows)
