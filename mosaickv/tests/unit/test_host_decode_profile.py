import pytest

from mosaickv.evaluation.host_decode_profile import summarize


def test_cpu_profile_summary_separates_self_and_nested_time():
    report = summarize({("fixture.py", 5, "outer"): (1, 1, 0.1, 0.5, {})})
    assert report["top_self"][0]["self_seconds"] == 0.1
    assert report["top_cumulative"][0]["cumulative_seconds"] == 0.5
    assert report["cumulative_times_are_nested_do_NOT_sum"]
    assert report["C_function_time_can_include_GPU_wait"]
    assert report["instrumentation_NOT_request_latency_acceptance"]


@pytest.mark.parametrize(
    "counts", [(True, 1, 0.1, 0.2), (2, 1, 0.1, 0.2), (1, 1, -1, 0), (1, 1, 0, float("nan"))]
)
def test_bad_cpu_counters_rejected(counts):
    with pytest.raises(RuntimeError):
        summarize({("fixture.py", 5, "bad"): (*counts, {})})


def test_missing_profile_rejected():
    with pytest.raises(RuntimeError):
        summarize({})
