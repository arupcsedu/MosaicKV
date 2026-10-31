import copy

import pytest

from mosaickv.evaluation.request_statistics import compare_requests


def examples():
    rows = [
        dict(
            sample_id=str(i),
            end_to_end_seconds=10.0 + i,
            storage={
                f"{boundary}_active_kv_storage_bytes": 100 + i
                for boundary in ("initial", "final", "peak")
            },
            whole_request_peak_allocated_bytes=1000 + i,
        )
        for i in range(4)
    ]
    candidates = copy.deepcopy(rows)
    for r in candidates:
        r["end_to_end_seconds"] *= 0.9
        r["storage"] = {k: v // 2 for k, v in r["storage"].items()}
        r["whole_request_peak_allocated_bytes"] -= 20
    return candidates, rows


def test_paired_memory_latency_and_allocator_boundaries_remain_distinct():
    candidate, baseline = examples()
    a = compare_requests(candidate, baseline, resamples=200, seed=7)
    b = compare_requests(list(reversed(candidate)), baseline, resamples=200, seed=7)
    assert a == b
    assert a["latency_no_slower"] and a["mean_latency_ratio"] == pytest.approx(0.9)
    assert a["whole_request_allocated_peak"]["paired_delta_ci95"] == [-20.0, -20.0]
    assert all(m["all_inputs_smaller"] for m in a["active_storage"].values())
    assert a["not_a_shared_encoded_byte_comparison"]


def test_no_favorable_complete_intersection_or_duplicate_ids():
    c, b = examples()
    with pytest.raises(ValueError, match="complete unique"):
        compare_requests(c[:-1], b, resamples=10, seed=7)
    with pytest.raises(ValueError, match="complete unique"):
        compare_requests([*c, c[0]], b, resamples=10, seed=7)


def test_a_single_storage_regression_is_not_hidden_by_mean_savings():
    c, b = examples()
    c[0]["storage"]["peak_active_kv_storage_bytes"] = 101
    out = compare_requests(c, b, resamples=100, seed=7)
    assert not out["active_storage"]["peak_active_kv_storage_bytes"]["all_inputs_smaller"]
