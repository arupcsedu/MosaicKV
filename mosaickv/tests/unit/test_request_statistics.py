import copy
import json

import numpy as np
import pytest

from mosaickv.evaluation import request_statistics
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


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "1.0",
        None,
        [],
        {},
        float("nan"),
        float("inf"),
        float("-inf"),
        -1.0,
        0.0,
        10**400,
    ],
)
def test_latency_requires_raw_positive_finite_numbers_before_bootstrap(monkeypatch, side, value):
    cohorts = examples()
    cohorts[side][0]["end_to_end_seconds"] = value
    monkeypatch.setattr(
        request_statistics.np.random,
        "default_rng",
        lambda *_: pytest.fail("invalid observations reached bootstrap allocation"),
    )
    with pytest.raises(ValueError, match="latency"):
        compare_requests(*cohorts, resamples=100, seed=7)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize(
    "field", [*request_statistics.STORAGE_FIELDS, "whole_request_peak_allocated_bytes"]
)
@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "100",
        100.0,
        99.9,
        None,
        [],
        {},
        float("nan"),
        float("inf"),
        -1,
        0,
        2**63,
        np.int64(100),
    ],
)
def test_all_storage_boundaries_require_plain_positive_integer_bytes(
    monkeypatch, side, field, value
):
    cohorts = examples()
    row = cohorts[side][0]
    if field in request_statistics.STORAGE_FIELDS:
        row["storage"][field] = value
    else:
        row[field] = value
    monkeypatch.setattr(
        request_statistics.np.random,
        "default_rng",
        lambda *_: pytest.fail("invalid byte observations reached bootstrap allocation"),
    )
    with pytest.raises(ValueError, match="plain-integer"):
        compare_requests(*cohorts, resamples=100, seed=7)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("sample_id", [None, True, False, 0, 1, "", "  ", [], {}])
def test_pairing_requires_nonempty_string_identities(side, sample_id):
    cohorts = examples()
    cohorts[side][0]["sample_id"] = sample_id
    with pytest.raises(ValueError, match="complete unique"):
        compare_requests(*cohorts, resamples=100, seed=7)


@pytest.mark.parametrize(
    "parameter,value",
    [
        ("resamples", True),
        ("resamples", False),
        ("resamples", "100"),
        ("resamples", 100.0),
        ("resamples", None),
        ("resamples", float("inf")),
        ("resamples", 0),
        ("resamples", -1),
        ("seed", True),
        ("seed", False),
        ("seed", "7"),
        ("seed", 7.0),
        ("seed", None),
        ("seed", float("nan")),
        ("seed", -1),
    ],
)
def test_bootstrap_parameters_are_plain_integers(parameter, value):
    arguments = dict(resamples=100, seed=7)
    arguments[parameter] = value
    with pytest.raises(ValueError, match="bootstrap"):
        compare_requests(*examples(), **arguments)


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize(
    "failure",
    ["missing_latency", "missing_storage", "missing_boundary", "missing_peak", "row_shape"],
)
def test_incomplete_observations_are_rejected(side, failure):
    cohorts = examples()
    row = cohorts[side][0]
    if failure == "missing_latency":
        del row["end_to_end_seconds"]
    elif failure == "missing_storage":
        del row["storage"]
    elif failure == "missing_boundary":
        del row["storage"][request_statistics.STORAGE_FIELDS[0]]
    elif failure == "missing_peak":
        del row["whole_request_peak_allocated_bytes"]
    else:
        cohorts[side][0] = True
    with pytest.raises(ValueError):
        compare_requests(*cohorts, resamples=100, seed=7)


@pytest.mark.parametrize("timing", [1, 1.0, 0.25])
def test_valid_integer_and_float_timings_preserve_no_slower_threshold_and_scope(timing):
    cohorts = examples()
    for rows in cohorts:
        for row in rows:
            row["end_to_end_seconds"] = timing
    original = copy.deepcopy(cohorts)
    result = compare_requests(*cohorts, resamples=100, seed=0)
    assert cohorts == original
    assert result["mean_latency_ratio"] == 1.0
    assert result["paired_mean_latency_ratio_ci95"] == [1.0, 1.0]
    assert result["latency_no_slower"] is True
    assert result["not_a_shared_encoded_byte_comparison"] is True
    assert result["not_a_whole_request_KV_only_peak"] is True
    json.dumps(result, allow_nan=False)


def test_minimum_valid_resamples_and_seed_are_preserved():
    result = compare_requests(*examples(), resamples=1, seed=0)
    assert result["unique_sample_count"] == 4
    assert len(result["paired_mean_latency_ratio_ci95"]) == 2


@pytest.mark.parametrize("side", [0, 1])
def test_finite_inputs_cannot_emit_nonfinite_overflowed_statistics(side):
    cohorts = examples()
    for row in cohorts[side]:
        row["end_to_end_seconds"] = 1e308
    with pytest.raises(ValueError, match="finite representable"):
        compare_requests(*cohorts, resamples=100, seed=7)


def test_confirmed_boolean_pass_repro_now_fails_closed():
    candidate, baseline = examples()
    for row in candidate:
        row["end_to_end_seconds"] = True
        row["storage"] = dict.fromkeys(request_statistics.STORAGE_FIELDS, True)
        row["whole_request_peak_allocated_bytes"] = True
    with pytest.raises(ValueError):
        compare_requests(candidate, baseline, resamples=100, seed=7)
