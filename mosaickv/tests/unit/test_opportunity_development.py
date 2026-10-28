"""CPU synthetic fixtures validate analysis, not measured quality claims."""

import copy

import pytest
from evicastkv_opportunity_development import (
    expected_keys,
    quality_decision,
    registered_comparisons,
    validate_rows,
)


def rows_fixture():
    ids = [f"consumed-{i}" for i in range(64)]
    return ids, [
        dict(
            sample_id=sid,
            method=method,
            budget_bytes=cap,
            variant=variant,
            status="completed",
            workload="coco",
            repeat=0,
            emitted_tokens=64,
            forward_steps=63,
            initial_kv_bytes=cap or 100_000_000,
            final_kv_bytes=(cap or 100_000_000) + 8257536,
            peak_kv_bytes=(cap or 100_000_000) + 8257536,
            e2e_wall_seconds=1.0,
            effective_method=method
            if variant == "complete"
            else "mosaickv_exact__evidence_"
            + {
                "evidence-routing-v1": "routing_only",
                "evidence-coverage-v1": "coverage_only",
                "evidence-both-v1": "both",
            }[variant]
            + "_v1",
            input_sha=sid,
            processor_sha="fixed",
            scorer_sha="fixed",
            references=["fixture"],
            media_group_ids=[sid],
            gpu_uuid="fixture-gpu",
            source_sha="fixture-source",
        )
        for sid, method, cap, variant in sorted(expected_keys(ids), key=str)
    ]


def test_complete_registered_matrix_has_exact_byte_pairs_and_no_fresh_ids():
    ids, rows = rows_fixture()
    assert len(rows) == 832 and len(validate_rows(rows, ids)) == 128
    assert len(registered_comparisons()) == len(set(registered_comparisons())) == 18
    for a, b in registered_comparisons():
        assert a.rsplit("/", 1)[1] == b.rsplit("/", 1)[1]


@pytest.mark.parametrize("count", [0, 2, 63, 65])
def test_wrong_cohort_size_rejected(count):
    with pytest.raises(ValueError, match="64"):
        expected_keys([str(i) for i in range(count)])


@pytest.mark.parametrize("change", ["missing", "duplicate", "failed", "unknown_variant"])
def test_missing_failures_or_duplicate_rows_cannot_disappear(change):
    ids, rows = rows_fixture()
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "failed":
        rows[0]["status"] = "failed"
    else:
        rows[0]["variant"] = "unregistered"
    with pytest.raises(ValueError):
        validate_rows(rows, ids)


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_kv_bytes", 1),
        ("final_kv_bytes", 1),
        ("peak_kv_bytes", float("nan")),
        ("e2e_wall_seconds", float("nan")),
        ("forward_steps", 62),
        ("emitted_tokens", 63),
        ("effective_method", "fullkv_fallback"),
        ("processor_sha", "different"),
        ("gpu_uuid", "different"),
        ("references", ["different"]),
    ],
)
def test_drift_or_unequal_measured_bytes_fail_closed(field, value):
    ids, rows = rows_fixture()
    row = next(r for r in rows if r["budget_bytes"] is not None)
    row[field] = value
    with pytest.raises(ValueError):
        validate_rows(rows, ids)


@pytest.mark.parametrize("lower,expected", [(-0.1, False), (0.0, False), (0.1, True)])
def test_zero_lower_bound_is_not_superiority(lower, expected):
    report = dict(
        contrasts=[
            dict(reference=a, candidate=b, simultaneous_ci=[lower, 1.0])
            for a, b in registered_comparisons()
        ]
    )
    assert all(
        r["positive_development_signal"] is expected for r in quality_decision(report).values()
    )


def test_one_positive_budget_or_frozen_only_gain_is_not_all_baseline_success():
    contrasts = [
        dict(reference=a, candidate=b, simultaneous_ci=[1.0, 2.0])
        for a, b in registered_comparisons()
    ]
    contrasts[0]["simultaneous_ci"] = [-1.0, 1.0]
    decisions = quality_decision(dict(contrasts=contrasts))
    assert not decisions["evidence-routing-v1"]["positive_development_signal"]
    assert decisions["evidence-coverage-v1"]["positive_development_signal"]
