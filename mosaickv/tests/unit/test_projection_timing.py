"""Synthetic schema fixtures, not measured model performance."""

import pytest

from mosaickv.evaluation.projection_timing import VARIANTS, orders, summarize


def fixtures():
    return [
        dict(
            block=block,
            warmup=block < 0,
            variant=variant,
            bitwise_reference_parity=True,
            workspace_audit_enabled=False,
            peak_from_timed_trial_admissible=False,
            construction_seconds=1.0,
            decode_wall_seconds=1.0,
            decode_cuda_ms=1000.0,
            persistent_storage={"initial": 100, "final": 110},
        )
        for block in range(-5, 20)
        for variant in VARIANTS
    ]


def test_order_is_complete_randomized_deterministic_and_prespecified():
    assert orders(7) == orders(7)
    assert len({tuple(o) for _, o in orders(7)}) > 1
    assert [b for b, _ in orders(7)] == list(range(-5, 20))
    assert all(set(o) == set(VARIANTS) for _, o in orders(7))


def test_analysis_rejects_incomplete_numerically_changed_and_instrumented_trials():
    summaries = summarize(fixtures())
    assert len(summaries) == 3
    assert all(
        t["decode_cuda_ms_ratio_vs_encoded_KV"]["paired_ci95"] == [1.0, 1.0] for t in summaries
    )
    for field, value in (
        ("workspace_audit_enabled", True),
        ("bitwise_reference_parity", False),
        ("decode_cuda_ms", float("nan")),
    ):
        trials = fixtures()
        trials[0][field] = value
        with pytest.raises(ValueError, match="complete finite"):
            summarize(trials)
    with pytest.raises(ValueError, match="complete finite"):
        summarize(fixtures()[:-1])
    trials = fixtures()
    trials[0]["persistent_storage"] = {"initial": 101, "final": 110}
    with pytest.raises(ValueError, match="storage changed"):
        summarize(trials)


def test_raw_and_encoded_comparators_are_both_reported():
    trials = fixtures()
    for t in trials:
        if t["variant"] == "encoded_KV":
            t["decode_cuda_ms"] *= 1.2
        if t["variant"] == "hybrid_source":
            t["decode_cuda_ms"] *= 1.22
    hybrid = next(t for t in summarize(trials) if t["variant"] == "hybrid_source")
    assert hybrid["decode_cuda_ms_ratio_vs_encoded_KV"]["mean_ratio"] < 1.05
    assert hybrid["decode_cuda_ms_ratio_vs_raw_separate"]["mean_ratio"] > 1.05
