"""Synthetic checks of exhaustive retrospective pairing and latency inference."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mosaickv.evaluation.cider_resampling import paired_draw_counts

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/analyze_coco_cross_budget_retrospective.py"
SPEC = importlib.util.spec_from_file_location("coco_retrospective", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_every_cross_budget_pair_and_unavailable_registration_retained():
    primary = [
        dict(candidate_cap_mib=c, baseline_cap_mib=b) for c, b in ((10, 20), (25, 40), (40, 80))
    ]
    catalog = MODULE.pair_catalog([10, 20, 40, 80], primary)
    assert len(catalog) == 14
    assert sum(p["available"] for p in catalog) == 12
    unavailable = [p for p in catalog if not p["available"]]
    assert {(p["candidate_cap_mib"], p["baseline_cap_mib"]) for p in unavailable} == {(25, 40)}
    assert (
        len({(p["baseline"], p["candidate_cap_mib"], p["baseline_cap_mib"]) for p in catalog}) == 14
    )


def test_adjusted_intervals_are_simultaneous_and_wider():
    values = np.arange(10000, dtype=float)
    ci = MODULE.interval(values, 70)
    assert ci["adjusted_low"] < ci["ci95_low"] < ci["ci95_high"] < ci["adjusted_high"]
    assert ci["adjusted_low"] == pytest.approx(np.quantile(values, 0.025 / 70))


def test_latency_ratio_is_ratio_of_paired_means_not_mean_ratio():
    baseline = np.array([1.0, 10.0, 20.0, 30.0])
    candidate = baseline / 2
    counts = paired_draw_counts(4, 100, 9)
    result = MODULE.latency_statistics(baseline, candidate, counts, 70)
    assert result["ratio"] == 0.5
    assert result["ratio_interval"]["adjusted_low"] == 0.5
    assert result["ratio_interval"]["adjusted_high"] == 0.5
    candidate = np.array([1.0, 10.0, 10.0, 10.0])
    result = MODULE.latency_statistics(baseline, candidate, counts, 70)
    assert result["ratio"] == pytest.approx(candidate.mean() / baseline.mean())
    assert result["ratio"] != pytest.approx((candidate / baseline).mean())
    expected = (counts @ candidate - counts @ baseline) / 4
    assert result["difference_interval"] == MODULE.interval(expected, 70)


@pytest.mark.parametrize("bad", [np.nan, 0.0, -1.0])
def test_invalid_timings_block_analysis(bad):
    with pytest.raises(ValueError, match="positive and finite"):
        MODULE.latency_statistics(
            np.array([1.0, 2.0]), np.array([1.0, bad]), paired_draw_counts(2, 2, 0), 1
        )
