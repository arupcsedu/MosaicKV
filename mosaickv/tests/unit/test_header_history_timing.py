from collections import Counter

import header_history_timing as study
import pytest


def test_schedule_exact_counts_randomized_and_reproducible():
    a = study.schedule(3)
    assert a == study.schedule(3) and a != study.schedule(4)
    assert Counter((r["warmup"], r["arm"]) for r in a) == Counter(
        {(warmup, arm): 5 if warmup else 20 for warmup in (True, False) for arm in study.ARMS}
    )
    for first, second in zip(a[::2], a[1::2], strict=True):
        assert first["repetition"] == second["repetition"]
        assert first["arm"] != second["arm"]


def test_paired_ratio_uses_identical_trials_and_strict_gate():
    values = [list(range(1, 21)), [x * 0.9 for x in range(1, 21)]]
    result = study.paired(values, seed=3)
    assert result["equal_or_faster"] and result["mean_latency_ratio"] == pytest.approx(0.9)
    assert (
        study.paired([values[0], [x * 1.01 for x in values[0]]], seed=3)["equal_or_faster"] is False
    )
    with pytest.raises(ValueError, match="20 complete"):
        study.paired([values[0][:-1], values[1]], seed=3)
