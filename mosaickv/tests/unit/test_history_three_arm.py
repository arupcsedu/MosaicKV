from collections import Counter

import header_history_timing as timing
import history_three_arm as study
import pytest


def test_three_arm_schedule_and_frozen_policy_meanings():
    jobs = study.registered_schedule(3)
    assert len(jobs) == 75
    assert Counter((r["warmup"], r["arm"]) for r in jobs) == Counter(
        {(warm, arm): 5 if warm else 20 for warm in (True, False) for arm in study.ARMS}
    )
    assert study.POLICIES[study.ARMS[0]] == dict(history=False, mixed=False)
    assert study.POLICIES[study.ARMS[1]] == dict(history=True, mixed=False)
    assert study.POLICIES[study.ARMS[2]] == dict(history=True, mixed=True)
    assert study.registered_schedule(3) == jobs


def test_multiple_endpoints_do_not_narrow_intervals():
    a = list(range(1, 21))
    b = [v * (0.9 + (i % 3) * 0.05) for i, v in enumerate(a)]
    old = timing.paired([a, b], seed=3, family_comparisons=2)
    adjusted = timing.paired([a, b], seed=3, family_comparisons=12)
    assert adjusted["ratio_adjusted_CI"][0] <= old["ratio_adjusted_CI"][0]
    assert adjusted["ratio_adjusted_CI"][1] >= old["ratio_adjusted_CI"][1]
    with pytest.raises(ValueError, match="comparison family"):
        timing.paired([a, b], seed=3, family_comparisons=True)
