import pytest

from mosaickv.evaluation.latency_strongest_control import analyze


def test_reuse_only_success_cannot_hide_a_faster_prefix_arm():
    rows = [
        dict(method=m, arm=a, trial=t, measurement=dict(total_wall_seconds=s))
        for m, a, s in [
            ("mosaickv_exact", "original", 12.0),
            ("mosaickv_exact", "prefetch", 10.2),
            ("prefixkv_reimpl", "reuse", 10.0),
            ("prefixkv_reimpl", "hybrid-6291456", 9.0),
        ]
        for t in range(-5, 20)
    ]
    report = analyze(rows, seed=4)
    candidate = report["contrasts"][0]
    assert candidate["own_improvement_gate"]
    assert not candidate["similar_to_all_tested_prefix_runtimes"]
    assert not candidate["no_slower_than_all_tested_prefix_runtimes"]
    assert report["fastest_observed_prefix_arm"] == "hybrid-6291456"
    assert report["retrospective"]
    with pytest.raises(ValueError, match="unique"):
        analyze(rows[:-1], seed=4)
