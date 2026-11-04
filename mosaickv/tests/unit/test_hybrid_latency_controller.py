import json

import evicastkv_hybrid_latency as controller
import pytest


def trials():
    return [
        dict(method=m, arm=a, trial=t, measurement=dict(total_wall_seconds=s))
        for m, a, s in [
            ("mosaickv_exact", "original", 12.0),
            ("mosaickv_exact", "reuse", 10.0),
            ("prefixkv_reimpl", "original", 11.0),
            ("prefixkv_reimpl", "reuse", 10.5),
        ]
        for t in range(-5, 20)
    ]


def test_complete_paired_trials_and_no_missing_or_duplicate_rows():
    result = controller.contrasts(trials(), seed=3)
    candidate = next(r for r in result["comparisons"] if r["method"] == "mosaickv_exact")
    assert candidate["own_speedup_gate"] and candidate["prefix_similar_gate"]
    assert candidate["prefix_no_slower_gate"]
    assert result == controller.contrasts(trials(), seed=3)
    for rows in (trials()[:-1], trials() + trials()[:1]):
        with pytest.raises(ValueError, match="unique"):
            controller.contrasts(rows, seed=3)
    rows = trials()
    rows[-1]["measurement"]["total_wall_seconds"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        controller.contrasts(rows, seed=3)


def test_failed_or_different_input_arms_cannot_promote(monkeypatch, tmp_path):
    monkeypatch.setattr(controller.driver, "checked", lambda *_: dict(seed=3))

    def row(arm):
        return dict(
            comparisons=[
                dict(
                    method="mosaickv_exact",
                    arm=arm,
                    own_speedup_gate=True,
                    prefix_similar_gate=True,
                )
            ]
        )

    monkeypatch.setattr(
        controller.driver, "load_producers", lambda *_: ([row("reuse"), row("hybrid-6291456")], [])
    )
    monkeypatch.setattr(controller, "submit_stage", lambda *_: pytest.fail("no common passing arm"))
    controller.timing_analysis(tmp_path, "x")
    report = json.loads((tmp_path / "analysis/decision.json").read_text())
    assert not report["all_input_latency_gate"] and report["selected_engineering_arm"] is None
    assert not report["paper_superiority_established"]


def test_failed_producers_never_launch_timing(monkeypatch, tmp_path):
    monkeypatch.setattr(controller.driver, "checked", lambda *_: {})
    monkeypatch.setattr(controller.driver, "load_producers", lambda *_: ([], ["worker timed out"]))
    monkeypatch.setattr(controller, "submit_stage", lambda *_: pytest.fail("no GPU retries"))
    controller.study_analysis(tmp_path, "x")
    report = json.loads((tmp_path / "analysis/decision.json").read_text())
    assert not report["complete"] and report["failures"]
