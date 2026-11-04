import evicastkv_direct_text_latency as controller
import pytest


def rows():
    return [
        dict(method=m, arm=a, trial=t, measurement=dict(total_wall_seconds=v))
        for m, a, v in (
            ("mosaickv_exact", "cached", 11),
            ("mosaickv_exact", "direct", 10),
            ("prefixkv_reimpl", "cached", 10.2),
            ("prefixkv_reimpl", "direct", 10.1),
            ("prefixkv_reimpl", "whole-6291456", 9.8),
            ("lookm_reimpl", "direct", 9),
        )
        for t in range(-5, 20)
    ]


def test_fastest_other_baseline_is_not_hidden_by_prefix_similarity_gate():
    report = controller.contrasts(rows(), 3)
    assert report["own_speedup_gate"] and report["strongest_prefix_similar_gate"]
    assert not report["all_baselines_no_slower_gate"]
    assert len(report["baselines"]) == 4
    assert report["family_maximum_engineering_contrasts"] == 48


def test_missing_or_duplicate_rows_and_missing_strongest_control_rejected():
    for observations in (rows()[:-1], [*rows(), rows()[0]]):
        with pytest.raises(ValueError, match="complete unique"):
            controller.contrasts(observations, 3)
    with pytest.raises(ValueError, match="registered PrefixKV"):
        controller.contrasts([r for r in rows() if r["arm"] != "whole-6291456"], 3)


def test_import_does_not_mutate_other_controller_launchers():
    assert controller.SCRIPT.name == "evicastkv_direct_text_latency.sbatch"
    assert controller.common.SCRIPT.name == "evicastkv_cached_text_latency.sbatch"
    assert controller.common.previous.SCRIPT.name == "evicastkv_hybrid_latency.sbatch"


def test_incomplete_controls_cannot_launch_timing(monkeypatch, tmp_path):
    monkeypatch.setattr(controller.driver, "checked", lambda *_: {})
    monkeypatch.setattr(controller.driver, "load_producers", lambda *_: ([], ["worker failed"]))
    monkeypatch.setattr(
        controller, "stage", lambda *_: pytest.fail("failed controls cannot advance")
    )
    controller.study_analysis(tmp_path, "x")
