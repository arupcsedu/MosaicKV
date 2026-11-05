import evicastkv_lt_model_profile as controller
import pytest

from mosaickv.evaluation.lt_model_profile import STEPS, phase_summary


def test_windows_are_after_initial_graph_capture_and_include_later_prefixes():
    assert STEPS == (1, 64, 126)
    assert 0 not in STEPS


def test_phase_summary_is_host_only_and_nested_not_wall_acceptance():
    assert phase_summary({"capture": [0.2, 0.3], "replay": []}) == {
        "capture": dict(calls=2, total_host_seconds=0.5, maximum_host_seconds=0.3),
        "replay": dict(calls=0, total_host_seconds=0, maximum_host_seconds=0.0),
    }


def test_compares_new_reader_to_original_evicast_and_fast_prefix_control():
    assert controller.ARMS == (
        ("mosaickv_exact", "graph"),
        ("mosaickv_exact", "graph-lt"),
        ("prefixkv_reimpl", "graph-original"),
    )


@pytest.mark.parametrize(
    "complete,passed,released",
    [(True, False, True), (True, True, False), (False, False, False), (False, True, False)],
)
def test_profile_released_only_after_complete_negative_parent(
    tmp_path, monkeypatch, complete, passed, released
):
    parent = tmp_path / "parent"
    root = tmp_path / "diagnostic"
    monkeypatch.setattr(controller, "PARENT", parent)
    decision = parent / "paired-timing/analysis/decision.json"
    controller.write_once(decision, dict(complete=complete, four_percent_latency_gate=passed))
    monkeypatch.setattr(controller.driver, "checked", lambda *a: {"source_sha": "a" * 40})
    monkeypatch.setattr(controller.driver, "cpu_proof", lambda *a: None)
    calls = []

    def submit(root, action, dependency=None):
        calls.append((action, dependency))
        return "123"

    monkeypatch.setattr(controller, "submit", submit)
    controller.release(root, "b" * 64)
    record = controller.driver.read(root / "release.json")
    assert record["profile_released"] is released
    assert record["parent_sha256"] == controller.sha(decision)
    assert calls == ([("profile", None), ("analysis", "123")] if released else [])
