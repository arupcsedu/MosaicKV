import json

import evicastkv_replay_descriptor_followup as driver
import pytest


def test_dependency_prefers_terminal_closeout_over_earlier_analysis(tmp_path):
    with pytest.raises(ValueError, match="dependency"):
        driver.parent_dependency(tmp_path)
    (tmp_path / "submission.json").write_text(json.dumps(dict(cpu_analysis="1")))
    assert driver.parent_dependency(tmp_path) == "1"
    (tmp_path / "dispatch/timing").mkdir(parents=True)
    (tmp_path / "dispatch/timing/submission.json").write_text(json.dumps(dict(cpu_analysis="2")))
    assert driver.parent_dependency(tmp_path) == "2"
    (tmp_path / "dispatch/registered_closeout.json").write_text(json.dumps(dict(job_id="3")))
    assert driver.parent_dependency(tmp_path) == "3"


def test_phase_gate_uses_only_complete_label_blind_replay_cost():
    rows = [
        dict(
            method="mosaickv_exact",
            input_index=i,
            phases={
                "middle/code_replay": dict(exclusive_wall_seconds=0.2),
                "prefill/code_replay": dict(exclusive_wall_seconds=999),
            },
        )
        for i in (0, 0, 1, 1)
    ]
    assert driver.opportunity(rows) == [0.2] * 4
    for bad in ([], rows[:-1], [dict(r, phases={}) for r in rows]):
        with pytest.raises(ValueError):
            driver.opportunity(bad)


def test_waits_on_new_dependency_once_then_stops_instead_of_polling(monkeypatch, tmp_path):
    parent, root = tmp_path / "parent", tmp_path / "out"
    parent.mkdir()
    monkeypatch.setattr(driver, "PARENT", parent)
    monkeypatch.setattr(driver, "clean_source", lambda *a: "revision")
    monkeypatch.setattr(driver, "parent_dependency", lambda: "100")
    calls = []
    monkeypatch.setattr(driver, "schedule", lambda *a: calls.append(a))
    driver.watch(root, "revision", 0)
    assert calls == [(root, "revision", "100", 1)]
    launch = root.with_name(root.name + "-launch")
    (launch / "watch-1.json").write_text(json.dumps(dict(dependency="100")))
    driver.watch(root, "revision", 1)
    assert len(calls) == 1 and (launch / "blocked.json").exists()
    with pytest.raises(ValueError, match="finite"):
        driver.watch(tmp_path / "invalid", "revision", 6)


def test_negative_or_successful_parent_cannot_launch_a_larger_campaign(monkeypatch, tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    monkeypatch.setattr(driver, "PARENT", parent)
    monkeypatch.setattr(driver, "clean_source", lambda *a: "revision")
    monkeypatch.setattr(
        driver.driver.common.previous, "cpu", lambda *a: pytest.fail("no further execution")
    )
    for status in (False, True):
        (parent / "engineering_closeout.json").write_text(
            json.dumps(dict(complete=status, failures=[], all_method_no_slower_gate=status))
        )
        root = tmp_path / str(status)
        driver.watch(root, "revision", 0)
        launch = root.with_name(root.name + "-launch")
        assert (launch / ("not_needed.json" if status else "blocked.json")).exists()
