import evicastkv_joint_row_feasibility as control


def test_current_probe_only_adds_an_observer_and_common_controls(monkeypatch):
    seen = []

    class State:
        def capture(self, packets):
            pass

    state = State()
    monkeypatch.setattr(control.probe, "current", lambda: state)
    monkeypatch.setattr(control, "ORIGINAL_EXECUTE", lambda *a, **kw: seen.append((a, kw)))
    record = {}
    control.execute(None, None, None, None, None, "mosaickv_exact", record)
    assert seen[0][1]["final_validator"] == state.capture
    assert seen[0][0][-1]["graph_accounting"] == "event_sum"
    assert seen[0][0][-1]["reuse_patch_dispatch"]
    assert record == {}


def test_one_dependency_deferral_submits_no_GPU_and_does_not_poll(monkeypatch, tmp_path):
    checked = dict(source_sha="source")
    monkeypatch.setattr(control, "WAIT", tmp_path / "previous")
    monkeypatch.setattr(control.driver, "checked", lambda *a: checked)
    monkeypatch.setattr(control.driver, "cpu_proof", lambda *a: None)
    monkeypatch.setattr(control.driver, "read", lambda *a: dict(analysis="123"))
    jobs, records = [], []
    monkeypatch.setattr(control, "submit", lambda *a: jobs.append(a) or "456")
    monkeypatch.setattr(control, "write_once", lambda *a: records.append(a))
    control.release(tmp_path, "plan")
    assert jobs == [(tmp_path, "release", "123")]
    assert records[0][0].name == "deferred_release.json"
    assert records[0][1] == dict(release="456", afterany="123")
