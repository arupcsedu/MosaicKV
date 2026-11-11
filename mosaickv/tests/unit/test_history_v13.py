"""Event-driven finite continuation, mocked scheduler; no real scans or jobs."""

import json

import evicastkv_history_v13 as study
import pytest


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def context(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    put(parent / "registration.json", dict(source_sha=study.PARENT_SHA, source_root=str(tmp_path)))
    put(parent / "jobs/after-frontier-submission.json", dict(job="100"))
    monkeypatch.setattr(study, "SCRATCH", tmp_path)
    monkeypatch.setattr(study, "clean_source", lambda *a: "b" * 40)
    monkeypatch.setattr(study, "cpu_gate", lambda *a: dict(status="passed"))
    commands = []
    monkeypatch.setattr(
        study.subprocess,
        "check_output",
        lambda command, **k: commands.append(command) or str(500 + len(commands)),
    )
    root = tmp_path / "new"
    return root, parent, commands


def register(context, **kwargs):
    root, parent, _ = context
    return study.register(root, cpu_gate_root=root.parent / "cpu", parent_campaign=parent, **kwargs)


def test_register_submits_only_afterany_current_actual_parent_analyzer(context):
    result = register(context)
    root, _, commands = context
    assert result["callback"] == "501"
    command = commands[0]
    assert "--dependency=afterany:100" in command
    assert "--cpus-per-task=4" in command
    assert not any("gpu" in x.lower() for x in command)
    assert (root / "registration.json").is_file()


def test_already_selected_parent_does_not_depend_on_purged_historical_scheduler_job(
    context, monkeypatch
):
    _, parent, commands = context
    put(parent / "frontier-selection.json", dict(index=4))  # validated later by attach
    monkeypatch.setenv("SLURM_JOB_ID", "900")
    register(context)
    assert "--dependency=afterany:900" in commands[0]
    assert "--dependency=afterany:100" not in commands[0]


def test_dry_registration_does_not_scan_or_submit(context):
    result = register(context, execute=False)
    assert result["jobs"] == 0 and not context[2]
    with pytest.raises(ValueError, match="executable"):
        study.load(context[0])


def test_failed_CPU_gate_prevents_registration_and_all_jobs(context, monkeypatch):
    monkeypatch.setattr(
        study, "cpu_gate", lambda *a: (_ for _ in ()).throw(ValueError("CPU failed"))
    )
    with pytest.raises(ValueError, match="CPU failed"):
        register(context)
    assert not context[0].exists() and not context[2]


def test_reattach_uses_next_existing_analyzer_without_poll_or_sleep(context):
    register(context)
    root, parent, commands = context
    put(parent / "frontier-segment-2-jobs.json", dict(worker="200", analyzer="201"))
    value = study.attach(root, 0)
    assert value["reattached_to"] == "201"
    assert "--dependency=afterany:201" in commands[-1]
    assert len(commands) == 2


def test_parent_no_progress_or_exhausted_hop_budget_stops_without_resubmission(context):
    register(context)
    value = study.attach(context[0], 0)
    assert value["stop_reason"] == "parent_selection_unavailable"
    assert not value["fresh_confirmation_authorized"]
    assert len(context[2]) == 1


def test_missing_parent_selection_with_hop_three_is_bounded(context):
    register(context)
    root, parent, commands = context
    put(root / "parent-attachments/3.json", dict(index=3))
    put(parent / "frontier-segment-4-jobs.json", dict(worker="400", analyzer="401"))
    assert study.attach(root, 3)["stop_reason"] == "parent_selection_unavailable"
    assert len(commands) == 1


def test_ready_parent_imports_exact_selected_states_and_one_eight_CPU_array(context, monkeypatch):
    register(context)
    root, parent, commands = context
    selected = parent / "frontier"
    put(
        selected / study.GATE,
        dict(source_sha=study.PARENT_SHA, missing_shards=[], frontier_exhausted=False),
    )
    put(selected / "plan.json", dict(source_sha=study.PARENT_SHA))
    put(
        parent / "frontier-selection.json",
        dict(index=1, path=str(selected), gate_sha256=study.sha(selected / study.GATE)),
    )
    calls = []
    monkeypatch.setattr(
        study.frontier, "prepare", lambda out, **kwargs: calls.append((out, kwargs))
    )
    study.attach(root, 0)
    assert calls[0][0] == root / "segment-1" and calls[0][1]["parent"] == selected
    worker, analyzer = commands[-2:]
    assert "--array=0-7%8" in worker and "--cpus-per-task=1" in worker
    assert "--time=00:05:00" in worker
    assert any(x.startswith("--dependency=afterany:") for x in analyzer)
    assert all("--partition=bii" in cmd for cmd in commands)


@pytest.mark.parametrize(
    "missing,changed,exhausted,index,expected",
    [
        ([2], False, False, 1, "missing_worker_states"),
        ([], True, False, 24, "maximum_segments"),
        ([], False, False, 1, "no_state_progress"),
        ([], True, True, 1, "frontier_exhausted"),
    ],
)
def test_new_segments_have_hard_bound_and_never_skip_missing_state(
    context, monkeypatch, missing, changed, exhausted, index, expected
):
    register(context)
    root, _, commands = context
    gate = dict(
        missing_shards=missing, frontier_exhausted=exhausted, original_and_new_coverage_errors=9
    )
    put(root / f"segment-{index}" / study.GATE, gate)
    monkeypatch.setattr(study.frontier, "analyze", lambda *a: gate)
    monkeypatch.setattr(study, "progress", lambda *a: changed)
    calls = []
    monkeypatch.setattr(
        study, "launch_yaml", lambda r, p, reason: calls.append(reason) or dict(stop=reason)
    )
    result = study.advance(root, index)
    if missing:
        assert result["stop_reason"] == expected and not calls
    else:
        assert calls == [expected]
    assert len(commands) == 1  # no additional frontier array


def test_actual_state_progress_excludes_error_only_changes_and_rejects_error_deletion(tmp_path):
    for index in range(8):
        old = dict(errors=[dict(error="preserved")], frontier=[dict(offset=0)])
        new = dict(errors=[dict(error="preserved"), dict(error="new")], frontier=[dict(offset=0)])
        put(tmp_path / f"imports/shard-{index}.json", dict(state=old))
        put(tmp_path / f"workers/shard-{index}.json", dict(state=new))
    assert not study.progress(tmp_path)
    path = tmp_path / "workers/shard-2.json"
    value = study.read(path)
    value["state"]["frontier"][0]["offset"] = 1
    put(path, value)
    assert study.progress(tmp_path)
    value["state"]["errors"] = []
    put(path, value)
    with pytest.raises(ValueError, match="removed"):
        study.progress(tmp_path)


def test_YAML_array_then_afterany_terminal_completion_binds_static_and_current_jobs(
    context, monkeypatch
):
    root, _parent, commands = context
    completion = ["sbatch", "--parsable", "--partition=bii", "dummy-root-controller.sbatch"]
    register(context, completion_argv=completion, completion_dependencies=["700", "701"])
    monkeypatch.setenv("SLURM_JOB_ID", "702")
    terminal = study.finish(root, reason="max_bound_reached")
    assert not terminal["complete_historical_inventory_certified"]
    assert "--dependency=afterany:700:701:702" in commands[-1]
    count = len(commands)
    study.finish(root, reason="duplicate_callback")
    assert len(commands) == count


def test_completion_rejects_duplicate_dependency_or_GPU_request(context):
    root, _, _ = context
    with pytest.raises(ValueError, match="CPU-only"):
        study.submit(root, "unsafe", "unused", completion=["sbatch", "--gres=gpu:a100:1"])
    with pytest.raises(ValueError, match="duplicate"):
        study.submit(root, "unsafe", "unused", completion=["sbatch", "--dependency=afterok:1"])


def test_single_YAML_recovery_is_afterany_and_cannot_be_registered_twice(context, monkeypatch):
    register(context)
    root, parent, commands = context
    preparations = []
    monkeypatch.setattr(
        study.yaml_events, "prepare", lambda out, **kwargs: preparations.append((out, kwargs))
    )
    result = study.launch_yaml(root, parent, "no_state_progress")
    assert len(preparations) == 1
    assert "--array=0-7%8" in commands[-2]
    assert f"--dependency=afterany:{result['yaml_worker']}" in commands[-1]
    with pytest.raises(FileExistsError):
        study.launch_yaml(root, parent, "duplicate")


def test_scheduler_lost_ack_is_not_blindly_retried(context, monkeypatch):
    root, _, _ = context
    monkeypatch.setattr(
        study.subprocess,
        "check_output",
        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("unknown acknowledgement")),
    )
    with pytest.raises(TimeoutError):
        register(context)
    assert (root / "jobs/attach-0-request.json").exists()
    assert (root / "jobs/attach-0-submission-error.json").exists()
    with pytest.raises(FileExistsError):
        study.submit(root, "attach-0", "attach")


def test_callback_failure_still_emits_terminal_error_gate(context, monkeypatch):
    register(context)
    monkeypatch.setattr(
        study, "attach", lambda *a: (_ for _ in ()).throw(ValueError("preserved failure"))
    )
    gate = study.run(context[0], "attach", 0)
    assert gate["status"] == "blocked_history_continuation"
    assert "preserved failure" in gate["error"]
    assert gate["error_waivers"] == 0
