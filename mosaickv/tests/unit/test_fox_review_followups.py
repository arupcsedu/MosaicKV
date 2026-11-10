"""Synthetic finite callback and six-GPU admission tests; no Slurm access."""

import json
import sys
from types import SimpleNamespace

import evicastkv_review_followups as module
import pytest

from mosaickv.evaluation.fox_campaign import sha, write_once


@pytest.fixture
def plan(tmp_path, monkeypatch):
    source, parent, mechanisms, mile = [
        tmp_path / name for name in ("source", "parent", "mechanisms", "mile")
    ]
    monkeypatch.setattr(module, "SOURCE", source)
    monkeypatch.setattr(module, "clean_source", lambda *args: "c" * 40)
    for name in (
        "evicastkv_review_followups.py",
        "evicastkv_caption_audit.py",
        "evicastkv_transport_scores.py",
        "evicastkv_graph_profile.py",
        "evicastkv_window_study.py",
        "evicastkv_mile_parser_audit.py",
    ):
        path = source / "mosaickv/scripts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source fixture")
    wrapper = source / "mosaickv/slurm/evicastkv_review_followups.sbatch"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text("synthetic wrapper fixture")
    write_once(parent / "protocol_lock.json", dict(source_root=str(source), source_sha="c" * 40))
    for stage in ("correctness", "scoring", "profiling", "measurement"):
        write_once(
            parent / f"analysis/{stage}/GATE_{stage}.json",
            dict(
                status="passed",
                source_sha="c" * 40,
                protocol_sha256=sha(parent / "protocol_lock.json"),
                blockers=[],
            ),
        )
    write_once(
        mechanisms / "protocol_lock.json",
        dict(
            source_root=str(source),
            source_sha="c" * 40,
            parent_protocol_sha256=sha(parent / "protocol_lock.json"),
        ),
    )
    write_once(
        mechanisms / "jobs/mechanisms-submission.json", dict(analysis="12345", array="12344")
    )
    for name in ("GATE_scoring.json", "registration.json", "paired_statistics.json"):
        write_once(mile / name, dict(synthetic=True))
    commands = []

    def scheduler(command, **_kwargs):
        commands.append(command)
        return str(20000 + len(commands)) + "\n"

    monkeypatch.setattr(module.subprocess, "check_output", scheduler)
    return SimpleNamespace(
        root=tmp_path / "new",
        source=source,
        parent=parent,
        mechanisms=mechanisms,
        mile=mile,
        commands=commands,
    )


def register(plan, *, execute=False):
    return module.register(plan.root, plan.parent, plan.mechanisms, plan.mile, execute=execute)


def test_dry_run_is_default_and_cannot_advance(plan):
    result = register(plan)
    assert result["status"] == "DRY_RUN_ONLY"
    assert plan.commands == []
    assert result["fresh_confirmation_authorized"] is False
    with pytest.raises(ValueError, match="dry-run"):
        module.load(plan.root)
    with pytest.raises(FileExistsError):
        register(plan, execute=True)


def test_execution_submits_cpu_first_and_afterany_callbacks_only(plan):
    result = register(plan, execute=True)
    assert result["status"] == "submitted_finite_chain"
    assert len(plan.commands) == 3
    assert all(not any("gres=" in arg for arg in cmd) for cmd in plan.commands)
    assert plan.commands[0][-1] == "tests"
    assert "--dependency=afterany:20001" in plan.commands[1]
    assert "--dependency=afterany:20002:12345" in plan.commands[2]
    assert all("afterok:" not in " ".join(command) for command in plan.commands)


@pytest.mark.parametrize("stage", ["correctness", "scoring", "profiling", "measurement"])
def test_any_failed_parent_control_blocks_registration(plan, stage):
    path = plan.parent / f"analysis/{stage}/GATE_{stage}.json"
    data = json.loads(path.read_text())
    data["status"] = "blocked"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=stage):
        register(plan, execute=True)
    assert plan.commands == []
    assert not plan.root.exists()


def test_registered_source_or_historical_hash_change_blocks_callbacks(plan):
    register(plan, execute=True)
    (plan.mile / "paired_statistics.json").write_text("changed fixture")
    with pytest.raises(ValueError, match="evidence changed"):
        module.load(plan.root)


def test_pending_own_gpu_jobs_are_waited_for_before_new_admission():
    snapshot = dict(
        active_gpus=4,
        active=[dict(job="100_1", gpus=1), dict(job="100_2", gpus=3)],
        pending=[dict(job="200_3", gpus=1)],
    )
    gate = module.admission(snapshot, 6)
    assert gate["wait_for_existing_own_gpu_jobs"] == ["100", "200"]
    assert gate["admitted_array_concurrency"] == 6
    assert gate["maximum_a100s"] == 6


def test_empty_scheduler_allows_one_graph_or_six_windows_only():
    snapshot = dict(active_gpus=0, active=[], pending=[])
    assert module.admission(snapshot, 1)["admitted_array_concurrency"] == 1
    assert module.admission(snapshot, 6)["admitted_array_concurrency"] == 6
    with pytest.raises(ValueError):
        module.admission(snapshot, 8)


def test_gpu_snapshot_counts_active_and_retains_pending(monkeypatch):
    monkeypatch.setenv("USER", "synthetic")
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *a, **kw: (
            "10_0|gpu:a100:1|RUNNING\n11|(null)|RUNNING\n12_0|gpu:a100:1|PENDING\n13|gpu:2|COMPLETING\n"
        ),
    )
    snapshot = module.gpu_snapshot()
    assert snapshot["active_gpus"] == 3
    assert len(snapshot["pending"]) == 1
    assert snapshot["command"][snapshot["command"].index("-u") + 1] == "synthetic"


def test_finite_submission_cannot_retry_in_place(plan):
    register(plan, execute=True)
    module.submit(plan.root, "unique", "after-graph", ["123"])
    count = len(plan.commands)
    with pytest.raises(FileExistsError):
        module.submit(plan.root, "unique", "after-graph", ["123"])
    assert len(plan.commands) == count


def test_gpu_resource_contract_is_one_a100_twelve_cpus_two_hours(plan):
    register(plan, execute=True)
    module.submit(
        plan.root,
        "synthetic-gpu",
        "graph-worker",
        ["123"],
        array="0%1",
        spec_sha="a" * 64,
        gpu=True,
    )
    command = plan.commands[-1]
    for flag in (
        "--gres=gpu:a100:1",
        "--cpus-per-task=12",
        "--time=02:00:00",
        "--mem=64G",
        "--reservation=bi_fox_dgx",
        "--array=0%1",
        "--dependency=afterany:123",
    ):
        assert flag in command


def mechanism_gate(plan, *, status):
    write_once(plan.root / "GATE_cpu_audits.json", dict(status="passed", bindings={}))
    write_once(
        plan.mechanisms / "analysis/mechanisms/GATE_mechanisms.json",
        dict(
            status=status, protocol_sha256=sha(plan.mechanisms / "protocol_lock.json"), blockers=[]
        ),
    )


def test_failed_mechanisms_stop_before_any_followup_gpu(plan):
    register(plan, execute=True)
    mechanism_gate(plan, status="blocked")
    count = len(plan.commands)
    with pytest.raises(ValueError, match="empirical gate"):
        module.run(plan.root, "after-mechanisms")
    assert len(plan.commands) == count


def test_negative_fullroute_pilot_is_not_inherited_as_original_policy_pass(plan, monkeypatch):
    register(plan, execute=True)
    mechanism_gate(plan, status="passed")
    write_once(
        plan.mechanisms / "analysis/full_graph_pilot/GATE_full_graph_pilot.json",
        dict(status="blocked", blockers=["synthetic negative pilot"]),
    )

    def prepare(root, parent):
        assert parent == plan.parent
        write_once(root / "jobs/graph_cpu_profile.json", dict(jobs=[dict(indices=[0, 1])]))

    monkeypatch.setitem(
        sys.modules,
        "evicastkv_graph_profile",
        SimpleNamespace(STAGE="graph_cpu_profile", prepare=prepare),
    )
    monkeypatch.setattr(module, "gpu_snapshot", lambda: dict(active_gpus=0, active=[], pending=[]))
    module.run(plan.root, "after-fullgraph")
    decision = json.loads((plan.root / "decisions/fullgraph.json").read_text())
    assert decision["full_graph_pilot_status"] == "blocked"
    assert decision["inherited_as_original_policy_control"] is False
    assert plan.commands[-1][-1] == "after-graph"
    assert sum("--gres=gpu:a100:1" in cmd for cmd in plan.commands) == 1


def test_worker_entry_refuses_more_than_six_own_gpus(plan, monkeypatch):
    register(plan, execute=True)
    monkeypatch.setattr(module, "gpu_snapshot", lambda: dict(active_gpus=7, active=[], pending=[]))
    with pytest.raises(RuntimeError, match="six-A100"):
        module.run(plan.root, "graph-worker", spec_sha="a" * 64)


def test_graph_failure_blocks_windows(plan, monkeypatch):
    register(plan, execute=True)

    def analyze(root):
        write_once(
            root / "analysis/graph_cpu_profile/GATE_graph_cpu_profile.json",
            dict(status="blocked", blockers=["synthetic failure"]),
        )

    monkeypatch.setitem(sys.modules, "evicastkv_graph_profile", SimpleNamespace(analyze=analyze))
    count = len(plan.commands)
    with pytest.raises(ValueError, match="empirical gate"):
        module.run(plan.root, "after-graph")
    assert len(plan.commands) == count


def test_cpu_audits_bind_every_derived_gate_without_gpu_submission(plan, monkeypatch):
    register(plan, execute=True)
    monkeypatch.setattr(
        module, "cpu_gate", lambda *args: dict(status="passed", counts=dict(tests=5))
    )
    calls = []

    def derived(name, gate_name):
        def audit(parent, output):
            calls.append((name, parent))
            write_once(output / gate_name, dict(status="passed_fixture"))

        return audit

    monkeypatch.setitem(
        sys.modules,
        "evicastkv_caption_audit",
        SimpleNamespace(audit=derived("captions", "GATE_caption_audit.json")),
    )
    monkeypatch.setitem(
        sys.modules,
        "evicastkv_transport_scores",
        SimpleNamespace(reexport=derived("quality", "GATE_quality_statistics.json")),
    )
    monkeypatch.setitem(
        sys.modules,
        "evicastkv_mile_parser_audit",
        SimpleNamespace(audit=derived("parser", "GATE_parser_audit.json")),
    )
    count = len(plan.commands)
    module.run(plan.root, "cpu-audits")
    assert len(plan.commands) == count
    assert [call[0] for call in calls] == ["captions", "quality", "parser"]
    gate = module.require_cpu_audits(plan.root)
    assert len(gate["bindings"]) == 4
    (plan.root / "mile-parser/GATE_parser_audit.json").write_text("changed fixture")
    with pytest.raises(ValueError, match="gate changed"):
        module.require_cpu_audits(plan.root)


def test_window_completion_without_registered_boundary_is_explicitly_scoped(plan, monkeypatch):
    register(plan, execute=True)

    def analyze(root):
        write_once(root / "protocol_lock.json", dict(synthetic=True))
        write_once(
            root / "analysis/window_sensitivity/GATE_window_sensitivity.json",
            dict(
                status="passed_exploratory_sensitivity_not_confirmation",
                protocol_sha256=sha(root / "protocol_lock.json"),
            ),
        )

    monkeypatch.setitem(
        sys.modules,
        "evicastkv_window_study",
        SimpleNamespace(STAGE="window_sensitivity", analyze=analyze),
    )
    count = len(plan.commands)
    report = module.run(plan.root, "after-window")
    assert len(plan.commands) == count
    assert report["status"] == "COMPLETED_CONSUMED_DIAGNOSTICS_ONLY"
    assert report["boundary_operator_diagnostic"].startswith("READY_NOT_RUN")
    assert report["superiority_established"] is False
    assert report["fresh_confirmation_authorized"] is False


def test_negative_boundary_control_does_not_reverse_separate_a_study(plan, monkeypatch):
    (plan.source / "mosaickv/scripts/evicastkv_boundary_operators.py").write_text(
        "synthetic fixture"
    )
    register(plan, execute=True)

    def analyze(root):
        write_once(
            root / "GATE_boundary_operators.json",
            dict(status="passed_diagnostic_capture", fullkv_chunked_logit_control_all_passed=False),
        )

    monkeypatch.setitem(
        sys.modules, "evicastkv_boundary_operators", SimpleNamespace(analyze=analyze)
    )
    report = module.run(plan.root, "after-boundary")
    assert report["status"] == "COMPLETED_CONSUMED_A_STUDIES_BOUNDARY_SEPARATE"
    assert report["fullkv_chunked_logit_control_all_passed"] is False
    assert report["boundary_quality_inference_authorized"] is False
    assert report["superiority_established"] is False


def test_failed_boundary_analysis_remains_explicit_without_reversing_a_study(plan, monkeypatch):
    (plan.source / "mosaickv/scripts/evicastkv_boundary_operators.py").write_text(
        "synthetic fixture"
    )
    register(plan, execute=True)

    def analyze(root):
        raise ValueError("synthetic missing capture")

    monkeypatch.setitem(
        sys.modules, "evicastkv_boundary_operators", SimpleNamespace(analyze=analyze)
    )
    report = module.run(plan.root, "after-boundary")
    assert report["boundary_diagnostic_status"] == "blocked_diagnostic_execution"
    assert "synthetic missing capture" in report["boundary_analysis_error"]
    assert report["superiority_established"] is False


@pytest.mark.parametrize("bad", ["skipped", "failures", "errors"])
def test_complete_cpu_suite_requires_zero_failures_errors_and_skips(tmp_path, bad):
    root = tmp_path
    junit = root / "full-cpu/cpu-tests.xml"
    junit.parent.mkdir(parents=True)
    junit.write_text(f'<testsuites><testsuite tests="5" {bad}="1"/></testsuites>')
    log = junit.with_name("cpu-tests.log")
    log.write_text("synthetic test fixture")
    write_once(
        junit.with_name("cpu-tests.json"),
        dict(
            status="passed",
            returncode=0,
            source_sha="c" * 40,
            log_sha256=sha(log),
            junit_sha256=sha(junit),
        ),
    )
    with pytest.raises(ValueError, match="CPU suite"):
        module.cpu_gate(root, "c" * 40)
