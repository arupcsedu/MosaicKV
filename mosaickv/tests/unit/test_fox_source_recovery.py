from __future__ import annotations

import json
from pathlib import Path

import evicastkv_review_source_recovery as control
import pytest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def registration(root):
    value = dict(
        source_sha="new",
        producer_source="/old-source",
        producer_sha=control.OLD_SHA,
        old_root="/old-root",
        old_final_job="123",
        execute=True,
        bindings={},
    )
    write(root / "registration.json", value)
    return value


@pytest.mark.parametrize("bridge,tests", [(True, False), (False, False), (False, True)])
def test_submission_bounds_and_afterany(tmp_path, monkeypatch, bridge, tests):
    registration(tmp_path)
    monkeypatch.setattr(control.subprocess, "check_output", lambda *args, **kwargs: "1234;cluster")
    assert control.submit(tmp_path, "once", "recover", [100], bridge=bridge, tests=tests) == "1234"
    command = control.read(tmp_path / "jobs/once-request.json")["command"]
    assert "--dependency=afterany:100" in command
    assert "--cpus-per-task=12" in command
    assert "--account=bii_dsc_community" in command
    if bridge:
        for option in (
            "--array=0-1%2",
            "--gres=gpu:a100:1",
            "--reservation=bi_fox_dgx",
            "--time=02:00:00",
            "--partition=bii-gpu",
        ):
            assert option in command
        assert command[-3:] == ["/old-source", "/old-root", "bridge-worker"]
    else:
        assert "--partition=bii" in command and "--time=00:30:00" in command
        assert not any("--gres" in option or "--reservation" in option for option in command)
    with pytest.raises(FileExistsError):
        control.submit(tmp_path, "once", "recover", tests=tests, bridge=bridge)


def passing_gate():
    return dict(
        complete=True,
        actual_rows=14,
        completed_rows=14,
        failures=[],
        identical_reference_conditioning_verified=True,
        all_source_variants_pass_registered_guard=False,
        recomputation_verifications=[dict(exact_guard_booleans_checked=4) for _ in range(14)],
    )


def clone_rows():
    return [
        dict(
            arm="fullkv_forced_clone_control",
            status="completed",
            diagnostics=dict(prefix_128=dict(token_agreement=1.0, maximum_logit_difference=0.0)),
        )
        for _ in range(2)
    ]


def test_lossy_source_failures_are_outcomes_not_bridge_admission_requirements():
    assert control.recovered_gate_allows_bridge(passing_gate(), clone_rows())


@pytest.mark.parametrize("change", ["rows", "failure", "clone", "guard_verification"])
def test_incomplete_or_changed_guard_blocks_bridge(change):
    gate, rows = passing_gate(), clone_rows()
    if change == "rows":
        gate["actual_rows"] = 13
    elif change == "failure":
        gate["failures"] = ["different token"]
    elif change == "guard_verification":
        gate["recomputation_verifications"][0]["exact_guard_booleans_checked"] = 3
    else:
        rows[1]["diagnostics"]["prefix_128"]["maximum_logit_difference"] = 0.00101
    assert not control.recovered_gate_allows_bridge(gate, rows)


def test_failed_cpu_never_runs_analysis_or_gpu(tmp_path, monkeypatch):
    record = registration(tmp_path)
    monkeypatch.setattr(control, "load", lambda root: record)

    def fail(*args):
        raise ValueError("CPU failure")

    monkeypatch.setattr(control, "cpu_gate", fail)
    monkeypatch.setattr(control, "gpu_snapshot", lambda: pytest.fail("CPU failure admitted GPU"))
    calls = []
    monkeypatch.setattr(
        control, "submit", lambda *args, **kwargs: calls.append((args, kwargs)) or "321"
    )
    result = control.recover(tmp_path)
    assert result["status"] == "blocked"
    assert len(calls) == 1 and calls[0][0][1] == "final"
    assert "CPU failure" in control.read(tmp_path / "recovery_error.json")["error"]


def test_failed_analysis_never_submits_gpu_and_preserves_raw(tmp_path, monkeypatch):
    import evicastkv_review_source_prefix as analyzer

    old, target = tmp_path / "old", tmp_path / "derived"
    target.mkdir()
    record = registration(target)
    record["old_root"] = str(old)
    plan = old / "source-prefix/plan.json"
    write(plan, dict(immutable_plan=True))
    (old / "source-prefix/workers").mkdir()
    original = plan.read_bytes()
    monkeypatch.setattr(control, "load", lambda root: record)
    monkeypatch.setattr(control, "cpu_gate", lambda *args: dict(status="passed"))

    def fail(*args, **kwargs):
        raise ValueError("guard changed")

    monkeypatch.setattr(analyzer, "analyze", fail)
    monkeypatch.setattr(
        control, "gpu_snapshot", lambda: pytest.fail("failed analyzer admitted GPU")
    )
    calls = []
    monkeypatch.setattr(control, "submit", lambda *args, **kwargs: calls.append(kwargs) or "321")
    assert control.recover(target)["status"] == "blocked"
    assert plan.read_bytes() == original
    assert (target / "source-prefix/plan.json").read_bytes() == original
    assert (target / "source-prefix/workers").is_symlink()
    assert not any(call.get("bridge") for call in calls)


def test_load_checks_both_clean_sources_and_binding_hashes(tmp_path, monkeypatch):
    record = registration(tmp_path)
    calls = []
    monkeypatch.setattr(control, "clean_source", lambda *args: calls.append(args))
    assert control.load(tmp_path) == record
    assert calls[1] == (Path("/old-source"), control.OLD_SHA)
    record["execute"] = False
    write(tmp_path / "registration.json", record)
    with pytest.raises(ValueError, match="dry-run"):
        control.load(tmp_path)


def test_derived_finish_never_reclassifies_original_negative_result(tmp_path, monkeypatch):
    old, root = tmp_path / "old", tmp_path / "new"
    root.mkdir()
    record = registration(root)
    record["old_root"] = str(old)
    write(old / "FINAL_report.json", dict(shape_numerical_gate=False))
    monkeypatch.setattr(control, "load", lambda _: record)
    result = control.finish(root)
    assert result["original_negative_boundary_preserved"]
    assert not result["superiority_established"]
    assert not result["confirmation_authorized"]
    read_original = control.read(old / "FINAL_report.json")
    assert read_original == dict(shape_numerical_gate=False)


def test_launcher_sets_cpu_isa_before_python_and_no_gpu():
    launcher = control.SOURCE / "mosaickv/slurm/evicastkv_review_source_recovery.sbatch"
    source = launcher.read_text()
    assert source.index("ATEN_CPU_CAPABILITY=avx2") < source.index("exec /scratch")
    assert 'CUDA_VISIBLE_DEVICES=""' in source
    assert "OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12" in source


def test_dry_run_binds_existing_workers_without_scheduler_calls(tmp_path, monkeypatch):
    old, root = tmp_path / "old", tmp_path / "new"
    write(
        old / "registration.json",
        dict(source_root="old-clean", source_sha=control.OLD_SHA, bindings={}),
    )
    for index in range(2):
        write(
            old / f"source-prefix/workers/{index}/result.json",
            dict(
                status="completed",
                failures=[],
                source_sha=control.OLD_SHA,
                outputs={
                    name: "bound"
                    for name in (
                        "raw.jsonl",
                        "logits.pt",
                        "manifest.json",
                        "reference_prefix.json",
                        "parity_controls.json",
                    )
                },
            ),
        )
    write(old / "jobs/final-submission.json", dict(job="123"))
    monkeypatch.setattr(control, "safe_root", Path)
    monkeypatch.setattr(control, "clean_source", lambda *args: "newsha")
    monkeypatch.setattr(control, "sha", lambda *args: "bound")
    monkeypatch.setattr(control, "submit", lambda *args, **kwargs: pytest.fail("dry-run submitted"))
    assert control.register(root, old)["submitted_jobs"] == 0
    sealed = control.read(root / "registration.json")
    assert not sealed["execute"]
    assert sealed["source_gpu_retries"] == 0
    assert not sealed["fresh_confirmation_authorized"]


def test_valid_recovery_admits_only_old_prepared_bridge_after_other_gpu_jobs(tmp_path, monkeypatch):
    import evicastkv_review_source_prefix as analyzer

    old, root = tmp_path / "old", tmp_path / "new"
    root.mkdir()
    record = registration(root)
    record["old_root"] = str(old)
    write(old / "source-prefix/plan.json", dict(immutable=True))
    for index in range(2):
        path = old / f"source-prefix/workers/{index}/raw.jsonl"
        path.parent.mkdir(parents=True)
        records = [clone_rows()[index], *(dict(arm=f"source-{j}") for j in range(6))]
        path.write_text("\n".join(json.dumps(row) for row in records))
    monkeypatch.setattr(control, "load", lambda _: record)
    monkeypatch.setattr(control, "cpu_gate", lambda *args: dict(status="passed"))
    monkeypatch.setattr(
        analyzer,
        "analyze",
        lambda target, **kwargs: control.write_once(
            target / "analysis/GATE_source_fixed_prefix.json", passing_gate()
        ),
    )
    monkeypatch.setattr(
        control,
        "gpu_snapshot",
        lambda: dict(active=[dict(job="999_0")], pending=[dict(job="998_1")], active_gpus=1),
    )
    calls = []
    monkeypatch.setattr(
        control,
        "submit",
        lambda *args, **kwargs: calls.append((args, kwargs)) or str(500 + len(calls)),
    )
    result = control.recover(root)
    assert result["status"] == "bridge_submitted"
    assert len(calls) == 2
    assert calls[0][0][1:4] == ("bridge", "bridge-worker", ["998", "999"])
    assert calls[0][1] == dict(bridge=True)
    assert calls[1][0][1:3] == ("bridge-analysis", "bridge-analyze")
    assert not (old / "source-prefix/analysis").exists()
