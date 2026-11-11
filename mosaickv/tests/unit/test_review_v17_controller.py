"""Synthetic scheduler/control fixtures; never submit real jobs or access labels."""

from types import SimpleNamespace

import evicastkv_review_v17 as control
import pytest

from mosaickv.evaluation.fox_campaign import write_once


@pytest.fixture(autouse=True)
def clear_scheduler_environment(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)


@pytest.fixture
def ready(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: dict(source_sha="test"))
    return tmp_path


def test_all_commands_are_cpu_only_with_exact_resource_profiles(ready):
    cases = [
        ("tests", dict(tests=True), 12, 20),
        ("prepare", {}, 4, 15),
        ("oversize-worker", dict(workers=2), 1, 5),
        ("after-oversize", {}, 4, 15),
        ("finish", {}, 4, 15),
    ]
    for mode, kwargs, cores, minutes in cases:
        cmd = control.command(ready, mode, mode, **kwargs)
        assert "--partition=bii,standard" in cmd
        assert f"--cpus-per-task={cores}" in cmd
        assert "--mem=16G" in cmd and f"--time=00:{minutes:02d}:00" in cmd
        assert not any(
            token.startswith(("--gres", "--gpus", "--reservation", "--tres-per-task", "-G"))
            or "bii-gpu" in token
            for token in cmd
        )
    assert "--array=0-1%2" in control.command(ready, "worker", "oversize-worker", workers=2)


@pytest.mark.parametrize(
    "mode,kwargs",
    [
        ("ownership-worker", {}),
        ("tests", {}),
        ("register", {}),
        ("prepare", dict(workers=2)),
        ("oversize-worker", dict(workers=8)),
        ("oversize-worker", {}),
    ],
)
def test_no_unregistered_mode_or_larger_array(ready, mode, kwargs):
    with pytest.raises(ValueError):
        control.command(ready, "bad", mode, **kwargs)


def test_write_once_submission_and_afterany_dependencies(ready, monkeypatch):
    calls = []
    monkeypatch.setattr(
        control.subprocess, "check_output", lambda cmd, **kw: calls.append(cmd) or "123;cluster\n"
    )
    assert (
        control.submit(ready, "after", "after-oversize", dependencies=["11", "11", "12"]) == "123"
    )
    assert "--dependency=afterany:11:12" in calls[0]
    assert control.read(ready / "jobs/after-submission.json")["job"] == "123"
    with pytest.raises(FileExistsError):
        control.submit(ready, "after", "after-oversize")
    assert len(calls) == 1


def test_unacknowledged_request_is_recorded_and_not_retried(ready, monkeypatch):
    def fail(*args, **kwargs):
        raise TimeoutError("scheduler ACK unavailable")

    monkeypatch.setattr(control.subprocess, "check_output", fail)
    with pytest.raises(control.shared.SubmissionError):
        control.submit(ready, "producer", "oversize-worker", workers=2)
    with pytest.raises(FileExistsError):
        control.submit(ready, "producer", "oversize-worker", workers=2)
    report = control.finish(ready)
    assert report["unacknowledged_requests"] == ["producer"]
    assert report["status"] == "blocked_unacknowledged_CPU_submission"
    assert report["GPU_requests"] == 0 and not report["fresh_confirmation_established"]


def test_cpu_failure_stops_before_recovery_preparation(ready, monkeypatch):
    def fail(*args):
        raise ValueError("canonical CPU suite incomplete")

    monkeypatch.setattr(control.shared, "cpu_gate", fail)
    monkeypatch.setattr(control, "module", lambda: pytest.fail("CPU failure must block producer"))
    result = control.prepare(ready)
    assert result["recovery_evidence"]["status"] == "missing_recovery_gate_not_scientific_success"
    assert result["errors"] and result["GPU_requests"] == 0


@pytest.mark.parametrize("slurm_id", [None, "92731"])
def test_normal_chain_has_finite_array_then_afterany(ready, monkeypatch, slurm_id):
    if slurm_id is not None:
        monkeypatch.setenv("SLURM_JOB_ID", slurm_id)
    monkeypatch.setattr(control.shared, "cpu_gate", lambda *args: dict(status="passed"))
    seen = []
    monkeypatch.setattr(
        control,
        "module",
        lambda: SimpleNamespace(prepare=lambda root, **kw: dict(workers=2, pending_files=15)),
    )

    def submit(root, label, mode, **kwargs):
        seen.append((label, mode, kwargs))
        return "10" if label == "oversize" else "11"

    monkeypatch.setattr(control, "submit", submit)
    result = control.prepare(ready)
    assert result["producer"] == "10" and result["analyzer"] == "11"
    assert seen[0] == ("oversize", "oversize-worker", dict(workers=2))
    assert seen[1] == (
        "after-oversize",
        "after-oversize",
        dict(dependencies=["10"] + ([slurm_id] if slurm_id else [])),
    )


def test_wrong_prepare_worker_count_fails_before_submission(ready, monkeypatch):
    monkeypatch.setattr(control.shared, "cpu_gate", lambda *args: dict(status="passed"))
    monkeypatch.setattr(
        control, "module", lambda: SimpleNamespace(prepare=lambda *a, **kw: dict(workers=3))
    )
    monkeypatch.setattr(control, "submit", lambda *a, **kw: pytest.fail("unregistered array"))
    assert control.prepare(ready)["errors"]


def test_worker_cannot_run_without_cpu_gate_or_prepared_array(ready, monkeypatch):
    monkeypatch.setattr(control.shared, "cpu_gate", lambda *args: dict(status="passed"))
    seen = []
    monkeypatch.setattr(
        control, "module", lambda: SimpleNamespace(worker=lambda *a: seen.append(a))
    )
    with pytest.raises(ValueError):
        control.worker(ready, True)
    with pytest.raises(FileNotFoundError):
        control.worker(ready, 0)
    write_once(ready / "oversize-prepared.json", dict(workers=2))
    control.worker(ready, 1)
    assert seen == [(ready / "oversize", 1)]


def test_afterany_analysis_preserves_negative_gate_without_promoting_claims(ready, monkeypatch):
    def analyze(root):
        write_once(root / control.RECOVERY_GATE, dict(status="bounded_negative", remaining=7))

    monkeypatch.setattr(control, "module", lambda: SimpleNamespace(analyze=analyze))
    report = control.after(ready)
    assert report["recovery_evidence"]["gate"]["remaining"] == 7
    for key, value in control.NO_CLAIMS.items():
        assert report[key] == value
    assert report["typed_KV_refinement_not_run_by_controller"] is True


def test_analyzer_exception_is_finalized_as_failure_not_pass(ready, monkeypatch):
    def fail(*args):
        raise ValueError("incomplete recovery worker")

    monkeypatch.setattr(control, "module", lambda: SimpleNamespace(analyze=fail))
    report = control.after(ready)
    assert report["errors"]
    assert report["recovery_evidence"]["status"] == "missing_recovery_gate_not_scientific_success"


def test_wrapper_disables_gpus_and_preserves_canonical_unit_threads():
    source = (control.SOURCE / "mosaickv/slurm/evicastkv_review_v17.sbatch").read_text()
    assert 'export CUDA_VISIBLE_DEVICES=""' in source
    tests = source.split('if [[ "$3" == "tests" ]]; then', 1)[1].split("fi", 1)[0]
    assert "export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2" in tests
    assert "evicastkv_review_closure_v2.py tests" in tests
