import json

import evicastkv_review_freshness_retry as control
import pytest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.mark.parametrize("tests,array", [(True, False), (False, True), (False, False)])
def test_cpu_only_resources_and_afterany(tmp_path, monkeypatch, tests, array):
    monkeypatch.setattr(control.subprocess, "check_output", lambda *args, **kwargs: "123;cluster")
    assert control.submit(tmp_path, "once", "worker", [1], tests=tests, array=array) == "123"
    args = control.read(tmp_path / "jobs/once-request.json")["command"]
    for expected in (
        "--partition=bii",
        "--cpus-per-task=4",
        "--mem=16G",
        "--time=00:15:00",
        "--dependency=afterany:1",
    ):
        assert expected in args
    assert not any("gres" in arg or "reservation" in arg for arg in args)
    assert ("--array=0-7%4" in args) is array
    with pytest.raises(FileExistsError):
        control.submit(tmp_path, "once", "worker")


def test_failed_cpu_suite_submits_no_scanner(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: dict(source_sha="abc"))

    def failure(*args):
        raise ValueError("failed suite")

    monkeypatch.setattr(control, "cpu_gate", failure)
    monkeypatch.setattr(control, "submit", lambda *args, **kwargs: pytest.fail("CPU gate bypassed"))
    result = control.run(tmp_path, "launch")
    assert result["status"] == "blocked_cpu"
    assert not result["fresh_confirmation_authorized"]


def test_passing_cpu_submits_bounded_workers_then_afterany_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: dict(source_sha="abc"))
    monkeypatch.setattr(control, "cpu_gate", lambda *args: dict(status="passed"))
    calls = []
    monkeypatch.setattr(
        control,
        "submit",
        lambda *args, **kwargs: calls.append((args, kwargs)) or str(100 + len(calls)),
    )
    result = control.run(tmp_path, "launch")
    assert result == dict(workers="101", analysis="102")
    assert calls[0][1] == dict(array=True)
    assert calls[1][0][-1] == ["101"]


def test_aggregation_retains_incomplete_status_without_promotion(tmp_path, monkeypatch):
    import evicastkv_review_freshness_worker as scanner

    monkeypatch.setattr(control, "load", lambda root: dict(source_sha="abc"))
    monkeypatch.setattr(
        scanner, "aggregate", lambda root: dict(status="incomplete", missing_shards=[7])
    )
    result = control.run(tmp_path, "analyze")
    assert result["scanner_gate"]["missing_shards"] == [7]
    assert not result["fresh_confirmation_authorized"]
    assert not result["cohort_selected"]
    assert result["GPU_jobs"] == 0


def test_dry_run_lock_cannot_execute(tmp_path, monkeypatch):
    write(tmp_path / "registration.json", dict(source_sha="abc", execute=False))
    monkeypatch.setattr(control, "clean_source", lambda *args: "abc")
    with pytest.raises(ValueError, match="dry-run"):
        control.load(tmp_path)


def test_launch_template_disables_gpu_and_bounds_threads():
    text = (control.SOURCE / "mosaickv/slurm/evicastkv_review_freshness_retry.sbatch").read_text()
    assert 'CUDA_VISIBLE_DEVICES=""' in text
    assert "OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4" in text
    assert "assert_clean_worktree.sh" in text
