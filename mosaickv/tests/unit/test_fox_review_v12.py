from pathlib import Path
from types import SimpleNamespace

import evicastkv_review_v12 as control
import pytest


@pytest.mark.parametrize(
    "mode,gpu,array,cpus,minutes",
    [
        ("precision-worker", True, False, 12, 20),
        ("after-precision", False, False, 12, 15),
        ("media-worker", False, True, 1, 5),
        ("after-media", False, False, 4, 15),
    ],
)
def test_resource_bounds_and_afterany(tmp_path, monkeypatch, mode, gpu, array, cpus, minutes):
    calls = []
    monkeypatch.setattr(
        control.subprocess,
        "check_output",
        lambda args, **kwargs: calls.append(args) or "12345;cluster\n",
    )
    assert control.submit(tmp_path, "test", mode, deps=[111], gpu=gpu, array=array) == "12345"
    command = calls[0]
    assert "--dependency=afterany:111" in command
    assert f"--cpus-per-task={cpus}" in command
    assert f"--time=00:{minutes:02d}:00" in command
    assert ("--gres=gpu:a100:1" in command) is gpu
    assert ("--reservation=bi_fox_dgx" in command) is gpu
    assert ("--array=0-7%8" in command) is array
    assert control.read(tmp_path / "jobs/test-submission.json")["job"] == "12345"


def test_unacknowledged_submission_is_preserved_and_not_retried(tmp_path, monkeypatch):
    count = []

    def fail(*args, **kwargs):
        count.append(1)
        raise OSError("lost scheduler acknowledgement")

    monkeypatch.setattr(control.subprocess, "check_output", fail)
    with pytest.raises(control.shared.SubmissionError):
        control.submit(tmp_path, "test", "media-worker", array=True)
    assert len(count) == 1
    assert control.read(tmp_path / "jobs/test-submission-error.json")["automatic_retry"] is False


def test_prepare_preserves_dag_and_uses_real_module_signatures(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    control.write_once(tmp_path / "ownership-retry-decision.json", dict(rerun=False))
    monkeypatch.setattr(control, "load", lambda root: dict(source_sha="snapshot"))
    monkeypatch.setattr(control.shared, "cpu_gate", lambda *args: dict(passed=True))
    monkeypatch.setattr(control.shared, "gpu_snapshot", lambda: dict(active=[], pending=[]))
    calls, preparations = [], []

    def module(lane):
        def prepare(root, **kwargs):
            if lane in {"grouped", "yaml", "frontier"}:
                assert set(kwargs) == {"cpu_gate_root", "parent"}
            elif lane in {"lookm", "lossless", "historical_media"}:
                assert set(kwargs) == {"cpu_gate_root"}
            else:
                assert kwargs == {}
            preparations.append(lane)
            return dict(prepared=True)

        return SimpleNamespace(prepare=prepare)

    def submit(root, label, mode, **kwargs):
        job = str(100 + len(calls))
        calls.append(dict(label=label, mode=mode, job=job, **kwargs))
        return job

    original_import = control.importlib.import_module
    monkeypatch.setattr(control, "module", module)
    monkeypatch.setattr(control, "submit", submit)
    monkeypatch.setattr(
        control.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(inventory=lambda p: dict(available=True))
            if name == "evicastkv_source_trace_adjudication"
            else original_import(name)
        ),
    )
    result = control.prepare(tmp_path)
    assert preparations == [
        "grouped",
        "frontier",
        "historical_media",
        "precision",
        "lookm",
        "lossless",
    ]
    by_name = {r["label"]: r for r in calls}
    assert by_name["after-frontier"]["deps"] == [
        by_name["frontier"]["job"],
        by_name["after-grouped"]["job"],
    ]
    assert "prepare-media" not in by_name and "yaml" not in by_name
    assert result == dict(frontier_controller=by_name["after-frontier"]["job"], bounded_segments=4)
    assert not (tmp_path / "prepared-media.json").exists()


def test_failed_cpu_gate_cannot_submit_work(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: dict(source_sha="snapshot"))
    monkeypatch.setattr(
        control.shared, "cpu_gate", lambda *a: (_ for _ in ()).throw(ValueError("failed"))
    )
    monkeypatch.setattr(control, "submit", lambda *a, **k: pytest.fail("CPU failure admitted work"))
    monkeypatch.setattr(control, "finish", lambda root: dict(blocked=True))
    assert control.prepare(tmp_path) == dict(blocked=True)
    assert (tmp_path / "errors/cpu.json").exists()


def test_gpu_ceiling_rejected_before_model_import(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {})
    monkeypatch.setattr(control.shared, "gpu_snapshot", lambda: dict(active_gpus=7))
    monkeypatch.setattr(control, "module", lambda *a: pytest.fail("over-ceiling model import"))
    with pytest.raises(ValueError, match="six-A100"):
        control.run(tmp_path, "precision-worker")


def test_worker_failure_still_receives_analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {})

    def analyze(root):
        raise FileNotFoundError("worker was killed")

    monkeypatch.setattr(control, "module", lambda *a: SimpleNamespace(analyze=analyze))
    result = control.run(tmp_path, "after-precision")
    assert result["status"] == "blocked"
    assert "worker was killed" in result["error"]
    assert (tmp_path / "errors/analyze-precision.json").is_file()


def test_media_waits_for_merged_history_and_keeps_all_final_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {})
    control.write_once(tmp_path / "stage-jobs.json", dict(dependencies=["11", "12"]))
    observed, jobs = [], []

    def prepare(root, **kwargs):
        observed.append(kwargs)
        return dict(prepared=True)

    def submit(root, label, mode, **kwargs):
        jobs.append(dict(label=label, **kwargs))
        return str(20 + len(jobs))

    monkeypatch.setattr(control, "module", lambda lane: SimpleNamespace(prepare=prepare))
    monkeypatch.setattr(control, "submit", submit)
    control.prepare_media(tmp_path)
    assert observed[0]["history_root"] == tmp_path / "yaml"
    assert observed[0]["history_gate_filename"] == control.CPU["yaml"][1]
    assert jobs[-1]["deps"] == ["11", "12", "21", "22"]
    assert Path(tmp_path / "prepared-media.json").is_file()


def test_final_report_preserves_missing_gates_and_no_claim(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: dict(source_sha="immutable"))
    report = control.finish(tmp_path)
    assert all(r["status"] == "missing_or_blocked" for r in report["artifacts"].values())
    assert report["missing_output_is_not_a_pass"]
    assert report["superiority_established"] is False
    assert report["paper_campaign_submitted"] is False
    text = (tmp_path / "SUMMARY.md").read_text()
    assert "missing_or_blocked" in text and "No paper campaign was submitted" in text
    with pytest.raises(FileExistsError):
        control.finish(tmp_path)


@pytest.mark.parametrize("index,expect_next", [(1, True), (4, False)])
def test_finite_frontier_progression_never_restarts_or_exceeds_budget(
    tmp_path, monkeypatch, index, expect_next
):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(control, "load", lambda root: {})
    path = control.frontier_path(tmp_path, index)
    for shard in range(8):
        control.write_once(path / f"workers/shard-{shard}.json", dict(state_sha256="advanced"))
        control.write_once(path / f"imports/shard-{shard}.json", dict(state_sha256="old"))
    preparations, submissions = [], []

    def module(lane):
        def analyze(root, **kwargs):
            assert root == path
            value = dict(missing_shards=[], frontier_exhausted=False)
            control.write_once(path / control.CPU["frontier"][1], value)
            return value

        def prepare(root, **kwargs):
            preparations.append((lane, root, kwargs))
            assert kwargs["parent"] == path
            assert kwargs["parent_gate_filename"] == control.CPU["frontier"][1]

        return SimpleNamespace(analyze=analyze, prepare=prepare)

    def submit(root, label, mode, **kwargs):
        submissions.append((label, mode, kwargs))
        return str(300 + len(submissions))

    monkeypatch.setattr(control, "module", module)
    monkeypatch.setattr(control, "submit", submit)
    result = control.frontier_followup(tmp_path, index)
    if expect_next:
        assert result["next_segment"] == 2
        assert preparations[0][0] == "frontier"
        assert preparations[0][1] == tmp_path / "frontier-2"
        assert submissions[0][1] == "frontier-2-worker"
        assert not (tmp_path / "frontier-selection.json").exists()
    else:
        assert preparations[0][0] == "yaml"
        assert preparations[0][1] == tmp_path / "yaml"
        assert submissions[0][1] == "yaml-worker"
        assert all("frontier-5" not in mode for _, mode, _ in submissions)
        assert control.read(tmp_path / "frontier-selection.json")["index"] == 4
        assert "media_preparation" in result


def test_first_frontier_worker_is_not_misparsed_as_segment_number(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "load", lambda root: {})
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    called = []
    monkeypatch.setattr(
        control,
        "module",
        lambda lane: SimpleNamespace(worker=lambda path, index: called.append((lane, path, index))),
    )
    control.run(tmp_path, "frontier-worker")
    assert called == [("frontier", tmp_path / "frontier", 3)]
    with pytest.raises(ValueError, match="bounded"):
        control.run(tmp_path, "frontier-5-worker")


def test_frontier_analyzer_failure_still_schedules_final_not_more_walks(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setattr(
        control, "frontier_followup", lambda *a: (_ for _ in ()).throw(ValueError("invalid source"))
    )
    control.write_once(tmp_path / "jobs/precision-submission.json", dict(job="123"))
    called = []
    monkeypatch.setattr(
        control, "submit", lambda *args, **kwargs: called.append((args, kwargs)) or "234"
    )
    assert control.frontier_event(tmp_path, 1) == dict(final="234")
    assert called[0][0][1:] == ("final", "final")
    assert called[0][1]["deps"] == ["123"]
    assert (tmp_path / "errors/frontier-controller-1.json").exists()
