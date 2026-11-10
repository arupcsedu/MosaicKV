"""Continuation orchestration is bounded; all scanner/scheduler calls mocked."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import evicastkv_review_freshness_continue as cont
import pytest


@pytest.fixture
def setup(tmp_path, monkeypatch):
    audits = tmp_path / "audits"
    audits.mkdir()
    parent, old = audits / "parent", tmp_path / "old-source"
    scanner = old / "mosaickv/scripts/evicastkv_review_freshness_stream.py"
    scanner.parent.mkdir(parents=True)
    scanner.write_text("# preserved scanner source\n")
    parent.mkdir()
    (parent / "registration.json").write_text("{}")
    (parent / "freshness").mkdir()
    (parent / "freshness/GATE_freshness_workers.json").write_text(
        json.dumps(dict(status="blocked_missing_or_incomplete_shards"))
    )
    for index in range(8):
        path = parent / "freshness/shards" / f"shard-{index:03d}"
        path.mkdir(parents=True)
        (path / "registration.json").write_text(
            json.dumps(dict(source_sha=cont.OLD_SHA, shard_index=index, shard_count=8))
        )
        (path / "reviewed_provenance.json").write_text("{}")
        (path / "checkpoint-000002.json").write_text(
            json.dumps(
                dict(
                    state=dict(traversal_complete=False, processed_files=100, projected_records=200)
                )
            )
        )
    monkeypatch.setattr(cont, "AUDITS_ROOT", audits)
    monkeypatch.setattr(cont, "OLD_SOURCE", old)
    monkeypatch.setattr(cont, "clean_source", lambda path, expected=None: expected or "b" * 40)
    monkeypatch.setattr(cont, "cpu_gate", lambda root, revision: dict(status="passed"))
    return audits / "round", parent


def active(setup, monkeypatch):
    root, parent = setup
    cont.register(root, parent)
    record = cont.read(root / "registration.json")
    record["execute"] = True  # Test-only authorization fixture, no real execution.
    monkeypatch.setattr(cont, "load", lambda path: record)
    return root, record


def test_default_registration_is_dry_and_binds_initial_checkpoints(setup, monkeypatch):
    root, parent = setup
    calls = []
    monkeypatch.setattr(cont.subprocess, "run", lambda *a, **k: calls.append(a))
    result = cont.register(root, parent)
    assert calls == [] and result["jobs_submitted"] == 0
    record = cont.read(root / "registration.json")
    assert len(record["shards"]) == 8
    assert record["maximum_bytes_per_invocation"] == 2 * 2**30
    assert record["maximum_GPU_workers"] == 0
    assert record["fresh_confirmation_authorized"] is False
    assert all(row["checkpoint"].endswith("checkpoint-000002.json") for row in record["shards"])
    assert all(row["checkpoint"] in record["bindings"] for row in record["shards"])


def test_verified_cpu_gate_and_finite_array_before_afterany_analysis(setup, monkeypatch):
    root, parent = setup
    with pytest.raises(ValueError, match="verified current-source CPU gate"):
        cont.register(root, parent, execute=True)
    cpu = parent / "full-cpu"
    cpu.mkdir()
    for name in ("cpu-tests.json", "cpu-tests.log", "cpu-tests.xml"):
        (cpu / name).write_text("synthetic verified CPU fixture")
    checked = []
    monkeypatch.setattr(
        cont, "cpu_gate", lambda path, sha: checked.append((path, sha)) or dict(status="passed")
    )
    calls = []

    def submit(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=f"{123 + len(calls)}\n")

    monkeypatch.setattr(cont.subprocess, "run", submit)
    result = cont.register(root, parent, execute=True, cpu_gate_root=parent)
    assert checked == [(parent, "b" * 40)]
    assert result["workers"] == "124" and result["analysis"] == "125"
    assert "--array=0-7%4" in calls[0]
    assert not any(arg.startswith("--dependency=") for arg in calls[0])
    assert "--dependency=afterany:124" in calls[1]
    assert all("--cpus-per-task=4" in args and "--time=00:15:00" in args for args in calls)
    assert all(not any("gres" in arg or "gpus=" in arg for arg in args) for args in calls)
    assert all("CUDA_VISIBLE_DEVICES=" in args[-1] for args in calls)
    assert all("MOSAICKV_CACHE_ROOT=/scratch/djy8hg/cache/mosaickv" in args[-1] for args in calls)


def test_changed_parent_binding_rejected_without_rescanning(setup):
    root, parent = setup
    cont.register(root, parent)
    # Make execution true only in the synthetic fixture; binding content must
    # still be checked rather than trusted from a historical filename.
    record = cont.read(root / "registration.json")
    record["execute"] = True
    record["cpu_gate_root"] = str(parent)
    (root / "registration.json").write_text(json.dumps(record))
    # Source/CPU validity are independent of the deliberately mutated checkpoint.
    Path(record["shards"][0]["checkpoint"]).write_text("changed")
    with pytest.raises(ValueError, match="binding changed"):
        cont.load(root)


def test_worker_continues_three_calls_without_rewriting_parent_summaries(setup, monkeypatch):
    root, record = active(setup, monkeypatch)
    calls = []

    def invoke(path, mode, **kwargs):
        calls.append((mode, kwargs))
        number = len(calls)
        return dict(
            gate=dict(
                shard_index=0,
                source_sha=cont.OLD_SHA,
                processed_files=100 + number,
                projected_records=200 + number * 2,
                traversal_complete=False,
            ),
            stdout_sha256="c" * 64,
        )

    monkeypatch.setattr(cont, "invoke", invoke)
    result = cont.worker(root, 0)
    assert len(calls) == 3
    assert result["files_added"] == 3 and result["records_added"] == 6
    assert result["failure"] is None
    assert Path(record["shards"][0]["checkpoint"]).read_text().startswith("{")
    assert len(list((root / "progress").glob("*.json"))) == 3


def test_terminal_stops_early_and_advanced_initial_checkpoint_rejects_race(setup, monkeypatch):
    root, record = active(setup, monkeypatch)
    calls = []

    def invoke(*args, **kwargs):
        calls.append(True)
        return dict(
            gate=dict(
                shard_index=0,
                source_sha=cont.OLD_SHA,
                processed_files=101,
                projected_records=201,
                traversal_complete=True,
            )
        )

    monkeypatch.setattr(cont, "invoke", invoke)
    cont.worker(root, 0)
    assert len(calls) == 1
    path = Path(record["shards"][1]["path"]) / "checkpoint-000003.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="already advanced"):
        cont.worker(root, 1)


def test_worker_subprocess_error_retained_without_recursive_retry(setup, monkeypatch):
    root, _record = active(setup, monkeypatch)
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("subprocess failed")

    monkeypatch.setattr(cont, "invoke", fail)
    result = cont.worker(root, 0)
    assert len(calls) == 1
    assert result["failure"]["retryable"] is False
    assert result["progress"] == []


def test_aggregate_can_publish_conservative_partial_inventory(setup, monkeypatch):
    root, _record = active(setup, monkeypatch)
    calls = []

    def merge(path, mode):
        calls.append(mode)
        return dict(gate=dict(traversal_complete=False, fresh_confirmation_authorized=False))

    monkeypatch.setattr(cont, "invoke", merge)
    result = cont.aggregate(root)
    assert calls == ["merge"]
    assert len(result["missing_workers"]) == 8
    assert not result["merged"]["gate"]["traversal_complete"]
    assert result["fresh_confirmation_authorized"] is False


def test_subprocess_uses_old_source_and_captures_transcript_hashes(setup, monkeypatch):
    root, _parent = setup
    calls = []

    def run(arguments, **kwargs):
        calls.append((arguments, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"traversal_complete":false}\n', stderr="")

    monkeypatch.setattr(cont.subprocess, "run", run)
    result = cont.invoke(root, "step", index=2, attempt=0)
    args, options = calls[0]
    assert args[1:3] == ["-c", cont.CHILD_BOOTSTRAP]
    assert "child-step" in args and options["cwd"] == cont.OLD_SOURCE
    assert options["timeout"] == 250
    assert options["env"]["PYTHONPATH"].startswith(str(cont.OLD_SOURCE))
    assert options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert result["stdout_sha256"]
    assert len(list((root / "transcripts").glob("*.json"))) == 2


def test_failed_cpu_gate_cannot_be_bypassed_with_execution(setup, monkeypatch):
    root, parent = setup

    def reject(*args):
        raise ValueError("complete current-source CPU suite did not pass")

    monkeypatch.setattr(cont, "cpu_gate", reject)
    with pytest.raises(ValueError, match="CPU suite did not pass"):
        cont.register(root, parent, execute=True, cpu_gate_root=parent)
    assert not root.exists()


def test_subprocess_timeout_preserves_partial_transcripts(setup, monkeypatch):
    root, _parent = setup

    def timeout(*args, **kwargs):
        raise cont.subprocess.TimeoutExpired(
            args[0], 250, output=b"partial gate", stderr=b"interrupted"
        )

    monkeypatch.setattr(cont.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="124"):
        cont.invoke(root, "step", index=0)
    path = root / "transcripts/step-0-0.stdout.json"
    assert json.loads(path.read_text()) == "partial gate"
    assert json.loads(path.with_name("step-0-0.stderr.json").read_text()) == "interrupted"


def test_new_driver_cannot_shadow_old_scanner_import_precedence(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "synthetic_scanner.py").write_text("VALUE = 'pinned-old'\n")
    (new / "synthetic_scanner.py").write_text("VALUE = 'wrong-new'\n")
    driver = new / "driver.py"
    driver.write_text("import synthetic_scanner\nprint(synthetic_scanner.VALUE)\n")
    result = subprocess.run(
        [cont.PYTHON, "-c", cont.CHILD_BOOTSTRAP, str(driver)],
        cwd=old,
        env=dict(os.environ, PYTHONPATH=str(old), PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "pinned-old"


@pytest.mark.parametrize(
    "stage,uncertain", [("worker", False), ("aggregate", False), ("aggregate", True)]
)
def test_submission_failure_preserves_requests_and_known_worker_ids(
    setup, monkeypatch, stage, uncertain
):
    root, parent = setup
    cpu = parent / "full-cpu"
    cpu.mkdir()
    for name in ("cpu-tests.json", "cpu-tests.log", "cpu-tests.xml"):
        (cpu / name).write_text("synthetic verified CPU fixture")
    calls = []

    def scheduler(arguments, **kwargs):
        current = "worker" if not calls else "aggregate"
        assert (root / "jobs" / f"{current}-request.json").is_file()
        calls.append(arguments)
        if current == stage:
            if uncertain:
                raise subprocess.TimeoutExpired(arguments, 30, output=b"unknown scheduler reply")
            raise subprocess.CalledProcessError(
                1,
                arguments,
                output="",
                stderr="sbatch: error: Batch job submission failed: Invalid account",
            )
        return SimpleNamespace(stdout="123\n", stderr="")

    monkeypatch.setattr(cont.subprocess, "run", scheduler)
    result = cont.register(root, parent, execute=True, cpu_gate_root=parent)
    assert result["status"] == "blocked_submission"
    assert result["workers"] == ("123" if stage == "aggregate" else None)
    assert result["analysis"] is None
    assert result["analysis_submission_uncertain"] is uncertain
    assert result["analysis_definitely_not_submitted"] is not uncertain
    assert (root / "jobs" / f"{stage}-error.json").is_file()
    assert (root / "jobs/submission-result.json").is_file()
    assert len(calls) == (2 if stage == "aggregate" else 1)
    with pytest.raises(ValueError, match="write-once"):
        cont.register(root, parent, execute=True, cpu_gate_root=parent)
    assert len(calls) == (2 if stage == "aggregate" else 1)
