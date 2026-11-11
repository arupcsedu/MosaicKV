"""Synthetic event-driven receipts only; no jobs, GPU or datasets are used."""

import json
from types import SimpleNamespace

import evicastkv_review_stages_v32 as stages
import pytest

REV = "a" * 40


def test_v34_pins_actual_packed_producer_without_expanding_scope():
    producer = "422a67994d06ffb0b69943727063a7dfc885b6d2"
    parent = "/scratch/djy8hg/workdir/fox-review-v33/run-20261111-1"
    plan_sha = "3bbd8e053073bf6c878e1618923c45f02ea2f5325b04a582d1ee648e2aa28745"
    assert stages.whole.PRODUCER == stages.numerical.BRIDGE_PRODUCER == producer
    assert str(stages.whole.PARENT) == str(stages.numerical.BRIDGE_ROOT) == parent
    assert stages.whole.PARENT_PLAN_SHA == stages.numerical.BRIDGE_PLAN_SHA == plan_sha
    assert stages.DISPATCH_JOB == "22834999"
    assert stages.POLICY["id"] == "V34_packed_raw_whole_COCO_provenance_callbacks_v1"
    assert stages.POLICY["actual_producer_sha"] == producer
    assert stages.POLICY["actual_parent_plan_sha256"] == plan_sha
    assert stages.POLICY["direct_GPU_jobs"] == 0
    for name in (
        "fresh_confirmation_authorized",
        "new_identity_selection_authorized",
        "materialization_authorized",
        "paper_campaign_authorized",
        "automatic_retry",
        "continuous_polling",
    ):
        assert stages.POLICY[name] is False
    assert len(stages.numerical.CONSUMED_IDS) == 2
    assert stages.numerical.POLICY["predictions"] == 128
    assert stages.numerical.POLICY["workers"] == 2
    assert stages.numerical.POLICY["tolerance"] == 1e-4
    assert stages.numerical.POLICY["scoring"] is False
    assert stages.provenance.POLICY["original_worker_indices"] == [1, 7]
    assert stages.provenance.POLICY["selected_directories"] == 521
    assert stages.provenance.POLICY["worker_seconds"] == 210
    assert stages.provenance.POLICY["worker_bytes"] == 2 * 2**30


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def ack(path, mode, *, producer=None, job="123"):
    return write(
        path,
        dict(
            source_sha=producer or stages.whole.PRODUCER,
            job=job,
            command=["sbatch", mode, producer or stages.whole.PRODUCER],
        ),
    )


def plan_fixture(monkeypatch, tmp_path):
    root = tmp_path / "controller"
    parent = tmp_path / "parent"
    monkeypatch.setattr(stages.whole, "PARENT", parent)
    path = write(parent / "plan.json", dict(source_sha=stages.whole.PRODUCER, bindings={}))
    monkeypatch.setattr(stages.whole, "PARENT_PLAN_SHA", stages.sha(path))
    ack(parent / "jobs/dispatch-submission.json", "dispatch", job=stages.DISPATCH_JOB)
    plan = dict(
        source_sha=REV,
        policy=stages.POLICY,
        policy_sha256=stages.digest(stages.POLICY),
        bindings={},
    )
    write(root / "plan.json", plan)
    monkeypatch.setattr(stages, "load", lambda root: plan)
    monkeypatch.setattr(stages, "cpu_gate", lambda *args: dict(status="passed"))
    return root, parent, plan


@pytest.mark.parametrize(
    "mode,cpus,memory,minutes",
    [
        ("tests", 4, "8G", 15),
        ("after-dispatch", 4, "8G", 5),
        ("after-whole", 12, "24G", 20),
        ("after-coco", 4, "8G", 5),
    ],
)
def test_callbacks_are_cpu_only_with_exact_resources(
    monkeypatch, tmp_path, mode, cpus, memory, minutes
):
    calls = []
    monkeypatch.setattr(
        stages.subprocess,
        "check_output",
        lambda command, **kwargs: calls.append(command) or "123\n",
    )
    stages.submit(tmp_path, REV, mode, dependencies=("111", "222"))
    command = calls[0]
    for option in (
        "--partition=interactive",
        "--nodes=1",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time=00:{minutes}:00",
        "--dependency=afterany:111:222",
    ):
        assert option in command
    assert not any(item.startswith(("--gres", "--reservation", "--array")) for item in command)


def test_uncertain_acknowledgement_not_retried(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        stages.subprocess, "check_output", lambda *args, **kwargs: calls.append(args) or "uncertain"
    )
    with pytest.raises(ValueError, match="uncertain"):
        stages.submit(tmp_path, REV, "tests")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "state,active", [("RUNNING", True), ("PENDING", True), ("COMPLETED", False), ("FAILED", False)]
)
def test_single_scheduler_observation_handles_terminal_jobs(monkeypatch, state, active):
    calls = []
    monkeypatch.setattr(
        stages.subprocess,
        "run",
        lambda *args, **kwargs: (
            calls.append(args)
            or SimpleNamespace(returncode=0, stdout=f"JobState={state}", stderr="")
        ),
    )
    assert stages.active_job("123") is active
    assert len(calls) == 1


def test_min_job_age_tombstone_is_not_requeued_or_polled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        stages.subprocess,
        "run",
        lambda *args, **kwargs: (
            calls.append(args)
            or SimpleNamespace(returncode=1, stdout="", stderr="Invalid job id specified")
        ),
    )
    assert stages.active_job("123") is False
    assert len(calls) == 1


def test_scheduler_outage_is_not_treated_as_completed(monkeypatch):
    monkeypatch.setattr(
        stages.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="controller unreachable"
        ),
    )
    with pytest.raises(ValueError, match="unavailable"):
        stages.active_job("123")


def dispatch_fixture(parent):
    ack(parent / "jobs/gpu-submission.json", "worker", job="101")
    ack(parent / "jobs/analysis-submission.json", "analyze", job="102")
    write(parent / "jobs/dispatch-result.json", dict(blockers=[], workers="101", analysis="102"))


def test_after_dispatch_uses_actual_acknowledged_analysis_dependency(monkeypatch, tmp_path):
    root, parent, _plan = plan_fixture(monkeypatch, tmp_path)
    dispatch_fixture(parent)
    monkeypatch.setattr(stages, "original_gate_ready", lambda: False)
    monkeypatch.setattr(stages, "active_job", lambda job: job == "102")
    calls = []
    monkeypatch.setattr(
        stages, "submit", lambda *args, **kwargs: calls.append((args, kwargs)) or "103"
    )
    decision = stages.run(root, "after-dispatch")
    assert decision["status"] == "completed_stage"
    assert calls[0][0][-1] == "after-whole" and calls[0][1]["dependencies"] == ("102",)


def test_completed_verified_gate_skips_stale_afterany(monkeypatch, tmp_path):
    root, parent, _plan = plan_fixture(monkeypatch, tmp_path)
    dispatch_fixture(parent)
    monkeypatch.setattr(stages, "original_gate_ready", lambda: True)
    monkeypatch.setattr(
        stages, "active_job", lambda *args: pytest.fail("verified completed gate needs no query")
    )
    calls = []
    monkeypatch.setattr(stages, "submit", lambda *args, **kwargs: calls.append(kwargs) or "103")
    assert stages.run(root, "after-dispatch")["status"] == "completed_stage"
    assert calls == [dict(dependencies=())]


@pytest.mark.parametrize("failure", ["cpu", "dispatch", "missing_analysis", "wrong_ack"])
def test_failed_predecessor_blocks_all_successors(monkeypatch, tmp_path, failure):
    root, parent, _plan = plan_fixture(monkeypatch, tmp_path)
    dispatch_fixture(parent)
    if failure == "cpu":
        monkeypatch.setattr(
            stages, "cpu_gate", lambda *args: (_ for _ in ()).throw(ValueError("CPU failed"))
        )
    elif failure == "dispatch":
        write(
            parent / "jobs/dispatch-result.json",
            dict(blockers=["failed"], workers=None, analysis="102"),
        )
    elif failure == "missing_analysis":
        write(parent / "jobs/dispatch-result.json", dict(blockers=[], workers="101"))
    else:
        ack(parent / "jobs/analysis-submission.json", "analyze", producer="c" * 40, job="102")
    monkeypatch.setattr(
        stages, "submit", lambda *args, **kwargs: pytest.fail("must not queue successor")
    )
    decision = stages.run(root, "after-dispatch")
    assert decision["status"] == "blocked_stage" and decision["failures"]


def test_original_ready_gate_checks_actual_worker_hashes(monkeypatch, tmp_path):
    _root, parent, _plan = plan_fixture(monkeypatch, tmp_path)
    write(
        parent / f"analysis/{stages.numerical.BRIDGE_GATE_NAME}",
        dict(
            source_sha=stages.whole.PRODUCER,
            plan_sha256=stages.whole.PARENT_PLAN_SHA,
            failures=[],
            complete=True,
            source_fidelity_128_pass=True,
            whole_request_KV_byte_match_pass=True,
        ),
    )
    for index in range(4):
        out = parent / f"workers/{index}"
        word = write(out / "owned.json", dict(synthetic=True))
        write(
            out / "result.json",
            dict(
                source_sha=stages.whole.PRODUCER,
                plan_sha256=stages.whole.PARENT_PLAN_SHA,
                failures=[],
                outputs={"owned.json": stages.sha(word)},
            ),
        )
    assert stages.original_gate_ready() is True
    write(parent / "workers/3/owned.json", dict(synthetic="changed"))
    with pytest.raises(ValueError, match="GPU output changed"):
        stages.original_gate_ready()


def whole_fixture(monkeypatch, tmp_path, *, passed=True):
    root, _parent, plan = plan_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(stages, "original_gate_ready", lambda: True)
    gate = dict(
        complete=passed,
        source_fidelity_128_pass=passed,
        whole_request_KV_byte_match_pass=passed,
        whole_request_KV_peak_certified=passed,
        failures=[],
    )

    def adjudicate(path):
        write(path / "GATE_bounded_reader_v27.json", gate)
        return gate

    monkeypatch.setattr(stages.whole, "adjudicate", adjudicate)
    monkeypatch.setattr(stages.numerical, "require_whole_gate", lambda *args: {})
    return root, plan


def test_negative_whole_replay_never_launches_coco(monkeypatch, tmp_path):
    root, _plan = whole_fixture(monkeypatch, tmp_path, passed=False)
    monkeypatch.setattr(stages.numerical, "launch", lambda *args: pytest.fail("no GPU"))
    assert stages.run(root, "after-whole")["status"] == "blocked_stage"


def test_positive_whole_replay_uses_existing_coco_launcher_then_afterany(monkeypatch, tmp_path):
    root, _plan = whole_fixture(monkeypatch, tmp_path)
    called = []

    def launch(path, gate):
        called.append((path, gate))
        ack(path / "jobs/analysis-submission.json", "analyze", producer=REV, job="202")

    monkeypatch.setattr(stages.numerical, "launch", launch)
    monkeypatch.setattr(stages, "active_job", lambda job: job == "202")
    queued = []
    monkeypatch.setattr(
        stages, "submit", lambda *args, **kwargs: queued.append((args, kwargs)) or "203"
    )
    assert stages.run(root, "after-whole")["status"] == "completed_stage"
    assert called == [(root / "coco", root / "whole-replay/GATE_bounded_reader_v27.json")]
    assert queued[0][0][-1] == "after-coco" and queued[0][1]["dependencies"] == ("202",)


@pytest.mark.parametrize("failure", ["numeric_plan", "numeric_gate"])
def test_failed_coco_controls_block_provenance_scan(monkeypatch, tmp_path, failure):
    root, _parent, _plan = plan_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        stages.numerical,
        "validate",
        lambda *args: (
            (_ for _ in ()).throw(ValueError("plan failed")) if failure == "numeric_plan" else None
        ),
    )
    monkeypatch.setattr(
        stages.provenance,
        "empirical_bindings",
        lambda *args: (_ for _ in ()).throw(ValueError("numerical failed")),
    )
    monkeypatch.setattr(stages.provenance, "launch", lambda *args, **kwargs: pytest.fail("no scan"))
    assert stages.run(root, "after-coco")["status"] == "blocked_stage"


def test_positive_coco_launches_only_bounded_cpu_chain_and_reuses_suite(monkeypatch, tmp_path):
    root, _parent, _plan = plan_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(stages.numerical, "validate", lambda *args: None)
    monkeypatch.setattr(
        stages.provenance,
        "empirical_bindings",
        lambda *args: ({"binding": "hash"}, stages.whole.PRODUCER),
    )
    calls = []
    monkeypatch.setattr(
        stages.provenance,
        "launch",
        lambda *args, **kwargs: (
            calls.append((args, kwargs)) or dict(cpu=None, after_cpu="301", GPU_jobs=0)
        ),
    )
    decision = stages.run(root, "after-coco")
    assert decision["status"] == "completed_stage"
    assert calls[0][0][0] == root / "provenance" and calls[0][1] == dict(cpu_gate_root=root)
    assert decision["fresh_confirmation_authorized"] is False


def test_stage_is_write_once_and_cannot_duplicate_successor(monkeypatch, tmp_path):
    root, _parent, _plan = plan_fixture(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(stages, "after_coco", lambda *args: calls.append(args) or {})
    assert stages.run(root, "after-coco")["status"] == "completed_stage"
    with pytest.raises(FileExistsError):
        stages.run(root, "after-coco")
    assert len(calls) == 1


def test_wrapper_sets_scratch_and_gpu_invisibility_before_python():
    script = (stages.SOURCE / "mosaickv/slurm/fox_review_stages_v32.sbatch").read_text()
    assert script.index('CUDA_VISIBLE_DEVICES=""') < script.index("exec /scratch")
    assert "--cpu-timeout-seconds 840 --cpu-capture-mode sys" in script
    assert "gcc/11.4.0" in script and "assert_clean_worktree.sh" in script
    branch = script.index('if [[ "$MODE" == after-whole ]]')
    assert script.index("OMP_NUM_THREADS=2") < branch
    assert branch < script.index("OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12")
    assert script.index("OMP_NUM_THREADS=12") < script.rindex("exec /scratch")
    assert "ATEN_CPU_CAPABILITY=avx2" in script


@pytest.mark.parametrize(
    "ready,active,expected_dispatch",
    [
        (False, True, True),
        (True, False, False),
        (False, False, False),
    ],
)
def test_launch_waits_for_cpu_and_live_dispatch_without_stale_dependencies(
    monkeypatch, tmp_path, ready, active, expected_dispatch
):
    parent = tmp_path / "parent"
    monkeypatch.setattr(stages.whole, "PARENT", parent)
    path = write(parent / "plan.json", dict(source_sha=stages.whole.PRODUCER))
    monkeypatch.setattr(stages.whole, "PARENT_PLAN_SHA", stages.sha(path))
    ack(parent / "jobs/dispatch-submission.json", "dispatch", job=stages.DISPATCH_JOB)
    monkeypatch.setattr(stages, "SCRATCH", tmp_path)
    monkeypatch.setattr(stages, "clean_source", lambda *args: REV)
    monkeypatch.setattr(stages.numerical, "execution_base_proof", lambda *args: {})
    monkeypatch.setattr(stages, "original_gate_ready", lambda: ready)
    observed = []
    monkeypatch.setattr(stages, "active_job", lambda job: observed.append(job) or active)
    calls = []
    monkeypatch.setattr(
        stages,
        "submit",
        lambda *args, **kwargs: calls.append((args, kwargs)) or str(100 + len(calls)),
    )
    root = tmp_path / "new-controller"
    stages.launch(root)
    assert calls[0][0][-1] == "tests"
    expected = ("101", stages.DISPATCH_JOB) if expected_dispatch else ("101",)
    assert calls[1][0][-1] == "after-dispatch" and calls[1][1]["dependencies"] == expected
    assert observed == ([] if ready else [stages.DISPATCH_JOB])


def test_negative_existing_gate_stops_launch_without_jobs(monkeypatch, tmp_path):
    parent = tmp_path / "parent"
    monkeypatch.setattr(stages.whole, "PARENT", parent)
    path = write(parent / "plan.json", dict(source_sha=stages.whole.PRODUCER))
    monkeypatch.setattr(stages.whole, "PARENT_PLAN_SHA", stages.sha(path))
    ack(parent / "jobs/dispatch-submission.json", "dispatch", job=stages.DISPATCH_JOB)
    monkeypatch.setattr(stages, "SCRATCH", tmp_path)
    monkeypatch.setattr(stages, "clean_source", lambda *args: REV)
    monkeypatch.setattr(stages.numerical, "execution_base_proof", lambda *args: {})
    monkeypatch.setattr(
        stages,
        "original_gate_ready",
        lambda: (_ for _ in ()).throw(ValueError("negative whole gate")),
    )
    monkeypatch.setattr(stages, "submit", lambda *args, **kwargs: pytest.fail("no new job"))
    root = tmp_path / "new-controller"
    with pytest.raises(ValueError, match="negative"):
        stages.launch(root)
    decision = stages.read(root / "decisions/launch.json")
    assert decision["status"] == "blocked_stage" and decision["failures"]
