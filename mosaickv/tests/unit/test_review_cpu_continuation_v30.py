"""Synthetic finite-controller tests: no Slurm jobs or historical scans run."""

import json
from pathlib import Path

import evicastkv_review_cpu_continuation_v30 as controller
import pytest

REV = "a" * 40
WHOLE = "b" * 40


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def gates(tmp_path):
    whole = write(
        tmp_path / "whole/gate.json",
        dict(
            source_sha=WHOLE,
            complete=True,
            source_fidelity_128_pass=True,
            whole_request_KV_byte_match_pass=True,
            whole_request_KV_peak_certified=True,
            failures=[],
        ),
    )
    coco_root = tmp_path / "coco"
    plan = write(coco_root / "plan.json", dict(source_sha=REV))
    plan_sha = controller.sha(plan)
    reports = [
        dict(sample_id=sample, predictions=128, prediction_agreement=128, pass_128=True)
        for sample in controller.numerical.CONSUMED_IDS
    ]
    coco = write(
        coco_root / "analysis/gate.json",
        dict(
            source_sha=REV,
            complete=True,
            corrected_boundary_128_pass=True,
            failures=[],
            plan_sha256=plan_sha,
            reports=reports,
            source_scope="FP16_GQA_COCO_FullKV_evaluation_only_boundary_control",
        ),
    )
    for i in (0, 1):
        output = write(coco_root / f"workers/{i}/summary.json", dict(synthetic=True))
        write(
            coco_root / f"workers/{i}/result.json",
            dict(
                source_sha=REV,
                failures=[],
                plan_sha256=plan_sha,
                outputs={"summary.json": controller.sha(output)},
            ),
        )
    return whole, coco


@pytest.fixture
def empirical(monkeypatch, tmp_path):
    monkeypatch.setattr(controller, "SCRATCH", tmp_path)
    monkeypatch.setattr(controller.numerical, "require_whole_gate", lambda gate, revision: {})
    return gates(tmp_path)


def test_both_empirical_gates_are_bound_before_queuing(empirical):
    bindings, producer = controller.empirical_bindings(*empirical, REV)
    assert producer == WHOLE
    assert len(bindings) == 7
    assert all(controller.sha(Path(name)) == value for name, value in bindings.items())


@pytest.mark.parametrize(
    "which,field",
    [
        (0, "source_fidelity_128_pass"),
        (0, "whole_request_KV_peak_certified"),
        (1, "corrected_boundary_128_pass"),
        (1, "complete"),
    ],
)
def test_failed_predecessor_rejects_cpu_scope(empirical, which, field):
    path = empirical[which]
    record = controller.read(path)
    record[field] = False
    write(path, record)
    with pytest.raises(ValueError):
        controller.empirical_bindings(*empirical, REV)


def test_coco_worker_missingness_cannot_disappear(empirical):
    path = empirical[1].parent.parent / "workers/1/result.json"
    record = controller.read(path)
    record["failures"] = ["failed"]
    write(path, record)
    with pytest.raises(ValueError, match="completeness"):
        controller.empirical_bindings(*empirical, REV)


def test_worker_output_hashes_are_rechecked(empirical):
    write(empirical[1].parent.parent / "workers/0/summary.json", dict(synthetic="changed"))
    with pytest.raises(ValueError, match="evidence changed"):
        controller.empirical_bindings(*empirical, REV)


@pytest.mark.parametrize(
    "mode,cpus,memory,minutes",
    [
        ("tests", 4, "8G", 15),
        ("after-cpu", 4, "8G", 5),
        ("worker", 12, "8G", 5),
        ("analyze", 12, "16G", 5),
    ],
)
def test_all_jobs_are_bounded_cpu_only(monkeypatch, tmp_path, mode, cpus, memory, minutes):
    calls = []
    monkeypatch.setattr(
        controller.subprocess,
        "check_output",
        lambda command, **kwargs: calls.append(command) or "12345\n",
    )
    controller.submit(tmp_path, REV, mode, mode, dependency="999")
    command = calls[0]
    assert "--nodes=1" in command and "--partition=bii,standard,interactive" in command
    assert f"--cpus-per-task={cpus}" in command and f"--mem={memory}" in command
    assert f"--time=00:{minutes}:00" in command
    assert "--dependency=afterany:999" in command
    assert not any(arg.startswith(("--gres", "--reservation", "--constraint")) for arg in command)
    assert ("--array=1,7%2" in command) == (mode == "worker")


def test_uncertain_acknowledgement_is_not_retried(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        controller.subprocess,
        "check_output",
        lambda command, **kwargs: calls.append(command) or "unknown",
    )
    with pytest.raises(ValueError, match="uncertain"):
        controller.submit(tmp_path, REV, "cpu", "tests")
    assert len(calls) == 1


def launch_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(controller, "SCRATCH", tmp_path)
    monkeypatch.setattr(controller, "clean_source", lambda *args: REV)
    monkeypatch.setattr(controller, "empirical_bindings", lambda *args: ({}, WHOLE))
    calls = []
    monkeypatch.setattr(
        controller,
        "submit",
        lambda *args, **kwargs: calls.append((args, kwargs)) or str(100 + len(calls)),
    )
    return tmp_path / "workdir/new", calls


def test_missing_cpu_suite_schedules_tests_then_afterany(monkeypatch, tmp_path):
    root, calls = launch_fixture(monkeypatch, tmp_path)
    result = controller.launch(root, tmp_path / "whole", tmp_path / "coco")
    assert result["cpu"] == "101" and result["after_cpu"] == "102"
    assert calls[0][0][-1] == "tests"
    assert calls[1][0][-1] == "after-cpu" and calls[1][1]["dependency"] == "101"


def test_passed_current_cpu_suite_is_reused_not_rerun(monkeypatch, tmp_path):
    root, calls = launch_fixture(monkeypatch, tmp_path)
    cpu = tmp_path / "passed"
    for name in ("cpu-tests.json", "cpu-tests.log", "cpu-tests.xml"):
        write(cpu / "full-cpu" / name, dict(source_sha=REV))
    monkeypatch.setattr(controller, "cpu_gate", lambda *args: dict(status="passed"))
    result = controller.launch(root, tmp_path / "whole", tmp_path / "coco", cpu_gate_root=cpu)
    assert result["cpu"] is None and len(calls) == 1
    plan = controller.read(root / "plan.json")
    assert plan["reuse_passed_current_CPU"] is True and len(plan["bindings"]) == 3


def test_old_cpu_source_never_relabelled_current(monkeypatch, tmp_path):
    root, calls = launch_fixture(monkeypatch, tmp_path)
    cpu = tmp_path / "old"
    write(cpu / "full-cpu/cpu-tests.json", dict(source_sha="c" * 40))
    monkeypatch.setattr(
        controller, "cpu_gate", lambda *args: pytest.fail("must not reuse old suite")
    )
    controller.launch(root, tmp_path / "whole", tmp_path / "coco", cpu_gate_root=cpu)
    assert calls[0][0][-1] == "tests"


def after_fixture(monkeypatch, tmp_path):
    plan = dict(
        source_sha=REV,
        cpu_gate_root=str(tmp_path),
        whole_gate=str(tmp_path / "whole"),
        coco_gate=str(tmp_path / "coco"),
        whole_producer_sha=WHOLE,
        media_recovery_root=None,
    )
    write(tmp_path / "plan.json", plan)
    monkeypatch.setattr(controller, "load", lambda root: plan)
    return plan


def test_failed_cpu_gets_error_analysis_but_no_workers(monkeypatch, tmp_path):
    after_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        controller, "cpu_gate", lambda *args: (_ for _ in ()).throw(ValueError("CPU failed"))
    )
    monkeypatch.setattr(
        controller.history, "register", lambda *args, **kwargs: pytest.fail("must not register")
    )
    calls = []
    monkeypatch.setattr(
        controller, "submit", lambda *args, **kwargs: calls.append((args, kwargs)) or "123"
    )
    result = controller.after_cpu(tmp_path)
    assert result["workers"] is None and len(calls) == 1 and calls[0][0][-1] == "analyze"
    assert controller.read(tmp_path / "decisions/after-cpu.json")["admitted"] is False


def test_failed_integrity_still_gets_error_analysis(monkeypatch, tmp_path):
    after_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(
        controller, "load", lambda root: (_ for _ in ()).throw(ValueError("binding changed"))
    )
    calls = []
    monkeypatch.setattr(
        controller, "submit", lambda *args, **kwargs: calls.append((args, kwargs)) or "123"
    )
    controller.after_cpu(tmp_path)
    assert len(calls) == 1 and calls[0][0][-1] == "analyze"


def test_success_registers_only_reviewed_subset_then_two_workers(monkeypatch, tmp_path):
    after_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(controller, "cpu_gate", lambda *args: dict(status="passed"))
    registrations = []
    monkeypatch.setattr(
        controller.history, "register", lambda *args, **kwargs: registrations.append(kwargs)
    )
    calls = []
    monkeypatch.setattr(
        controller,
        "submit",
        lambda *args, **kwargs: calls.append((args, kwargs)) or str(100 + len(calls)),
    )
    controller.after_cpu(tmp_path)
    assert registrations[0]["execute"] is True
    assert registrations[0]["whole_producer_sha"] == WHOLE
    assert calls[0][0][-1] == "worker" and calls[1][0][-1] == "analyze"
    assert calls[1][1]["dependency"] == "101"


def test_blocked_admission_cannot_run_scan_or_media(monkeypatch, tmp_path):
    after_fixture(monkeypatch, tmp_path)
    write(
        tmp_path / "decisions/after-cpu.json",
        dict(admitted=False, failures=[dict(error="CPU failed")]),
    )
    monkeypatch.setattr(controller.history, "analyze", lambda *args: pytest.fail("no scan"))
    monkeypatch.setattr(controller, "recover_saved_media", lambda *args: pytest.fail("no media"))
    controller.analyze(tmp_path)
    report = controller.read(tmp_path / "FINAL_CPU_continuation.json")
    assert report["status"] == "blocked_or_partial_metadata_work"
    assert report["fresh_confirmation_authorized"] is False and report["GPU_jobs_submitted"] == 0


def test_missing_scan_worker_is_not_complete_success(monkeypatch, tmp_path):
    after_fixture(monkeypatch, tmp_path)
    write(tmp_path / "decisions/after-cpu.json", dict(admitted=True, failures=[]))
    monkeypatch.setattr(
        controller.history,
        "analyze",
        lambda *args: dict(missing_original_worker_indices=[7], new_coverage_errors=0),
    )
    monkeypatch.setattr(controller, "recover_saved_media", lambda *args: dict(reused=True))
    controller.analyze(tmp_path)
    report = controller.read(tmp_path / "FINAL_CPU_continuation.json")
    assert report["status"] == "blocked_or_partial_metadata_work"
    assert report["failures"][0]["missing"] == [7]


def test_bounded_completion_never_promotes_global_freshness(monkeypatch, tmp_path):
    after_fixture(monkeypatch, tmp_path)
    write(tmp_path / "decisions/after-cpu.json", dict(admitted=True, failures=[]))
    monkeypatch.setattr(
        controller.history,
        "analyze",
        lambda *args: dict(
            missing_original_worker_indices=[],
            new_coverage_errors=0,
            subset_frontier_exhausted=False,
        ),
    )
    monkeypatch.setattr(controller, "recover_saved_media", lambda *args: dict(reused=False))
    controller.analyze(tmp_path)
    report = controller.read(tmp_path / "FINAL_CPU_continuation.json")
    assert report["status"] == "completed_bounded_metadata_work"
    for field in (
        "globally_fresh",
        "new_identity_selection_authorized",
        "materialization_authorized",
        "fresh_confirmation_authorized",
        "large_campaign_authorized",
    ):
        assert report[field] is False


def test_sbatch_disables_gpu_before_every_python_command():
    script = (
        controller.SOURCE / "mosaickv/slurm/fox_review_cpu_continuation_v30.sbatch"
    ).read_text()
    assert script.index('CUDA_VISIBLE_DEVICES=""') < script.index("exec /scratch")
    assert "--cpu-timeout-seconds 840 --cpu-capture-mode sys" in script
    assert "gcc/11.4.0" in script and "assert_clean_worktree.sh" in script


def media_fixture(tmp_path):
    from mosaickv.evaluation.coco_hash_recovery import DATASET_REVISION, PRODUCER

    root = tmp_path / "media"
    original = write(tmp_path / "old_hash_rows.json", dict(synthetic=True))
    write(
        root / "registration.json",
        dict(
            reader_source_sha=REV,
            original_producer_sha=PRODUCER,
            inputs=dict(worker=dict(path=str(original), sha256=controller.sha(original))),
        ),
    )
    write(
        root / "GATE_saved_COCO_hash_recovery.json",
        dict(
            reader_source_sha=REV,
            original_producer_sha=PRODUCER,
            dataset_revision=DATASET_REVISION,
            complete_saved_hash_reduction=True,
            hash_rows=40504,
            globally_fresh=False,
            fresh_confirmation_authorized=False,
            materialization_authorized=False,
            candidate_ids_selected_or_exported=False,
        ),
    )
    return root, original


def test_existing_media_reduction_is_strictly_hash_bound(monkeypatch, tmp_path):
    monkeypatch.setattr(controller, "SCRATCH", tmp_path)
    root, original = media_fixture(tmp_path)
    bindings = controller.saved_media_bindings(root, REV)
    assert len(bindings) == 3 and bindings[str(original)] == controller.sha(original)
    write(original, dict(synthetic="changed"))
    with pytest.raises(ValueError, match="inputs changed"):
        controller.saved_media_bindings(root, REV)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reader_source_sha", "c" * 40),
        ("original_producer_sha", "c" * 40),
        ("complete_saved_hash_reduction", 1),
        ("globally_fresh", True),
        ("fresh_confirmation_authorized", True),
        ("materialization_authorized", True),
        ("candidate_ids_selected_or_exported", True),
    ],
)
def test_media_reuse_cannot_relabel_source_or_promote_scope(monkeypatch, tmp_path, field, value):
    monkeypatch.setattr(controller, "SCRATCH", tmp_path)
    root, _ = media_fixture(tmp_path)
    path = root / "GATE_saved_COCO_hash_recovery.json"
    record = controller.read(path)
    record[field] = value
    write(path, record)
    with pytest.raises(ValueError):
        controller.saved_media_bindings(root, REV)
