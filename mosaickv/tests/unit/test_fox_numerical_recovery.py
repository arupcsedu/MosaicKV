"""Recovery diagnosis does not waive thresholds or promote missing evidence."""

import json
from pathlib import Path

import evicastkv_numerical_recovery as study


def test_parity_needs_both_complete_checks_and_original_tolerance():
    item = dict(generated_tokens=16, token_agreement=1.0, maximum_logit_difference=0.0)
    checks = dict(ordinary_generate=item, reinjection=item)
    assert study.parity_passed(checks)
    assert not study.parity_passed({})
    assert not study.parity_passed(dict(ordinary_generate=item))
    for change in (
        dict(generated_tokens=15),
        dict(token_agreement=0.99),
        dict(maximum_logit_difference=0.0002),
    ):
        assert not study.parity_passed(checks | dict(reinjection=item | change))


def test_missing_gpu_artifact_is_incomplete_not_a_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "clean_source", lambda *_: "a" * 40)
    (tmp_path / "protocol.json").write_text(
        json.dumps(
            dict(
                source_sha="a" * 40,
                workloads=dict(milebench=dict(sample_ids=["used"]), coco=dict(sample_ids=["used"])),
            )
        )
    )
    study.analyze(tmp_path)
    result = json.loads((tmp_path / "analysis.json").read_text())
    assert not result["complete"]
    assert not result["thresholds_changed"] and not result["old_failures_reclassified"]
    assert not result["fresh_confirmation_authorized"]
    assert all(c["status"] == "failed_or_incomplete" for c in result["controls"])


def test_cpu_analysis_has_no_gpu_request_and_worker_sets_determinism_before_python():
    root = Path(__file__).resolve().parents[2]
    cpu = (root / "slurm/evicastkv_numerical_recovery_analysis.sbatch").read_text()
    assert "#SBATCH --gres" not in cpu and 'CUDA_VISIBLE_DEVICES=""' in cpu
    gpu = (root / "slurm/evicastkv_numerical_recovery.sbatch").read_text()
    assert "#SBATCH --gres=gpu:a100:1" in gpu and "#SBATCH --cpus-per-task=12" in gpu
    assert gpu.index("CUBLAS_WORKSPACE_CONFIG=:4096:8") < gpu.index("exec /scratch")


def test_submission_is_consumed_only_and_failure_aware_without_changing_existing_jobs(
    tmp_path, monkeypatch
):
    parents = {}
    for workload, ids in (("milebench", list(study.CASES)), ("coco", ["old1", "old2"])):
        parent = tmp_path / workload
        parent.mkdir()
        (parent / "protocol_lock.json").write_text(
            json.dumps(
                dict(
                    cohorts={workload: dict(validation=dict(sample_ids=ids))},
                )
            )
        )
        parents[workload] = parent
    monkeypatch.setattr(study, "clean_source", lambda *_: "a" * 40)
    monkeypatch.setattr(study, "exclusion_inventory", lambda *_: None)
    monkeypatch.setattr(study, "confirmation_feasibility", lambda *_: None)
    calls = []

    def fake_submit(command, **kwargs):
        calls.append(command)
        return str(1000 + len(calls))

    monkeypatch.setattr(study.subprocess, "check_output", fake_submit)
    root = tmp_path / "new"
    study.submit(root, parents["milebench"], parents["coco"], "old-gpu-array")
    assert len(calls) == 2 and all(c[0] == "sbatch" for c in calls)
    assert "--dependency=afterany:old-gpu-array" in calls[0]
    assert "--dependency=afterany:1001" in calls[1]
    assert all(not token.startswith("--gres") for token in calls[1])
    assert any("_analysis.sbatch" in token for token in calls[1])
    protocol = json.loads((root / "protocol.json").read_text())
    assert not protocol["fresh_inputs"] and not protocol["automatic_promotion"]
    assert protocol["numerical_tolerances"]["A_B_original"] == 1e-4
