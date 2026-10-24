"""Synthetic phase inventories only; never read real benchmark artifacts."""

import importlib
import json
from pathlib import Path

import pytest


@pytest.fixture
def recovery(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("recover_bounded_history_v4")


def trace(sample="a", teacher=False, cap=100):
    return dict(
        status="completed",
        sample_id=sample,
        effective_method="synthetic",
        teacher_forced_evaluation=dict(
            enabled=teacher,
            negative_log_likelihood_sum=2.0,
            evaluated_tokens=1,
            target_was_not_truncated=True,
        ),
        cache_lifecycle={f"{b}_active_kv_storage_bytes": cap for b in ("initial", "final", "peak")},
    )


def write(folder, name, payload):
    path = folder / name
    path.write_text(json.dumps(payload))
    return path


def test_retries_only_invalid_ppl_and_reuses_predictions(recovery, tmp_path):
    paths = [
        write(tmp_path, "gen-a.json", trace()),
        write(tmp_path, "gen-b.json", trace("b")),
        write(tmp_path, "ppl-a.json", trace(teacher=True)),
        write(tmp_path, "ppl-b.json", trace("b", teacher=True, cap=200)),
    ]
    valid, retry = recovery.phase_inventory(paths, ["a", "b"], 100)
    assert len(valid) == 3
    assert [r["sample_id"] for r in retry] == ["b"]
    assert retry[0]["rejected"][0]["sha256"] == recovery.digest(paths[-1])
    assert "generation" in {phase for _, phase in valid}


def test_missing_ppl_is_explicit_but_missing_generation_blocks_scope(recovery, tmp_path):
    generation = write(tmp_path, "gen.json", trace())
    _, retry = recovery.phase_inventory([generation], ["a"], 100)
    assert retry[0]["rejected"] == []
    with pytest.raises(ValueError, match="generation is not reusable"):
        recovery.phase_inventory([], ["a"], 100)


def test_duplicate_and_unregistered_traces_fail_closed(recovery, tmp_path):
    paths = [write(tmp_path, "one.json", trace()), write(tmp_path, "two.json", trace())]
    with pytest.raises(ValueError, match="multiple admissible"):
        recovery.phase_inventory(paths, ["a"], 100)
    with pytest.raises(ValueError, match="unregistered sample"):
        recovery.phase_inventory(paths, ["unknown"], 100)


def test_fullkv_quality_reference_is_not_subject_to_compressed_cap(recovery, tmp_path):
    paths = [
        write(tmp_path, "gen.json", trace()),
        write(tmp_path, "ppl.json", trace(teacher=True, cap=200)),
    ]
    valid, retry = recovery.phase_inventory(paths, ["a"], None)
    assert not retry and len(valid) == 2


def test_submission_registers_single_globally_controlled_six_a100_array(
    recovery, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(recovery, "validate", lambda *_: dict(source_sha="a" * 40, jobs=[{}, {}]))

    def submitted(cmd, **_kwargs):
        calls.append(cmd)
        return str(100 + len(calls)) + "\n"

    monkeypatch.setattr(recovery.subprocess, "check_output", submitted)
    recovery.submit(tmp_path, "b" * 64)
    assert len(calls) == 3
    assert "--hold" in calls[0] and "--array=0-1%6" in calls[0]
    assert "--dependency=afterany:101" in calls[1]
    policy = json.loads((tmp_path / "scheduler/a100_schedule.json").read_text())
    assert policy["maximum_a100s"] == 6 and policy["arrays"] == ["101"]
    assert policy["adopt_existing_holds"] == ["101"]
    summary = json.loads((tmp_path / "submission.json").read_text())
    assert summary["generation_reruns"] == 0 and not summary["paper_campaign_submitted"]


def test_cpu_prediction_recovery_and_incomplete_analysis_are_write_once(
    recovery,
    tmp_path,
    monkeypatch,
):
    """160 tiny synthetic rows, no models, real samples, jobs or downloads."""
    from dataclasses import replace

    import pandas as pd

    from mosaickv.config import canonical_config, config_sha256, load_config
    from mosaickv.evaluation.model import GenerationMetrics
    from mosaickv.evaluation.prefixkv_metrics import FullKVReferenceArtifact
    from mosaickv.evaluation.results import EvaluationResult

    parent_root = tmp_path / "synthetic-parent"
    parent_root.mkdir()
    (parent_root / "analysis").mkdir()
    write(parent_root / "analysis", "report.json", dict(classification="synthetic_negative"))
    base = load_config(
        recovery.SOURCE / "mosaickv/configs/development/llava_bounded_history_v1.yaml"
    )
    tasks = ["llava_description", "mmvet_prefixkv"]
    ids = [f"synthetic-{i}" for i in range(8)]
    jobs = []
    for task in tasks:
        ref = FullKVReferenceArtifact(
            "fixture",
            base.model.id,
            base.model.revision,
            base.model.revision,
            "a" * 64,
            {s: "synthetic answer" for s in ids},
        )
        reference = write(parent_root, f"{task}-ref.json", ref.to_json_object())
        arms = [
            ("full_kv", None),
            *(
                (name, cap)
                for cap in (32, 40, 80)
                for name in ("prompt_attention_topk", "mosaickv_exact", "mosaickv_direct_error_v4")
            ),
        ]
        for name, cap in arms:
            index = len(jobs)
            output = parent_root / "outputs" / str(index)
            (output / "traces").mkdir(parents=True)
            cfg = replace(base, cache=replace(base.cache, budget_value=(cap or 100) * 2**20))
            config = write(parent_root, f"cfg-{index}.json", canonical_config(cfg))
            rows = []
            for sample_id in ids:
                row = EvaluationResult.from_generation(
                    run_id=f"fixture-{index}",
                    sample_id=sample_id,
                    task=task,
                    model=cfg.model.id,
                    backend="huggingface",
                    method=name,
                    retention_ratio=1.0,
                    answer="synthetic answer",
                    reference="synthetic answer",
                    task_score=None,
                    metrics=GenerationMetrics(
                        generated_tokens=1024,
                        initial_active_kv_storage_bytes=cfg.cache.budget_value,
                        final_active_kv_storage_bytes=cfg.cache.budget_value,
                        peak_active_kv_storage_bytes=cfg.cache.budget_value,
                    ),
                    manifest_path=str(output / "missing-original-manifest.json"),
                )
                rows.append(json.dumps(row.to_json_object()))
                write(
                    output / "traces",
                    f"{sample_id}-gen.json",
                    dict(trace(sample_id, cap=cfg.cache.budget_value), effective_method=name),
                )
                bytes_used = cfg.cache.budget_value * (
                    2 if index == 11 and sample_id == ids[0] else 1
                )
                write(
                    output / "traces",
                    f"{sample_id}-ppl.json",
                    trace(sample_id, teacher=True, cap=bytes_used),
                )
            (output / "results.jsonl").write_text("\n".join(rows) + "\n")
            jobs.append(
                dict(
                    index=index,
                    method=name,
                    cap_mib=cap,
                    task=task,
                    config=str(config),
                    config_sha=config_sha256(cfg),
                    output=str(output),
                    reference=str(reference),
                    sample_ids=ids,
                )
            )
    parent = dict(
        source_sha="c" * 40,
        bound_files={},
        jobs=jobs,
        protocol=dict(experiment_id="bounded_history_memory_v4", tasks=tasks),
    )
    write(parent_root, "plan.json", parent)
    monkeypatch.setattr(recovery, "PARENT", parent_root)
    monkeypatch.setattr(recovery, "PARENT_SHA", recovery.digest(parent_root / "plan.json"))
    monkeypatch.setattr(recovery, "clean_source", lambda *_: "d" * 40)
    monkeypatch.setattr(recovery, "inputs", lambda *_a, **_k: parent)
    output = tmp_path / "recovery"
    recovery.prepare(output)
    restored = pd.read_parquet(output / "recovered_rouge.parquet")
    # The pinned rouge implementation uses an epsilon in its F1 denominator.
    assert len(restored) == 160 and (abs(restored.metric_value - 1.0) < 1e-7).all()
    assert not restored.parent_manifest_exists.any()
    plan = json.loads((output / "plan.json").read_text())
    assert [j["kind"] for j in plan["jobs"]] == ["ppl", "diagnostic", "diagnostic"]
    assert plan["jobs"][0]["sample_ids"] == ids[:1]
    assert len(json.loads((output / "reused-ppl.json").read_text())) == 159
    recovery.analyze(output, recovery.digest(output / "plan.json"))
    report = json.loads((output / "analysis/report.json").read_text())
    assert report["complete_ppl"] == 159 and report["expected_ppl"] == 160
    assert len(report["failures"]) == 3 and report["v4_retained_as_negative"]
    assert not report["larger_campaign_authorized"]
    assert (output / "analysis/report.md").is_file()
    with pytest.raises(FileExistsError):
        recovery.analyze(output, recovery.digest(output / "plan.json"))
