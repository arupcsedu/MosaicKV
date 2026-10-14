from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from PIL import Image

from mosaickv.config import load_config
from mosaickv.evaluation.model import (
    EvaluationRequest,
    GenerationMetrics,
    ModelGeneration,
)
from mosaickv.evaluation.paper_native_lanes import (
    cache_operating_point,
    load_paper_native_ledger,
    resolve_protocol,
    run_materialized_paper_native_evaluation,
    validate_byte_matched_operating_points,
    validate_protocol_run,
)
from mosaickv.evaluation.prefixkv_metrics import TeacherForcedNll
from mosaickv.evaluation.published_protocols import load_published_protocols
from mosaickv.evaluation.results import EvaluationResult
from mosaickv.evaluation.tasks import default_task_registry
from mosaickv.types import ResultCategory, ResultOrigin


@dataclass
class _FakeNativeModel:
    model_id: str = "llava-hf/llava-1.5-7b-hf"
    backend: str = "hf"
    method: str = "full_kv"
    retention_ratio: float = 1.0
    supports_video: bool = False
    answer: str = "yes"

    def generate(self, request: EvaluationRequest) -> ModelGeneration:
        del request
        return ModelGeneration(
            self.answer,
            GenerationMetrics(
                generated_tokens=1,
                active_kv_bytes=1_024,
                residual_kv_bytes=0,
            ),
        )

    def teacher_forced_nll(self, request: EvaluationRequest, target: str) -> TeacherForcedNll:
        del request, target
        return TeacherForcedNll(1.0, 2)


def _materialized_ledger(
    root: Path,
    *,
    task: str,
    dataset_id: str,
    split: str,
    prefixkv: bool,
) -> tuple[Path, str]:
    image_path = root / "sample.png"
    Image.new("RGB", (2, 2), (10, 20, 30)).save(image_path)
    image_sha = hashlib.sha256(image_path.read_bytes()).hexdigest()
    sample: dict[str, object] = {
        "sample_id": "sample-1",
        "prompt": "Answer the question.",
        "references": ["yes"],
        "media": [{"kind": "image", "path": "sample.png", "sha256": image_sha}],
    }
    if prefixkv:
        sample["teacher_forcing_target"] = "human target"
    else:
        sample.update(
            {
                "milebench_dataset": "ActionLocalization",
                "milebench_question_type": "multi-choice",
                "choices": ["no", "yes"],
            }
        )
    payload = {
        "schema_version": 1,
        "task": task,
        "dataset_id": dataset_id,
        "dataset_revision": "a" * 40,
        "split": split,
        "samples": [sample],
    }
    path = root / f"{task}.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _run_paths(root: Path, prefix: str) -> dict[str, Path]:
    return {
        "raw_output": root / f"{prefix}.jsonl",
        "parquet_output": root / f"{prefix}-raw.parquet",
        "metric_output": root / f"{prefix}-metrics.jsonl",
        "metric_observations_parquet_output": root / f"{prefix}-observations.parquet",
        "metric_aggregates_parquet_output": root / f"{prefix}-aggregates.parquet",
    }


def test_prefixkv_materialized_lane_freezes_and_consumes_reference(tmp_path: Path) -> None:
    task = default_task_registry().resolve("llava_description")
    ledger_path, ledger_sha = _materialized_ledger(
        tmp_path,
        task=task.name,
        dataset_id=task.dataset_id,
        split=task.split,
        prefixkv=True,
    )
    ledger = load_paper_native_ledger(
        ledger_path,
        expected_sha256=ledger_sha,
        task=task,
        dataset_revision="a" * 40,
    )
    reference_path = tmp_path / "reference.json"
    full_model = _FakeNativeModel()
    summary = run_materialized_paper_native_evaluation(
        run_id="full",
        task=task,
        ledger=ledger,
        model=full_model,
        manifest_path=str(tmp_path / "full-manifest.json"),
        seed=3,
        subset_size=None,
        evaluator_commit="b" * 40,
        model_revision="c" * 40,
        tokenizer_revision="c" * 40,
        generation_config_sha256="d" * 64,
        fullkv_reference_output=reference_path,
        **_run_paths(tmp_path, "full"),
    )
    assert summary["failed_samples"] == 0
    assert summary["cache_operating_point"]["mean_active_kv_bytes"] == 1_024
    assert reference_path.is_file()

    candidate = _FakeNativeModel(method="prefixkv_reimpl", retention_ratio=0.5)
    candidate_summary = run_materialized_paper_native_evaluation(
        run_id="candidate",
        task=task,
        ledger=ledger,
        model=candidate,
        manifest_path=str(tmp_path / "candidate-manifest.json"),
        seed=3,
        subset_size=None,
        evaluator_commit="b" * 40,
        model_revision="c" * 40,
        tokenizer_revision="c" * 40,
        generation_config_sha256="d" * 64,
        fullkv_reference_input=reference_path,
        fullkv_reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        **_run_paths(tmp_path, "candidate"),
    )
    assert candidate_summary["failed_samples"] == 0
    metric_lines = (tmp_path / "candidate-metrics.jsonl").read_text(encoding="utf-8").splitlines()
    metrics = [json.loads(line) for line in metric_lines]
    assert {row["metric_name"] for row in metrics} == {
        "rouge_l_f1",
        "teacher_forced_ppl",
    }
    assert any(row["metric_scope"] == "corpus" for row in metrics)


def test_milebench_materialized_lane_emits_samples_and_hierarchy(tmp_path: Path) -> None:
    task = default_task_registry().resolve("milebench")
    ledger_path, ledger_sha = _materialized_ledger(
        tmp_path,
        task=task.name,
        dataset_id=task.dataset_id,
        split=task.split,
        prefixkv=False,
    )
    ledger = load_paper_native_ledger(
        ledger_path,
        expected_sha256=ledger_sha,
        task=task,
        dataset_revision="a" * 40,
    )
    summary = run_materialized_paper_native_evaluation(
        run_id="mile",
        task=task,
        ledger=ledger,
        model=_FakeNativeModel(method="lookm_reimpl", retention_ratio=0.5),
        manifest_path=str(tmp_path / "manifest.json"),
        seed=5,
        subset_size=None,
        evaluator_commit="e" * 40,
        model_revision="c" * 40,
        tokenizer_revision="c" * 40,
        generation_config_sha256="d" * 64,
        **_run_paths(tmp_path, "mile"),
    )
    assert summary["failed_samples"] == 0
    metrics = [
        json.loads(line)
        for line in (tmp_path / "mile-metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        row["metric_scope"] == "sample" and row["metric_name"] == "accuracy" for row in metrics
    )
    assert any(
        row["metric_scope"] == "corpus"
        and row["metric_name"] == "milebench/Realistic Temporal/accuracy"
        for row in metrics
    )


def _result(method: str, sample_id: str, active_bytes: int) -> EvaluationResult:
    return EvaluationResult.from_generation(
        run_id=f"run-{method}",
        sample_id=sample_id,
        task="docvqa",
        model="model",
        backend="hf",
        method=method,
        retention_ratio=0.5,
        answer="answer",
        reference="answer",
        task_score=1.0,
        metrics=GenerationMetrics(active_kv_bytes=active_bytes),
        manifest_path="manifest.json",
    )


@pytest.mark.parametrize("method", ["prompt_attention_topk", "lookm_reimpl", "mosaickv_exact"])
def test_unified_milebench_keeps_all_result_and_metric_categories(tmp_path, method):
    task = default_task_registry().resolve("milebench")
    path, digest = _materialized_ledger(
        tmp_path, task=task.name, dataset_id=task.dataset_id, split=task.split, prefixkv=False
    )
    ledger = load_paper_native_ledger(
        path, expected_sha256=digest, task=task, dataset_revision="a" * 40
    )
    kwargs = dict(
        run_id="unified",
        task=task,
        ledger=ledger,
        model=_FakeNativeModel(method=method, retention_ratio=0.5),
        manifest_path=str(tmp_path / "manifest.json"),
        seed=5,
        subset_size=None,
        evaluator_commit="e" * 40,
        model_revision="c" * 40,
        tokenizer_revision="c" * 40,
        generation_config_sha256="d" * 64,
        **_run_paths(tmp_path, "unified"),
    )
    summary = run_materialized_paper_native_evaluation(
        **kwargs, result_category=ResultCategory.UNIFIED_COMPARISON
    )
    assert summary["failed_samples"] == 0
    assert summary["result_category"] == "unified_comparison"
    assert summary["cache_operating_point"]["result_category"] == "unified_comparison"
    for name in ("unified.jsonl", "unified-metrics.jsonl"):
        rows = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
        assert rows
        assert {r["result_category"] for r in rows} == {"unified_comparison"}
        assert {r["result_origin"] for r in rows} == {"locally_measured"}
    # Neither a completed result nor its metric can silently become a reproduction.
    with pytest.raises(ValueError, match=r"another result category|not enabled"):
        run_materialized_paper_native_evaluation(**kwargs)


def test_unified_milebench_failure_preserves_category_and_denominator(tmp_path):
    class Broken(_FakeNativeModel):
        def generate(self, request):
            raise RuntimeError("synthetic model failure")

    task = default_task_registry().resolve("milebench")
    path, digest = _materialized_ledger(
        tmp_path, task=task.name, dataset_id=task.dataset_id, split=task.split, prefixkv=False
    )
    ledger = load_paper_native_ledger(
        path, expected_sha256=digest, task=task, dataset_revision="a" * 40
    )
    summary = run_materialized_paper_native_evaluation(
        run_id="broken",
        task=task,
        ledger=ledger,
        model=Broken(),
        manifest_path=str(tmp_path / "manifest.json"),
        seed=5,
        subset_size=None,
        evaluator_commit="e" * 40,
        model_revision="c" * 40,
        tokenizer_revision="c" * 40,
        generation_config_sha256="d" * 64,
        result_category=ResultCategory.UNIFIED_COMPARISON,
        **_run_paths(tmp_path, "broken"),
    )
    assert summary["selected_samples"] == summary["failed_samples"] == 1
    assert summary["metric_output"] is None
    row = json.loads((tmp_path / "broken.jsonl").read_text())
    assert row["result_category"] == "unified_comparison"
    assert row["status"] == "failed" and "synthetic model failure" in row["error"]


def test_actual_active_bytes_are_required_and_byte_match_is_paired() -> None:
    left = (_result("mosaickv_exact", "a", 100), _result("mosaickv_exact", "b", 200))
    right = (_result("prompt_attention_topk", "a", 100), _result("prompt_attention_topk", "b", 201))
    assert cache_operating_point(left).mean_active_kv_bytes == 150
    validate_byte_matched_operating_points(left, right, maximum_relative_difference=0.01)
    with pytest.raises(ValueError, match="active-byte mismatch"):
        validate_byte_matched_operating_points(
            left, (_result("other", "a", 80), _result("other", "b", 160))
        )
    paper_right = tuple(
        replace(
            row,
            result_category=ResultCategory.PAPER_REPRODUCTION,
            result_origin=ResultOrigin.LOCALLY_REPRODUCED,
        )
        for row in right
    )
    with pytest.raises(ValueError, match="cannot mix result categories"):
        validate_byte_matched_operating_points(left, paper_right)


def test_protocol_guard_rejects_disabled_and_wrong_operating_points() -> None:
    root = Path(__file__).resolve().parents[2]
    suite = load_published_protocols(
        root / "configs/experiments/published_paper_native_metrics_v1.yaml"
    )
    protocol = resolve_protocol(suite, "prefixkv_hf_reimpl_native_metrics_v1")
    task = default_task_registry().resolve("llava_description")
    task_revision = next(item.dataset_revision for item in protocol.tasks if item.name == task.name)
    assert task_revision is not None
    config = load_config(root / "configs/hf_prefixkv_reimpl.yaml")
    config = replace(
        config,
        dataset=replace(
            config.dataset,
            id=task.dataset_id,
            revision=task_revision,
            split=task.split,
        ),
    )
    lock_sha = hashlib.sha256((root / "env/common/requirements.lock").read_bytes()).hexdigest()
    validate_protocol_run(
        protocol,
        task=task,
        config=config,
        environment_lock_sha256=lock_sha,
    )
    with pytest.raises(ValueError, match="disabled"):
        validate_protocol_run(
            protocol,
            task=task,
            config=config,
            environment_lock_sha256=lock_sha,
            method_label="prefixkv_official",
        )
    with pytest.raises(ValueError, match="retention ratio"):
        validate_protocol_run(
            protocol,
            task=task,
            config=replace(
                config,
                cache=replace(config.cache, retention_ratio=0.55),
            ),
            environment_lock_sha256=lock_sha,
        )
    with pytest.raises(ValueError, match="dataset/split"):
        validate_protocol_run(
            protocol,
            task=task,
            config=replace(config, dataset=replace(config.dataset, id="wrong/dataset")),
            environment_lock_sha256=lock_sha,
        )
    with pytest.raises(ValueError, match="dataset revision"):
        validate_protocol_run(
            protocol,
            task=task,
            config=replace(config, dataset=replace(config.dataset, revision="a" * 40)),
            environment_lock_sha256=lock_sha,
        )
    with pytest.raises(ValueError, match="supported backend/attention/precision/lock"):
        validate_protocol_run(
            protocol,
            task=task,
            config=config,
            environment_lock_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="cache budget convention"):
        validate_protocol_run(
            protocol,
            task=task,
            config=replace(
                config,
                cache=replace(config.cache, accounting_spec_sha="0" * 64),
            ),
            environment_lock_sha256=lock_sha,
        )


def test_ledger_hash_and_task_specific_metadata_fail_closed(tmp_path: Path) -> None:
    task = default_task_registry().resolve("llava_description")
    path, digest = _materialized_ledger(
        tmp_path,
        task=task.name,
        dataset_id=task.dataset_id,
        split=task.split,
        prefixkv=True,
    )
    with pytest.raises(ValueError, match="ledger SHA mismatch"):
        load_paper_native_ledger(
            path,
            expected_sha256="0" * 64,
            task=task,
            dataset_revision="a" * 40,
        )
    assert digest != "0" * 64
