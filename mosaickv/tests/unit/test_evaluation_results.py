from __future__ import annotations

from pathlib import Path

import pytest

from mosaickv.evaluation.model import GenerationMetrics
from mosaickv.evaluation.results import RESULT_COLUMNS, EvaluationResult
from mosaickv.evaluation.storage import (
    JsonlResultStore,
    ResultConflictError,
    load_jsonl,
    merge_jsonl,
    write_parquet_aggregate,
)


def _row(run_id: str, sample_id: str, answer: str = "blue") -> EvaluationResult:
    return EvaluationResult.from_generation(
        run_id=run_id,
        sample_id=sample_id,
        task="synthetic_ci",
        model="synthetic/model",
        backend="synthetic",
        method="ci_fixture",
        retention_ratio=1.0,
        answer=answer,
        reference="blue",
        task_score=1.0,
        metrics=GenerationMetrics(generated_tokens=1),
        manifest_path="/tmp/manifest.json",
    )


def test_result_schema_contains_all_required_research_fields() -> None:
    required = {
        "sample_id",
        "model",
        "backend",
        "method",
        "retention_ratio",
        "answer",
        "reference",
        "task_score",
        "ttft",
        "prefill_time",
        "compression_time",
        "decode_time",
        "end_to_end_time",
        "generated_tokens",
        "active_kv_bytes",
        "residual_kv_bytes",
        "peak_gpu_memory",
        "repair_count",
        "repaired_bytes",
        "token_agreement",
        "logit_kl",
        "attention_output_error",
        "manifest_path",
    }
    assert required <= set(RESULT_COLUMNS)


def test_jsonl_store_resumes_and_rejects_conflicts(tmp_path: Path) -> None:
    store = JsonlResultStore(tmp_path / "raw.jsonl")
    first = _row("run-a", "sample-a")
    assert store.append(first) is True
    assert store.append(first) is False
    assert store.completed_sample_ids("run-a") == frozenset({"sample-a"})
    with pytest.raises(ResultConflictError, match="refusing to replace"):
        store.append(_row("run-a", "sample-a", answer="red"))


def test_merge_deduplicates_identical_run_sample_keys(tmp_path: Path) -> None:
    left = JsonlResultStore(tmp_path / "left.jsonl")
    right = JsonlResultStore(tmp_path / "right.jsonl")
    duplicate = _row("run-a", "sample-a")
    left.append(duplicate)
    right.append(duplicate)
    right.append(_row("run-a", "sample-b"))
    destination = merge_jsonl((left.path, right.path), tmp_path / "merged.jsonl")
    rows = load_jsonl(destination)
    assert [(row.run_id, row.sample_id) for row in rows] == [
        ("run-a", "sample-a"),
        ("run-a", "sample-b"),
    ]


def test_parquet_materialization_is_explicitly_optional(tmp_path: Path) -> None:
    output = tmp_path / "aggregate.parquet"
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="pyarrow"):
            write_parquet_aggregate((_row("run", "sample"),), output)
    else:
        assert write_parquet_aggregate((_row("run", "sample"),), output) == output
        assert output.stat().st_size > 0


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_result_rejects_invalid_retention_ratio(value: float) -> None:
    with pytest.raises(ValueError, match="retention_ratio"):
        EvaluationResult.from_generation(
            run_id="run",
            sample_id="sample",
            task="synthetic_ci",
            model="model",
            backend="synthetic",
            method="method",
            retention_ratio=value,
            answer="answer",
            reference="answer",
            task_score=1.0,
            metrics=GenerationMetrics(),
            manifest_path="manifest.json",
        )
