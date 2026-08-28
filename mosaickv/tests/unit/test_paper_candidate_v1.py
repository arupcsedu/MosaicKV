from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mosaickv.config import load_config
from mosaickv.evaluation.paper_candidate import (
    PAPER_CANDIDATE_MODEL_ID,
    PAPER_CANDIDATE_STORAGE_QUANTUM_BYTES,
    PAPER_CANDIDATE_TASKS,
    PaperCandidateContract,
    evaluate_paper_candidate_gate,
)
from mosaickv.evaluation.results import EvaluationResult, ResultStatus


def _row(
    task: str,
    method: str,
    *,
    score: float,
    active_bytes: int,
    sample_id: str | None = None,
) -> EvaluationResult:
    return EvaluationResult(
        run_id=f"run-{task}-{method}",
        sample_id=sample_id or f"{task}:1",
        task=task,
        status=ResultStatus.COMPLETED,
        error=None,
        model=PAPER_CANDIDATE_MODEL_ID,
        backend="hf",
        method=method,
        retention_ratio=0.2,
        answer="answer",
        reference="answer",
        task_score=score,
        ttft=1.0,
        prefill_time=1.0,
        compression_time=1.0,
        decode_time=1.0,
        end_to_end_time=1.0,
        generated_tokens=1,
        active_kv_bytes=active_bytes,
        residual_kv_bytes=0,
        peak_gpu_memory=1,
        repair_count=0,
        repaired_bytes=0,
        token_agreement=1.0,
        logit_kl=0.0,
        attention_output_error=0.0,
        manifest_path="/scratch/test-manifest.json",
    )


def test_budget_is_aligned_and_strictly_more_than_thirty_percent_smaller() -> None:
    contract = PaperCandidateContract()
    baseline = 40_370_176
    budget = contract.active_budget(baseline, PAPER_CANDIDATE_STORAGE_QUANTUM_BYTES)
    assert budget == 53 * PAPER_CANDIDATE_STORAGE_QUANTUM_BYTES
    assert 1 - budget / baseline > 0.30
    assert budget <= baseline * 0.699


def test_candidate_source_resolves_only_the_registered_model_and_tasks() -> None:
    package = Path(__file__).parents[2]
    template = load_config(package / "configs/paper_candidate_llava_v1.yaml")
    resolved = PaperCandidateContract().resolve_run_config(
        template,
        task="textvqa",
        baseline_active_bytes=40_000_000,
        storage_quantum_bytes=32_768,
    )
    assert resolved.model.id == PAPER_CANDIDATE_MODEL_ID
    assert resolved.dataset.id == "lmms-lab/textvqa"
    assert resolved.dataset.revision == "9c0699cd19768ac5ab97568f6b3cbac4c0062884"
    assert resolved.cache.budget_value < 28_000_000
    with pytest.raises(ValueError, match="unsupported paper-candidate task"):
        PaperCandidateContract().resolve_run_config(
            template,
            task="chartqa",
            baseline_active_bytes=40_000_000,
            storage_quantum_bytes=32_768,
        )


def test_gate_requires_quality_noninferiority_and_memory_on_every_task() -> None:
    baseline = tuple(
        _row(task, "prompt_attention_topk", score=0.75, active_bytes=1_000_000)
        for task in PAPER_CANDIDATE_TASKS
    )
    candidate = tuple(
        _row(task, "mosaickv_full", score=0.75, active_bytes=690_000)
        for task in PAPER_CANDIDATE_TASKS
    )
    report = evaluate_paper_candidate_gate(baseline, candidate)
    assert report.passed
    assert all(task.memory_passed and task.quality_passed for task in report.tasks)

    degraded = list(candidate)
    degraded[1] = _row("textvqa", "mosaickv_full", score=0.74, active_bytes=690_000)
    failed = evaluate_paper_candidate_gate(baseline, tuple(degraded))
    assert not failed.passed
    assert not next(task for task in failed.tasks if task.task == "textvqa").quality_passed


def test_protocol_has_exactly_one_model_three_tasks_and_two_primary_gates() -> None:
    package = Path(__file__).parents[2]
    protocol = yaml.safe_load(
        (package / "configs/experiments/paper_quality_memory_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["scope"]["model_id"] == PAPER_CANDIDATE_MODEL_ID
    assert tuple(protocol["scope"]["tasks"]) == PAPER_CANDIDATE_TASKS
    assert protocol["scope"]["nearest_baseline"] == "prompt_attention_topk"
    assert (
        protocol["budget_contract"]["target_active_fraction_of_prompt_attention_topk"]
        == 0.699
    )
    assert protocol["budget_contract"]["minimum_active_reduction_fraction"] == 0.30
    assert protocol["quality_contract"]["margin"] == 0.0
    assert protocol["execution_authorization"]["gpu_preflight_authorized"] is True
    assert protocol["execution_authorization"]["paper_experiment_authorized"] is False
    assert protocol["preflight"]["sample_count_per_task"] == 2
    assert protocol["preflight"]["methods"] == [
        "full_kv",
        "prompt_attention_topk",
        "mosaickv_full",
    ]
    assert protocol["preflight"]["automatic_paper_submission"] is False
