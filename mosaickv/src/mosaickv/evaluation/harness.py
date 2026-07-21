"""Deterministic local evaluation runner with failure-preserving resume."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

from mosaickv.evaluation.messages import build_multimodal_messages
from mosaickv.evaluation.model import EvaluationRequest, GenerationMetrics, LocalEvaluationModel
from mosaickv.evaluation.results import EvaluationResult, ResultStatus
from mosaickv.evaluation.storage import JsonlResultStore, write_parquet_aggregate
from mosaickv.evaluation.tasks import (
    TaskRegistry,
    TaskSample,
    default_task_registry,
    select_samples,
)
from mosaickv.types import JsonObject


@dataclass(frozen=True, slots=True)
class EvaluationRunSummary:
    """Non-scientific execution summary for orchestration and CI."""

    run_id: str
    task: str
    selected_samples: int
    resumed_samples: int
    completed_samples: int
    failed_samples: int
    raw_output: str
    parquet_output: str | None
    synthetic: bool

    def to_json_object(self) -> JsonObject:
        return cast("JsonObject", asdict(self))


class EvaluationHarness:
    """Run local task samples and preserve every terminal outcome."""

    def __init__(self, registry: TaskRegistry | None = None) -> None:
        self.registry = registry or default_task_registry()

    def run(
        self,
        *,
        run_id: str,
        task_name: str,
        samples: tuple[TaskSample, ...],
        model: LocalEvaluationModel,
        raw_output: str | Path,
        manifest_path: str,
        seed: int,
        subset_size: int | None = None,
        parquet_output: str | Path | None = None,
    ) -> EvaluationRunSummary:
        """Run or resume a deterministic subset.

        This local runner is intentionally restricted to tasks with an explicit
        local scorer. Public benchmarks flow through :mod:`lmms_adapter`.
        """

        task = self.registry.resolve(task_name)
        if task.local_scorer is None:
            raise RuntimeError(
                f"task {task_name!r} must run through lmms-eval; no local scorer is defined"
            )
        if task.requires_video and not model.supports_video:
            raise ValueError(f"model {model.model_id!r} does not declare video support")
        selected = select_samples(samples, seed=seed, subset_size=subset_size)
        store = JsonlResultStore(raw_output)
        prior = store.completed_sample_ids(run_id)
        resumed = 0
        for sample in selected:
            if sample.sample_id in prior:
                resumed += 1
                continue
            started = time.perf_counter()
            request = EvaluationRequest(
                run_id=run_id,
                sample_id=sample.sample_id,
                task=task.name,
                messages=build_multimodal_messages(sample.prompt, sample.media),
                generation_kwargs={},
            )
            try:
                generation = model.generate(request)
                metrics = generation.metrics
                if metrics.end_to_end_time is None:
                    metrics = replace(
                        metrics, end_to_end_time=max(0.0, time.perf_counter() - started)
                    )
                score = task.score_local(generation.answer, sample.references)
                result = EvaluationResult.from_generation(
                    run_id=run_id,
                    sample_id=sample.sample_id,
                    task=task.name,
                    model=model.model_id,
                    backend=model.backend,
                    method=generation.effective_method or model.method,
                    retention_ratio=model.retention_ratio,
                    answer=generation.answer,
                    reference=sample.canonical_reference(),
                    task_score=score,
                    metrics=metrics,
                    manifest_path=manifest_path,
                )
            except Exception as error:  # Each sample must become a failure row.
                result = EvaluationResult.failed(
                    run_id=run_id,
                    sample_id=sample.sample_id,
                    task=task.name,
                    model=model.model_id,
                    backend=model.backend,
                    method=model.method,
                    retention_ratio=model.retention_ratio,
                    reference=sample.canonical_reference(),
                    error=f"{type(error).__name__}: {error}",
                    manifest_path=manifest_path,
                    metrics=GenerationMetrics(
                        end_to_end_time=max(0.0, time.perf_counter() - started)
                    ),
                )
            store.append(result)

        rows = store.results(run_id=run_id)
        selected_ids = {sample.sample_id for sample in selected}
        selected_rows = tuple(row for row in rows if row.sample_id in selected_ids)
        completed = sum(row.status == ResultStatus.COMPLETED for row in selected_rows)
        failed = sum(row.status == ResultStatus.FAILED for row in selected_rows)
        parquet_path: str | None = None
        if parquet_output is not None:
            parquet_path = str(write_parquet_aggregate(selected_rows, parquet_output).resolve())
        return EvaluationRunSummary(
            run_id=run_id,
            task=task.name,
            selected_samples=len(selected),
            resumed_samples=resumed,
            completed_samples=completed,
            failed_samples=failed,
            raw_output=str(Path(raw_output).resolve()),
            parquet_output=parquet_path,
            synthetic=task.name == "synthetic_ci",
        )


__all__ = ["EvaluationHarness", "EvaluationRunSummary"]
