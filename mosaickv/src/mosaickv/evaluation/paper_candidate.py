"""Narrow quality--memory contract for the first paper candidate.

This module intentionally contains only the two gates needed by the paper
candidate: measured attention-active KV bytes and official task score.  It
does not use benchmark answers to construct the cache.  The scorer consumes
answers only after all paired generations have completed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from statistics import fmean

from mosaickv.config import DatasetConfig, RunConfig
from mosaickv.evaluation.results import EvaluationResult, ResultStatus
from mosaickv.types import BudgetUnit, MosaicKVMethod

PAPER_CANDIDATE_MODEL_ID = "llava-hf/llava-1.5-7b-hf"
PAPER_CANDIDATE_TASKS = ("mmstar", "textvqa", "docvqa")
PAPER_CANDIDATE_METHOD = MosaicKVMethod.MOSAICKV_FULL
PAPER_CANDIDATE_BASELINE = MosaicKVMethod.PROMPT_ATTENTION_TOPK
# 32 layers * 32 KV heads * 128 head width * K/V * FP16 bytes.  The worker
# verifies the pinned model layout before using this dense-position quantum.
PAPER_CANDIDATE_STORAGE_QUANTUM_BYTES = 524_288
PAPER_CANDIDATE_TASK_ROUTES = {
    "mmstar": DatasetConfig(
        "Lin-Chen/MMStar",
        "bc98d668301da7b14f648724866e57302778ab27",
        "val",
    ),
    "textvqa": DatasetConfig(
        "lmms-lab/textvqa",
        "9c0699cd19768ac5ab97568f6b3cbac4c0062884",
        "validation",
    ),
    "docvqa": DatasetConfig(
        "lmms-lab/DocVQA",
        "539088ef8a8ada01ac8e2e6d4e372586748a265e",
        "validation",
    ),
}


@dataclass(frozen=True, slots=True)
class PaperCandidateContract:
    """Immutable definition of the first narrow paper claim."""

    minimum_active_reduction_fraction: float = 0.30
    target_active_fraction: float = 0.699
    score_noninferiority_margin: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.minimum_active_reduction_fraction < 1:
            raise ValueError("minimum active reduction must be in (0, 1)")
        if not 0 < self.target_active_fraction < 1:
            raise ValueError("target active fraction must be in (0, 1)")
        if self.target_active_fraction >= 1 - self.minimum_active_reduction_fraction:
            raise ValueError(
                "target active fraction must provide a strict safety margin beyond "
                "the minimum reduction"
            )
        if not math.isfinite(self.score_noninferiority_margin):
            raise ValueError("score noninferiority margin must be finite")
        if self.score_noninferiority_margin < 0:
            raise ValueError("score noninferiority margin must be nonnegative")

    def active_budget(self, baseline_active_bytes: int, storage_quantum_bytes: int) -> int:
        """Return a block-aligned cap strictly below 70% of baseline bytes."""

        for name, value in (
            ("baseline_active_bytes", baseline_active_bytes),
            ("storage_quantum_bytes", storage_quantum_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        unaligned = math.floor(baseline_active_bytes * self.target_active_fraction)
        aligned = (unaligned // storage_quantum_bytes) * storage_quantum_bytes
        if aligned < storage_quantum_bytes:
            raise ValueError("baseline cache is too small for one candidate storage quantum")
        realized = 1 - aligned / baseline_active_bytes
        if realized <= self.minimum_active_reduction_fraction:
            raise ValueError("aligned candidate budget does not satisfy the strict memory gate")
        return aligned

    def resolve_run_config(
        self,
        template: RunConfig,
        *,
        task: str,
        baseline_active_bytes: int,
        storage_quantum_bytes: int,
    ) -> RunConfig:
        """Bind one candidate template to a measured paired-baseline budget."""

        if task not in PAPER_CANDIDATE_TASKS:
            raise ValueError(f"unsupported paper-candidate task: {task!r}")
        if template.model.id != PAPER_CANDIDATE_MODEL_ID:
            raise ValueError("paper candidate is restricted to the pinned LLaVA-1.5-7B model")
        if template.method is not PAPER_CANDIDATE_METHOD:
            raise ValueError("paper candidate template must use mosaickv_full")
        if template.cache.budget_unit is not BudgetUnit.BYTES:
            raise ValueError("paper candidate requires byte-based cache accounting")
        budget = self.active_budget(baseline_active_bytes, storage_quantum_bytes)
        total_storage = template.cache.total_storage_budget_bytes
        if total_storage is not None and total_storage < budget:
            raise ValueError("candidate total-storage cap cannot be below its active cap")
        return replace(
            template,
            dataset=PAPER_CANDIDATE_TASK_ROUTES[task],
            cache=replace(template.cache, budget_value=budget),
        )


@dataclass(frozen=True, slots=True)
class PaperCandidateTaskGate:
    """Per-task outcome; incompatible task metrics are never averaged together."""

    task: str
    sample_count: int
    baseline_score: float
    candidate_score: float
    score_delta: float
    mean_active_reduction_fraction: float
    minimum_sample_active_reduction_fraction: float
    memory_passed: bool
    quality_passed: bool

    @property
    def passed(self) -> bool:
        return self.memory_passed and self.quality_passed


@dataclass(frozen=True, slots=True)
class PaperCandidateGateReport:
    """Complete three-task result for one immutable candidate source."""

    tasks: tuple[PaperCandidateTaskGate, ...]

    @property
    def passed(self) -> bool:
        return len(self.tasks) == len(PAPER_CANDIDATE_TASKS) and all(
            task.passed for task in self.tasks
        )


def _completed_by_identity(
    rows: tuple[EvaluationResult, ...],
    *,
    expected_method: MosaicKVMethod,
) -> dict[tuple[str, str], EvaluationResult]:
    indexed: dict[tuple[str, str], EvaluationResult] = {}
    for row in rows:
        if row.model != PAPER_CANDIDATE_MODEL_ID:
            raise ValueError(f"unexpected model in paper-candidate row: {row.model!r}")
        if row.task not in PAPER_CANDIDATE_TASKS:
            raise ValueError(f"unexpected task in paper-candidate row: {row.task!r}")
        if row.method != expected_method.value:
            raise ValueError(
                f"expected method {expected_method.value!r}, observed {row.method!r}"
            )
        if row.status is not ResultStatus.COMPLETED:
            raise ValueError(f"paper-candidate row failed: {row.task}/{row.sample_id}")
        if row.task_score is None or row.active_kv_bytes is None:
            raise ValueError("paper-candidate rows require task_score and active_kv_bytes")
        identity = (row.task, row.sample_id)
        if identity in indexed:
            raise ValueError(f"duplicate paper-candidate row: {identity}")
        indexed[identity] = row
    return indexed


def _required_score_and_bytes(row: EvaluationResult) -> tuple[float, int]:
    """Return metrics already required by the completed-row validator."""

    if row.task_score is None or row.active_kv_bytes is None:  # pragma: no cover
        raise ValueError("paper-candidate rows require task_score and active_kv_bytes")
    return float(row.task_score), int(row.active_kv_bytes)


def evaluate_paper_candidate_gate(
    baseline_rows: tuple[EvaluationResult, ...],
    candidate_rows: tuple[EvaluationResult, ...],
    *,
    contract: PaperCandidateContract | None = None,
) -> PaperCandidateGateReport:
    """Evaluate only active-memory reduction and official task non-inferiority."""

    contract = contract or PaperCandidateContract()
    baseline = _completed_by_identity(
        baseline_rows,
        expected_method=PAPER_CANDIDATE_BASELINE,
    )
    candidate = _completed_by_identity(
        candidate_rows,
        expected_method=PAPER_CANDIDATE_METHOD,
    )
    if not baseline or set(baseline) != set(candidate):
        raise ValueError("baseline and candidate must contain identical nonempty sample identities")
    observed_tasks = {task for task, _sample_id in baseline}
    if observed_tasks != set(PAPER_CANDIDATE_TASKS):
        raise ValueError(
            "paper candidate requires MMStar, TextVQA, and DocVQA in one paired report"
        )

    task_reports: list[PaperCandidateTaskGate] = []
    for task in PAPER_CANDIDATE_TASKS:
        identities = sorted(identity for identity in baseline if identity[0] == task)
        baseline_metrics = [_required_score_and_bytes(baseline[item]) for item in identities]
        candidate_metrics = [_required_score_and_bytes(candidate[item]) for item in identities]
        baseline_scores = [score for score, _active_bytes in baseline_metrics]
        candidate_scores = [score for score, _active_bytes in candidate_metrics]
        reductions = [
            1 - candidate_bytes / baseline_bytes
            for (_candidate_score, candidate_bytes), (_baseline_score, baseline_bytes) in zip(
                candidate_metrics,
                baseline_metrics,
                strict=True,
            )
        ]
        baseline_score = fmean(baseline_scores)
        candidate_score = fmean(candidate_scores)
        score_delta = candidate_score - baseline_score
        minimum_reduction = min(reductions)
        task_reports.append(
            PaperCandidateTaskGate(
                task=task,
                sample_count=len(identities),
                baseline_score=baseline_score,
                candidate_score=candidate_score,
                score_delta=score_delta,
                mean_active_reduction_fraction=fmean(reductions),
                minimum_sample_active_reduction_fraction=minimum_reduction,
                memory_passed=(
                    minimum_reduction > contract.minimum_active_reduction_fraction
                ),
                quality_passed=(
                    score_delta >= -contract.score_noninferiority_margin
                ),
            )
        )
    return PaperCandidateGateReport(tuple(task_reports))


__all__ = [
    "PAPER_CANDIDATE_BASELINE",
    "PAPER_CANDIDATE_METHOD",
    "PAPER_CANDIDATE_MODEL_ID",
    "PAPER_CANDIDATE_STORAGE_QUANTUM_BYTES",
    "PAPER_CANDIDATE_TASKS",
    "PAPER_CANDIDATE_TASK_ROUTES",
    "PaperCandidateContract",
    "PaperCandidateGateReport",
    "PaperCandidateTaskGate",
    "evaluate_paper_candidate_gate",
]
