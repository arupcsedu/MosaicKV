"""Unified evaluation interfaces for local and lmms-eval-backed tasks."""

from mosaickv.evaluation.harness import EvaluationHarness, EvaluationRunSummary
from mosaickv.evaluation.messages import MediaItem, MediaKind, MultimodalMessage
from mosaickv.evaluation.model import (
    EvaluationRequest,
    GenerationMetrics,
    LocalEvaluationModel,
    ModelGeneration,
)
from mosaickv.evaluation.paper_candidate import (
    PaperCandidateContract,
    PaperCandidateGateReport,
    PaperCandidateTaskGate,
    PaperCandidateV2Contract,
    evaluate_paper_candidate_gate,
)
from mosaickv.evaluation.result_categories import (
    LocallyReproducedResult,
    PaperReportedResult,
    ReproductionComparison,
    validate_locally_reproduced_manifests,
    validate_unified_comparison_manifests,
    write_reproduction_comparisons,
)
from mosaickv.evaluation.results import EvaluationResult, ResultStatus
from mosaickv.evaluation.tasks import TaskRegistry, TaskSample, TaskSpec, default_task_registry

__all__ = [
    "EvaluationHarness",
    "EvaluationRequest",
    "EvaluationResult",
    "EvaluationRunSummary",
    "GenerationMetrics",
    "LocalEvaluationModel",
    "LocallyReproducedResult",
    "MediaItem",
    "MediaKind",
    "ModelGeneration",
    "MultimodalMessage",
    "PaperCandidateContract",
    "PaperCandidateGateReport",
    "PaperCandidateTaskGate",
    "PaperCandidateV2Contract",
    "PaperReportedResult",
    "ReproductionComparison",
    "ResultStatus",
    "TaskRegistry",
    "TaskSample",
    "TaskSpec",
    "default_task_registry",
    "evaluate_paper_candidate_gate",
    "validate_locally_reproduced_manifests",
    "validate_unified_comparison_manifests",
    "write_reproduction_comparisons",
]
