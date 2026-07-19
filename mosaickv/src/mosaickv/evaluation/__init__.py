"""Unified evaluation interfaces for local and lmms-eval-backed tasks."""

from mosaickv.evaluation.harness import EvaluationHarness, EvaluationRunSummary
from mosaickv.evaluation.messages import MediaItem, MediaKind, MultimodalMessage
from mosaickv.evaluation.model import (
    EvaluationRequest,
    GenerationMetrics,
    LocalEvaluationModel,
    ModelGeneration,
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
    "MediaItem",
    "MediaKind",
    "ModelGeneration",
    "MultimodalMessage",
    "ResultStatus",
    "TaskRegistry",
    "TaskSample",
    "TaskSpec",
    "default_task_registry",
]
