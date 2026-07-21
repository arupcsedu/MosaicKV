"""Evaluation task definitions and deterministic development subsets."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib.resources import files

from mosaickv.evaluation.messages import MediaItem, MediaKind

LocalScorer = Callable[[str, tuple[str, ...]], float]


@dataclass(frozen=True, slots=True)
class TaskSample:
    """One task sample before message construction."""

    sample_id: str
    prompt: str
    references: tuple[str, ...]
    media: tuple[MediaItem, ...]

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if not self.references or any(not item.strip() for item in self.references):
            raise ValueError("references must contain non-empty strings")

    def canonical_reference(self) -> str:
        """Serialize one or more references without losing multiple annotations."""

        if len(self.references) == 1:
            return self.references[0]
        return json.dumps(self.references, ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Static benchmark routing and scoring metadata."""

    name: str
    dataset_id: str
    split: str
    lmms_task: str | None
    development_lmms_task: str | None
    sample_metric: str
    media_kinds: frozenset[MediaKind]
    scoring_owner: str
    scoring_deterministic: bool
    local_scorer: LocalScorer | None = None

    @property
    def requires_video(self) -> bool:
        return MediaKind.VIDEO in self.media_kinds

    def score_local(self, answer: str, references: tuple[str, ...]) -> float:
        """Score only tasks explicitly owned by the local CI harness."""

        if self.local_scorer is None:
            raise RuntimeError(
                f"task {self.name!r} must be scored by {self.scoring_owner}; "
                "local fallback scoring is intentionally unavailable"
            )
        return self.local_scorer(answer, references)


class TaskRegistry:
    """Fail-closed registry for supported evaluation tasks."""

    def __init__(self, tasks: Iterable[TaskSpec] = ()) -> None:
        self._tasks: dict[str, TaskSpec] = {}
        for task in tasks:
            self.register(task)

    def register(self, task: TaskSpec) -> None:
        if task.name in self._tasks:
            raise ValueError(f"task {task.name!r} is already registered")
        self._tasks[task.name] = task

    def resolve(self, name: str) -> TaskSpec:
        if name == "synthetic_smoke":
            name = "synthetic_ci"
        try:
            return self._tasks[name]
        except KeyError as error:
            known = ", ".join(self.names())
            raise LookupError(f"unknown evaluation task {name!r}; known tasks: {known}") from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tasks))


def _normalize_ci_answer(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold()).strip()


def _score_ci(answer: str, references: tuple[str, ...]) -> float:
    normalized = _normalize_ci_answer(answer)
    return float(any(normalized == _normalize_ci_answer(reference) for reference in references))


def default_task_registry() -> TaskRegistry:
    """Return task routes audited against lmms-eval 0.7.2."""

    image = frozenset({MediaKind.IMAGE})
    return TaskRegistry(
        [
            TaskSpec(
                "mmstar",
                "Lin-Chen/MMStar",
                "test",
                "mmstar",
                "mmstar",
                "average",
                image,
                "lmms_eval",
                True,
            ),
            TaskSpec(
                "mmvet",
                "lmms-lab/MMVet",
                "test",
                "mmvet",
                "mmvet",
                "gpt_eval_score",
                image,
                "lmms_eval_external_judge",
                False,
            ),
            TaskSpec(
                "textvqa",
                "lmms-lab/textvqa",
                "validation",
                "textvqa_val",
                "textvqa_val_lite",
                "exact_match",
                image,
                "lmms_eval",
                True,
            ),
            TaskSpec(
                "docvqa",
                "lmms-lab/DocVQA",
                "validation",
                "docvqa_val",
                "docvqa_val_lite",
                "anls",
                image,
                "lmms_eval",
                True,
            ),
            TaskSpec(
                "chartqa",
                "lmms-lab/ChartQA",
                "test",
                "chartqa",
                "chartqa_lite",
                "relaxed_overall",
                image,
                "lmms_eval",
                True,
            ),
            TaskSpec(
                "video_mme",
                "lmms-lab/Video-MME",
                "test",
                "videomme",
                "videomme",
                "videomme_perception_score",
                frozenset({MediaKind.VIDEO}),
                "lmms_eval",
                True,
            ),
            TaskSpec(
                "synthetic_ci",
                "mosaickv/synthetic-ci",
                "test",
                None,
                None,
                "exact_match",
                image,
                "mosaickv_ci_only",
                True,
                _score_ci,
            ),
        ]
    )


def deterministic_sample_ids(
    sample_ids: Sequence[str],
    *,
    seed: int,
    subset_size: int | None,
) -> tuple[str, ...]:
    """Select a stable subset using SHA-256 ordering, independent of input order."""

    if seed < 0:
        raise ValueError("seed must be >= 0")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs must be unique before selection")
    if subset_size is not None and subset_size < 1:
        raise ValueError("subset_size must be >= 1 when provided")

    def key(sample_id: str) -> tuple[bytes, str]:
        payload = f"mosaickv-subset-v1\0{seed}\0{sample_id}".encode()
        return hashlib.sha256(payload).digest(), sample_id

    ordered = sorted(sample_ids, key=key)
    if subset_size is not None:
        ordered = ordered[: min(subset_size, len(ordered))]
    return tuple(ordered)


def select_samples(
    samples: Sequence[TaskSample],
    *,
    seed: int,
    subset_size: int | None,
) -> tuple[TaskSample, ...]:
    """Return samples in deterministic selected order."""

    by_id = {sample.sample_id: sample for sample in samples}
    if len(by_id) != len(samples):
        raise ValueError("task samples contain duplicate sample IDs")
    selected = deterministic_sample_ids(tuple(by_id), seed=seed, subset_size=subset_size)
    return tuple(by_id[sample_id] for sample_id in selected)


def load_synthetic_samples() -> tuple[TaskSample, ...]:
    """Load the packaged, weight-free CI fixture."""

    resource = files("mosaickv.evaluation.data").joinpath("synthetic_ci.jsonl")
    samples: list[TaskSample] = []
    for line_number, line in enumerate(resource.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        try:
            pixel = tuple(int(value) for value in payload["pixel_rgb"])
            samples.append(
                TaskSample(
                    sample_id=str(payload["sample_id"]),
                    prompt=str(payload["prompt"]),
                    references=tuple(str(value) for value in payload["references"]),
                    media=(MediaItem(MediaKind.IMAGE, pixel),),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid synthetic fixture row {line_number}: {error}") from error
    if not samples:
        raise ValueError("synthetic fixture is empty")
    return tuple(samples)


__all__ = [
    "TaskRegistry",
    "TaskSample",
    "TaskSpec",
    "default_task_registry",
    "deterministic_sample_ids",
    "load_synthetic_samples",
    "select_samples",
]
