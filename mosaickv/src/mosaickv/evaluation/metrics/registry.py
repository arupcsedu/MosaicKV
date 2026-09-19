"""Fail-closed registry for paper-native metric definitions."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from mosaickv.evaluation.metrics.scorers import (
    aggregate_cider,
    aggregate_mean,
    aggregate_perplexity,
)
from mosaickv.evaluation.metrics.types import MetricDirection, MetricSample, MetricScope

MetricReducer = Callable[[Sequence[MetricSample]], float]


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Identity and reduction contract for one metric."""

    name: str
    direction: MetricDirection
    scope: MetricScope
    implementation: str
    version: str
    unit: str
    reducer: MetricReducer
    evaluator_owner: str

    def __post_init__(self) -> None:
        for name in ("name", "implementation", "version", "unit", "evaluator_owner"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")

    def aggregate(self, samples: Sequence[MetricSample]) -> float:
        return self.reducer(samples)


class MetricRegistry:
    """Registry that rejects unknown or duplicate metric identities."""

    def __init__(self, definitions: Iterable[MetricDefinition] = ()) -> None:
        self._definitions: dict[str, MetricDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: MetricDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"metric {definition.name!r} is already registered")
        self._definitions[definition.name] = definition

    def resolve(self, name: str) -> MetricDefinition:
        try:
            return self._definitions[name]
        except KeyError as error:
            raise LookupError(
                f"unknown metric {name!r}; known metrics: {', '.join(self.names())}"
            ) from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._definitions))


def default_metric_registry() -> MetricRegistry:
    mean = (
        ("accuracy", "lmms_eval.task_owned", "fraction"),
        ("vqa_accuracy", "lmms_eval.textvqa", "fraction"),
        ("anls", "lmms_eval.docvqa", "fraction"),
        ("rouge_l_f1", "mosaickv.rouge_l_lcs", "fraction"),
        ("mathvista_accuracy", "lmms_eval.mathvista", "percentage_points"),
    )
    definitions = [
        MetricDefinition(
            name,
            MetricDirection.HIGHER,
            MetricScope.SAMPLE,
            implementation,
            "1",
            unit,
            aggregate_mean,
            "lmms_eval" if implementation.startswith("lmms_eval") else "mosaickv",
        )
        for name, implementation, unit in mean
    ]
    definitions.extend(
        [
            MetricDefinition(
                "perplexity",
                MetricDirection.LOWER,
                MetricScope.CORPUS,
                "mosaickv.teacher_forced_token_nll",
                "1",
                "perplexity",
                aggregate_perplexity,
                "prefixkv_paper_protocol",
            ),
            MetricDefinition(
                "cider",
                MetricDirection.HIGHER,
                MetricScope.CORPUS,
                "pycocoevalcap.cider.Cider_x100",
                "1.2",
                "score_points",
                aggregate_cider,
                "lmms_eval",
            ),
        ]
    )
    return MetricRegistry(definitions)


__all__ = ["MetricDefinition", "MetricRegistry", "default_metric_registry"]
