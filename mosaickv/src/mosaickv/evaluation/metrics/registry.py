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
        self._aliases: dict[str, str] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: MetricDefinition) -> None:
        if definition.name in self._definitions or definition.name in self._aliases:
            raise ValueError(f"metric {definition.name!r} is already registered")
        self._definitions[definition.name] = definition

    def register_alias(self, alias: str, canonical_name: str) -> None:
        if not alias.strip():
            raise ValueError("metric alias must be non-empty")
        if alias in self._definitions or alias in self._aliases:
            raise ValueError(f"metric {alias!r} is already registered")
        if canonical_name not in self._definitions:
            raise LookupError(f"cannot alias unknown metric {canonical_name!r}")
        self._aliases[alias] = canonical_name

    def resolve(self, name: str) -> MetricDefinition:
        try:
            canonical = self._aliases.get(name, name)
            return self._definitions[canonical]
        except KeyError as error:
            raise LookupError(
                f"unknown metric {name!r}; known metrics: {', '.join(self.names())}"
            ) from error

    def names(self) -> tuple[str, ...]:
        return tuple(sorted((*self._definitions, *self._aliases)))

    def canonical_name(self, name: str) -> str:
        return self.resolve(name).name


def default_metric_registry() -> MetricRegistry:
    mean = (
        ("accuracy", "lmms_eval.task_owned", "fraction"),
        ("vqa_accuracy", "lmms_eval.textvqa", "fraction"),
        ("anls", "lmms_eval.api.metrics.anls@0.7.2", "fraction"),
        ("rouge_l_f1", "rouge==1.0.1.rouge-l.f", "fraction"),
        ("mathvista_accuracy", "lmms_eval.mathvista@0.7.2", "percentage_points"),
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
                "teacher_forced_ppl",
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
                "lmms_eval.coco_cap.PTBTokenizer+Cider_x100@0.7.2",
                "1.2",
                "score_points",
                aggregate_cider,
                "lmms_eval",
            ),
        ]
    )
    registry = MetricRegistry(definitions)
    registry.register_alias("perplexity", "teacher_forced_ppl")
    return registry


__all__ = ["MetricDefinition", "MetricRegistry", "default_metric_registry"]
