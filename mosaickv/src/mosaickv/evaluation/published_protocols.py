"""Strict loader for published-baseline evaluation contracts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from mosaickv.evaluation.metrics.registry import MetricRegistry, default_metric_registry
from mosaickv.types import JsonObject

_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True, slots=True)
class PublishedTaskContract:
    """Dataset, metric, and sample population from one paper protocol."""

    name: str
    dataset: str
    split: str
    sample_count: int | None
    metrics: tuple[str, ...]
    aggregation: str

    def __post_init__(self) -> None:
        for field_name in ("name", "dataset", "split", "aggregation"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"task contract {field_name} must be non-empty")
        if self.sample_count is not None and self.sample_count < 1:
            raise ValueError("task contract sample_count must be positive")
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("task contract metrics must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class PublishedMethodContract:
    """One method label and its implementation provenance."""

    name: str
    source: str
    status: str
    repository_commit: str | None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.source.strip():
            raise ValueError("method name and source must be non-empty")
        if self.status not in {
            "official_pinned",
            "local_reimplementation",
            "local_reference",
            "unsupported_unpinned",
        }:
            raise ValueError(f"unsupported method provenance status {self.status!r}")
        if self.status == "official_pinned" and (
            self.repository_commit is None or _SHA.fullmatch(self.repository_commit) is None
        ):
            raise ValueError("official_pinned methods require a 40-character commit SHA")
        if self.status == "local_reimplementation" and not self.name.endswith("_reimpl"):
            raise ValueError("local paper reimplementations must end in _reimpl")


@dataclass(frozen=True, slots=True)
class PublishedBaselineProtocol:
    """Versioned paper-native protocol loaded from YAML."""

    protocol_id: str
    paper: str
    primary_source: str
    model: str
    model_revision: str
    retention_ratios: tuple[float, ...]
    tasks: tuple[PublishedTaskContract, ...]
    methods: tuple[PublishedMethodContract, ...]
    execution_authorized: bool

    def __post_init__(self) -> None:
        for field_name in ("protocol_id", "paper", "primary_source", "model"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"protocol {field_name} must be non-empty")
        if not self.model_revision.strip():
            raise ValueError("protocol model_revision must be non-empty")
        if not self.retention_ratios or any(not 0 < ratio <= 1 for ratio in self.retention_ratios):
            raise ValueError("protocol retention ratios must lie in (0, 1]")
        if tuple(sorted(set(self.retention_ratios), reverse=True)) != self.retention_ratios:
            raise ValueError("protocol retention ratios must be unique and descending")
        if not self.tasks or len({task.name for task in self.tasks}) != len(self.tasks):
            raise ValueError("protocol tasks must be non-empty and unique")
        if not self.methods or len({method.name for method in self.methods}) != len(self.methods):
            raise ValueError("protocol methods must be non-empty and unique")

    def validate_metrics(self, registry: MetricRegistry | None = None) -> None:
        selected = registry or default_metric_registry()
        for task in self.tasks:
            for metric in task.metrics:
                selected.resolve(metric)


@dataclass(frozen=True, slots=True)
class PublishedProtocolSuite:
    schema_version: int
    suite_id: str
    protocols: tuple[PublishedBaselineProtocol, ...]
    sha256: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("published protocol schema_version must be 1")
        if not self.suite_id.strip():
            raise ValueError("published protocol suite_id must be non-empty")
        if len({protocol.protocol_id for protocol in self.protocols}) != len(self.protocols):
            raise ValueError("published protocol IDs must be unique")
        for protocol in self.protocols:
            protocol.validate_metrics()


def _required_string(data: Mapping[str, object], key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string")
    return value


def _mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be an object with string keys")
    return cast("Mapping[str, object]", value)


def _sequence(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def _parse_task(value: object, path: str) -> PublishedTaskContract:
    data = _mapping(value, path)
    metrics = _sequence(data.get("metrics"), f"{path}.metrics")
    if not all(isinstance(metric, str) for metric in metrics):
        raise ValueError(f"{path}.metrics must contain strings")
    count = data.get("sample_count")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int)):
        raise ValueError(f"{path}.sample_count must be an integer or null")
    return PublishedTaskContract(
        name=_required_string(data, "name", path),
        dataset=_required_string(data, "dataset", path),
        split=_required_string(data, "split", path),
        sample_count=count,
        metrics=tuple(cast("list[str]", metrics)),
        aggregation=_required_string(data, "aggregation", path),
    )


def _parse_method(value: object, path: str) -> PublishedMethodContract:
    data = _mapping(value, path)
    commit = data.get("repository_commit")
    if commit is not None and not isinstance(commit, str):
        raise ValueError(f"{path}.repository_commit must be a string or null")
    return PublishedMethodContract(
        name=_required_string(data, "name", path),
        source=_required_string(data, "source", path),
        status=_required_string(data, "status", path),
        repository_commit=commit,
    )


def load_published_protocols(path: str | Path) -> PublishedProtocolSuite:
    source = Path(path)
    raw = source.read_bytes()
    payload = yaml.safe_load(raw)
    data = _mapping(payload, "root")
    version = data.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("root.schema_version must be an integer")
    protocols: list[PublishedBaselineProtocol] = []
    for index, value in enumerate(_sequence(data.get("protocols"), "root.protocols")):
        path_prefix = f"root.protocols[{index}]"
        item = _mapping(value, path_prefix)
        raw_ratios = _sequence(item.get("retention_ratios"), f"{path_prefix}.retention_ratios")
        ratios: list[float] = []
        for ratio in raw_ratios:
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
                raise ValueError(f"{path_prefix}.retention_ratios must contain numbers")
            ratios.append(float(ratio))
        authorized = item.get("execution_authorized")
        if not isinstance(authorized, bool):
            raise ValueError(f"{path_prefix}.execution_authorized must be boolean")
        protocols.append(
            PublishedBaselineProtocol(
                protocol_id=_required_string(item, "protocol_id", path_prefix),
                paper=_required_string(item, "paper", path_prefix),
                primary_source=_required_string(item, "primary_source", path_prefix),
                model=_required_string(item, "model", path_prefix),
                model_revision=_required_string(item, "model_revision", path_prefix),
                retention_ratios=tuple(ratios),
                tasks=tuple(
                    _parse_task(task, f"{path_prefix}.tasks[{task_index}]")
                    for task_index, task in enumerate(
                        _sequence(item.get("tasks"), f"{path_prefix}.tasks")
                    )
                ),
                methods=tuple(
                    _parse_method(method, f"{path_prefix}.methods[{method_index}]")
                    for method_index, method in enumerate(
                        _sequence(item.get("methods"), f"{path_prefix}.methods")
                    )
                ),
                execution_authorized=authorized,
            )
        )
    return PublishedProtocolSuite(
        schema_version=version,
        suite_id=_required_string(data, "suite_id", "root"),
        protocols=tuple(protocols),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def protocol_summary(suite: PublishedProtocolSuite) -> JsonObject:
    return cast(
        "JsonObject",
        {
            "schema_version": suite.schema_version,
            "suite_id": suite.suite_id,
            "sha256": suite.sha256,
            "protocols": [
                {
                    "protocol_id": protocol.protocol_id,
                    "paper": protocol.paper,
                    "model": protocol.model,
                    "tasks": [task.name for task in protocol.tasks],
                    "metrics": sorted(
                        {metric for task in protocol.tasks for metric in task.metrics}
                    ),
                    "execution_authorized": protocol.execution_authorized,
                }
                for protocol in suite.protocols
            ],
        },
    )


__all__ = [
    "PublishedBaselineProtocol",
    "PublishedMethodContract",
    "PublishedProtocolSuite",
    "PublishedTaskContract",
    "load_published_protocols",
    "protocol_summary",
]
