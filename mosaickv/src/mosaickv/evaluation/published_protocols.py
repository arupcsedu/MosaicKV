"""Strict loader for published-baseline evaluation contracts."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
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
    execution_enabled: bool = True

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
        if self.status == "unsupported_unpinned" and self.execution_enabled:
            raise ValueError("unsupported_unpinned methods cannot be execution-enabled")


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
        if self.execution_authorized and _SHA.fullmatch(self.model_revision) is None:
            raise ValueError("authorized protocols require an exact 40-character model SHA")
        if not self.retention_ratios or any(not 0 < ratio <= 1 for ratio in self.retention_ratios):
            raise ValueError("protocol retention ratios must lie in (0, 1]")
        if tuple(sorted(set(self.retention_ratios), reverse=True)) != self.retention_ratios:
            raise ValueError("protocol retention ratios must be unique and descending")
        if not self.tasks or len({task.name for task in self.tasks}) != len(self.tasks):
            raise ValueError("protocol tasks must be non-empty and unique")
        if not self.methods or len({method.name for method in self.methods}) != len(self.methods):
            raise ValueError("protocol methods must be non-empty and unique")
        if not any(method.execution_enabled for method in self.methods):
            raise ValueError("protocol must enable at least one method")

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
    execution_enabled = data.get("execution_enabled", True)
    if not isinstance(execution_enabled, bool):
        raise ValueError(f"{path}.execution_enabled must be boolean")
    return PublishedMethodContract(
        name=_required_string(data, "name", path),
        source=_required_string(data, "source", path),
        status=_required_string(data, "status", path),
        repository_commit=commit,
        execution_enabled=execution_enabled,
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
                    "enabled_methods": [
                        method.name for method in protocol.methods if method.execution_enabled
                    ],
                    "excluded_methods": [
                        method.name for method in protocol.methods if not method.execution_enabled
                    ],
                    "execution_authorized": protocol.execution_authorized,
                }
                for protocol in suite.protocols
            ],
        },
    )


def metric_readiness_issues(suite: PublishedProtocolSuite) -> tuple[str, ...]:
    """Return code/dependency blockers without weakening execution authorization."""

    issues: list[str] = []
    required_distributions = {
        "Levenshtein": "0.27.5",
        "lmms-eval": "0.7.2",
        "pycocoevalcap": "1.2",
        "rouge": "1.0.1",
    }
    for distribution, expected in required_distributions.items():
        try:
            installed = version(distribution)
        except PackageNotFoundError:
            issues.append(f"dependency {distribution} is not installed")
            continue
        if installed != expected:
            issues.append(f"dependency {distribution}=={installed}, expected {expected}")

    evaluator_imports = {
        "ANLS": "lmms_eval.api.metrics",
        "CIDEr": "lmms_eval.tasks.coco_cap.utils",
        "ROUGE-L": "rouge",
    }
    for evaluator, module_name in evaluator_imports.items():
        try:
            __import__(module_name)
        except Exception as error:
            issues.append(f"{evaluator} evaluator import failed: {type(error).__name__}: {error}")
    # lmms-eval constructs its MathVista judge at module import time.  The
    # dummy provider exercises the exact extraction/scoring module without
    # requiring credentials or changing the caller's eventual judge choice.
    mathvista_environment = os.environ.copy()
    mathvista_environment["API_TYPE"] = "dummy"
    mathvista_import = subprocess.run(
        [sys.executable, "-c", "import lmms_eval.tasks.mathvista.utils"],
        env=mathvista_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if mathvista_import.returncode != 0:
        detail = mathvista_import.stderr.strip().splitlines()
        message = detail[-1] if detail else "unknown import failure"
        issues.append(f"MathVista evaluator import failed: {message}")

    registry = default_metric_registry()
    from mosaickv.evaluation.milebench import MILEBENCH_DATASET_TO_GROUP
    from mosaickv.evaluation.tasks import default_task_registry

    task_registry = default_task_registry()
    for protocol in suite.protocols:
        for task in protocol.tasks:
            contract_metrics = {registry.canonical_name(name) for name in task.metrics}
            if task.name in task_registry.names():
                routed = task_registry.resolve(task.name)
                routed_metrics = {registry.canonical_name(name) for name in routed.paper_metrics}
                if not contract_metrics <= routed_metrics:
                    missing = sorted(contract_metrics - routed_metrics)
                    issues.append(
                        f"{protocol.protocol_id}/{task.name}: task route lacks metrics {missing}"
                    )
                if (routed.dataset_id, routed.split) != (task.dataset, task.split):
                    issues.append(
                        f"{protocol.protocol_id}/{task.name}: task route dataset/split "
                        "does not match protocol"
                    )
                if routed.lmms_task is None and not routed.scoring_owner.startswith(
                    "paper_native_"
                ):
                    issues.append(
                        f"{protocol.protocol_id}/{task.name}: route has no executable owner"
                    )
            else:
                issues.append(f"{protocol.protocol_id}/{task.name}: no metric execution route")
            if task.name == "milebench" and len(MILEBENCH_DATASET_TO_GROUP) != 29:
                issues.append(
                    f"{protocol.protocol_id}/milebench: incomplete official dataset routing"
                )
    return tuple(issues)


__all__ = [
    "PublishedBaselineProtocol",
    "PublishedMethodContract",
    "PublishedProtocolSuite",
    "PublishedTaskContract",
    "load_published_protocols",
    "metric_readiness_issues",
    "protocol_summary",
]
