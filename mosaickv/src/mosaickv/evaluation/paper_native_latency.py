"""Fail-closed protocol and trace utilities for paper-native latency trials."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


class PaperNativeLatencyProtocolError(ValueError):
    """Raised when a paper-native latency protocol is incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class PersistentTimingProtocol:
    """Registered paired timing procedure executed in one model process."""

    lane: str
    methods: tuple[str, str]
    sample_count: int
    warmups_per_method: int
    timed_repetitions_per_method: int
    trial_order: str
    persistent_model_process: bool
    synchronize_cuda_at_method_boundaries: bool
    primary_statistic: str
    bootstrap_resamples: int
    maximum_latency_ratio: float


@dataclass(frozen=True, slots=True)
class PaperNativeLatencyProtocol:
    """Validated subset of the versioned paper-native experiment protocol."""

    experiment_id: str
    classification: str
    source_protocol: str
    timing: PersistentTimingProtocol
    expected_mosaickv_mean_active_kv_bytes: int
    expected_mean_active_kv_byte_tolerance: int


def _mapping(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PaperNativeLatencyProtocolError(f"{path} must be an object")
    return {str(key): item for key, item in value.items()}


def _exact_fields(value: Mapping[str, Any], expected: set[str], path: str) -> None:
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unknown:
            details.append(f"unknown {', '.join(unknown)}")
        raise PaperNativeLatencyProtocolError(f"{path} fields are invalid: {'; '.join(details)}")


def load_paper_native_latency_protocol(path: str | Path) -> PaperNativeLatencyProtocol:
    """Load and strictly validate the registered persistent timing procedure."""

    source = Path(path).resolve(strict=True)
    payload = _mapping(yaml.safe_load(source.read_text(encoding="utf-8")), "protocol")
    _exact_fields(
        payload,
        {
            "schema_version",
            "experiment_id",
            "classification",
            "parent_experiment",
            "source_protocol",
            "selection_policy",
            "generation",
            "methods",
            "mosaickv_block_one_profile",
            "timing_protocol",
            "acceptance_gates",
            "reporting",
        },
        "protocol",
    )
    if payload["schema_version"] != 1:
        raise PaperNativeLatencyProtocolError("protocol.schema_version must be 1")
    experiment_id = str(payload["experiment_id"])
    if experiment_id != "paper_native_latency_v3":
        raise PaperNativeLatencyProtocolError(
            "protocol.experiment_id must be paper_native_latency_v3"
        )

    selection = _mapping(payload["selection_policy"], "selection_policy")
    prefix = _mapping(selection.get("prefixkv"), "selection_policy.prefixkv")
    if prefix.get("sample_count") != 4:
        raise PaperNativeLatencyProtocolError("PrefixKV timing requires exactly four samples")

    methods = _mapping(payload["methods"], "methods")
    prefix_methods = methods.get("prefixkv")
    if not isinstance(prefix_methods, list):
        raise PaperNativeLatencyProtocolError("methods.prefixkv must be an array")
    method_names = tuple(
        str(_mapping(item, f"methods.prefixkv[{index}]").get("name"))
        for index, item in enumerate(prefix_methods)
    )
    if method_names != ("full_kv", "prefixkv_reimpl", "mosaickv_exact"):
        raise PaperNativeLatencyProtocolError("methods.prefixkv has an unexpected method order")

    profile = _mapping(payload["mosaickv_block_one_profile"], "mosaickv_block_one_profile")
    if (
        profile.get("stable_head_selector") != "vectorized_unit_block_topk"
        or profile.get("contiguous_prompt_score_reuse") is not True
        or float(profile.get("routed_head_fraction", -1.0)) != 1.0 / 64.0
        or profile.get("uniform_physical_capacity_per_layer_head") is not True
    ):
        raise PaperNativeLatencyProtocolError(
            "MosaicKV latency profile must retain the audited vectorized 1/64-head path"
        )

    timing = _mapping(payload["timing_protocol"], "timing_protocol")
    _exact_fields(
        timing,
        {
            "lane",
            "methods",
            "sample_count",
            "warmups_per_method",
            "timed_repetitions_per_method",
            "trial_order",
            "persistent_model_process",
            "synchronize_cuda_at_method_boundaries",
            "primary_statistic",
            "bootstrap_resamples",
            "maximum_latency_ratio",
        },
        "timing_protocol",
    )
    timing_methods = timing["methods"]
    if not isinstance(timing_methods, list) or tuple(timing_methods) != (
        "prefixkv_reimpl",
        "mosaickv_exact",
    ):
        raise PaperNativeLatencyProtocolError(
            "timing_protocol.methods must be PrefixKV followed by MosaicKV"
        )
    registered = PersistentTimingProtocol(
        lane=str(timing["lane"]),
        methods=cast("tuple[str, str]", tuple(str(item) for item in timing_methods)),
        sample_count=int(timing["sample_count"]),
        warmups_per_method=int(timing["warmups_per_method"]),
        timed_repetitions_per_method=int(timing["timed_repetitions_per_method"]),
        trial_order=str(timing["trial_order"]),
        persistent_model_process=bool(timing["persistent_model_process"]),
        synchronize_cuda_at_method_boundaries=bool(timing["synchronize_cuda_at_method_boundaries"]),
        primary_statistic=str(timing["primary_statistic"]),
        bootstrap_resamples=int(timing["bootstrap_resamples"]),
        maximum_latency_ratio=float(timing["maximum_latency_ratio"]),
    )
    if (
        registered.lane != "prefixkv"
        or registered.sample_count != 4
        or registered.warmups_per_method < 5
        or registered.timed_repetitions_per_method < 20
        or registered.trial_order != "alternating_forward_reverse"
        or not registered.persistent_model_process
        or not registered.synchronize_cuda_at_method_boundaries
        or registered.primary_statistic != "paired_median_repetition_mean_ratio"
        or registered.bootstrap_resamples < 10_000
        or registered.maximum_latency_ratio != 1.05
    ):
        raise PaperNativeLatencyProtocolError(
            "timing_protocol does not satisfy the registered paper measurement contract"
        )

    gates = _mapping(payload["acceptance_gates"], "acceptance_gates")
    latency = _mapping(gates.get("latency"), "acceptance_gates.latency")
    if float(latency.get("maximum_relative_ratio", -1.0)) != registered.maximum_latency_ratio:
        raise PaperNativeLatencyProtocolError("latency gate and timing protocol disagree")
    physical = _mapping(gates.get("physical_layout"), "acceptance_gates.physical_layout")
    expected_bytes = int(physical.get("expected_prefixkv_mosaickv_mean_active_kv_bytes", -1))
    byte_tolerance = int(physical.get("expected_mean_active_kv_byte_tolerance", -1))
    if expected_bytes != 42_860_544 or byte_tolerance != 0:
        raise PaperNativeLatencyProtocolError(
            "physical-layout gate must preserve the registered 40.875 MiB MosaicKV cache"
        )
    return PaperNativeLatencyProtocol(
        experiment_id=experiment_id,
        classification=str(payload["classification"]),
        source_protocol=str(payload["source_protocol"]),
        timing=registered,
        expected_mosaickv_mean_active_kv_bytes=expected_bytes,
        expected_mean_active_kv_byte_tolerance=byte_tolerance,
    )


def paired_method_order(protocol: PersistentTimingProtocol, repetition: int) -> tuple[str, str]:
    """Return the preregistered order, reversing every other repetition."""

    if repetition < 0:
        raise ValueError("repetition must be nonnegative")
    return protocol.methods if repetition % 2 == 0 else (protocol.methods[1], protocol.methods[0])


def canonical_inference_traces(
    paths: tuple[Path, ...],
) -> tuple[dict[str, Any], ...]:
    """Select one non-teacher-forced inference trace per sample, fail closed."""

    selected: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        record = _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise PaperNativeLatencyProtocolError(f"trace has no sample_id: {path}")
        teacher = record.get("teacher_forced_evaluation")
        is_teacher_forced = bool(isinstance(teacher, Mapping) and teacher.get("enabled") is True)
        if is_teacher_forced:
            continue
        if sample_id in selected:
            raise PaperNativeLatencyProtocolError(
                f"multiple canonical inference traces for sample {sample_id}"
            )
        selected[sample_id] = record
    if not selected:
        raise PaperNativeLatencyProtocolError("no canonical inference traces were found")
    return tuple(selected[sample_id] for sample_id in sorted(selected))


__all__ = [
    "PaperNativeLatencyProtocol",
    "PaperNativeLatencyProtocolError",
    "PersistentTimingProtocol",
    "canonical_inference_traces",
    "load_paper_native_latency_protocol",
    "paired_method_order",
]
