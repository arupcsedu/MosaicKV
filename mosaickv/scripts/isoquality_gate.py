"""Fail-closed gates for lower-memory-at-equal-quality comparisons.

The functions in this module do not score benchmark outputs.  Metric-specific
callers must first compute a paired confidence interval with the registered
official reducer.  This module then combines that interval with *measured*
per-sample initial, final, and peak live-cache storage.

Candidate-cap selection is deliberately label blind: for each baseline point,
choose the largest registered candidate cap whose measured storage is at least
``minimum_reduction`` smaller at every boundary for every paired sample.  Only
after that choice is frozen may the quality interval be inspected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from mosaickv.evaluation.metrics.statistics import PairedMetricComparison

STORAGE_FIELDS = tuple(
    f"{boundary}_active_kv_storage_bytes" for boundary in ("initial", "final", "peak")
)


class QualityDirection(StrEnum):
    """Direction used to normalize a candidate-minus-baseline interval."""

    HIGHER = "higher"
    LOWER = "lower"


@dataclass(frozen=True, slots=True)
class StorageReduction:
    """Measured reduction for one cache boundary."""

    boundary: str
    baseline_mean_bytes: float
    candidate_mean_bytes: float
    aggregate_reduction: float
    worst_sample_reduction: float


@dataclass(frozen=True, slots=True)
class IsoqualityGate:
    """Joint memory and paired-quality decision for one registered pair."""

    baseline_method: str
    baseline_cap_mib: float
    candidate_method: str
    candidate_cap_mib: float
    metric_name: str
    quality_direction: str
    quality_margin: float
    minimum_memory_reduction: float
    sample_count: int
    reductions: tuple[StorageReduction, ...]
    quality_ci_low: float
    quality_ci_high: float
    memory_pass: bool
    quality_pass: bool
    joint_pass: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reductions"] = [asdict(item) for item in self.reductions]
        return payload


def _by_sample(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in indexed:
            raise ValueError("storage rows require unique nonempty sample IDs")
        indexed[sample_id] = row
    if len(indexed) < 2:
        raise ValueError("iso-quality comparison requires at least two paired samples")
    return indexed


def storage_reductions(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> tuple[StorageReduction, ...]:
    """Compute aggregate and worst-paired-sample physical storage reductions."""

    baseline = _by_sample(baseline_rows)
    candidate = _by_sample(candidate_rows)
    if baseline.keys() != candidate.keys():
        raise ValueError("storage comparisons require identical sample IDs")
    reductions: list[StorageReduction] = []
    for field in STORAGE_FIELDS:
        baseline_values: list[int] = []
        candidate_values: list[int] = []
        paired: list[float] = []
        for sample_id in sorted(baseline):
            baseline_value = baseline[sample_id].get(field)
            candidate_value = candidate[sample_id].get(field)
            if (
                type(baseline_value) is not int
                or type(candidate_value) is not int
                or baseline_value <= 0
                or candidate_value <= 0
            ):
                raise ValueError(f"missing/nonpositive measured storage: {field}")
            baseline_values.append(baseline_value)
            candidate_values.append(candidate_value)
            paired.append(1.0 - candidate_value / baseline_value)
        baseline_mean = sum(baseline_values) / len(baseline_values)
        candidate_mean = sum(candidate_values) / len(candidate_values)
        reductions.append(
            StorageReduction(
                boundary=field.removesuffix("_active_kv_storage_bytes"),
                baseline_mean_bytes=baseline_mean,
                candidate_mean_bytes=candidate_mean,
                aggregate_reduction=1.0 - candidate_mean / baseline_mean,
                worst_sample_reduction=min(paired),
            )
        )
    return tuple(reductions)


def memory_gate(
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    *,
    minimum_reduction: float = 0.30,
) -> tuple[bool, tuple[StorageReduction, ...]]:
    """Require the threshold at every boundary and on every paired sample."""

    if not 0 < minimum_reduction < 1:
        raise ValueError("minimum_reduction must lie in (0, 1)")
    reductions = storage_reductions(baseline_rows, candidate_rows)
    passed = all(
        item.aggregate_reduction >= minimum_reduction
        and item.worst_sample_reduction >= minimum_reduction
        for item in reductions
    )
    return passed, reductions


def quality_gate(
    comparison: PairedMetricComparison,
    *,
    direction: QualityDirection,
    margin: float = 0.0,
) -> bool:
    """Apply a paired non-inferiority gate without changing interval semantics.

    ``comparison`` is always candidate minus baseline.  For higher-is-better
    metrics, the lower confidence bound may not cross ``-margin``.  For
    lower-is-better metrics such as PPL, the upper bound may not cross
    ``margin``.
    """

    if margin < 0:
        raise ValueError("quality margin must be nonnegative")
    if direction is QualityDirection.HIGHER:
        return comparison.ci_low >= -margin
    if direction is QualityDirection.LOWER:
        return comparison.ci_high <= margin
    raise ValueError(f"unsupported quality direction: {direction}")


def evaluate_isoquality_pair(
    *,
    baseline_method: str,
    baseline_cap_mib: float,
    baseline_rows: Sequence[Mapping[str, Any]],
    candidate_method: str,
    candidate_cap_mib: float,
    candidate_rows: Sequence[Mapping[str, Any]],
    comparison: PairedMetricComparison,
    direction: QualityDirection,
    quality_margin: float = 0.0,
    minimum_memory_reduction: float = 0.30,
) -> IsoqualityGate:
    """Combine an official paired metric interval with physical storage gates."""

    if not baseline_method or not candidate_method or candidate_method == baseline_method:
        raise ValueError("candidate and baseline methods must be distinct")
    if candidate_cap_mib >= baseline_cap_mib:
        raise ValueError("candidate cap must be strictly below the baseline cap")
    if comparison.sample_count != len(_by_sample(baseline_rows)):
        raise ValueError("metric and storage denominators differ")
    memory_pass, reductions = memory_gate(
        baseline_rows,
        candidate_rows,
        minimum_reduction=minimum_memory_reduction,
    )
    quality_pass = quality_gate(comparison, direction=direction, margin=quality_margin)
    return IsoqualityGate(
        baseline_method=baseline_method,
        baseline_cap_mib=baseline_cap_mib,
        candidate_method=candidate_method,
        candidate_cap_mib=candidate_cap_mib,
        metric_name=comparison.metric_name,
        quality_direction=direction.value,
        quality_margin=quality_margin,
        minimum_memory_reduction=minimum_memory_reduction,
        sample_count=comparison.sample_count,
        reductions=reductions,
        quality_ci_low=comparison.ci_low,
        quality_ci_high=comparison.ci_high,
        memory_pass=memory_pass,
        quality_pass=quality_pass,
        joint_pass=memory_pass and quality_pass,
    )


def choose_candidate_cap_without_quality(
    *,
    baseline_rows: Sequence[Mapping[str, Any]],
    candidates: Mapping[float, Sequence[Mapping[str, Any]]],
    baseline_cap_mib: float,
    minimum_memory_reduction: float = 0.30,
) -> float | None:
    """Choose the largest physically eligible cap without reading quality.

    This deterministic rule maximizes candidate capacity (and therefore gives
    the quality test its most conservative chance) while retaining the exact
    registered memory claim.  Returning ``None`` records physical infeasibility
    rather than manufacturing a comparison.
    """

    eligible = []
    for cap, rows in candidates.items():
        if cap >= baseline_cap_mib:
            continue
        passed, _ = memory_gate(
            baseline_rows,
            rows,
            minimum_reduction=minimum_memory_reduction,
        )
        if passed:
            eligible.append(float(cap))
    return max(eligible) if eligible else None


def validate_isoquality_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the preregistration without authorizing or materializing it."""

    if protocol.get("schema_version") != 1:
        raise ValueError("unrecognized iso-quality protocol schema")
    claim = protocol.get("claim", {})
    if (
        claim.get("minimum_reduction") != 0.30
        or claim.get("quality_noninferiority_margin") != 0.0
        or claim.get("required_boundaries") != ["initial", "final", "peak"]
        or claim.get("candidate_point_selection")
        != "largest_measured_mosaickv_cap_satisfying_all_memory_boundaries_before_quality_is_read"
        or not claim.get("no_interpolation")
        or not claim.get("no_nominal_retention_comparison")
    ):
        raise ValueError("memory/quality claim gate was weakened")
    resources = protocol.get("resources", {})
    if resources != {
        "max_global_a100s": 4,
        "gpus_per_worker": 1,
        "cpus_per_worker": 12,
        "time_limit_seconds": 7200,
        "shard_size": 16,
        "immutable_write_once_outputs": True,
        "resume_only_missing_or_failed_samples": True,
    }:
        raise ValueError("unexpected execution resources")
    lane_summary: dict[str, Any] = {}
    for lane_name, lane in protocol.get("lanes", {}).items():
        if lane_name == "prefixkv_native":
            if (
                lane.get("status")
                != "wait_for_active_prefixkv_campaign_then_plan_storage_only_pairs"
            ):
                raise ValueError("PrefixKV lane must not alter the active campaign")
            continue
        methods = lane.get("methods", [])
        baselines = lane.get("named_baselines", [])
        caps = lane.get("compressed_caps_mib", [])
        if (
            len(methods) != len(set(methods))
            or not {"full_kv", "prompt_attention_topk", "mosaickv_exact"}.issubset(methods)
            or not baselines
            or any(method not in methods or method == "mosaickv_exact" for method in baselines)
            or len(caps) != len(set(caps))
            or caps != sorted(caps)
            or any(not isinstance(cap, int) or cap <= 0 for cap in caps)
        ):
            raise ValueError(f"invalid method/cap registration: {lane_name}")
        for pair in lane.get("primary_registered_pairs", []):
            baseline_cap = pair.get("baseline_cap_mib")
            candidate_cap = pair.get("candidate_cap_mib")
            if (
                baseline_cap not in caps
                or candidate_cap not in caps
                or candidate_cap >= baseline_cap
            ):
                raise ValueError(f"unregistered primary pair: {lane_name}")
        directions = {metric.get("direction") for metric in lane.get("metrics", [])}
        if not directions or not directions.issubset({"higher", "lower"}):
            raise ValueError(f"metric direction unresolved: {lane_name}")
        sample_count = lane["sample_count"]
        validation_sample_count = lane.get("validation_sample_count")
        shard_size = resources["shard_size"]
        if not isinstance(sample_count, int) or sample_count < 2 or sample_count % shard_size:
            raise ValueError(f"sample count must form complete shards: {lane_name}")
        if (
            not isinstance(validation_sample_count, int)
            or validation_sample_count < shard_size
            or validation_sample_count % shard_size
        ):
            raise ValueError(f"validation sample count must form complete shards: {lane_name}")
        compressed_rows = sample_count * (len(methods) - 1) * len(caps)
        fullkv_rows = sample_count
        lane_summary[lane_name] = {
            "sample_count": sample_count,
            "validation_sample_count": validation_sample_count,
            "compressed_methods": len(methods) - 1,
            "caps": len(caps),
            "quality_metrics": len(lane["metrics"]),
            "planned_fullkv_rows": fullkv_rows,
            "planned_compressed_method_cap_rows": compressed_rows,
            "planned_total_rows": fullkv_rows + compressed_rows,
            "planned_worker_jobs": (sample_count // shard_size)
            * (1 + (len(methods) - 1) * len(caps)),
            "planned_validation_worker_jobs": (validation_sample_count // shard_size)
            * (1 + (len(methods) - 1) * len(caps)),
        }
    if set(lane_summary) != {"vl_cache_coco", "lookm_milebench"}:
        raise ValueError("required executable baseline lanes are missing")
    return {
        "status": "registered_execution_disabled"
        if not protocol.get("execution_authorized")
        else "registered_execution_authorized",
        "source_sealed": isinstance(protocol.get("source_sha"), str),
        "lanes": lane_summary,
    }


__all__ = [
    "STORAGE_FIELDS",
    "IsoqualityGate",
    "QualityDirection",
    "StorageReduction",
    "choose_candidate_cap_without_quality",
    "evaluate_isoquality_pair",
    "memory_gate",
    "quality_gate",
    "storage_reductions",
    "validate_isoquality_protocol",
]
