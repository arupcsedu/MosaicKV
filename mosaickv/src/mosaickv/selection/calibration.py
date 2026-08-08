"""Label-free fidelity calibration and immutable profile selection."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class CalibrationStabilityError(ValueError):
    """No candidate met the frozen bootstrap selection-frequency threshold."""

    def __init__(self, *, best_frequency: float, required_frequency: float) -> None:
        self.best_frequency = best_frequency
        self.required_frequency = required_frequency
        super().__init__(
            "no calibration variant satisfies the bootstrap stability gate: "
            f"best_frequency={best_frequency:.6f}, required={required_frequency:.6f}"
        )


@dataclass(frozen=True, slots=True)
class FidelityObservation:
    """One paired compressed-vs-FullKV diagnostic, never a task label or score."""

    variant: str
    sample_id: str
    active_kv_bytes: int
    token_agreement: float
    logit_kl: float
    attention_output_error: float
    forecast_regret: float

    def __post_init__(self) -> None:
        if not self.variant.strip() or not self.sample_id.strip():
            raise ValueError("calibration variant and sample_id must be non-empty")
        if self.active_kv_bytes < 1:
            raise ValueError("calibration active_kv_bytes must be positive")
        if not 0 <= self.token_agreement <= 1:
            raise ValueError("calibration token_agreement must lie in [0, 1]")
        for name in ("logit_kl", "attention_output_error", "forecast_regret"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"calibration {name} must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class FidelityCalibrationProfile:
    """Frozen label-free choice and its disjoint-split provenance."""

    selected_variant: str
    parameters: dict[str, Any]
    calibration_dataset_id: str
    calibration_dataset_revision: str
    calibration_split: str
    calibration_sample_ids: tuple[str, ...]
    evaluation_sample_ids: tuple[str, ...]
    target_active_kv_bytes: int
    objective_value: float
    mean_token_agreement: float
    mean_logit_kl: float
    mean_attention_output_error: float
    mean_forecast_regret: float
    mean_active_kv_bytes: float
    tail_fidelity_loss: float = 0.0
    bootstrap_selection_frequency: float = 1.0
    bootstrap_resamples: int = 0
    bootstrap_seed: int = 0
    tail_weight: float = 0.0
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("fidelity profile schema_version must equal 2")
        if not self.selected_variant.strip():
            raise ValueError("selected calibration variant must be non-empty")
        if self.target_active_kv_bytes < 1:
            raise ValueError("target_active_kv_bytes must be positive")
        if set(self.calibration_sample_ids) & set(self.evaluation_sample_ids):
            raise ValueError("calibration and evaluation sample IDs must be disjoint")
        if not self.calibration_sample_ids or not self.evaluation_sample_ids:
            raise ValueError("calibration and evaluation ledgers must both be non-empty")
        if not 0 <= self.bootstrap_selection_frequency <= 1:
            raise ValueError("bootstrap selection frequency must lie in [0, 1]")
        if self.bootstrap_resamples < 0 or self.bootstrap_seed < 0:
            raise ValueError("bootstrap calibration provenance must be nonnegative")
        if self.tail_weight < 0 or self.tail_fidelity_loss < 0:
            raise ValueError("tail calibration controls must be nonnegative")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_record(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = {**self.to_record(), "profile_sha256": self.sha256}
        destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def select_fidelity_profile(
    observations: tuple[FidelityObservation, ...],
    *,
    parameters_by_variant: dict[str, dict[str, Any]],
    calibration_dataset_id: str,
    calibration_dataset_revision: str,
    calibration_split: str,
    evaluation_sample_ids: tuple[str, ...],
    target_active_kv_bytes: int,
    tail_weight: float = 0.25,
    bootstrap_resamples: int = 200,
    bootstrap_seed: int = 2027,
    minimum_selection_frequency: float = 0.0,
    minimum_calibration_samples: int = 2,
) -> FidelityCalibrationProfile:
    """Choose the lowest FullKV-fidelity loss under a measured byte ceiling.

    The objective is fixed and label-free: ``1-token agreement + log1p(KL) +
    attention-output error + forecast regret``. Every candidate must cover the
    identical calibration ledger; ties prefer fewer measured bytes and then a
    lexicographically smaller variant name.
    """

    if not observations:
        raise ValueError("fidelity calibration requires observations")
    if target_active_kv_bytes < 1:
        raise ValueError("target_active_kv_bytes must be positive")
    if tail_weight < 0:
        raise ValueError("tail_weight must be nonnegative")
    if bootstrap_resamples < 0 or bootstrap_seed < 0:
        raise ValueError("bootstrap controls must be nonnegative")
    if not 0 <= minimum_selection_frequency <= 1:
        raise ValueError("minimum_selection_frequency must lie in [0, 1]")
    if minimum_calibration_samples < 2:
        raise ValueError("minimum_calibration_samples must be >= 2")
    by_variant: defaultdict[str, list[FidelityObservation]] = defaultdict(list)
    for observation in observations:
        by_variant[observation.variant].append(observation)
    reference_variant = min(by_variant)
    expected_ids = tuple(
        sorted(item.sample_id for item in observations if item.variant == reference_variant)
    )
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("calibration observations cannot duplicate variant/sample pairs")
    if len(expected_ids) < minimum_calibration_samples:
        raise ValueError(
            "fidelity calibration ledger is too small for the declared stability gate"
        )
    if set(expected_ids) & set(evaluation_sample_ids):
        raise ValueError("calibration and evaluation sample IDs must be disjoint")

    rows_by_variant: dict[str, tuple[FidelityObservation, ...]] = {}
    candidates: list[tuple[float, float, str, dict[str, float]]] = []
    for variant, rows in sorted(by_variant.items()):
        sample_ids = tuple(sorted(row.sample_id for row in rows))
        if sample_ids != expected_ids:
            raise ValueError("every calibration variant must cover the identical sample ledger")
        if variant not in parameters_by_variant:
            raise ValueError(f"missing immutable parameters for calibration variant {variant!r}")
        count = len(rows)
        ordered_rows = tuple(sorted(rows, key=lambda row: row.sample_id))
        rows_by_variant[variant] = ordered_rows
        means = {
            "token": sum(row.token_agreement for row in rows) / count,
            "kl": sum(row.logit_kl for row in rows) / count,
            "attention": sum(row.attention_output_error for row in rows) / count,
            "regret": sum(row.forecast_regret for row in rows) / count,
            "bytes": sum(row.active_kv_bytes for row in rows) / count,
        }
        if means["bytes"] > target_active_kv_bytes:
            continue
        losses = tuple(
            1.0
            - row.token_agreement
            + math.log1p(row.logit_kl)
            + row.attention_output_error
            + row.forecast_regret
            for row in ordered_rows
        )
        tail = _quantile(losses, 0.9)
        objective = statistics.fmean(losses) + tail_weight * tail
        means["tail"] = tail
        candidates.append((objective, means["bytes"], variant, means))
    if not candidates:
        raise ValueError("no calibration variant satisfies the measured active-byte ceiling")
    eligible_variants = {candidate[2] for candidate in candidates}
    selection_counts = {variant: 0 for variant in eligible_variants}
    if bootstrap_resamples:
        generator = random.Random(bootstrap_seed)
        sample_count = len(expected_ids)
        for _replicate in range(bootstrap_resamples):
            indices = tuple(generator.randrange(sample_count) for _ in range(sample_count))
            replicate_candidates: list[tuple[float, float, str]] = []
            for _candidate_objective, _bytes, candidate_variant, _means in candidates:
                bootstrap_rows = rows_by_variant[candidate_variant]
                measured_bytes = statistics.fmean(
                    bootstrap_rows[index].active_kv_bytes for index in indices
                )
                # Candidate byte eligibility is established once on the full
                # calibration ledger above. Reapplying the ceiling to a
                # resampled mean can reject every method simultaneously when
                # all methods share identical sample-dependent cache sizes.
                # Those empty replicates made selection frequencies sum below
                # one and measured byte noise rather than selector stability.
                losses = tuple(
                    1.0
                    - bootstrap_rows[index].token_agreement
                    + math.log1p(bootstrap_rows[index].logit_kl)
                    + bootstrap_rows[index].attention_output_error
                    + bootstrap_rows[index].forecast_regret
                    for index in indices
                )
                robust_loss = statistics.fmean(losses) + tail_weight * _quantile(losses, 0.9)
                replicate_candidates.append((robust_loss, measured_bytes, candidate_variant))
            if replicate_candidates:
                selection_counts[min(replicate_candidates)[2]] += 1
    frequencies = {
        variant: (
            selection_counts[variant] / bootstrap_resamples
            if bootstrap_resamples
            else 1.0
        )
        for variant in eligible_variants
    }
    stable_candidates = tuple(
        candidate
        for candidate in candidates
        if frequencies[candidate[2]] >= minimum_selection_frequency
    )
    if not stable_candidates:
        best_frequency = max(frequencies.values(), default=0.0)
        raise CalibrationStabilityError(
            best_frequency=best_frequency,
            required_frequency=minimum_selection_frequency,
        )
    objective, _bytes, variant, means = min(stable_candidates)
    return FidelityCalibrationProfile(
        selected_variant=variant,
        parameters=parameters_by_variant[variant],
        calibration_dataset_id=calibration_dataset_id,
        calibration_dataset_revision=calibration_dataset_revision,
        calibration_split=calibration_split,
        calibration_sample_ids=expected_ids,
        evaluation_sample_ids=tuple(sorted(evaluation_sample_ids)),
        target_active_kv_bytes=target_active_kv_bytes,
        objective_value=objective,
        mean_token_agreement=means["token"],
        mean_logit_kl=means["kl"],
        mean_attention_output_error=means["attention"],
        mean_forecast_regret=means["regret"],
        mean_active_kv_bytes=means["bytes"],
        tail_fidelity_loss=means["tail"],
        bootstrap_selection_frequency=frequencies[variant],
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        tail_weight=tail_weight,
    )


def _quantile(values: tuple[float, ...], probability: float) -> float:
    if not values:
        raise ValueError("a calibration quantile requires values")
    if not 0 <= probability <= 1:
        raise ValueError("quantile probability must lie in [0, 1]")
    ordered = tuple(sorted(float(value) for value in values))
    offset = probability * (len(ordered) - 1)
    lower = math.floor(offset)
    upper = math.ceil(offset)
    if lower == upper:
        return ordered[lower]
    fraction = offset - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@dataclass(frozen=True, slots=True)
class HeadFidelityObservation:
    """Label-free exact-vs-prototype attention error for one calibration sample/head."""

    sample_id: str
    layer: int
    kv_head: int
    exact_error: float
    prototype_error: float

    def __post_init__(self) -> None:
        if not self.sample_id.strip() or min(self.layer, self.kv_head) < 0:
            raise ValueError("head calibration identity is invalid")
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.exact_error, self.prototype_error)
        ):
            raise ValueError("head calibration errors must be finite and nonnegative")

    @property
    def prototype_damage(self) -> float:
        return self.prototype_error - self.exact_error


@dataclass(frozen=True, slots=True)
class AdaptiveHeadRouting:
    """Frozen per-head exact/small-cluster routing derived without task labels."""

    exact_head_routes: tuple[str, ...]
    head_group_sizes: tuple[str, ...]
    head_position_spans: tuple[str, ...]
    mean_damage_by_head: tuple[tuple[int, int, float], ...]
    calibration_sample_ids: tuple[str, ...]


def derive_adaptive_head_routing(
    observations: tuple[HeadFidelityObservation, ...],
    *,
    default_group_size: int,
    sensitive_group_size: int,
    sensitive_position_span: int,
    damage_threshold: float,
    exact_head_fraction: float = 0.02,
) -> AdaptiveHeadRouting:
    """Freeze bounded per-head routing from paired label-free fidelity diagnostics."""

    if not observations:
        raise ValueError("adaptive head routing requires observations")
    if not 1 <= sensitive_group_size <= default_group_size:
        raise ValueError("sensitive group size must lie in [1, default_group_size]")
    if sensitive_position_span < 0 or damage_threshold < 0:
        raise ValueError("head routing thresholds must be nonnegative")
    if not 0 <= exact_head_fraction <= 1:
        raise ValueError("exact_head_fraction must lie in [0, 1]")
    grouped: defaultdict[tuple[int, int], list[HeadFidelityObservation]] = defaultdict(list)
    sample_ids = tuple(sorted({row.sample_id for row in observations}))
    for row in observations:
        grouped[(row.layer, row.kv_head)].append(row)
    if any(tuple(sorted(row.sample_id for row in rows)) != sample_ids for rows in grouped.values()):
        raise ValueError("every routed head must cover the identical calibration ledger")
    damage = tuple(
        (layer, head, statistics.fmean(row.prototype_damage for row in rows))
        for (layer, head), rows in sorted(grouped.items())
    )
    sensitive = tuple(item for item in damage if item[2] > damage_threshold)
    exact_count = min(
        len(sensitive),
        math.floor(len(damage) * exact_head_fraction),
    )
    exact_identities = {
        (layer, head)
        for layer, head, _value in sorted(
            sensitive, key=lambda item: (-item[2], item[0], item[1])
        )[:exact_count]
    }
    group_routes = tuple(
        f"{layer}:{head}:{sensitive_group_size}"
        for layer, head, _value in sensitive
        if (layer, head) not in exact_identities
    )
    span_routes = tuple(
        f"{layer}:{head}:{sensitive_position_span}"
        for layer, head, _value in sensitive
        if (layer, head) not in exact_identities
    )
    return AdaptiveHeadRouting(
        exact_head_routes=tuple(f"{layer}:{head}" for layer, head in sorted(exact_identities)),
        head_group_sizes=group_routes,
        head_position_spans=span_routes,
        mean_damage_by_head=damage,
        calibration_sample_ids=sample_ids,
    )


__all__ = [
    "AdaptiveHeadRouting",
    "CalibrationStabilityError",
    "FidelityCalibrationProfile",
    "FidelityObservation",
    "HeadFidelityObservation",
    "derive_adaptive_head_routing",
    "select_fidelity_profile",
]
