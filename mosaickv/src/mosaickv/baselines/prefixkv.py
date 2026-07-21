"""Paper-faithful local PrefixKV reimplementation.

This is not official PrefixKV code.  The implementation follows Wang et al.
(NeurIPS 2025) and keeps the official source pinned under
``third_party/PrefixKV`` for independent parity runs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from mosaickv.cache_state import FullKVState
from mosaickv.config import CacheConfig, PrefixKVConfig
from mosaickv.types import BudgetUnit, JsonObject, JsonValue, PrefixKVProfileMode


class PrefixKVReimplementationError(RuntimeError):
    """Raised when a PrefixKV run cannot preserve the paper's assumptions."""


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_sha(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


def _finite_ratios(values: tuple[float, ...], name: str) -> None:
    if not values:
        raise PrefixKVReimplementationError(f"{name} cannot be empty")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise PrefixKVReimplementationError(f"{name} must contain finite values in [0, 1]")


@dataclass(frozen=True, slots=True)
class PrefixKVCalibrationObservation:
    """Prompt-side, head-averaged attention importance for one calibration sample."""

    sample_id: str
    layer_scores: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise PrefixKVReimplementationError("calibration sample_id must be non-empty")
        if not self.layer_scores:
            raise PrefixKVReimplementationError("calibration observation has no layers")
        if any(not scores for scores in self.layer_scores):
            raise PrefixKVReimplementationError("calibration layer scores cannot be empty")
        if any(
            not math.isfinite(score) or score < 0
            for scores in self.layer_scores
            for score in scores
        ):
            raise PrefixKVReimplementationError(
                "calibration attention scores must be finite and nonnegative"
            )


@dataclass(frozen=True, slots=True)
class PrefixKVOfflineProfile:
    """Immutable adaptive layer recipe estimated outside the evaluation set."""

    model_id: str
    model_revision: str
    target_retention_ratio: float
    layer_forget_ratios: tuple[float, ...]
    calibration_dataset_id: str
    calibration_dataset_revision: str
    calibration_split: str
    calibration_sample_ids: tuple[str, ...]
    calibration_seed: int
    start_size: int
    protect_size: int
    official_repository_sha: str = "597f1ab032704951550f93bcc8a23f1454b80aa4"
    source_kind: str = "calibration_generated"
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("model_id", "model_revision", "source_kind"):
            if not str(getattr(self, name)).strip():
                raise PrefixKVReimplementationError(f"profile.{name} must be non-empty")
        if not 0 < self.target_retention_ratio <= 1:
            raise PrefixKVReimplementationError("profile.target_retention_ratio must be in (0, 1]")
        _finite_ratios(self.layer_forget_ratios, "profile.layer_forget_ratios")
        if self.start_size < 0 or self.protect_size < 1:
            raise PrefixKVReimplementationError(
                "profile start_size must be >= 0 and protect_size must be >= 1"
            )
        if self.calibration_seed < 0:
            raise PrefixKVReimplementationError("profile.calibration_seed must be nonnegative")
        if not _valid_sha(self.official_repository_sha, 40):
            raise PrefixKVReimplementationError(
                "profile.official_repository_sha must be a lowercase git SHA"
            )
        if self.schema_version != 1:
            raise PrefixKVReimplementationError("profile.schema_version must equal 1")
        if self.source_kind == "calibration_generated":
            for name in (
                "calibration_dataset_id",
                "calibration_dataset_revision",
                "calibration_split",
            ):
                if not str(getattr(self, name)).strip():
                    raise PrefixKVReimplementationError(f"profile.{name} must be non-empty")
            if not self.calibration_sample_ids:
                raise PrefixKVReimplementationError(
                    "a generated profile must record calibration sample IDs"
                )
        if self.calibration_sample_ids != tuple(sorted(set(self.calibration_sample_ids))):
            raise PrefixKVReimplementationError(
                "profile calibration sample IDs must be sorted and unique"
            )

    @property
    def profile_sha256(self) -> str:
        """Digest of the canonical profile payload."""

        return _sha256_json(self.to_json_object(include_digest=False))

    @property
    def layer_retention_ratios(self) -> tuple[float, ...]:
        return tuple(1.0 - value for value in self.layer_forget_ratios)

    @property
    def calibration_sample_ids_sha256(self) -> str:
        return _sha256_json(list(self.calibration_sample_ids))

    def assert_evaluation_disjoint(self, evaluation_sample_ids: tuple[str, ...]) -> None:
        """Reject calibration/evaluation overlap before an evaluation row is emitted."""

        overlap = set(self.calibration_sample_ids).intersection(evaluation_sample_ids)
        if overlap:
            examples = ", ".join(sorted(overlap)[:5])
            raise PrefixKVReimplementationError(
                "PrefixKV calibration and evaluation samples overlap: " + examples
            )

    def to_json_object(self, *, include_digest: bool = True) -> JsonObject:
        payload: JsonObject = {
            "schema_version": self.schema_version,
            "implementation": "prefixkv_reimpl_profile",
            "source_kind": self.source_kind,
            "official_repository_sha": self.official_repository_sha,
            "model": {"id": self.model_id, "revision": self.model_revision},
            "target_retention_ratio": self.target_retention_ratio,
            "official_forget_ratio": 1.0 - self.target_retention_ratio,
            "layer_forget_ratios": list(self.layer_forget_ratios),
            "layer_retention_ratios": list(self.layer_retention_ratios),
            "calibration": {
                "dataset_id": self.calibration_dataset_id,
                "dataset_revision": self.calibration_dataset_revision,
                "split": self.calibration_split,
                "sample_ids": list(self.calibration_sample_ids),
                "sample_ids_sha256": self.calibration_sample_ids_sha256,
                "seed": self.calibration_seed,
            },
            "search": {
                "algorithm": "global_cumulative_priority_binary_search",
                "sample_aggregation": "mean_layer_forget_ratio",
                "start_size": self.start_size,
                "protect_size": self.protect_size,
            },
        }
        if include_digest:
            payload["profile_sha256"] = self.profile_sha256
        return payload

    def write(self, path: str | Path) -> Path:
        """Atomically write a profile without environment-token interpolation."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(self.to_json_object(), indent=2, sort_keys=True) + "\n"
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        except Exception:
            with suppress(FileNotFoundError):
                os.unlink(temporary)
            raise
        return destination

    @classmethod
    def from_json_object(cls, payload: JsonObject) -> PrefixKVOfflineProfile:
        expected = {
            "schema_version",
            "implementation",
            "source_kind",
            "official_repository_sha",
            "model",
            "target_retention_ratio",
            "official_forget_ratio",
            "layer_forget_ratios",
            "layer_retention_ratios",
            "calibration",
            "search",
            "profile_sha256",
        }
        unknown = set(payload) - expected
        if unknown:
            raise PrefixKVReimplementationError(
                "profile contains unknown fields: " + ", ".join(sorted(unknown))
            )
        if payload.get("implementation") != "prefixkv_reimpl_profile":
            raise PrefixKVReimplementationError(
                "profile implementation must be 'prefixkv_reimpl_profile'"
            )
        model = cast("dict[str, object]", payload.get("model"))
        calibration = cast("dict[str, object]", payload.get("calibration"))
        search = cast("dict[str, object]", payload.get("search"))
        profile = cls(
            model_id=str(model["id"]),
            model_revision=str(model["revision"]),
            target_retention_ratio=float(cast("float", payload["target_retention_ratio"])),
            layer_forget_ratios=tuple(
                float(value) for value in cast("list[float | int]", payload["layer_forget_ratios"])
            ),
            calibration_dataset_id=str(calibration["dataset_id"]),
            calibration_dataset_revision=str(calibration["dataset_revision"]),
            calibration_split=str(calibration["split"]),
            calibration_sample_ids=tuple(
                sorted(str(value) for value in cast("list[JsonValue]", calibration["sample_ids"]))
            ),
            calibration_seed=int(cast("int", calibration["seed"])),
            start_size=int(cast("int", search["start_size"])),
            protect_size=int(cast("int", search["protect_size"])),
            official_repository_sha=str(payload["official_repository_sha"]),
            source_kind=str(payload["source_kind"]),
            schema_version=int(cast("int", payload["schema_version"])),
        )
        declared = payload.get("profile_sha256")
        if declared is not None and declared != profile.profile_sha256:
            raise PrefixKVReimplementationError("profile_sha256 does not match profile content")
        declared_retention = tuple(
            float(value) for value in cast("list[float | int]", payload["layer_retention_ratios"])
        )
        if any(
            not math.isclose(actual, expected_value, rel_tol=0, abs_tol=1e-12)
            for actual, expected_value in zip(
                declared_retention, profile.layer_retention_ratios, strict=True
            )
        ):
            raise PrefixKVReimplementationError(
                "profile layer retention and forget ratios are inconsistent"
            )
        official_forget = float(cast("float", payload["official_forget_ratio"]))
        if not math.isclose(
            official_forget, 1 - profile.target_retention_ratio, rel_tol=0, abs_tol=1e-12
        ):
            raise PrefixKVReimplementationError(
                "profile official_forget_ratio is inconsistent with target retention"
            )
        return profile


def load_prefixkv_profile(
    path: str | Path,
    *,
    model_id: str,
    model_revision: str,
    target_retention_ratio: float,
    start_size: int,
    protect_size: int,
) -> PrefixKVOfflineProfile:
    """Load a native profile or an official raw per-layer forget-ratio JSON file."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PrefixKVReimplementationError(
            f"cannot load PrefixKV profile {source}: {error}"
        ) from error
    if isinstance(payload, list):
        profile = PrefixKVOfflineProfile(
            model_id=model_id,
            model_revision=model_revision,
            target_retention_ratio=target_retention_ratio,
            layer_forget_ratios=tuple(float(value) for value in payload),
            calibration_dataset_id="not_recorded_by_official_artifact",
            calibration_dataset_revision="not_recorded_by_official_artifact",
            calibration_split="not_recorded_by_official_artifact",
            calibration_sample_ids=(),
            calibration_seed=0,
            start_size=start_size,
            protect_size=protect_size,
            source_kind="official_repository_config_calibration_ids_unavailable",
        )
    elif isinstance(payload, dict):
        profile = PrefixKVOfflineProfile.from_json_object(cast("JsonObject", payload))
    else:
        raise PrefixKVReimplementationError("PrefixKV profile must be a JSON object or list")
    if profile.model_id != model_id or profile.model_revision != model_revision:
        raise PrefixKVReimplementationError(
            "PrefixKV profile model identity does not match the configured model/revision"
        )
    if not math.isclose(
        profile.target_retention_ratio, target_retention_ratio, rel_tol=0, abs_tol=5e-3
    ):
        raise PrefixKVReimplementationError(
            "PrefixKV profile target retention ratio does not match cache.retention_ratio"
        )
    if profile.start_size != start_size or profile.protect_size != protect_size:
        raise PrefixKVReimplementationError(
            "PrefixKV profile protected-boundary settings do not match the run configuration"
        )
    return profile


def prefixkv_attention_scores(attention_weights: tuple[Any, ...]) -> tuple[tuple[float, ...], ...]:
    """Compute the paper importance: mean heads, then sum over prompt queries."""

    scores: list[tuple[float, ...]] = []
    for layer_index, attention in enumerate(attention_weights):
        if hasattr(attention, "detach"):
            # Match the pinned source exactly: head averaging occurs in the
            # incoming attention dtype, then the result is promoted to FP32
            # before query reduction. Casting before the mean can reorder
            # nearly tied positions and changes the selected cache.
            detached = attention.detach()
            if int(detached.ndim) != 4 or int(detached.shape[0]) != 1:
                raise PrefixKVReimplementationError(
                    f"PrefixKV attention layer {layer_index} must have shape [1,H,Q,K]"
                )
            pooled = detached.mean(dim=1).float().sum(dim=-2)[0].cpu().numpy()
        else:
            array = np.asarray(attention)
            if array.ndim != 4 or array.shape[0] != 1:
                raise PrefixKVReimplementationError(
                    f"PrefixKV attention layer {layer_index} must have shape [1,H,Q,K]"
                )
            pooled = np.asarray(array.mean(axis=1), dtype=np.float32).sum(
                axis=-2, dtype=np.float32
            )[0]
        if not bool(np.all(np.isfinite(pooled))) or bool(np.any(pooled < 0)):
            raise PrefixKVReimplementationError(
                f"PrefixKV attention layer {layer_index} is non-finite or negative"
            )
        scores.append(tuple(float(value) for value in pooled.tolist()))
    if not scores:
        raise PrefixKVReimplementationError("PrefixKV requires eager prompt attentions")
    return tuple(scores)


def prefixkv_calibration_observation(
    sample_id: str, attention_weights: tuple[Any, ...]
) -> PrefixKVCalibrationObservation:
    """Create one serializable calibration row from an eager HF prefill."""

    return PrefixKVCalibrationObservation(
        sample_id=sample_id,
        layer_scores=prefixkv_attention_scores(attention_weights),
    )


def _bounded_apportion(
    desired: tuple[float, ...],
    lower: tuple[int, ...],
    upper: tuple[int, ...],
    target: int,
) -> tuple[int, ...]:
    if not (len(desired) == len(lower) == len(upper)):
        raise PrefixKVReimplementationError("layer-budget vectors have different lengths")
    if not sum(lower) <= target <= sum(upper):
        raise PrefixKVReimplementationError(
            f"target layer budget {target} is outside mandatory bounds [{sum(lower)}, {sum(upper)}]"
        )
    clipped = tuple(
        min(float(high), max(float(low), value))
        for value, low, high in zip(desired, lower, upper, strict=True)
    )
    if target <= sum(clipped):
        base = tuple(float(value) for value in lower)
        span = tuple(value - low for value, low in zip(clipped, lower, strict=True))
    else:
        base = clipped
        span = tuple(high - value for high, value in zip(upper, clipped, strict=True))
    span_sum = sum(span)
    fraction = 0.0 if span_sum == 0 else (target - sum(base)) / span_sum
    continuous = tuple(value + fraction * width for value, width in zip(base, span, strict=True))
    result = [math.floor(value + 1e-12) for value in continuous]
    remaining = target - sum(result)
    ranking = sorted(
        range(len(result)),
        key=lambda index: (-(continuous[index] - result[index]), index),
    )
    for index in ranking:
        if remaining == 0:
            break
        if result[index] < upper[index]:
            result[index] += 1
            remaining -= 1
    if remaining:
        for index in range(len(result)):
            while remaining and result[index] < upper[index]:
                result[index] += 1
                remaining -= 1
    if remaining or sum(result) != target:
        raise PrefixKVReimplementationError("could not apportion the exact global layer budget")
    return tuple(result)


def search_prefixkv_layer_sizes(
    layer_scores: tuple[tuple[float, ...], ...],
    *,
    retention_ratio: float,
    start_size: int,
    protect_size: int,
) -> tuple[int, ...]:
    """Binary-search the shared cumulative-priority threshold from Equation (1)."""

    if not 0 < retention_ratio <= 1:
        raise PrefixKVReimplementationError("retention_ratio must be in (0, 1]")
    if not layer_scores:
        raise PrefixKVReimplementationError("PrefixKV search requires at least one layer")
    lengths = tuple(len(scores) for scores in layer_scores)
    lower = tuple(min(length, start_size + protect_size) for length in lengths)
    target = max(sum(lower), math.floor(sum(lengths) * retention_ratio + 1e-12))
    target = min(target, sum(lengths))
    cumulative: list[np.ndarray[Any, Any]] = []
    for scores in layer_scores:
        ranked = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
        total = float(ranked.sum())
        normalized = np.full_like(ranked, 1.0 / len(ranked)) if total <= 0 else ranked / total
        cumulative.append(np.cumsum(normalized))

    def sizes(threshold: float) -> tuple[int, ...]:
        return tuple(
            max(low, min(length, int(np.searchsorted(cdf, threshold, side="left")) + 1))
            for cdf, low, length in zip(cumulative, lower, lengths, strict=True)
        )

    left, right = 0.0, 1.0
    candidates: set[tuple[int, ...]] = {lower, lengths}
    for _ in range(64):
        middle = (left + right) / 2.0
        current = sizes(middle)
        candidates.add(current)
        if sum(current) < target:
            left = middle
        elif sum(current) > target:
            right = middle
        else:
            return current
    closest = min(candidates, key=lambda value: (abs(sum(value) - target), value))
    return _bounded_apportion(tuple(float(value) for value in closest), lower, lengths, target)


def generate_prefixkv_profile(
    observations: tuple[PrefixKVCalibrationObservation, ...],
    *,
    model_id: str,
    model_revision: str,
    dataset_id: str,
    dataset_revision: str,
    calibration_split: str,
    evaluation_sample_ids: tuple[str, ...],
    retention_ratio: float,
    seed: int,
    start_size: int = 1,
    protect_size: int = 1,
) -> PrefixKVOfflineProfile:
    """Estimate and aggregate per-sample global prefix configurations offline."""

    if not observations:
        raise PrefixKVReimplementationError("PrefixKV calibration observations cannot be empty")
    sample_ids = tuple(sorted(observation.sample_id for observation in observations))
    if len(sample_ids) != len(set(sample_ids)):
        raise PrefixKVReimplementationError("PrefixKV calibration sample IDs must be unique")
    overlap = set(sample_ids).intersection(evaluation_sample_ids)
    if overlap:
        raise PrefixKVReimplementationError(
            "PrefixKV calibration and evaluation samples overlap: " + ", ".join(sorted(overlap)[:5])
        )
    layer_count = len(observations[0].layer_scores)
    if any(len(observation.layer_scores) != layer_count for observation in observations):
        raise PrefixKVReimplementationError("calibration observations have different layer counts")
    sample_forget_ratios: list[tuple[float, ...]] = []
    for observation in observations:
        sizes = search_prefixkv_layer_sizes(
            observation.layer_scores,
            retention_ratio=retention_ratio,
            start_size=start_size,
            protect_size=protect_size,
        )
        sample_forget_ratios.append(
            tuple(
                (len(scores) - retained) / len(scores)
                for scores, retained in zip(observation.layer_scores, sizes, strict=True)
            )
        )
    means = tuple(
        sum(sample[layer] for sample in sample_forget_ratios) / len(sample_forget_ratios)
        for layer in range(layer_count)
    )
    return PrefixKVOfflineProfile(
        model_id=model_id,
        model_revision=model_revision,
        target_retention_ratio=retention_ratio,
        layer_forget_ratios=means,
        calibration_dataset_id=dataset_id,
        calibration_dataset_revision=dataset_revision,
        calibration_split=calibration_split,
        calibration_sample_ids=sample_ids,
        calibration_seed=seed,
        start_size=start_size,
        protect_size=protect_size,
    )


@dataclass(frozen=True, slots=True)
class PrefixKVLayerState:
    """Selection and fixed-distance decode schedule for one decoder layer."""

    layer: int
    source_positions: int
    retained_positions: int
    selected_physical_positions: tuple[int, ...]
    selected_logical_positions: tuple[int, ...]
    protected_positions: tuple[int, ...]
    attention_scores_sha256: str
    attention_score_sum: float
    bytes_per_position: int
    retained_bytes: int
    eviction_offset: int

    def __post_init__(self) -> None:
        if self.layer < 0 or self.source_positions < 1:
            raise PrefixKVReimplementationError("PrefixKV layer identity/length is invalid")
        if self.retained_positions != len(self.selected_physical_positions):
            raise PrefixKVReimplementationError("PrefixKV retained layer count is inconsistent")
        if self.selected_physical_positions != tuple(sorted(set(self.selected_physical_positions))):
            raise PrefixKVReimplementationError(
                "PrefixKV selected positions must be sorted and unique"
            )
        if not set(self.protected_positions) <= set(self.selected_physical_positions):
            raise PrefixKVReimplementationError("PrefixKV removed a protected boundary token")
        if not _valid_sha(self.attention_scores_sha256, 64):
            raise PrefixKVReimplementationError("PrefixKV attention score digest is invalid")
        if self.retained_bytes != self.retained_positions * self.bytes_per_position:
            raise PrefixKVReimplementationError("PrefixKV retained layer bytes are inconsistent")
        if not 0 <= self.eviction_offset < self.retained_positions:
            raise PrefixKVReimplementationError("PrefixKV eviction offset is outside the cache")

    @property
    def retention_ratio(self) -> float:
        return self.retained_positions / self.source_positions


@dataclass(frozen=True, slots=True)
class PrefixKVCompressionPlan:
    """Paper algorithm output consumed by MosaicKV's common cache packer."""

    full_state: FullKVState
    config: PrefixKVConfig
    profile: PrefixKVOfflineProfile | None
    layers: tuple[PrefixKVLayerState, ...]
    source_slots: int
    active_slots: int
    source_bytes: int
    retained_bytes: int
    target_retention_ratio: float
    implementation_label: str

    def __post_init__(self) -> None:
        if len(self.layers) != len(self.full_state.layers):
            raise PrefixKVReimplementationError("PrefixKV plan does not cover every layer")
        if tuple(layer.layer for layer in self.layers) != tuple(range(len(self.layers))):
            raise PrefixKVReimplementationError("PrefixKV layers are not in canonical order")
        expected_slots = sum(
            layer.retained_positions * self.full_state.layers[layer.layer].kv_heads
            for layer in self.layers
        )
        if self.active_slots != expected_slots or self.source_slots != len(self.full_state.blocks):
            raise PrefixKVReimplementationError("PrefixKV slot accounting is inconsistent")
        if self.retained_bytes != sum(layer.retained_bytes for layer in self.layers):
            raise PrefixKVReimplementationError("PrefixKV byte accounting is inconsistent")
        if self.source_bytes != self.full_state.active_bytes:
            raise PrefixKVReimplementationError("PrefixKV source byte accounting is inconsistent")
        if self.implementation_label not in {"prefixkv_reimpl", "generalized_prefixkv_reimpl"}:
            raise PrefixKVReimplementationError("PrefixKV implementation label is invalid")

    @property
    def layer_cache_sizes(self) -> tuple[int, ...]:
        return tuple(layer.retained_positions for layer in self.layers)

    def trace(self) -> JsonObject:
        return {
            "implementation": "prefixkv_reimpl",
            "implementation_label": self.implementation_label,
            "official_code": False,
            "paper": "Wang et al., NeurIPS 2025, Equation 1 and Algorithm 1",
            "official_repository_sha": self.config.official_repository_sha,
            "profile_mode": self.config.profile_mode.value,
            "profile_sha256": self.profile.profile_sha256 if self.profile is not None else None,
            "profile_source_kind": self.profile.source_kind if self.profile is not None else None,
            "calibration_sample_ids_sha256": (
                self.profile.calibration_sample_ids_sha256 if self.profile is not None else None
            ),
            "calibration_sample_count": (
                len(self.profile.calibration_sample_ids) if self.profile is not None else 0
            ),
            "target_retention_ratio": self.target_retention_ratio,
            "official_forget_ratio": 1.0 - self.target_retention_ratio,
            "realized_slot_retention_ratio": self.active_slots / self.source_slots,
            "source_slots": self.source_slots,
            "active_slots": self.active_slots,
            "source_bytes": self.source_bytes,
            "retained_bytes": self.retained_bytes,
            "start_size": self.config.start_size,
            "protect_size": self.config.protect_size,
            "eviction_distance": self.config.eviction_distance,
            "layers": [
                {
                    "layer": layer.layer,
                    "source_positions": layer.source_positions,
                    "retained_positions": layer.retained_positions,
                    "retention_ratio": layer.retention_ratio,
                    "selected_physical_positions": list(layer.selected_physical_positions),
                    "selected_logical_positions": list(layer.selected_logical_positions),
                    "protected_positions": list(layer.protected_positions),
                    "attention_scores_sha256": layer.attention_scores_sha256,
                    "attention_score_sum": layer.attention_score_sum,
                    "bytes_per_position": layer.bytes_per_position,
                    "retained_bytes": layer.retained_bytes,
                    "eviction_offset": layer.eviction_offset,
                }
                for layer in self.layers
            ],
        }


def _position_payload(
    tensor: Any,
    *,
    head_axis: int,
    head: int,
    sequence_axis: int,
    position: int,
) -> Any:
    index: list[Any] = [slice(None)] * int(tensor.ndim)
    index[head_axis] = slice(head, head + 1)
    index[sequence_axis] = slice(position, position + 1)
    result = tensor[tuple(index)]
    clone = getattr(result, "clone", None)
    return clone() if clone is not None else result.copy()


def _layer_bytes_per_position(full_state: FullKVState, layer_index: int) -> int:
    layer = full_state.layers[layer_index]
    if layer.byte_size % layer.sequence_length:
        raise PrefixKVReimplementationError("PrefixKV layer bytes are not token-aligned")
    return layer.byte_size // layer.sequence_length


def _select_positions(
    scores: tuple[float, ...], retained: int, start_size: int, protect_size: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    length = len(scores)
    protected = tuple(
        sorted(
            set(range(min(start_size, length))) | set(range(max(0, length - protect_size), length))
        )
    )
    remaining = retained - len(protected)
    if remaining < 0:
        raise PrefixKVReimplementationError("layer budget cannot retain protected PrefixKV tokens")
    candidates = tuple(position for position in range(length) if position not in set(protected))
    ranked = sorted(candidates, key=lambda position: (-scores[position], position))
    selected = tuple(sorted((*protected, *ranked[:remaining])))
    return selected, protected


def _eviction_offset(retained: int, distance: int) -> int:
    if retained < 1:
        raise PrefixKVReimplementationError("PrefixKV cannot schedule an empty cache")
    offset = distance if distance > 0 else retained + distance
    if offset < 1 and retained >= 3:
        offset = retained // 2
    return max(0, min(retained - 1, offset))


def build_prefixkv_reimpl_plan(
    full_state: FullKVState,
    attention_weights: tuple[Any, ...],
    prefixkv_config: PrefixKVConfig,
    cache_config: CacheConfig,
    *,
    model_id: str,
    model_revision: str,
    profile: PrefixKVOfflineProfile | None = None,
) -> PrefixKVCompressionPlan:
    """Select per-layer top-importance positions under one exact global budget."""

    if not prefixkv_config.enabled:
        raise PrefixKVReimplementationError("prefixkv.enabled must be true")
    if full_state.block_size != 1 or cache_config.block_size != 1:
        raise PrefixKVReimplementationError("PrefixKV requires token-sized cache blocks")
    if cache_config.budget_unit not in {BudgetUnit.BLOCKS, BudgetUnit.BYTES}:
        raise PrefixKVReimplementationError("PrefixKV supports block or byte cache budgets")
    layer_count = len(full_state.layers)
    lengths = tuple(layer.sequence_length for layer in full_state.layers)
    kv_heads = tuple(layer.kv_heads for layer in full_state.layers)
    if len(set(kv_heads)) != 1:
        raise PrefixKVReimplementationError(
            "generalized PrefixKV currently requires the same KV-head count in every layer"
        )
    head_count = kv_heads[0]
    lower = tuple(
        len(
            set(range(min(prefixkv_config.start_size, length)))
            | set(range(max(0, length - prefixkv_config.protect_size), length))
        )
        for length in lengths
    )
    source_slots = len(full_state.blocks)
    target_slots = math.floor(source_slots * cache_config.retention_ratio + 1e-12)
    if cache_config.budget_unit is BudgetUnit.BLOCKS:
        target_slots = min(target_slots, cache_config.budget_value)
    target_positions = target_slots // head_count
    if target_positions < sum(lower):
        raise PrefixKVReimplementationError(
            "PrefixKV global budget is smaller than the protected start/tail tokens"
        )
    if prefixkv_config.profile_mode is PrefixKVProfileMode.OFFLINE_PROFILE:
        if profile is None:
            if prefixkv_config.profile_path is None:
                raise PrefixKVReimplementationError("PrefixKV offline profile path is missing")
            profile = load_prefixkv_profile(
                prefixkv_config.profile_path,
                model_id=model_id,
                model_revision=model_revision,
                target_retention_ratio=cache_config.retention_ratio,
                start_size=prefixkv_config.start_size,
                protect_size=prefixkv_config.protect_size,
            )
        if len(profile.layer_forget_ratios) != layer_count:
            raise PrefixKVReimplementationError(
                "PrefixKV profile layer count does not match the decoder"
            )
        desired = tuple(
            length * (1.0 - forget)
            for length, forget in zip(lengths, profile.layer_forget_ratios, strict=True)
        )
    else:
        if profile is not None:
            raise PrefixKVReimplementationError("fixed-global PrefixKV cannot consume a profile")
        desired = tuple(length * cache_config.retention_ratio for length in lengths)
    retained = list(_bounded_apportion(desired, lower, lengths, target_positions))
    bytes_per_position = tuple(
        _layer_bytes_per_position(full_state, layer) for layer in range(layer_count)
    )
    if cache_config.budget_unit is BudgetUnit.BYTES:
        while (
            sum(count * size for count, size in zip(retained, bytes_per_position, strict=True))
            > cache_config.budget_value
        ):
            candidates = [layer for layer in range(layer_count) if retained[layer] > lower[layer]]
            if not candidates:
                raise PrefixKVReimplementationError(
                    "PrefixKV byte budget is smaller than protected boundary storage"
                )
            layer = min(
                candidates,
                key=lambda index: (
                    desired[index] - retained[index],
                    -bytes_per_position[index],
                    index,
                ),
            )
            retained[layer] -= 1
    scores = (
        prefixkv_attention_scores(attention_weights)
        if cache_config.retention_ratio < 1.0
        else tuple(tuple(0.0 for _ in range(length)) for length in lengths)
    )
    if len(scores) != layer_count or any(
        len(layer_scores) != length for layer_scores, length in zip(scores, lengths, strict=True)
    ):
        raise PrefixKVReimplementationError("PrefixKV attention/cache layer shapes do not match")
    layer_states: list[PrefixKVLayerState] = []
    for layer_index, (layer_scores, count, position_bytes) in enumerate(
        zip(scores, retained, bytes_per_position, strict=True)
    ):
        selected, protected = _select_positions(
            layer_scores,
            count,
            prefixkv_config.start_size,
            prefixkv_config.protect_size,
        )
        logical = tuple(
            full_state.logical_positions.logical_for_physical(position) for position in selected
        )
        score_digest = hashlib.sha256(
            np.asarray(layer_scores, dtype="<f8").tobytes(order="C")
        ).hexdigest()
        layer_states.append(
            PrefixKVLayerState(
                layer=layer_index,
                source_positions=lengths[layer_index],
                retained_positions=count,
                selected_physical_positions=selected,
                selected_logical_positions=logical,
                protected_positions=protected,
                attention_scores_sha256=score_digest,
                attention_score_sum=sum(layer_scores),
                bytes_per_position=position_bytes,
                retained_bytes=count * position_bytes,
                eviction_offset=_eviction_offset(count, prefixkv_config.eviction_distance),
            )
        )
    active_slots = sum(
        state.retained_positions * full_state.layers[state.layer].kv_heads for state in layer_states
    )
    retained_bytes = sum(state.retained_bytes for state in layer_states)
    if cache_config.budget_unit is BudgetUnit.BLOCKS and active_slots > cache_config.budget_value:
        raise PrefixKVReimplementationError("PrefixKV active slots exceed cache.budget_value")
    if cache_config.budget_unit is BudgetUnit.BYTES and retained_bytes > cache_config.budget_value:
        raise PrefixKVReimplementationError("PrefixKV retained bytes exceed cache.budget_value")
    official_family = "llava" in model_id.lower() and "1.5" in model_id.lower()
    return PrefixKVCompressionPlan(
        full_state=full_state,
        config=prefixkv_config,
        profile=profile,
        layers=tuple(layer_states),
        source_slots=source_slots,
        active_slots=active_slots,
        source_bytes=full_state.active_bytes,
        retained_bytes=retained_bytes,
        target_retention_ratio=cache_config.retention_ratio,
        implementation_label=(
            "prefixkv_reimpl" if official_family else "generalized_prefixkv_reimpl"
        ),
    )


def prefixkv_runtime_payloads(
    plan: PrefixKVCompressionPlan,
) -> tuple[
    dict[tuple[int, int], list[tuple[int, str, int, Any, Any]]],
    tuple[JsonObject, ...],
]:
    """Convert a PrefixKV plan to the shared exact-cache payload protocol."""

    payloads: dict[tuple[int, int], list[tuple[int, str, int, Any, Any]]] = {}
    records: list[JsonObject] = []
    for state in plan.layers:
        storage = plan.full_state.layers[state.layer]
        for head in range(storage.kv_heads):
            entries: list[tuple[int, str, int, Any, Any]] = []
            for slot, (physical, logical) in enumerate(
                zip(
                    state.selected_physical_positions,
                    state.selected_logical_positions,
                    strict=True,
                )
            ):
                key = _position_payload(
                    storage.key,
                    head_axis=storage.key_head_dimension,
                    head=head,
                    sequence_axis=storage.key_sequence_dimension,
                    position=physical,
                )
                value = _position_payload(
                    storage.value,
                    head_axis=storage.value_head_dimension,
                    head=head,
                    sequence_axis=storage.value_sequence_dimension,
                    position=physical,
                )
                entries.append((logical, "prefixkv_reimpl", physical, key, value))
                records.append(
                    {
                        "layer": state.layer,
                        "kv_head": head,
                        "slot": slot,
                        "logical_position": logical,
                        "physical_position": physical,
                        "tier": "prefixkv_reimpl_exact",
                        "source_id": physical,
                    }
                )
            payloads[(state.layer, head)] = entries
    return payloads, tuple(records)


__all__ = [
    "PrefixKVCalibrationObservation",
    "PrefixKVCompressionPlan",
    "PrefixKVLayerState",
    "PrefixKVOfflineProfile",
    "PrefixKVReimplementationError",
    "build_prefixkv_reimpl_plan",
    "generate_prefixkv_profile",
    "load_prefixkv_profile",
    "prefixkv_attention_scores",
    "prefixkv_calibration_observation",
    "prefixkv_runtime_payloads",
    "search_prefixkv_layer_sizes",
]
