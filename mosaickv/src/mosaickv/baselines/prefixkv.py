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
from dataclasses import asdict, dataclass, replace
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
    question_positions: tuple[int, ...] = ()
    question_layer_scores: tuple[tuple[float, ...], ...] = ()
    question_mapping_sha256: str | None = None

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
        if self.question_positions or self.question_layer_scores or self.question_mapping_sha256:
            _validate_question_positions(self.question_positions, min(map(len, self.layer_scores)))
            if not self.question_mapping_sha256 or not _valid_sha(self.question_mapping_sha256, 64):
                raise PrefixKVReimplementationError("question calibration requires a mapping hash")
            if len(self.question_layer_scores) != len(self.layer_scores) or any(
                len(a) != len(b) or any(not math.isfinite(x) or x < 0 for x in b)
                for a, b in zip(self.layer_scores, self.question_layer_scores, strict=True)
            ):
                raise PrefixKVReimplementationError("invalid question calibration score shapes")


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
    # Absent in historical JSON/hashes. New profiling must explicitly identify
    # the pinned upstream protected-count convention rather than relabel old
    # local profiles as upstream-compatible.
    search_policy: str = "legacy_cdf"
    calibration_byte_cap: int | None = None
    calibration_bytes_per_layer_position: int | None = None
    calibration_mean_layer_positions: tuple[float, ...] = ()
    prompt_reservation_policy: str = "boundaries"
    question_score_weight: float = 0.0
    calibration_evidence_sha256: str | None = None

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
        if self.search_policy not in {"legacy_cdf", "upstream_protected_cdf", "optional_cdf"}:
            raise PrefixKVReimplementationError("unsupported PrefixKV calibration search policy")
        if self.prompt_reservation_policy not in {"boundaries", "question_and_boundaries"}:
            raise PrefixKVReimplementationError("invalid profile prompt reservation policy")
        if (
            isinstance(self.question_score_weight, bool)
            or not isinstance(self.question_score_weight, (int, float))
            or not math.isfinite(self.question_score_weight)
            or not 0 <= self.question_score_weight <= 1
        ):
            raise PrefixKVReimplementationError("invalid profile question score weight")
        if self.evidence_variant and (
            self.calibration_byte_cap is None
            or self.calibration_evidence_sha256 is None
            or not _valid_sha(self.calibration_evidence_sha256, 64)
        ):
            raise PrefixKVReimplementationError(
                "evidence profile requires byte cap and capture hash"
            )
        if not self.evidence_variant and self.calibration_evidence_sha256 is not None:
            raise PrefixKVReimplementationError("unexpected evidence hash on legacy profile")
        if self.calibration_byte_cap is not None:
            width = self.calibration_bytes_per_layer_position
            if (
                type(self.calibration_byte_cap) is not int
                or self.calibration_byte_cap <= 0
                or type(width) is not int
                or width <= 0
                or self.source_kind != "calibration_generated"
                or len(self.calibration_mean_layer_positions) != len(self.layer_forget_ratios)
                or any(
                    not math.isfinite(value) or value <= 0
                    for value in self.calibration_mean_layer_positions
                )
            ):
                raise PrefixKVReimplementationError("invalid byte-target calibration metadata")
            if not math.isclose(
                sum(self.calibration_mean_layer_positions),
                self.calibration_byte_cap // width,
                rel_tol=0,
                abs_tol=1e-8,
            ):
                raise PrefixKVReimplementationError(
                    "calibration positions do not fill the byte cap"
                )
        elif (
            self.calibration_bytes_per_layer_position is not None
            or self.calibration_mean_layer_positions
        ):
            raise PrefixKVReimplementationError("byte-target calibration metadata requires a cap")
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
    def evidence_variant(self) -> bool:
        return (
            self.search_policy == "optional_cdf"
            or self.prompt_reservation_policy != "boundaries"
            or self.question_score_weight > 0
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
        if self.search_policy != "legacy_cdf":
            payload["search_policy"] = self.search_policy
        if self.calibration_byte_cap is not None:
            payload["byte_calibration"] = {
                "cap": self.calibration_byte_cap,
                "bytes_per_layer_position": self.calibration_bytes_per_layer_position,
                "mean_layer_positions": list(self.calibration_mean_layer_positions),
                "sample_aggregation": "mean_retained_positions_at_identical_byte_cap",
            }
        if self.evidence_variant:
            payload["evidence_policy"] = {
                "prompt_reservation": self.prompt_reservation_policy,
                "question_score_weight": self.question_score_weight,
                "calibration_evidence_sha256": self.calibration_evidence_sha256,
                "scoring": "optional_normalized_original_question_mixture_v1",
            }
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
            "search_policy",
            "byte_calibration",
            "evidence_policy",
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
        byte = payload.get("byte_calibration")
        if byte is not None and (
            not isinstance(byte, dict)
            or set(byte)
            != {"cap", "bytes_per_layer_position", "mean_layer_positions", "sample_aggregation"}
            or byte.get("sample_aggregation") != "mean_retained_positions_at_identical_byte_cap"
            or type(byte.get("cap")) is not int
            or type(byte.get("bytes_per_layer_position")) is not int
            or not isinstance(byte.get("mean_layer_positions"), list)
        ):
            raise PrefixKVReimplementationError("invalid byte-target calibration metadata")
        byte_fields = cast("dict[str, Any]", byte) if byte is not None else {}
        evidence = payload.get("evidence_policy", {})
        if not isinstance(evidence, dict) or (
            evidence
            and (
                set(evidence)
                != {
                    "prompt_reservation",
                    "question_score_weight",
                    "calibration_evidence_sha256",
                    "scoring",
                }
                or evidence.get("scoring") != "optional_normalized_original_question_mixture_v1"
            )
        ):
            raise PrefixKVReimplementationError("invalid profile evidence policy")
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
            search_policy=str(payload.get("search_policy", "legacy_cdf")),
            calibration_byte_cap=byte_fields.get("cap"),
            calibration_bytes_per_layer_position=byte_fields.get("bytes_per_layer_position"),
            calibration_mean_layer_positions=tuple(byte_fields.get("mean_layer_positions", ())),
            prompt_reservation_policy=str(evidence.get("prompt_reservation", "boundaries")),
            question_score_weight=cast("float", evidence.get("question_score_weight", 0.0)),
            calibration_evidence_sha256=cast(
                "str | None", evidence.get("calibration_evidence_sha256")
            ),
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
    target_retention_ratio: float | None,
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
        if target_retention_ratio is None:
            raise PrefixKVReimplementationError(
                "a raw PrefixKV profile requires profile_reference_retention_ratio "
                "when rescaled to a byte budget"
            )
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
    if target_retention_ratio is not None and not math.isclose(
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
    sample_id: str,
    attention_weights: tuple[Any, ...],
    *,
    question_positions: tuple[int, ...] = (),
    question_mapping_sha256: str | None = None,
) -> PrefixKVCalibrationObservation:
    """Create one serializable calibration row from an eager HF prefill."""

    return PrefixKVCalibrationObservation(
        sample_id=sample_id,
        layer_scores=prefixkv_attention_scores(attention_weights),
        question_positions=question_positions,
        question_layer_scores=(
            prefixkv_question_scores(attention_weights, question_positions)
            if question_positions
            else ()
        ),
        question_mapping_sha256=question_mapping_sha256,
    )


def _validate_question_positions(positions: tuple[int, ...], length: int) -> None:
    if (
        not positions
        or any(type(p) is not int or not 0 <= p < length for p in positions)
        or positions != tuple(sorted(set(positions)))
    ):
        raise PrefixKVReimplementationError(
            "question positions must be nonempty, sorted, unique and in range"
        )


def prefixkv_question_scores(
    attention_weights: tuple[Any, ...], question_positions: tuple[int, ...]
) -> tuple[tuple[float, ...], ...]:
    """Same head reduction, but only actual mapped user-instruction query rows."""
    scores = []
    for attention in attention_weights:
        if (
            len(attention.shape) != 4
            or attention.shape[0] != 1
            or attention.shape[-2] != attention.shape[-1]
        ):
            raise PrefixKVReimplementationError(
                "question scoring requires complete square prefill attentions"
            )
        _validate_question_positions(question_positions, int(attention.shape[-1]))
        if hasattr(attention, "detach"):
            pooled = (
                attention.detach()[:, :, list(question_positions), :]
                .mean(dim=1)
                .float()
                .sum(dim=-2)[0]
                .cpu()
                .numpy()
            )
        else:
            pooled = (
                np.asarray(attention)[:, :, list(question_positions), :]
                .mean(axis=1)
                .astype(np.float32)
                .sum(axis=-2)[0]
            )
        if not np.isfinite(pooled).all() or (pooled < 0).any():
            raise PrefixKVReimplementationError("invalid question attention scores")
        scores.append(tuple(float(x) for x in pooled))
    if not scores:
        raise PrefixKVReimplementationError("question scoring requires prefill attention")
    return tuple(scores)


def mix_prefixkv_scores(
    original: tuple[tuple[float, ...], ...],
    question: tuple[tuple[float, ...], ...],
    protected: tuple[tuple[int, ...], ...],
    weight: float,
) -> tuple[tuple[float, ...], ...]:
    """Normalize each source over OPTIONAL positions, then mix; uniform on zero mass."""
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or not 0 <= weight <= 1
    ):
        raise PrefixKVReimplementationError("question mixture weight must be in [0, 1]")
    if len(original) != len(question) or len(original) != len(protected):
        raise PrefixKVReimplementationError("question mixture layer count mismatch")
    result = []
    for a, q, fixed in zip(original, question, protected, strict=True):
        if len(a) != len(q):
            raise PrefixKVReimplementationError("question mixture score length mismatch")
        if any(type(p) is not int or not 0 <= p < len(a) for p in fixed) or fixed != tuple(
            sorted(set(fixed))
        ):
            raise PrefixKVReimplementationError("invalid protected mixture positions")
        eligible = np.ones(len(a), dtype=bool)
        eligible[list(fixed)] = False
        distributions = []
        for values in (a, q):
            array = np.asarray(values, dtype=np.float64).copy()
            if not np.isfinite(array).all() or (array < 0).any():
                raise PrefixKVReimplementationError(
                    "question mixture scores must be finite and nonnegative"
                )
            array[~eligible] = 0
            total = float(array.sum())
            distributions.append(
                array / total if total > 0 else eligible.astype(float) / max(1, int(eligible.sum()))
            )
        result.append(
            tuple(float(x) for x in ((1 - weight) * distributions[0] + weight * distributions[1]))
        )
    return tuple(result)


def _protected_positions(
    length: int, start: int, tail: int, extra: tuple[int, ...] = ()
) -> tuple[int, ...]:
    return tuple(
        sorted(
            set(range(min(start, length))) | set(range(max(0, length - tail), length)) | set(extra)
        )
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
    search_policy: str = "legacy_cdf",
    target_positions: int | None = None,
    protected_positions: tuple[tuple[int, ...], ...] | None = None,
) -> tuple[int, ...]:
    """Search a shared priority threshold, with explicitly versioned protection.

    ``upstream_protected_cdf`` follows obtain_cdf_num's kept-count convention
    (CDF prefix PLUS protected allowance). Exact integer apportionment remains
    a local adaptation; this is not a bitwise clone of upstream's <5 tolerance,
    forgotten-count rescaling and truncation. ``target_positions`` permits
    exact per-sample byte-derived calibration rather than a mean-length ratio.
    """

    if not 0 < retention_ratio <= 1:
        raise PrefixKVReimplementationError("retention_ratio must be in (0, 1]")
    if not layer_scores:
        raise PrefixKVReimplementationError("PrefixKV search requires at least one layer")
    if search_policy not in {"legacy_cdf", "upstream_protected_cdf", "optional_cdf"}:
        raise PrefixKVReimplementationError("unsupported PrefixKV calibration search policy")
    if start_size < 0 or protect_size < 1:
        raise PrefixKVReimplementationError("invalid protected boundary sizes")
    if any(
        not scores or any(not math.isfinite(value) or value < 0 for value in scores)
        for scores in layer_scores
    ):
        raise PrefixKVReimplementationError("layer scores must be nonempty, finite and nonnegative")
    lengths = tuple(len(scores) for scores in layer_scores)
    lower = tuple(min(length, start_size + protect_size) for length in lengths)
    fixed = protected_positions or tuple(
        _protected_positions(n, start_size, protect_size) for n in lengths
    )
    if len(fixed) != len(lengths) or any(
        p != tuple(sorted(set(p))) or any(type(i) is not int or not 0 <= i < n for i in p)
        for p, n in zip(fixed, lengths, strict=True)
    ):
        raise PrefixKVReimplementationError("invalid protected positions for calibration")
    if protected_positions is not None or search_policy == "optional_cdf":
        lower = tuple(min(n, len(p) + 1) for p, n in zip(fixed, lengths, strict=True))
    target = max(sum(lower), math.floor(sum(lengths) * retention_ratio + 1e-12))
    target = min(target, sum(lengths))
    if target_positions is not None:
        if type(target_positions) is not int or not sum(lower) <= target_positions <= sum(lengths):
            raise PrefixKVReimplementationError("target_positions is outside mandatory bounds")
        target = target_positions
    cumulative: list[np.ndarray[Any, Any]] = []
    for scores, protected in zip(layer_scores, fixed, strict=True):
        values = np.asarray(scores, dtype=np.float64)
        if search_policy == "optional_cdf":
            values = np.delete(values, protected)
        ranked = np.sort(values)[::-1]
        total = float(ranked.sum())
        normalized = (
            np.full_like(ranked, 1.0 / max(1, len(ranked))) if total <= 0 else ranked / total
        )
        cumulative.append(np.cumsum(normalized))

    def sizes(threshold: float) -> tuple[int, ...]:
        return tuple(
            max(
                low,
                min(
                    length,
                    int(np.searchsorted(cdf, threshold, side="left"))
                    + 1
                    + (len(protected) if search_policy != "legacy_cdf" else 0),
                ),
            )
            for cdf, low, length, protected in zip(cumulative, lower, lengths, fixed, strict=True)
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
    search_policy: str = "legacy_cdf",
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
            search_policy=search_policy,
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
        search_policy=search_policy,
    )


def generate_prefixkv_byte_profile(
    observations: tuple[PrefixKVCalibrationObservation, ...],
    *,
    byte_cap: int,
    bytes_per_layer_position: int,
    model_id: str,
    model_revision: str,
    dataset_id: str,
    dataset_revision: str,
    calibration_split: str,
    evaluation_sample_ids: tuple[str, ...],
    seed: int,
    start_size: int = 1,
    protect_size: int = 1,
    exclude_protected_from_calibration: bool = False,
    prompt_reservation_policy: str = "boundaries",
    question_score_weight: float = 0.0,
) -> PrefixKVOfflineProfile:
    """Label-blind local profile at the ACTUAL shared cap for every observation.

    Unlike averaging cap/full-size ratios and profiling at that one ratio,
    every sample has the same integer retained-position target. Freeze mean
    retained counts (not length-dependent forget ratios) for later inference.
    Full-cache/too-short observations fail rather than changing the objective.
    """
    if (
        type(byte_cap) is not int
        or byte_cap <= 0
        or type(bytes_per_layer_position) is not int
        or bytes_per_layer_position <= 0
        or not observations
    ):
        raise PrefixKVReimplementationError("byte calibration needs positive integer cap/width")
    if type(exclude_protected_from_calibration) is not bool:
        raise PrefixKVReimplementationError("exclude_protected_from_calibration must be boolean")
    if prompt_reservation_policy not in {"boundaries", "question_and_boundaries"}:
        raise PrefixKVReimplementationError("invalid prompt reservation policy")
    if (
        isinstance(question_score_weight, bool)
        or not isinstance(question_score_weight, (int, float))
        or not math.isfinite(question_score_weight)
        or not 0 <= question_score_weight <= 1
    ):
        raise PrefixKVReimplementationError("question_score_weight must be finite in [0, 1]")
    evidence = (
        exclude_protected_from_calibration
        or question_score_weight > 0
        or prompt_reservation_policy != "boundaries"
    )
    needs_question = (
        question_score_weight > 0 or prompt_reservation_policy == "question_and_boundaries"
    )
    if needs_question and any(
        not o.question_positions or not o.question_mapping_sha256 for o in observations
    ):
        raise PrefixKVReimplementationError(
            "evidence calibration requires mapped question captures"
        )
    lengths = [
        sum(len(scores) for scores in observation.layer_scores) for observation in observations
    ]
    target = byte_cap // bytes_per_layer_position
    if any(target >= length for length in lengths):
        raise PrefixKVReimplementationError("byte calibration must compress every observation")
    # Reuse identity, provenance and disjointness validation. The ratio is
    # descriptive only; the explicit byte-derived counts below are authoritative.
    ratio = sum(target / length for length in lengths) / len(lengths)
    base = generate_prefixkv_profile(
        observations,
        model_id=model_id,
        model_revision=model_revision,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        calibration_split=calibration_split,
        evaluation_sample_ids=evaluation_sample_ids,
        retention_ratio=ratio,
        seed=seed,
        start_size=start_size,
        protect_size=protect_size,
        search_policy="upstream_protected_cdf",
    )
    counts = []
    search_policy = (
        "optional_cdf" if exclude_protected_from_calibration else "upstream_protected_cdf"
    )
    for observation in observations:
        fixed = tuple(
            _protected_positions(
                len(scores),
                start_size,
                protect_size,
                observation.question_positions
                if prompt_reservation_policy == "question_and_boundaries"
                else (),
            )
            for scores in observation.layer_scores
        )
        scores = (
            mix_prefixkv_scores(
                observation.layer_scores,
                observation.question_layer_scores,
                fixed,
                question_score_weight,
            )
            if question_score_weight > 0
            else observation.layer_scores
        )
        counts.append(
            search_prefixkv_layer_sizes(
                scores,
                retention_ratio=ratio,
                start_size=start_size,
                protect_size=protect_size,
                search_policy=search_policy,
                target_positions=target,
                protected_positions=fixed if evidence else None,
            )
        )
    layer_count = len(base.layer_forget_ratios)
    return replace(
        base,
        layer_forget_ratios=tuple(
            sum(
                1.0 - count[layer] / len(observation.layer_scores[layer])
                for observation, count in zip(observations, counts, strict=True)
            )
            / len(observations)
            for layer in range(layer_count)
        ),
        calibration_byte_cap=byte_cap,
        calibration_bytes_per_layer_position=bytes_per_layer_position,
        calibration_mean_layer_positions=tuple(
            sum(count[layer] for count in counts) / len(counts) for layer in range(layer_count)
        ),
        search_policy=search_policy,
        prompt_reservation_policy=prompt_reservation_policy,
        question_score_weight=question_score_weight,
        calibration_evidence_sha256=_sha256_json([asdict(o) for o in observations])
        if evidence
        else None,
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
    decode_forget_ratio: float | None = None

    def __post_init__(self) -> None:
        if self.decode_forget_ratio is not None and not 0 <= self.decode_forget_ratio <= 1:
            raise PrefixKVReimplementationError("PrefixKV decode forget ratio must be in [0, 1]")
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

    @property
    def decode_retention_ratio(self) -> float:
        # Preserve upstream's exact floating arithmetic, not rounded count/N.
        return (
            self.retention_ratio
            if self.decode_forget_ratio is None
            else 1.0 - self.decode_forget_ratio
        )


def prefixkv_evidence_label(
    sink_neutral: bool, question_protected: bool, question_scored: bool
) -> str:
    parts = [
        name
        for enabled, name in (
            (sink_neutral, "sink_neutral"),
            (question_protected, "question_protected"),
            (question_scored, "question_scored"),
        )
        if enabled
    ]
    return "prefixkv_" + "_".join(parts) + "_reimpl"


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
    question_positions: tuple[int, ...] = ()
    prompt_reservation_policy: str = "boundaries"

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
        labels = {
            "prefixkv_reimpl",
            "generalized_prefixkv_reimpl",
            "prefixkv_selection_only_reimpl",
            "prefixkv_capacity_floor_reimpl",
            "prefixkv_byte_calibrated_reimpl",
            "prefixkv_uniform_layer_reimpl",
        } | {
            prefixkv_evidence_label(a, b, c)
            for a in (False, True)
            for b in (False, True)
            for c in (False, True)
            if a or b or c
        }
        if self.config.decode_policy == "append_only":
            labels |= {
                name.removesuffix("_reimpl") + "_selection_only_reimpl"
                for name in labels if "selection_only" not in name
            }
        if self.implementation_label not in labels:
            raise PrefixKVReimplementationError("PrefixKV implementation label is invalid")

    @property
    def layer_cache_sizes(self) -> tuple[int, ...]:
        return tuple(layer.retained_positions for layer in self.layers)

    @property
    def has_uniform_layer_capacity(self) -> bool:
        """Whether common-length HF cache transport can represent this plan exactly.

        Upstream PrefixKV allows a different retained prefix length for every
        decoder layer. MosaicKV's shared HF transport uses one physical prompt
        length and attention-mask extent across layers. Padding this plan to
        that common extent changes its measured active storage, so such a
        padded execution is not paper-faithful PrefixKV.
        """

        return len(set(self.layer_cache_sizes)) == 1

    def trace(self) -> JsonObject:
        return {
            "implementation": "prefixkv_reimpl",
            "implementation_label": self.implementation_label,
            "official_code": False,
            "paper": "Wang et al., NeurIPS 2025, Equation 1 and Algorithm 1",
            "official_repository_sha": self.config.official_repository_sha,
            "profile_mode": self.config.profile_mode.value,
            "byte_budget_profile_rescaling": self.config.rescale_offline_profile_to_byte_budget,
            "profile_reference_retention_ratio": (
                self.config.profile_reference_retention_ratio
                if self.config.rescale_offline_profile_to_byte_budget
                else None
            ),
            "profile_native_retention_ratio": (
                self.profile.target_retention_ratio if self.profile is not None else None
            ),
            "profile_sha256": self.profile.profile_sha256 if self.profile is not None else None,
            "profile_source_kind": self.profile.source_kind if self.profile is not None else None,
            "calibration_search_policy": (
                self.profile.search_policy if self.profile is not None else None
            ),
            "calibration_byte_cap": (
                self.profile.calibration_byte_cap if self.profile is not None else None
            ),
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
            "hf_cache_transport": self.config.hf_cache_transport,
            "decode_boundary_policy": self.config.decode_boundary_policy,
            "decode_policy": self.config.decode_policy,
            "allocation_policy": self.config.allocation_policy,
            "layer_budget_floor_fraction": self.config.layer_budget_floor_fraction,
            "evidence_ablation": {
                "exclude_protected_from_calibration": (
                    self.config.exclude_protected_from_calibration
                ),
                "prompt_reservation_policy": self.prompt_reservation_policy,
                "question_score_weight": self.config.question_score_weight,
                "question_positions": list(self.question_positions),
                "calibration_evidence_sha256": self.profile.calibration_evidence_sha256
                if self.profile
                else None,
                "protected_counts": [len(layer.protected_positions) for layer in self.layers],
            },
            # Small summaries survive light traces, unlike the selected-position lists.
            "layer_capacity_summary": {
                "positions": list(self.layer_cache_sizes),
                "minimum": min(self.layer_cache_sizes),
                "maximum": max(self.layer_cache_sizes),
                "total": sum(self.layer_cache_sizes),
                "layers_below_eviction_distance": sum(
                    count < abs(self.config.eviction_distance) for count in self.layer_cache_sizes
                ),
            },
            "paper_native_algorithm": self.config.allocation_policy == "upstream_round",
            "minimum_decode_capacity": (
                max(
                    min(layer.source_positions, len(layer.protected_positions) + 1)
                    for layer in self.layers
                )
                if self.prompt_reservation_policy == "question_and_boundaries"
                else max(3, self.config.start_size + self.config.protect_size + 1)
                if self.config.decode_boundary_policy == "protected"
                else None
            ),
            "layers": [
                {
                    "layer": layer.layer,
                    "source_positions": layer.source_positions,
                    "retained_positions": layer.retained_positions,
                    "retention_ratio": layer.retention_ratio,
                    "decode_retention_ratio": layer.decode_retention_ratio,
                    "decode_forget_ratio": layer.decode_forget_ratio,
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
    scores: tuple[float, ...],
    retained: int,
    start_size: int,
    protect_size: int,
    extra_protected: tuple[int, ...] = (),
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    length = len(scores)
    protected = _protected_positions(length, start_size, protect_size, extra_protected)
    remaining = retained - len(protected)
    if remaining < 0:
        raise PrefixKVReimplementationError("layer budget cannot retain protected PrefixKV tokens")
    candidates = tuple(position for position in range(length) if position not in set(protected))
    ranked = sorted(candidates, key=lambda position: (-scores[position], position))
    selected = tuple(sorted((*protected, *ranked[:remaining])))
    return selected, protected


def _eviction_offset(
    retained: int, distance: int, *, start_size: int = 0, protect_size: int = 0
) -> int:
    if retained < 1:
        raise PrefixKVReimplementationError("PrefixKV cannot schedule an empty cache")
    offset = distance if distance > 0 else retained + distance
    if offset < 1 and retained >= 3:
        offset = retained // 2
    offset = max(0, min(retained - 1, offset))
    if not start_size <= offset < retained - protect_size:
        raise PrefixKVReimplementationError(
            "PrefixKV fixed eviction offset would remove a protected boundary; "
            "increase feasible layer capacity or use a supported eviction distance"
        )
    return offset


def build_prefixkv_reimpl_plan(
    full_state: FullKVState,
    attention_weights: tuple[Any, ...],
    prefixkv_config: PrefixKVConfig,
    cache_config: CacheConfig,
    *,
    model_id: str,
    model_revision: str,
    profile: PrefixKVOfflineProfile | None = None,
    physical_prompt_capacity: int | None = None,
    question_positions: tuple[int, ...] = (),
) -> PrefixKVCompressionPlan:
    """Select per-layer top-importance positions under one exact global budget."""

    if not prefixkv_config.enabled:
        raise PrefixKVReimplementationError("prefixkv.enabled must be true")
    if full_state.block_size != 1 or cache_config.block_size != 1:
        raise PrefixKVReimplementationError("PrefixKV requires token-sized cache blocks")
    if cache_config.budget_unit not in {BudgetUnit.BLOCKS, BudgetUnit.BYTES}:
        raise PrefixKVReimplementationError("PrefixKV supports block or byte cache budgets")
    if (
        prefixkv_config.rescale_offline_profile_to_byte_budget
        and cache_config.budget_unit is not BudgetUnit.BYTES
    ):
        raise PrefixKVReimplementationError(
            "byte-capped offline-profile rescaling requires cache.budget_unit='bytes'"
        )
    layer_count = len(full_state.layers)
    lengths = tuple(layer.sequence_length for layer in full_state.layers)
    kv_heads = tuple(layer.kv_heads for layer in full_state.layers)
    if len(set(kv_heads)) != 1:
        raise PrefixKVReimplementationError(
            "generalized PrefixKV currently requires the same KV-head count in every layer"
        )
    head_count = kv_heads[0]
    preserve_question = cache_config.physical_budget_mandatory_policy == "question_and_boundaries"
    evidence_variant = (
        preserve_question
        or prefixkv_config.exclude_protected_from_calibration
        or prefixkv_config.question_score_weight > 0
    )
    if evidence_variant and (
        prefixkv_config.hf_cache_transport != "ragged_llama_eager_v1"
        or prefixkv_config.allocation_policy != "global_apportion"
        or prefixkv_config.decode_boundary_policy != "protected"
        or not prefixkv_config.rescale_offline_profile_to_byte_budget
    ):
        raise PrefixKVReimplementationError(
            "evidence ablations require protected byte-rescaled ragged PrefixKV"
        )
    if (
        preserve_question or prefixkv_config.question_score_weight > 0
    ) and cache_config.retention_ratio < 1:
        _validate_question_positions(question_positions, min(lengths))
    extra = question_positions if preserve_question else ()
    if extra and full_state.logical_positions.physical_to_logical != tuple(range(lengths[0])):
        raise PrefixKVReimplementationError(
            "question mapping requires the original contiguous prefill cache"
        )
    fixed_positions = tuple(
        _protected_positions(n, prefixkv_config.start_size, prefixkv_config.protect_size, extra)
        for n in lengths
    )
    if preserve_question and set(full_state.mandatory_logical_positions) != set(fixed_positions[0]):
        raise PrefixKVReimplementationError(
            "shared question mandatory policy differs from source blocks"
        )
    guarded = prefixkv_config.decode_boundary_policy == "protected"
    if cache_config.physical_budget_mandatory_policy == "first_and_last" and (
        prefixkv_config.start_size != 1 or prefixkv_config.protect_size != 1 or not guarded
    ):
        raise PrefixKVReimplementationError(
            "shared first_and_last requires protected PrefixKV with start_size=protect_size=1"
        )
    lower = tuple(len(p) for p in fixed_positions)
    if guarded:
        # Upstream's short-cache midpoint fallback needs an interior position.
        # Reallocate within the SAME global byte budget, never add extra bytes.
        minimum = max(3, prefixkv_config.start_size + prefixkv_config.protect_size + 1)
        lower = tuple(
            min(length, max(low + int(preserve_question), minimum))
            for length, low in zip(lengths, lower, strict=True)
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
                target_retention_ratio=(
                    prefixkv_config.profile_reference_retention_ratio
                    if prefixkv_config.rescale_offline_profile_to_byte_budget
                    else cache_config.retention_ratio
                ),
                start_size=prefixkv_config.start_size,
                protect_size=prefixkv_config.protect_size,
            )
        if (
            prefixkv_config.rescale_offline_profile_to_byte_budget
            and prefixkv_config.profile_reference_retention_ratio is not None
            and not math.isclose(
                profile.target_retention_ratio,
                prefixkv_config.profile_reference_retention_ratio,
                rel_tol=0,
                abs_tol=5e-3,
            )
        ):
            raise PrefixKVReimplementationError(
                "PrefixKV profile does not match profile_reference_retention_ratio"
            )
        if len(profile.layer_forget_ratios) != layer_count:
            raise PrefixKVReimplementationError(
                "PrefixKV profile layer count does not match the decoder"
            )
        if profile.model_id != model_id or profile.model_revision != model_revision:
            raise PrefixKVReimplementationError("PrefixKV profile model identity mismatch")
        if (profile.start_size, profile.protect_size) != (
            prefixkv_config.start_size,
            prefixkv_config.protect_size,
        ):
            raise PrefixKVReimplementationError("PrefixKV profile protected-boundary mismatch")
        if (
            (profile.search_policy == "optional_cdf")
            != prefixkv_config.exclude_protected_from_calibration
            or profile.question_score_weight != prefixkv_config.question_score_weight
            or profile.prompt_reservation_policy
            != ("question_and_boundaries" if preserve_question else "boundaries")
        ):
            raise PrefixKVReimplementationError("PrefixKV evidence profile/configuration mismatch")
        desired = tuple(
            length * (1.0 - forget)
            for length, forget in zip(lengths, profile.layer_forget_ratios, strict=True)
        )
    else:
        if profile is not None:
            raise PrefixKVReimplementationError("fixed-global PrefixKV cannot consume a profile")
        desired = tuple(length * cache_config.retention_ratio for length in lengths)
    upper = lengths
    ragged = prefixkv_config.hf_cache_transport == "ragged_llama_eager_v1"
    if physical_prompt_capacity is not None and not ragged:
        if physical_prompt_capacity < 1:
            raise PrefixKVReimplementationError("physical_prompt_capacity must be positive")
        upper = tuple(min(length, physical_prompt_capacity) for length in lengths)
        if any(low > high for low, high in zip(lower, upper, strict=True)):
            raise PrefixKVReimplementationError(
                "PrefixKV protected positions exceed the dense physical capacity"
            )
        maximum_desired = max(desired)
        scale = (
            1.0
            if maximum_desired <= physical_prompt_capacity
            else physical_prompt_capacity / maximum_desired
        )
        desired = tuple(value * scale for value in desired)
        target_positions = max(
            sum(lower),
            min(
                target_positions,
                sum(upper),
                math.floor(sum(desired) + 1e-12),
            ),
        )
    bytes_per_position = tuple(
        _layer_bytes_per_position(full_state, layer) for layer in range(layer_count)
    )
    if profile is not None and profile.calibration_byte_cap is not None:
        if (
            not ragged
            or not prefixkv_config.rescale_offline_profile_to_byte_budget
            or prefixkv_config.allocation_policy != "global_apportion"
            or prefixkv_config.decode_policy
            not in (
                {"effective_profile_ratio", "append_only"}
                if profile.evidence_variant
                else {"effective_profile_ratio"}
            )
            or cache_config.budget_unit is not BudgetUnit.BYTES
            or cache_config.budget_value != profile.calibration_byte_cap
            or any(
                width != profile.calibration_bytes_per_layer_position
                for width in bytes_per_position
            )
        ):
            raise PrefixKVReimplementationError(
                "byte-calibrated profile requires its recorded physical cap/width, ragged "
                "global_apportion and effective_profile_ratio; it is not a native ratio point"
            )
        desired = profile.calibration_mean_layer_positions
    if ragged:
        # A dense cap is a *total* shared-byte target, not an upper bound on
        # each layer. Capping each layer destroys the offline profile shape.
        if len(set(bytes_per_position)) != 1:
            raise PrefixKVReimplementationError("ragged PrefixKV requires equal layer byte widths")
        if physical_prompt_capacity is not None:
            if physical_prompt_capacity < 1:
                raise PrefixKVReimplementationError("physical_prompt_capacity must be positive")
            target_positions = min(target_positions, physical_prompt_capacity * layer_count)
        if cache_config.budget_unit is BudgetUnit.BYTES:
            target_positions = min(
                target_positions, cache_config.budget_value // bytes_per_position[0]
            )
    upstream_round = prefixkv_config.allocation_policy == "upstream_round"
    if prefixkv_config.layer_budget_floor_fraction > 0:
        floor = math.floor(
            prefixkv_config.layer_budget_floor_fraction * target_positions / layer_count
        )
        lower = tuple(max(low, min(high, floor)) for low, high in zip(lower, upper, strict=True))
        if sum(lower) > target_positions:
            raise PrefixKVReimplementationError("capacity floor exceeds the shared byte budget")
    if upstream_round and cache_config.retention_ratio < 1.0:
        assert profile is not None
        # Upstream rounds *forgotten* positions independently (numpy ties-to-even).
        # Do not globally apportion or silently repair native layer capacities.
        retained = [
            length - int(np.rint(forget * length))
            for length, forget in zip(lengths, profile.layer_forget_ratios, strict=True)
        ]
        if any(
            not low <= count <= high
            for count, low, high in zip(retained, lower, upper, strict=True)
        ):
            raise PrefixKVReimplementationError(
                "upstream rounded profile violates mandatory bounds"
            )
        if (
            physical_prompt_capacity is not None
            and sum(retained) > physical_prompt_capacity * layer_count
        ):
            raise PrefixKVReimplementationError(
                "upstream rounded profile exceeds physical byte cap"
            )
        if (
            cache_config.budget_unit is BudgetUnit.BYTES
            and sum(count * size for count, size in zip(retained, bytes_per_position, strict=True))
            > cache_config.budget_value
        ):
            raise PrefixKVReimplementationError("upstream rounded profile exceeds byte budget")
    else:
        retained = list(_bounded_apportion(desired, lower, upper, target_positions))
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
    if prefixkv_config.question_score_weight > 0 and cache_config.retention_ratio < 1:
        scores = mix_prefixkv_scores(
            scores,
            prefixkv_question_scores(attention_weights, question_positions),
            fixed_positions,
            prefixkv_config.question_score_weight,
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
            extra,
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
                decode_forget_ratio=(
                    profile.layer_forget_ratios[layer_index]
                    if upstream_round and profile is not None and cache_config.retention_ratio < 1.0
                    else 1.0 - count / lengths[layer_index]
                    if prefixkv_config.decode_policy == "effective_profile_ratio"
                    else None
                ),
                eviction_offset=_eviction_offset(
                    count,
                    prefixkv_config.eviction_distance,
                    # Identity layers never evict; short original prompts remain valid.
                    start_size=prefixkv_config.start_size
                    if guarded and count < lengths[layer_index]
                    else 0,
                    protect_size=prefixkv_config.protect_size
                    if guarded and count < lengths[layer_index]
                    else 0,
                ),
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
    implementation_label = (
        prefixkv_evidence_label(
            prefixkv_config.exclude_protected_from_calibration,
            preserve_question,
            prefixkv_config.question_score_weight > 0,
        )
        if evidence_variant
        else "prefixkv_capacity_floor_reimpl"
        if prefixkv_config.layer_budget_floor_fraction > 0
        else "prefixkv_byte_calibrated_reimpl"
        if profile is not None and profile.calibration_byte_cap is not None
        else "prefixkv_selection_only_reimpl"
        if prefixkv_config.decode_policy == "append_only"
        else "prefixkv_uniform_layer_reimpl"
        if prefixkv_config.profile_mode is PrefixKVProfileMode.FIXED_GLOBAL
        and prefixkv_config.decode_policy == "effective_profile_ratio"
        else "prefixkv_reimpl"
        if official_family
        else "generalized_prefixkv_reimpl"
    )
    if (
        prefixkv_config.decode_policy == "append_only"
        and "selection_only" not in implementation_label
    ):
        implementation_label = (
            implementation_label.removesuffix("_reimpl") + "_selection_only_reimpl"
        )
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
        implementation_label=implementation_label,
        question_positions=question_positions,
        prompt_reservation_policy="question_and_boundaries" if preserve_question else "boundaries",
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
