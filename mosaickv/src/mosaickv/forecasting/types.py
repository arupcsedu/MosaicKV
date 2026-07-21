"""Typed outputs and provenance for MosaicKV future-query forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mosaickv.types import ForecastCovariance, ForecastMode


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"forecast tensor has no shape: {type(value).__qualname__}")
    return tuple(int(item) for item in shape)


@dataclass(frozen=True, slots=True)
class ForecastTiming:
    """Separately measured online forecast overhead in seconds."""

    cache_clone: float
    draft_decode: float
    query_preparation: float
    prompt_statistics: float
    centroid_construction: float
    total: float
    synchronization_calls: int
    timing_backend: str

    def __post_init__(self) -> None:
        values = (
            self.cache_clone,
            self.draft_decode,
            self.query_preparation,
            self.prompt_statistics,
            self.centroid_construction,
            self.total,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("forecast timings must be finite and nonnegative")
        component_total = sum(values[:-1])
        if not math.isclose(self.total, component_total, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("forecast total must equal its separately measured components")
        if self.synchronization_calls < 0:
            raise ValueError("synchronization_calls must be nonnegative")
        if self.timing_backend not in {"host_perf_counter", "torch_cuda_event"}:
            raise ValueError("unsupported forecast timing backend")


@dataclass(frozen=True, slots=True)
class HeadForecastProvenance:
    """Exact query-head sources represented by one KV-head forecast."""

    layer: int
    kv_head: int
    query_head_start: int
    query_head_end: int
    prompt_positions: tuple[int, ...]
    prompt_sample_count: int
    draft_sample_count: int

    def __post_init__(self) -> None:
        if self.layer < 0 or self.kv_head < 0:
            raise ValueError("forecast layer and KV head must be nonnegative")
        if self.query_head_start < 0 or self.query_head_end <= self.query_head_start:
            raise ValueError("invalid query-head group")
        if any(position < 0 for position in self.prompt_positions):
            raise ValueError("prompt positions must be nonnegative")
        if self.prompt_sample_count < 0 or self.draft_sample_count < 0:
            raise ValueError("forecast sample counts must be nonnegative")


@dataclass(frozen=True, slots=True)
class ForecastProvenance:
    """Global online-data provenance for one forecast."""

    mode: ForecastMode
    requested_prompt_window: int
    actual_prompt_window: int
    requested_draft_steps: int
    completed_draft_steps: int
    covariance: ForecastCovariance
    centroid_method: str
    centroid_count_requested: int
    deterministic_greedy: bool
    reused_original_prefill: bool
    draft_cache_isolated: bool
    original_logical_sequence_length: int

    def __post_init__(self) -> None:
        integer_fields = (
            self.requested_prompt_window,
            self.actual_prompt_window,
            self.requested_draft_steps,
            self.completed_draft_steps,
            self.centroid_count_requested,
            self.original_logical_sequence_length,
        )
        if any(value < 0 for value in integer_fields):
            raise ValueError("forecast provenance counts must be nonnegative")
        if self.actual_prompt_window > self.requested_prompt_window:
            raise ValueError("actual prompt window exceeds its request")
        if self.completed_draft_steps != self.requested_draft_steps:
            raise ValueError("online draft rollout must complete its configured fixed horizon")
        if self.centroid_count_requested < 1:
            raise ValueError("centroid_count_requested must be positive")
        if self.original_logical_sequence_length < 1:
            raise ValueError("original logical sequence length must be positive")
        if not self.deterministic_greedy or not self.reused_original_prefill:
            raise ValueError("MosaicKV forecasting requires deterministic reuse of prefill")
        if self.requested_draft_steps and not self.draft_cache_isolated:
            raise ValueError("draft forecasting requires an isolated cache")
        if self.centroid_method not in {"deterministic_spherical_kmeans", "streaming_spherical"}:
            raise ValueError("unsupported centroid method")


@dataclass(frozen=True, slots=True)
class KVHeadQueryForecast:
    """Forecast statistics and mixture centroids for one layer and KV head."""

    prompt_mean: Any | None
    prompt_covariance: Any | None
    prompt_diagonal_variance: Any | None
    draft_query_samples: Any | None
    normalized_centroids: Any
    forecast_weights: Any
    provenance: HeadForecastProvenance

    def __post_init__(self) -> None:
        centroid_shape = _shape(self.normalized_centroids)
        weight_shape = _shape(self.forecast_weights)
        if len(centroid_shape) != 2 or centroid_shape[0] < 1 or centroid_shape[1] < 1:
            raise ValueError("normalized_centroids must have shape [centroids, head_dim]")
        if weight_shape != (centroid_shape[0],):
            raise ValueError("forecast_weights must align with normalized_centroids")
        if self.prompt_mean is None:
            if self.prompt_covariance is not None or self.prompt_diagonal_variance is not None:
                raise ValueError("prompt dispersion requires a prompt mean")
        else:
            mean_shape = _shape(self.prompt_mean)
            if mean_shape != (centroid_shape[1],):
                raise ValueError("prompt_mean must match centroid head dimension")
            if (self.prompt_covariance is None) == (self.prompt_diagonal_variance is None):
                raise ValueError("provide exactly one prompt covariance representation")
            if self.prompt_covariance is not None and _shape(self.prompt_covariance) != (
                centroid_shape[1],
                centroid_shape[1],
            ):
                raise ValueError("prompt_covariance has an invalid shape")
            if self.prompt_diagonal_variance is not None and _shape(
                self.prompt_diagonal_variance
            ) != (centroid_shape[1],):
                raise ValueError("prompt_diagonal_variance has an invalid shape")
        if self.draft_query_samples is not None:
            draft_shape = _shape(self.draft_query_samples)
            if len(draft_shape) != 2 or draft_shape[1] != centroid_shape[1]:
                raise ValueError("draft_query_samples must have shape [samples, head_dim]")


@dataclass(frozen=True, slots=True)
class QueryForecast:
    """All layer/KV-head forecasts produced before online cache compression."""

    layers: tuple[tuple[KVHeadQueryForecast, ...], ...]
    provenance: ForecastProvenance
    timing: ForecastTiming

    def __post_init__(self) -> None:
        if not self.layers or any(not layer for layer in self.layers):
            raise ValueError("QueryForecast must contain every layer and at least one KV head")
        for layer_index, layer in enumerate(self.layers):
            for head_index, forecast in enumerate(layer):
                if (
                    forecast.provenance.layer != layer_index
                    or forecast.provenance.kv_head != head_index
                ):
                    raise ValueError("forecast provenance does not match layer/KV-head nesting")

    def for_head(self, layer: int, kv_head: int) -> KVHeadQueryForecast:
        """Return one layer/KV-head forecast with bounds-checked indexing."""

        try:
            return self.layers[layer][kv_head]
        except IndexError as error:
            raise IndexError(
                f"forecast head does not exist: layer={layer}, kv_head={kv_head}"
            ) from error


__all__ = [
    "ForecastProvenance",
    "ForecastTiming",
    "HeadForecastProvenance",
    "KVHeadQueryForecast",
    "QueryForecast",
]
