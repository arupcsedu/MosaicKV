"""Backend-preserving prompt, draft, and hybrid query forecasting."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, TypeVar

from mosaickv.config import ForecastingConfig
from mosaickv.forecasting.types import (
    ForecastProvenance,
    ForecastTiming,
    HeadForecastProvenance,
    KVHeadQueryForecast,
    QueryForecast,
)
from mosaickv.measurements.timing import CudaEventTimer, SynchronizationAudit
from mosaickv.types import ForecastCovariance

ResultT = TypeVar("ResultT")


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"query tensor has no shape: {type(value).__qualname__}")
    return tuple(int(item) for item in shape)


def _is_torch(value: Any) -> bool:
    return type(value).__module__.startswith("torch")


def _working_float(value: Any) -> Any:
    if _is_torch(value):
        return value.detach().float().clone()
    import numpy as np

    return np.asarray(value, dtype=np.float32).copy()


def _clone(value: Any) -> Any:
    clone = getattr(value, "clone", None)
    if clone is not None:
        return clone()
    copy = getattr(value, "copy", None)
    if copy is not None:
        return copy()
    raise TypeError(f"query tensor cannot be copied: {type(value).__qualname__}")


def _concat(values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot concatenate an empty query sample collection")
    if _is_torch(values[0]):
        import torch

        return torch.cat(tuple(values), dim=0)
    import numpy as np

    return np.concatenate(tuple(values), axis=0)


def _mean(samples: Any) -> Any:
    return samples.mean(dim=0) if _is_torch(samples) else samples.mean(axis=0)


def _variance(samples: Any) -> Any:
    return samples.var(dim=0, unbiased=False) if _is_torch(samples) else samples.var(axis=0)


def _covariance(samples: Any) -> Any:
    centered = samples - _mean(samples)
    return _transpose2d(centered) @ centered / max(int(_shape(samples)[0]), 1)


def _transpose2d(value: Any) -> Any:
    return value.transpose(0, 1) if _is_torch(value) else value.T


def _row_norms(samples: Any) -> Any:
    if _is_torch(samples):
        import torch

        return torch.linalg.vector_norm(samples, dim=-1)
    import numpy as np

    return np.linalg.norm(samples, axis=-1)


def _normalize_rows(samples: Any) -> Any:
    norms = _row_norms(samples)
    if _is_torch(samples):
        import torch

        if not bool(torch.all(torch.isfinite(norms)).item()):
            raise ValueError("query samples contain non-finite norms")
        nonzero = norms > 0
        if not bool(torch.any(nonzero).item()):
            raise ValueError("query samples contain no nonzero vector")
        selected = samples[nonzero]
        return selected / norms[nonzero].unsqueeze(-1)
    import numpy as np

    if not bool(np.all(np.isfinite(norms))):
        raise ValueError("query samples contain non-finite norms")
    nonzero = norms > 0
    if not bool(np.any(nonzero)):
        raise ValueError("query samples contain no nonzero vector")
    return samples[nonzero] / norms[nonzero, None]


def _argmax(value: Any) -> int:
    result = value.argmax()
    return int(result.item() if hasattr(result, "item") else result)


def _argmin(value: Any) -> int:
    result = value.argmin()
    return int(result.item() if hasattr(result, "item") else result)


def _stack(values: Sequence[Any]) -> Any:
    if _is_torch(values[0]):
        import torch

        return torch.stack(tuple(values), dim=0)
    import numpy as np

    return np.stack(tuple(values), axis=0)


def _weights(counts: Sequence[int], exemplar: Any) -> Any:
    total = float(sum(counts))
    if _is_torch(exemplar):
        import torch

        return torch.tensor(counts, dtype=torch.float32, device=exemplar.device) / total
    import numpy as np

    return np.asarray(counts, dtype=np.float32) / total


def _deterministic_spherical_kmeans(
    samples: Any, centroid_count: int, iterations: int
) -> tuple[Any, Any]:
    normalized = _normalize_rows(samples)
    count = int(_shape(normalized)[0])
    clusters = min(centroid_count, count)
    chosen = [0]
    while len(chosen) < clusters:
        existing = normalized[chosen]
        similarities = normalized @ _transpose2d(existing)
        closest = (
            similarities.max(dim=1).values if _is_torch(similarities) else similarities.max(axis=1)
        )
        for index in chosen:
            closest[index] = 2.0
        chosen.append(_argmin(closest))
    centroids = _clone(normalized[chosen])
    assignments: Any = None
    counts: list[int] = []
    for _iteration in range(iterations):
        similarities = normalized @ _transpose2d(centroids)
        assignments = (
            similarities.argmax(dim=1) if _is_torch(similarities) else similarities.argmax(axis=1)
        )
        updated: list[Any] = []
        counts = []
        for cluster in range(clusters):
            mask = assignments == cluster
            members = normalized[mask]
            member_count = int(_shape(members)[0])
            counts.append(member_count)
            if member_count:
                updated.append(_normalize_rows(_mean(members).reshape(1, -1))[0])
            else:
                updated.append(centroids[cluster])
        centroids = _stack(updated)
    if assignments is None:  # pragma: no cover - iterations is strictly positive
        raise RuntimeError("centroid assignment did not run")
    return centroids, _weights(counts, centroids)


def _streaming_spherical(sources: Sequence[Any], centroid_count: int) -> tuple[Any, Any]:
    """Single-pass O(KD) auxiliary-memory spherical centroid approximation."""

    sums: list[Any] = []
    centroids: list[Any] = []
    counts: list[int] = []
    for source in sources:
        for index in range(int(_shape(source)[0])):
            row = source[index]
            norm = _row_norms(row.reshape(1, -1))[0]
            norm_value = float(norm.item() if hasattr(norm, "item") else norm)
            if not math.isfinite(norm_value):
                raise ValueError("query samples contain non-finite norms")
            if norm_value == 0.0:
                continue
            sample = row / norm
            if len(centroids) < centroid_count:
                sums.append(_clone(sample))
                centroids.append(_clone(sample))
                counts.append(1)
                continue
            similarities = _stack(centroids) @ sample
            assignment = _argmax(similarities)
            sums[assignment] = sums[assignment] + sample
            counts[assignment] += 1
            centroids[assignment] = _normalize_rows(sums[assignment].reshape(1, -1))[0]
    if not centroids:
        raise ValueError("query samples contain no nonzero vector")
    result = _stack(centroids)
    return result, _weights(counts, result)


def _measure(exemplar: Any, action: Callable[[], ResultT]) -> tuple[ResultT, float, int, str]:
    if _is_torch(exemplar) and str(getattr(exemplar, "device", "cpu")).startswith("cuda"):
        import torch

        audit = SynchronizationAudit()
        timer = CudaEventTimer(torch, exemplar.device, audit)
        timer.start()
        result = action()
        elapsed = timer.stop()
        return result, elapsed, audit.calls, "torch_cuda_event"
    started = perf_counter()
    result = action()
    return result, perf_counter() - started, 0, "host_perf_counter"


@dataclass(slots=True)
class _HeadSamples:
    provenance: HeadForecastProvenance
    prompt: Any | None
    draft: Any | None
    sources: tuple[Any, ...]
    combined: Any | None
    prompt_mean: Any | None = None
    prompt_covariance: Any | None = None
    prompt_variance: Any | None = None


def _flatten_group(value: Any, start: int, end: int, position_slice: slice) -> Any:
    selected = value[:, start:end, position_slice, :]
    return _working_float(selected).reshape(-1, int(_shape(selected)[-1]))


def build_query_forecast(
    prompt_query_layers: Sequence[Any],
    draft_query_layers: Sequence[Any],
    kv_head_counts: Sequence[int],
    config: ForecastingConfig,
    *,
    original_logical_sequence_length: int,
    draft_cache_isolated: bool,
) -> QueryForecast:
    """Build deterministic per-KV-head forecast statistics and centroids."""

    if not config.enabled:
        raise ValueError("cannot build a forecast when forecasting.enabled is false")
    prompt_layers = tuple(prompt_query_layers)
    draft_layers = tuple(draft_query_layers)
    kv_counts = tuple(int(value) for value in kv_head_counts)
    if not prompt_layers:
        raise ValueError("prefill query vectors are required even when prompt_window is zero")
    if len(prompt_layers) != len(kv_counts):
        raise ValueError("prompt query layer count does not match KV cache layers")
    if config.draft_steps and len(draft_layers) != len(prompt_layers):
        raise ValueError("draft query layer count does not match prompt layers")
    if not config.draft_steps and draft_layers:
        raise ValueError("draft query samples were supplied for a zero-step forecast")
    prompt_length = int(_shape(prompt_layers[0])[-2])
    actual_window = min(config.prompt_window, prompt_length)
    exemplar = prompt_layers[0]
    head_samples: list[list[_HeadSamples]] = []

    def prepare_samples() -> None:
        for layer_index, (prompt_layer, kv_heads) in enumerate(
            zip(prompt_layers, kv_counts, strict=True)
        ):
            prompt_shape = _shape(prompt_layer)
            if len(prompt_shape) != 4 or prompt_shape[0] != 1:
                raise ValueError(
                    "prompt queries must have shape [1, query_heads, sequence, head_dim]"
                )
            if prompt_shape[-2] != prompt_length:
                raise ValueError("prompt query layers have inconsistent sequence lengths")
            query_heads = prompt_shape[1]
            if kv_heads <= 0 or query_heads % kv_heads:
                raise ValueError("query-head count must be divisible by KV-head count")
            group_size = query_heads // kv_heads
            draft_layer = draft_layers[layer_index] if draft_layers else None
            if draft_layer is not None:
                draft_shape = _shape(draft_layer)
                if (
                    len(draft_shape) != 4
                    or draft_shape[0] != 1
                    or draft_shape[1] != query_heads
                    or draft_shape[2] != config.draft_steps
                    or draft_shape[3] != prompt_shape[3]
                ):
                    raise ValueError("draft queries do not align with prompt query geometry")
            layer_samples: list[_HeadSamples] = []
            for kv_head in range(kv_heads):
                start = kv_head * group_size
                end = start + group_size
                prompt = (
                    _flatten_group(
                        prompt_layer,
                        start,
                        end,
                        slice(prompt_length - actual_window, None),
                    )
                    if actual_window
                    else None
                )
                draft = (
                    _flatten_group(draft_layer, start, end, slice(None))
                    if draft_layer is not None
                    else None
                )
                available = tuple(item for item in (prompt, draft) if item is not None)
                if not available:
                    raise ValueError("forecast has no mathematically valid query source")
                combined = (
                    None
                    if config.low_memory_centroids
                    else available[0]
                    if len(available) == 1
                    else _concat(available)
                )
                prompt_positions = tuple(range(prompt_length - actual_window, prompt_length))
                layer_samples.append(
                    _HeadSamples(
                        HeadForecastProvenance(
                            layer_index,
                            kv_head,
                            start,
                            end,
                            prompt_positions,
                            0 if prompt is None else int(_shape(prompt)[0]),
                            0 if draft is None else int(_shape(draft)[0]),
                        ),
                        prompt,
                        draft,
                        available,
                        combined,
                    )
                )
            head_samples.append(layer_samples)

    _, preparation_seconds, preparation_syncs, timing_backend = _measure(exemplar, prepare_samples)

    def statistics() -> None:
        for layer in head_samples:
            for samples in layer:
                if samples.prompt is None:
                    continue
                samples.prompt_mean = _mean(samples.prompt)
                if config.covariance is ForecastCovariance.FULL:
                    samples.prompt_covariance = _covariance(samples.prompt)
                else:
                    samples.prompt_variance = _variance(samples.prompt)

    _, statistics_seconds, statistics_syncs, statistics_backend = _measure(exemplar, statistics)
    if statistics_backend != timing_backend:
        raise RuntimeError("forecast phases used inconsistent timing backends")

    centroids: list[list[tuple[Any, Any]]] = []

    def construct_centroids() -> None:
        for layer in head_samples:
            centroid_layer: list[tuple[Any, Any]] = []
            for samples in layer:
                if config.low_memory_centroids:
                    result = _streaming_spherical(samples.sources, config.centroid_count)
                else:
                    if samples.combined is None:  # pragma: no cover - guarded by config
                        raise RuntimeError("exact centroids require materialized query samples")
                    result = _deterministic_spherical_kmeans(
                        samples.combined, config.centroid_count, config.centroid_iterations
                    )
                centroid_layer.append(result)
            centroids.append(centroid_layer)

    _, centroid_seconds, centroid_syncs, centroid_backend = _measure(exemplar, construct_centroids)
    if centroid_backend != timing_backend:
        raise RuntimeError("forecast phases used inconsistent timing backends")
    forecast_layers: list[tuple[KVHeadQueryForecast, ...]] = []
    for layer_samples, layer_centroids in zip(head_samples, centroids, strict=True):
        forecasts = tuple(
            KVHeadQueryForecast(
                prompt_mean=samples.prompt_mean,
                prompt_covariance=samples.prompt_covariance,
                prompt_diagonal_variance=samples.prompt_variance,
                draft_query_samples=samples.draft,
                normalized_centroids=centroid[0],
                forecast_weights=centroid[1],
                provenance=samples.provenance,
            )
            for samples, centroid in zip(layer_samples, layer_centroids, strict=True)
        )
        forecast_layers.append(forecasts)
    method = (
        "streaming_spherical" if config.low_memory_centroids else "deterministic_spherical_kmeans"
    )
    timing = ForecastTiming(
        cache_clone=0.0,
        draft_decode=0.0,
        query_preparation=preparation_seconds,
        prompt_statistics=statistics_seconds,
        centroid_construction=centroid_seconds,
        total=preparation_seconds + statistics_seconds + centroid_seconds,
        synchronization_calls=preparation_syncs + statistics_syncs + centroid_syncs,
        timing_backend=timing_backend,
    )
    provenance = ForecastProvenance(
        mode=config.mode,
        requested_prompt_window=config.prompt_window,
        actual_prompt_window=actual_window,
        requested_draft_steps=config.draft_steps,
        completed_draft_steps=config.draft_steps,
        covariance=config.covariance,
        centroid_method=method,
        centroid_count_requested=config.centroid_count,
        deterministic_greedy=True,
        reused_original_prefill=True,
        draft_cache_isolated=draft_cache_isolated or config.draft_steps == 0,
        original_logical_sequence_length=original_logical_sequence_length,
    )
    return QueryForecast(tuple(forecast_layers), provenance, timing)


__all__ = ["build_query_forecast"]
