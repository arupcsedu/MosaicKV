"""Hugging Face bridge for isolated deterministic draft-query rollouts."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from mosaickv.adapters.huggingface.types import DecodeState, PrefillOutput
from mosaickv.config import ForecastingConfig
from mosaickv.forecasting.core import _measure, build_query_forecast
from mosaickv.forecasting.types import ForecastTiming, QueryForecast


def _clone_value(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    clone = getattr(value, "clone", None)
    if detach is not None and clone is not None:
        return detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    return copy.deepcopy(value)


def _tensor_equal(first: Any, second: Any) -> bool:
    equal = getattr(first, "equal", None)
    if equal is not None:
        return bool(equal(second))
    try:
        import numpy as np

        return bool(np.array_equal(first, second, equal_nan=True))
    except (TypeError, ValueError):
        comparison = first == second
        all_method = getattr(comparison, "all", None)
        return bool(all_method() if all_method is not None else comparison)


def _scalar_token_id(value: Any) -> int:
    detach = getattr(value, "detach", None)
    if detach is not None:
        value = detach().cpu()
    reshape = getattr(value, "reshape", None)
    tolist = getattr(value, "tolist", None)
    if reshape is None or tolist is None:
        raise TypeError("greedy token must support reshape and tolist")
    items = reshape(-1).tolist()
    if len(items) != 1:
        raise RuntimeError("draft greedy token is not scalar")
    return int(items[0])


def _snapshot_equal(first: Any, second: Any) -> bool:
    if (
        first.source_class is not second.source_class
        or first.source_kind != second.source_kind
        or first.active_sequence_length != second.active_sequence_length
        or first.cached_key_state != second.cached_key_state
        or len(first.layers) != len(second.layers)
    ):
        return False
    return all(
        left.sequence_dimension == right.sequence_dimension
        and _tensor_equal(left.key, right.key)
        and _tensor_equal(left.value, right.value)
        for left, right in zip(first.layers, second.layers, strict=True)
    )


def _clone_decode_state(adapter: Any, state: DecodeState) -> DecodeState:
    snapshot = adapter.extract_past_key_values(state.past_key_values)
    return DecodeState(
        past_key_values=adapter.inject_past_key_values(snapshot),
        attention_mask=_clone_value(state.attention_mask),
        active_cache_length=state.active_cache_length,
        logical_sequence_length=state.logical_sequence_length,
        next_decode_position=state.next_decode_position,
        modality_map=state.modality_map,
        model_state=_clone_value(state.model_state),
    )


def _concatenate_query_steps(steps: list[tuple[Any, ...]]) -> tuple[Any, ...]:
    if not steps:
        return ()
    layer_count = len(steps[0])
    if any(len(step) != layer_count for step in steps):
        raise RuntimeError("draft query capture returned inconsistent layer counts")
    first = steps[0][0]
    if type(first).__module__.startswith("torch"):
        import torch

        return tuple(
            torch.cat(tuple(step[layer] for step in steps), dim=-2) for layer in range(layer_count)
        )
    import numpy as np

    return tuple(
        np.concatenate(tuple(step[layer] for step in steps), axis=-2)
        for layer in range(layer_count)
    )


@dataclass(frozen=True, slots=True)
class IsolatedQueryRollout:
    """Temporary greedy tokens and queries collected from a cloned prefill state."""

    query_layers: tuple[Any, ...]
    attention_steps: tuple[tuple[Any, ...], ...]
    token_ids: tuple[int, ...]
    kv_head_counts: tuple[int, ...]
    cache_clone_seconds: float
    draft_decode_seconds: float
    synchronization_calls: int
    timing_backend: str
    original_cache_unchanged: bool

    def __post_init__(self) -> None:
        if not self.query_layers and self.token_ids:
            raise ValueError("rollout tokens require captured query layers")
        if self.cache_clone_seconds < 0 or self.draft_decode_seconds < 0:
            raise ValueError("rollout timings must be nonnegative")
        if not self.kv_head_counts or any(count <= 0 for count in self.kv_head_counts):
            raise ValueError("rollout must record a positive KV-head count for every layer")
        if not self.original_cache_unchanged:
            raise ValueError("isolated query rollout changed the original cache")


def _kv_head_counts(snapshot: Any) -> tuple[int, ...]:
    counts: list[int] = []
    for index, layer in enumerate(snapshot.layers):
        shape = tuple(int(item) for item in layer.key.shape)
        if len(shape) < 3:
            raise RuntimeError(f"cache layer {index} is too small to expose a KV-head axis")
        count = shape[-3]
        if count <= 0:
            raise RuntimeError(f"cache layer {index} has no KV heads")
        counts.append(count)
    return tuple(counts)


def run_isolated_greedy_query_rollout(
    adapter: Any,
    prefill: PrefillOutput,
    *,
    steps: int,
    capture_attentions: bool = False,
) -> IsolatedQueryRollout:
    """Collect future queries without another prefill or shared mutable cache state."""

    if steps < 0:
        raise ValueError("rollout steps must be nonnegative")
    if not prefill.query_vectors.layers:
        raise ValueError("prefill must capture query vectors before forecasting")
    exemplar = prefill.query_vectors.layers[0]
    if steps == 0:
        snapshot, clone_seconds, clone_syncs, backend = _measure(
            exemplar,
            lambda: adapter.extract_past_key_values(prefill.state.past_key_values),
        )
        return IsolatedQueryRollout(
            (),
            (),
            (),
            _kv_head_counts(snapshot),
            clone_seconds,
            0.0,
            clone_syncs,
            backend,
            True,
        )

    def clone_phase() -> tuple[Any, DecodeState, Any, tuple[int, ...]]:
        source_snapshot = adapter.extract_past_key_values(prefill.state.past_key_values)
        draft_state = _clone_decode_state(adapter, prefill.state)
        original_attention = _clone_value(prefill.state.attention_mask)
        return source_snapshot, draft_state, original_attention, _kv_head_counts(source_snapshot)

    (
        (source_snapshot, draft_state, original_attention, kv_head_counts),
        clone_seconds,
        clone_syncs,
        backend,
    ) = _measure(exemplar, clone_phase)
    query_steps: list[tuple[Any, ...]] = []
    attention_steps: list[tuple[Any, ...]] = []
    tokens: list[int] = []

    def draft_phase() -> None:
        nonlocal draft_state
        token = _clone_value(prefill.next_token_id)
        for _step in range(steps):
            decode_kwargs = {"capture_queries": True}
            if capture_attentions:
                decode_kwargs["capture_attentions"] = True
            output = adapter.decode_one_token(token, draft_state, **decode_kwargs)
            if not output.query_vectors.layers:
                raise RuntimeError("draft decode did not capture query vectors")
            query_steps.append(output.query_vectors.layers)
            if capture_attentions and not output.attention_weights:
                raise RuntimeError("draft decode did not capture attention probabilities")
            if capture_attentions:
                attention_steps.append(output.attention_weights)
            tokens.append(_scalar_token_id(token))
            token = output.next_token_id
            draft_state = output.state

    _, draft_seconds, draft_syncs, draft_backend = _measure(exemplar, draft_phase)
    if draft_backend != backend:
        raise RuntimeError("draft phases used inconsistent timing backends")

    def integrity_phase() -> bool:
        after = adapter.extract_past_key_values(prefill.state.past_key_values)
        return _snapshot_equal(source_snapshot, after) and _tensor_equal(
            original_attention, prefill.state.attention_mask
        )

    unchanged, integrity_seconds, integrity_syncs, integrity_backend = _measure(
        exemplar, integrity_phase
    )
    if integrity_backend != backend:
        raise RuntimeError("draft integrity check used an inconsistent timing backend")
    if not unchanged:
        raise RuntimeError("draft rollout mutated the original prefill cache or attention mask")
    return IsolatedQueryRollout(
        query_layers=_concatenate_query_steps(query_steps),
        attention_steps=tuple(attention_steps),
        token_ids=tuple(tokens),
        kv_head_counts=kv_head_counts,
        cache_clone_seconds=clone_seconds + integrity_seconds,
        draft_decode_seconds=draft_seconds,
        synchronization_calls=clone_syncs + draft_syncs + integrity_syncs,
        timing_backend=backend,
        original_cache_unchanged=True,
    )


def forecast_with_rollout(
    adapter: Any,
    prefill: PrefillOutput,
    config: ForecastingConfig,
    *,
    capture_attentions: bool = False,
) -> tuple[QueryForecast, IsolatedQueryRollout]:
    """Return a forecast and its one isolated rollout without a second prefill."""

    if not config.enabled:
        raise ValueError("forecasting is disabled")
    rollout = run_isolated_greedy_query_rollout(
        adapter,
        prefill,
        steps=config.draft_steps,
        capture_attentions=capture_attentions,
    )
    forecast = build_query_forecast(
        prefill.query_vectors.layers,
        rollout.query_layers,
        rollout.kv_head_counts,
        config,
        original_logical_sequence_length=prefill.state.logical_sequence_length,
        draft_cache_isolated=rollout.original_cache_unchanged,
    )
    timing = ForecastTiming(
        cache_clone=rollout.cache_clone_seconds,
        draft_decode=rollout.draft_decode_seconds,
        query_preparation=forecast.timing.query_preparation,
        prompt_statistics=forecast.timing.prompt_statistics,
        centroid_construction=forecast.timing.centroid_construction,
        total=(
            rollout.cache_clone_seconds
            + rollout.draft_decode_seconds
            + forecast.timing.query_preparation
            + forecast.timing.prompt_statistics
            + forecast.timing.centroid_construction
        ),
        synchronization_calls=(
            rollout.synchronization_calls + forecast.timing.synchronization_calls
        ),
        timing_backend=(
            forecast.timing.timing_backend if config.draft_steps == 0 else rollout.timing_backend
        ),
    )
    if config.draft_steps and rollout.timing_backend != forecast.timing.timing_backend:
        raise RuntimeError("draft and forecast construction used inconsistent timing backends")
    return replace(forecast, timing=timing), rollout


def forecast_from_prefill(
    adapter: Any,
    prefill: PrefillOutput,
    config: ForecastingConfig,
) -> QueryForecast:
    """Forecast from one existing prefill; this function never calls ``prefill`` itself."""

    forecast, _rollout = forecast_with_rollout(adapter, prefill, config)
    return forecast


__all__ = [
    "IsolatedQueryRollout",
    "forecast_from_prefill",
    "forecast_with_rollout",
    "run_isolated_greedy_query_rollout",
]
