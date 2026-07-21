from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from mosaickv.adapters.huggingface import AdapterCapabilities, CachedKeyState, QueryVectorState
from mosaickv.backends import build_compression_plan
from mosaickv.cache_state import FullKVState
from mosaickv.config import (
    CacheConfig,
    ForecastingConfig,
    ResidualConfig,
    RunConfig,
    SelectionConfig,
    UtilityConfig,
    synthetic_smoke_config,
)
from mosaickv.forecasting import QueryForecast, build_query_forecast
from mosaickv.types import BudgetUnit, ForecastMode, MosaicKVMethod


def _full() -> FullKVState:
    key = np.arange(8 * 4, dtype=np.float32).reshape(1, 1, 8, 4) + 1
    value = np.flip(key, axis=-1).copy()
    return FullKVState.from_tensors(
        ((key, value),),
        block_size=1,
        mandatory_logical_positions=(0, 7),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
    )


def _forecast() -> QueryForecast:
    queries = np.arange(1 * 2 * 8 * 4, dtype=np.float32).reshape(1, 2, 8, 4) + 1
    return build_query_forecast(
        (queries,),
        (),
        (1,),
        ForecastingConfig(
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=4,
            draft_steps=0,
            centroid_count=2,
        ),
        original_logical_sequence_length=8,
        draft_cache_isolated=True,
    )


def _capabilities(*, safe: bool) -> AdapterCapabilities:
    return AdapterCapabilities(
        model_family="synthetic",
        architectures=("Synthetic",),
        attention_implementations=("eager",),
        image=True,
        multi_image=True,
        video=True,
        cache_classes=("tuple",),
        cache_sequence_dimension=-2,
        cached_key_state=(CachedKeyState.NOT_APPLICABLE if safe else CachedKeyState.POST_ROPE),
        query_vector_state=QueryVectorState.Q_PROJ_PRE_ROPE,
        supports_prototype_merge=safe,
        supports_residual_repair=safe,
    )


def _config(method: MosaicKVMethod, ratio: float) -> RunConfig:
    return replace(
        synthetic_smoke_config(),
        method=method,
        cache=CacheConfig(8, BudgetUnit.BLOCKS, ratio, 1),
        forecasting=ForecastingConfig(
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=4,
            draft_steps=0,
            centroid_count=2,
        ),
        utility=UtilityConfig(lambda_q=1.0, lambda_v=0.0, lambda_o=0.0),
        selection=SelectionConfig(
            lambda_g=-0.25,
            lambda_m=-0.25,
            stop_on_nonpositive_gain=False,
        ),
        residual=ResidualConfig(require_pinned_memory=False),
    )


@pytest.mark.parametrize(
    "method",
    (
        MosaicKVMethod.MOSAICKV_EXACT,
        MosaicKVMethod.MOSAICKV_PROTO,
        MosaicKVMethod.MOSAICKV_FULL,
    ),
)
def test_tiny_synthetic_cache_plans_every_runtime_method(method: MosaicKVMethod) -> None:
    full = _full()
    plan = build_compression_plan(
        full,
        _forecast(),
        {node_id: float(8 - node_id) for node_id in range(8)},
        _capabilities(safe=True),
        _config(method, 0.5),
    )

    assert plan.method is method
    assert plan.construction.state.statistics.active_kv_bytes <= full.active_bytes
    assert plan.selection.budget_spent <= plan.active_budget_value
    assert {0, 7} <= set(plan.selection.selected_node_ids)


def test_post_rope_proto_and_full_are_explicit_exact_safety_fallbacks() -> None:
    full = _full()
    for method in (MosaicKVMethod.MOSAICKV_PROTO, MosaicKVMethod.MOSAICKV_FULL):
        plan = build_compression_plan(
            full,
            _forecast(),
            {node_id: float(8 - node_id) for node_id in range(8)},
            _capabilities(safe=False),
            _config(method, 0.5),
        )
        assert plan.effective_method.endswith("mosaickv_exact_safety_fallback")
        assert not plan.construction.prototypes
        assert not plan.construction.residual_storage.payloads


def test_retention_one_is_exact_and_active_bytes_are_monotonic() -> None:
    full = _full()
    active: list[int] = []
    for ratio in (0.25, 0.5, 0.75, 1.0):
        plan = build_compression_plan(
            full,
            _forecast(),
            {node_id: 1.0 for node_id in range(8)},
            _capabilities(safe=False),
            _config(MosaicKVMethod.MOSAICKV_EXACT, ratio),
        )
        active.append(plan.construction.state.statistics.active_kv_bytes)
        if ratio == 1.0:
            plan.construction.state.reconstruct_full_state(full)
            assert not plan.construction.prototypes
            assert not plan.construction.residual_storage.payloads
    assert active == sorted(active)
