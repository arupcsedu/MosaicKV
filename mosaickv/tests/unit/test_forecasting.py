from __future__ import annotations

import numpy as np
import pytest

import mosaickv.forecasting as online_forecasting
from mosaickv.adapters.huggingface.types import (
    CachedKeyState,
    CacheLayerSnapshot,
    CacheSnapshot,
    DecodeOutput,
    DecodeState,
    PrefillOutput,
    QueryVectors,
)
from mosaickv.config import ConfigurationError, ForecastingConfig
from mosaickv.evaluation.forecast_diagnostics import (
    AttentionDiagnosticInput,
    evaluate_forecast_quality,
)
from mosaickv.evaluation.oracle_queries import EvaluationOnlyOracleQueries
from mosaickv.forecasting import build_query_forecast, forecast_from_prefill
from mosaickv.types import ForecastCovariance, ForecastMode


def _queries(*, layers: int = 2, sequence: int = 6) -> tuple[np.ndarray, ...]:
    return tuple(
        np.arange(1 * 4 * sequence * 3, dtype=np.float32).reshape(1, 4, sequence, 3)
        + 1
        + layer * 100
        for layer in range(layers)
    )


class _DraftIsolationAdapter:
    def __init__(self) -> None:
        self.prefill_calls = 0

    def prefill(self, *_args: object, **_kwargs: object) -> None:
        self.prefill_calls += 1
        raise AssertionError("forecasting performed a forbidden second prefill")

    def extract_past_key_values(self, value: CacheSnapshot) -> CacheSnapshot:
        return CacheSnapshot(
            tuple(
                CacheLayerSnapshot(layer.key.copy(), layer.value.copy(), layer.sequence_dimension)
                for layer in value.layers
            ),
            value.source_class,
            value.source_kind,
            value.active_sequence_length,
            value.cached_key_state,
        )

    def inject_past_key_values(self, snapshot: CacheSnapshot) -> CacheSnapshot:
        return snapshot

    def decode_one_token(
        self, token: np.ndarray, state: DecodeState, *, capture_queries: bool
    ) -> DecodeOutput:
        assert capture_queries
        snapshot = state.past_key_values
        assert isinstance(snapshot, CacheSnapshot)
        # Deliberately mutate the supplied draft cache to prove it is not shared.
        snapshot.layers[0].key[...] += 1
        query = np.full((1, 4, 1, 3), float(token.reshape(-1)[0]), dtype=np.float32)
        updated = DecodeState(
            snapshot,
            np.concatenate((state.attention_mask, np.ones((1, 1), dtype=np.int64)), axis=-1),
            state.active_cache_length + 1,
            state.logical_sequence_length + 1,
            state.next_decode_position + 1,
            state.modality_map,
            dict(state.model_state),
        )
        next_token = token + 1
        return DecodeOutput(
            np.zeros((1, 8), dtype=np.float32),
            next_token,
            updated,
            QueryVectors((query,)),
        )


def _fake_prefill() -> PrefillOutput:
    key = np.arange(12, dtype=np.float32).reshape(1, 1, 4, 3)
    snapshot = CacheSnapshot(
        (CacheLayerSnapshot(key, key.copy(), -2),),
        tuple,
        "tuple",
        4,
        CachedKeyState.POST_ROPE,
    )
    state = DecodeState(snapshot, np.ones((1, 4), dtype=np.int64), 4, 4, 4, ())
    return PrefillOutput(
        np.zeros((1, 8), dtype=np.float32),
        np.asarray([[2]], dtype=np.int64),
        state,
        QueryVectors(_queries(layers=1, sequence=4)),
    )


def test_prompt_window_statistics_are_per_kv_head() -> None:
    prompt = _queries(layers=1, sequence=5)
    config = ForecastingConfig(
        mode=ForecastMode.PROMPT_WINDOW,
        prompt_window=2,
        draft_steps=0,
        centroid_count=3,
        covariance=ForecastCovariance.DIAGONAL,
    )
    forecast = build_query_forecast(
        prompt,
        (),
        (2,),
        config,
        original_logical_sequence_length=5,
        draft_cache_isolated=True,
    )

    first = forecast.for_head(0, 0)
    expected = prompt[0][:, 0:2, -2:, :].reshape(-1, 3)
    assert first.prompt_mean is not None
    assert first.prompt_diagonal_variance is not None
    assert np.array_equal(first.prompt_mean, expected.mean(axis=0))
    assert np.array_equal(first.prompt_diagonal_variance, expected.var(axis=0))
    assert first.prompt_covariance is None
    assert first.draft_query_samples is None
    assert first.provenance.query_head_start == 0
    assert first.provenance.query_head_end == 2
    assert first.provenance.prompt_positions == (3, 4)
    assert np.allclose(np.linalg.norm(first.normalized_centroids, axis=-1), 1.0)
    assert np.isclose(first.forecast_weights.sum(), 1.0)


def test_draft_only_allows_zero_prompt_window() -> None:
    prompt = _queries(layers=1)
    draft = _queries(layers=1, sequence=3)
    config = ForecastingConfig(
        mode=ForecastMode.DRAFT_ROLLOUT,
        prompt_window=0,
        draft_steps=3,
        centroid_count=2,
        covariance=ForecastCovariance.FULL,
    )
    forecast = build_query_forecast(
        prompt,
        draft,
        (2,),
        config,
        original_logical_sequence_length=6,
        draft_cache_isolated=True,
    )

    head = forecast.for_head(0, 1)
    assert head.prompt_mean is None
    assert head.prompt_covariance is None
    assert head.prompt_diagonal_variance is None
    assert head.draft_query_samples is not None
    assert head.draft_query_samples.shape == (6, 3)
    assert forecast.provenance.actual_prompt_window == 0
    assert forecast.provenance.completed_draft_steps == 3


def test_draft_forecast_reuses_prefill_and_does_not_mutate_original_cache() -> None:
    adapter = _DraftIsolationAdapter()
    prefill = _fake_prefill()
    original_key = prefill.state.past_key_values.layers[0].key.copy()
    forecast = forecast_from_prefill(
        adapter,
        prefill,
        ForecastingConfig(
            mode=ForecastMode.HYBRID,
            prompt_window=2,
            draft_steps=3,
            centroid_count=2,
        ),
    )

    assert adapter.prefill_calls == 0
    assert np.array_equal(prefill.state.past_key_values.layers[0].key, original_key)
    assert forecast.provenance.reused_original_prefill
    assert forecast.provenance.draft_cache_isolated
    draft_samples = forecast.for_head(0, 0).draft_query_samples
    assert draft_samples is not None
    assert draft_samples.shape == (12, 3)


def test_hybrid_full_covariance_and_centroids_are_reproducible() -> None:
    prompt = _queries()
    draft = _queries(sequence=4)
    config = ForecastingConfig(
        mode=ForecastMode.HYBRID,
        prompt_window=4,
        draft_steps=4,
        centroid_count=3,
        covariance=ForecastCovariance.FULL,
        low_memory_centroids=False,
        centroid_iterations=5,
    )
    first = build_query_forecast(
        prompt,
        draft,
        (2, 2),
        config,
        original_logical_sequence_length=6,
        draft_cache_isolated=True,
    )
    second = build_query_forecast(
        prompt,
        draft,
        (2, 2),
        config,
        original_logical_sequence_length=6,
        draft_cache_isolated=True,
    )

    assert first.provenance.centroid_method == "deterministic_spherical_kmeans"
    for first_layer, second_layer in zip(first.layers, second.layers, strict=True):
        for left, right in zip(first_layer, second_layer, strict=True):
            assert left.prompt_covariance is not None
            assert left.prompt_covariance.shape == (3, 3)
            assert np.array_equal(left.prompt_covariance, left.prompt_covariance.T)
            assert np.array_equal(left.normalized_centroids, right.normalized_centroids)
            assert np.array_equal(left.forecast_weights, right.forecast_weights)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"mode": ForecastMode.PROMPT_WINDOW, "prompt_window": 0, "draft_steps": 0},
            "prompt_window mode",
        ),
        (
            {"mode": ForecastMode.PROMPT_WINDOW, "prompt_window": 2, "draft_steps": 1},
            "prompt_window mode",
        ),
        (
            {"mode": ForecastMode.DRAFT_ROLLOUT, "prompt_window": 1, "draft_steps": 2},
            "draft_rollout mode",
        ),
        (
            {"mode": ForecastMode.DRAFT_ROLLOUT, "prompt_window": 0, "draft_steps": 0},
            "draft_rollout mode",
        ),
        (
            {"mode": ForecastMode.HYBRID, "prompt_window": 2, "draft_steps": 0},
            "hybrid forecasting",
        ),
    ],
)
def test_zero_windows_and_rollouts_fail_when_mode_is_not_valid(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        ForecastingConfig(**kwargs)  # type: ignore[arg-type]


def test_evaluation_only_oracle_is_not_exported_by_online_package() -> None:
    assert not hasattr(online_forecasting, "EvaluationOnlyOracleQueries")
    assert not hasattr(online_forecasting, "collect_evaluation_only_true_future_queries")


def test_forecast_quality_diagnostics_report_all_requested_metrics() -> None:
    prompt = _queries(layers=1, sequence=4)
    forecast = build_query_forecast(
        prompt,
        (),
        (2,),
        ForecastingConfig(
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=2,
            draft_steps=0,
            centroid_count=2,
        ),
        original_logical_sequence_length=4,
        draft_cache_isolated=True,
    )
    oracle = EvaluationOnlyOracleQueries(
        query_layers=(prompt[0][:, :, -2:, :].copy(),),
        greedy_token_ids=(8, 9),
        future_steps=2,
    )
    attention = {
        (0, 0): AttentionDiagnosticInput(
            predicted_attention=np.asarray([0.5, 0.3, 0.1, 0.1]),
            true_attention=np.asarray([0.4, 0.4, 0.1, 0.1]),
            block_positions=((0, 1), (2, 3)),
            selected_block_count=1,
        ),
        (0, 1): AttentionDiagnosticInput(
            predicted_attention=np.asarray([0.1, 0.2, 0.3, 0.4]),
            true_attention=np.asarray([0.4, 0.3, 0.2, 0.1]),
            block_positions=((0, 1), (2, 3)),
            selected_block_count=1,
        ),
    }
    diagnostics = evaluate_forecast_quality(forecast, oracle, attention)

    assert len(diagnostics.heads) == 2
    assert diagnostics.heads[0].attention_rank_correlation > 0
    assert diagnostics.heads[0].selected_block_recall == 1.0
    assert diagnostics.heads[0].forecast_regret == 0.0
    assert diagnostics.heads[1].attention_rank_correlation < 0
    assert diagnostics.heads[1].selected_block_recall == 0.0
    assert diagnostics.heads[1].forecast_regret > 0
    assert -1.0 <= diagnostics.mean_cosine_similarity <= 1.0
    assert diagnostics.oracle_source.startswith("evaluation_only")
