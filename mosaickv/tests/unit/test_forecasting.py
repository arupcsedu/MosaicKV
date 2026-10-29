from __future__ import annotations

from dataclasses import replace

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
from mosaickv.forecasting import (
    build_query_forecast,
    forecast_from_prefill,
    forecast_with_rollout,
)
from mosaickv.forecasting.core import (
    _batched_numpy_streaming_spherical,
    _normalize_rows,
    _streaming_spherical,
)
from mosaickv.forecasting.huggingface import (
    _prompt_directional_uncertainty,
    run_isolated_greedy_query_rollout,
)
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
        self.decode_calls = 0

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

    def get_cache_layout(self, cache: CacheSnapshot) -> object:
        from types import SimpleNamespace

        return SimpleNamespace(
            layers=tuple(SimpleNamespace(key_shape=layer.key.shape) for layer in cache.layers)
        )

    def decode_one_token(
        self, token: np.ndarray, state: DecodeState, *, capture_queries: bool
    ) -> DecodeOutput:
        assert capture_queries
        self.decode_calls += 1
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


def test_conditional_draft_skips_low_uncertainty_without_cache_clone_or_decode() -> None:
    adapter = _DraftIsolationAdapter()
    aligned = np.ones((1, 4, 4, 3), dtype=np.float32)
    prefill = replace(_fake_prefill(), query_vectors=QueryVectors((aligned,)))

    forecast, rollout = forecast_with_rollout(
        adapter,
        prefill,
        ForecastingConfig(
            mode=ForecastMode.HYBRID,
            prompt_window=4,
            draft_steps=3,
            centroid_count=2,
            conditional_draft_enabled=True,
            conditional_draft_uncertainty_threshold=0.5,
        ),
        capture_attentions=False,
    )

    assert adapter.decode_calls == 0
    assert rollout.token_ids == ()
    assert rollout.cache_clone_seconds == 0
    assert forecast.provenance.mode is ForecastMode.PROMPT_WINDOW
    assert forecast.provenance.configured_mode is ForecastMode.HYBRID
    assert forecast.provenance.requested_draft_steps == 3
    assert forecast.provenance.completed_draft_steps == 0
    assert forecast.provenance.conditional_draft_decision == ("skipped_low_prompt_uncertainty")
    assert forecast.provenance.prompt_uncertainty == pytest.approx(0.0, abs=1e-6)


def test_zero_step_prompt_forecast_reads_layout_without_cloning_or_decoding(monkeypatch):
    adapter = _DraftIsolationAdapter()
    prefill = _fake_prefill()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("zero-step rollout must not clone KV")

    monkeypatch.setattr(adapter, "extract_past_key_values", forbidden)
    before = tuple(layer.key.copy() for layer in prefill.state.past_key_values.layers)
    forecast, rollout = forecast_with_rollout(
        adapter,
        prefill,
        ForecastingConfig(mode=ForecastMode.PROMPT_WINDOW, prompt_window=4, draft_steps=0),
    )
    assert rollout.cache_clone_seconds == 0 and rollout.draft_decode_seconds == 0
    assert rollout.metadata_seconds >= 0 and rollout.kv_head_counts == (1,)
    assert adapter.decode_calls == 0 and rollout.original_cache_unchanged
    assert forecast.timing.cache_clone == 0
    assert forecast.timing.query_preparation >= rollout.metadata_seconds
    assert all(
        np.array_equal(x, layer.key)
        for x, layer in zip(before, prefill.state.past_key_values.layers, strict=True)
    )


def test_conditional_draft_executes_for_high_prompt_uncertainty() -> None:
    adapter = _DraftIsolationAdapter()
    opposed = np.zeros((1, 4, 4, 3), dtype=np.float32)
    opposed[:, :, ::2, 0] = 1
    opposed[:, :, 1::2, 0] = -1
    prefill = replace(_fake_prefill(), query_vectors=QueryVectors((opposed,)))

    forecast, rollout = forecast_with_rollout(
        adapter,
        prefill,
        ForecastingConfig(
            mode=ForecastMode.HYBRID,
            prompt_window=4,
            draft_steps=2,
            centroid_count=2,
            conditional_draft_enabled=True,
            conditional_draft_uncertainty_threshold=0.5,
        ),
        capture_distributions=True,
    )

    assert adapter.decode_calls == 2
    assert rollout.token_ids == (2, 3)
    assert len(rollout.draft_distributions) == 2
    assert np.allclose(rollout.draft_distributions[0].sum(axis=-1), 1.0)
    assert forecast.provenance.mode is ForecastMode.HYBRID
    assert forecast.provenance.completed_draft_steps == 2
    assert forecast.provenance.conditional_draft_decision == ("executed_high_prompt_uncertainty")
    assert forecast.provenance.prompt_uncertainty == pytest.approx(1.0, abs=1e-6)


def test_prompt_directional_uncertainty_is_scale_invariant_and_bounded() -> None:
    values = _queries(layers=1, sequence=4)
    original = _prompt_directional_uncertainty(values, 3)
    scaled = _prompt_directional_uncertainty((values[0] * 17.0,), 3)

    assert 0 <= original <= 1
    assert scaled == pytest.approx(original, abs=1e-7)


def test_evaluation_rollout_does_not_require_prompt_queries() -> None:
    adapter = _DraftIsolationAdapter()
    prefill = replace(_fake_prefill(), query_vectors=QueryVectors(()))

    rollout = run_isolated_greedy_query_rollout(adapter, prefill, steps=2)

    assert rollout.token_ids == (2, 3)
    assert rollout.draft_distributions == ()
    assert rollout.original_cache_unchanged


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


def test_low_memory_centroid_fast_path_is_bitwise_reference_equivalent() -> None:
    rng = np.random.default_rng(17)
    sources = tuple(rng.normal(size=(rows, 9)).astype(np.float32) for rows in (7, 3, 1))

    sums: list[np.ndarray] = []
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    for source in sources:
        for row in source:
            norm = np.linalg.norm(row.reshape(1, -1), axis=-1)[0]
            sample = row / norm
            if len(centroids) < 3:
                sums.append(sample.copy())
                centroids.append(sample.copy())
                counts.append(1)
                continue
            assignment = int(np.argmax(np.stack(centroids, axis=0) @ sample))
            sums[assignment] = sums[assignment] + sample
            counts[assignment] += 1
            centroids[assignment] = _normalize_rows(sums[assignment].reshape(1, -1))[0]
    expected_centroids = np.stack(centroids, axis=0)
    expected_weights = np.asarray(counts, dtype=np.float32) / float(sum(counts))

    actual_centroids, actual_weights = _streaming_spherical(sources, 3)

    assert np.array_equal(actual_centroids, expected_centroids)
    assert np.array_equal(actual_weights, expected_weights)


def test_layer_batched_centroids_are_bitwise_per_head_equivalent() -> None:
    rng = np.random.default_rng(31)
    samples = rng.normal(size=(12, 24, 17)).astype(np.float32)
    expected = tuple(
        _streaming_spherical((samples[head_index],), 4) for head_index in range(samples.shape[0])
    )

    actual = _batched_numpy_streaming_spherical(samples, 4)

    assert len(actual) == len(expected)
    for (actual_centroids, actual_weights), (expected_centroids, expected_weights) in zip(
        actual,
        expected,
        strict=True,
    ):
        assert np.array_equal(actual_centroids, expected_centroids)
        assert np.array_equal(actual_weights, expected_weights)


def test_layer_batched_centroids_fall_back_exactly_for_zero_queries() -> None:
    samples = np.ones((3, 8, 5), dtype=np.float32)
    samples[1, 2] = 0
    expected = tuple(
        _streaming_spherical((samples[head_index],), 3) for head_index in range(samples.shape[0])
    )

    actual = _batched_numpy_streaming_spherical(samples, 3)

    for actual_head, expected_head in zip(actual, expected, strict=True):
        assert np.array_equal(actual_head[0], expected_head[0])
        assert np.array_equal(actual_head[1], expected_head[1])


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
        (
            {
                "mode": ForecastMode.PROMPT_WINDOW,
                "prompt_window": 2,
                "draft_steps": 0,
                "conditional_draft_enabled": True,
            },
            "conditional_draft_enabled requires hybrid",
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


def test_forecast_diagnostic_can_score_actual_selector_blocks() -> None:
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
        identity: AttentionDiagnosticInput(
            predicted_attention=np.asarray([0.8, 0.1, 0.05, 0.05]),
            true_attention=np.asarray([0.7, 0.2, 0.05, 0.05]),
            block_positions=((0, 1), (2, 3)),
            selected_block_count=1,
            selected_block_indices=(1,),
        )
        for identity in ((0, 0), (0, 1))
    }

    diagnostics = evaluate_forecast_quality(forecast, oracle, attention)

    assert diagnostics.mean_selected_block_recall == 0.0
    assert diagnostics.mean_forecast_regret > 0
