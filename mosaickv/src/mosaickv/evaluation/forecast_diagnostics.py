"""Evaluation-only diagnostics comparing forecasts with FullKV future queries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from mosaickv.evaluation.oracle_queries import EvaluationOnlyOracleQueries
from mosaickv.forecasting import QueryForecast

HeadId = tuple[int, int]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _normalize_rows(value: np.ndarray) -> np.ndarray:
    if value.ndim != 2:
        raise ValueError("diagnostic query arrays must be rank two")
    norms = np.linalg.norm(value, axis=-1)
    nonzero = norms > 0
    if not np.any(nonzero):
        raise ValueError("diagnostic query arrays contain no nonzero vectors")
    normalized: np.ndarray = value[nonzero] / norms[nonzero, None]
    return normalized


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.ndim != 1 or not len(first):
        raise ValueError("attention rankings must be aligned non-empty vectors")
    first_rank = _rankdata(first)
    second_rank = _rankdata(second)
    first_centered = first_rank - first_rank.mean()
    second_centered = second_rank - second_rank.mean()
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator == 0.0:
        return 1.0 if np.array_equal(first_rank, second_rank) else 0.0
    return float(np.dot(first_centered, second_centered) / denominator)


@dataclass(frozen=True, slots=True)
class AttentionDiagnosticInput:
    """RoPE-aware predicted/true attention probabilities and block membership."""

    predicted_attention: Any
    true_attention: Any
    block_positions: tuple[tuple[int, ...], ...]
    selected_block_count: int

    def __post_init__(self) -> None:
        predicted = _to_numpy(self.predicted_attention).reshape(-1)
        true = _to_numpy(self.true_attention).reshape(-1)
        if predicted.shape != true.shape or not len(predicted):
            raise ValueError("predicted and true attention must be aligned non-empty vectors")
        if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(true)):
            raise ValueError("attention diagnostics require finite values")
        if np.any(predicted < 0) or np.any(true < 0):
            raise ValueError("attention diagnostics require nonnegative attention values")
        if not self.block_positions or self.selected_block_count <= 0:
            raise ValueError("attention diagnostics require blocks and a positive selection count")
        if self.selected_block_count > len(self.block_positions):
            raise ValueError("selected_block_count exceeds available blocks")
        flattened = [position for block in self.block_positions for position in block]
        if sorted(flattened) != list(range(len(predicted))):
            raise ValueError("block positions must partition every attention position exactly once")


@dataclass(frozen=True, slots=True)
class HeadForecastDiagnostics:
    """Forecast quality for one layer and KV head."""

    layer: int
    kv_head: int
    cosine_similarity_to_true_queries: float
    attention_rank_correlation: float
    selected_block_recall: float
    forecast_regret: float
    predicted_selected_blocks: tuple[int, ...]
    oracle_selected_blocks: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.layer < 0 or self.kv_head < 0:
            raise ValueError("diagnostic layer/KV head must be nonnegative")
        if not -1.0 <= self.cosine_similarity_to_true_queries <= 1.0:
            raise ValueError("cosine similarity is outside [-1, 1]")
        if not -1.0 <= self.attention_rank_correlation <= 1.0:
            raise ValueError("rank correlation is outside [-1, 1]")
        if not 0.0 <= self.selected_block_recall <= 1.0:
            raise ValueError("selected-block recall is outside [0, 1]")
        if not math.isfinite(self.forecast_regret) or self.forecast_regret < 0:
            raise ValueError("forecast regret must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ForecastQualityDiagnostics:
    """Per-head and macro-averaged evaluation-only forecast diagnostics."""

    heads: tuple[HeadForecastDiagnostics, ...]
    mean_cosine_similarity: float
    mean_attention_rank_correlation: float
    mean_selected_block_recall: float
    mean_forecast_regret: float
    measurement_type: str = "validation_smoke"
    oracle_source: str = "evaluation_only_fullkv_greedy_decode"

    def __post_init__(self) -> None:
        if not self.heads:
            raise ValueError("forecast diagnostics require at least one head")
        expected = (
            sum(item.cosine_similarity_to_true_queries for item in self.heads) / len(self.heads),
            sum(item.attention_rank_correlation for item in self.heads) / len(self.heads),
            sum(item.selected_block_recall for item in self.heads) / len(self.heads),
            sum(item.forecast_regret for item in self.heads) / len(self.heads),
        )
        observed = (
            self.mean_cosine_similarity,
            self.mean_attention_rank_correlation,
            self.mean_selected_block_recall,
            self.mean_forecast_regret,
        )
        if any(
            not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
            for left, right in zip(expected, observed, strict=True)
        ):
            raise ValueError("macro forecast diagnostics do not match per-head values")


def _selected_blocks(
    attention: np.ndarray, block_positions: tuple[tuple[int, ...], ...], count: int
) -> tuple[int, ...]:
    utilities = [float(attention[list(positions)].sum()) for positions in block_positions]
    return tuple(
        sorted(range(len(utilities)), key=lambda index: (-utilities[index], index))[:count]
    )


def evaluate_forecast_quality(
    forecast: QueryForecast,
    oracle: EvaluationOnlyOracleQueries,
    attention: Mapping[HeadId, AttentionDiagnosticInput],
) -> ForecastQualityDiagnostics:
    """Compare an online forecast with evaluation-only future-query evidence."""

    if len(forecast.layers) != len(oracle.query_layers):
        raise ValueError("forecast and oracle layer counts differ")
    head_results: list[HeadForecastDiagnostics] = []
    for layer_index, layer in enumerate(forecast.layers):
        oracle_layer = _to_numpy(oracle.query_layers[layer_index])
        for head_index, head in enumerate(layer):
            try:
                attention_input = attention[(layer_index, head_index)]
            except KeyError as error:
                raise ValueError(
                    f"missing attention diagnostics for layer={layer_index}, KV head={head_index}"
                ) from error
            provenance = head.provenance
            true_queries = oracle_layer[
                :, provenance.query_head_start : provenance.query_head_end, :, :
            ].reshape(-1, oracle_layer.shape[-1])
            normalized_true = _normalize_rows(true_queries)
            normalized_centroids = _normalize_rows(_to_numpy(head.normalized_centroids))
            similarity = normalized_true @ normalized_centroids.T
            cosine = float(similarity.max(axis=1).mean())
            predicted_attention = _to_numpy(attention_input.predicted_attention).reshape(-1)
            true_attention = _to_numpy(attention_input.true_attention).reshape(-1)
            rank_correlation = _spearman(predicted_attention, true_attention)
            predicted_blocks = _selected_blocks(
                predicted_attention,
                attention_input.block_positions,
                attention_input.selected_block_count,
            )
            oracle_blocks = _selected_blocks(
                true_attention,
                attention_input.block_positions,
                attention_input.selected_block_count,
            )
            recall = len(set(predicted_blocks) & set(oracle_blocks)) / len(oracle_blocks)
            oracle_utility = sum(
                float(true_attention[list(attention_input.block_positions[index])].sum())
                for index in oracle_blocks
            )
            forecast_utility = sum(
                float(true_attention[list(attention_input.block_positions[index])].sum())
                for index in predicted_blocks
            )
            regret = (
                max(0.0, oracle_utility - forecast_utility) / oracle_utility
                if oracle_utility > 0
                else 0.0
            )
            head_results.append(
                HeadForecastDiagnostics(
                    layer_index,
                    head_index,
                    cosine,
                    rank_correlation,
                    recall,
                    regret,
                    predicted_blocks,
                    oracle_blocks,
                )
            )
    count = len(head_results)
    return ForecastQualityDiagnostics(
        heads=tuple(head_results),
        mean_cosine_similarity=sum(item.cosine_similarity_to_true_queries for item in head_results)
        / count,
        mean_attention_rank_correlation=sum(
            item.attention_rank_correlation for item in head_results
        )
        / count,
        mean_selected_block_recall=sum(item.selected_block_recall for item in head_results) / count,
        mean_forecast_regret=sum(item.forecast_regret for item in head_results) / count,
    )


__all__ = [
    "AttentionDiagnosticInput",
    "ForecastQualityDiagnostics",
    "HeadForecastDiagnostics",
    "evaluate_forecast_quality",
]
