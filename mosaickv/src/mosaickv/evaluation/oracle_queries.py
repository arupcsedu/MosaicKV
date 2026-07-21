"""Evaluation-only collection of true future queries from FullKV decoding.

Nothing in ``mosaickv.forecasting`` imports this module. Online compression
must consume ``QueryForecast`` only and must never receive this oracle object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mosaickv.adapters.huggingface.types import PrefillOutput
from mosaickv.forecasting.huggingface import run_isolated_greedy_query_rollout


@dataclass(frozen=True, slots=True)
class EvaluationOnlyOracleQueries:
    """True full-cache future queries, prohibited from online compression APIs."""

    query_layers: tuple[Any, ...]
    greedy_token_ids: tuple[int, ...]
    future_steps: int
    source: str = "evaluation_only_fullkv_greedy_decode"

    def __post_init__(self) -> None:
        if self.future_steps <= 0:
            raise ValueError("oracle future_steps must be positive")
        if len(self.greedy_token_ids) != self.future_steps:
            raise ValueError("oracle token count does not match future_steps")
        if not self.query_layers:
            raise ValueError("oracle query layers cannot be empty")
        for layer in self.query_layers:
            shape = tuple(int(item) for item in layer.shape)
            if len(shape) != 4 or shape[0] != 1 or shape[-2] != self.future_steps:
                raise ValueError(
                    "oracle queries must have shape [1, query_heads, future_steps, head_dim]"
                )
        if self.source != "evaluation_only_fullkv_greedy_decode":
            raise ValueError("oracle source must remain explicitly evaluation-only")


def collect_evaluation_only_true_future_queries(
    adapter: Any,
    prefill: PrefillOutput,
    *,
    future_steps: int,
) -> EvaluationOnlyOracleQueries:
    """Collect FullKV future queries on an isolated state for diagnostics only."""

    if future_steps <= 0:
        raise ValueError("future_steps must be positive")
    rollout = run_isolated_greedy_query_rollout(adapter, prefill, steps=future_steps)
    return EvaluationOnlyOracleQueries(
        query_layers=rollout.query_layers,
        greedy_token_ids=rollout.token_ids,
        future_steps=future_steps,
    )


__all__ = [
    "EvaluationOnlyOracleQueries",
    "collect_evaluation_only_true_future_queries",
]
