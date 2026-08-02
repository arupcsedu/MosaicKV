from __future__ import annotations

import pytest

from mosaickv.generalization import (
    CalibrationCandidate,
    GeneralizationProtocolError,
    assert_disjoint_sample_sets,
    canonical_hyperparameters,
    is_completed_forecast_diagnostic,
    select_calibration_winner,
)


def _candidate(
    candidate_id: str,
    *,
    score: float = 1.0,
    agreement: float = 1.0,
    recall: float = 0.5,
    regret: float = 0.1,
    latency: float = 2.0,
) -> CalibrationCandidate:
    return CalibrationCandidate(candidate_id, score, agreement, recall, regret, latency)


def test_calibration_winner_uses_predeclared_lexicographic_rule() -> None:
    winner = select_calibration_winner(
        (
            _candidate("slow", recall=0.8, latency=3.0),
            _candidate("lower_recall", recall=0.7, latency=1.0),
            _candidate("winner", recall=0.8, latency=2.0),
        )
    )
    assert winner.candidate_id == "winner"


def test_calibration_quality_precedes_latency() -> None:
    winner = select_calibration_winner(
        (_candidate("fast", score=0.5, latency=0.1), _candidate("quality", score=1.0, latency=9.0))
    )
    assert winner.candidate_id == "quality"


def test_calibration_and_evaluation_must_be_disjoint() -> None:
    with pytest.raises(GeneralizationProtocolError, match="leakage"):
        assert_disjoint_sample_sets(("doc:1",), {"docvqa": ("doc:1", "doc:2")})
    assert_disjoint_sample_sets(("doc:1",), {"docvqa": ("doc:2",), "chartqa": ("chart:1",)})


def test_frozen_hyperparameters_reject_missing_and_unknown_fields() -> None:
    complete = {
        "retention_ratio": 0.2,
        "block_size": 16,
        "forecast_mode": "hybrid",
        "prompt_window": 16,
        "draft_steps": 1,
        "centroid_count": 4,
        "graph_max_neighbors": 24,
        "minimum_stratum_fraction": 0.0,
        "prototype_group_size": 12,
        "prototype_max_position_span": 256,
        "repair_policy": "prototype_risk",
        "repair_entropy_threshold": 0.01,
        "repair_prototype_risk_threshold": 1.0e-6,
        "repair_prototype_risk_min_entropy": 0.005,
        "repair_max_blocks_per_step": 1,
        "repair_candidate_ranking": "risk_weighted_reconstruction_error",
        "preserve_partial_prototype": True,
    }
    assert canonical_hyperparameters(complete)["draft_steps"] == 1
    with pytest.raises(GeneralizationProtocolError, match="fields differ"):
        canonical_hyperparameters(
            {key: value for key, value in complete.items() if key != "block_size"}
        )
    with pytest.raises(GeneralizationProtocolError, match="fields differ"):
        canonical_hyperparameters({**complete, "hidden_tuning": 1})


def test_completed_forecast_diagnostic_uses_metrics_and_oracle_isolation() -> None:
    diagnostic = {
        "mean_cosine_similarity": 0.8,
        "mean_attention_rank_correlation": 0.7,
        "mean_selected_block_recall": 0.6,
        "mean_forecast_regret": 0.1,
        "online_selection_isolation": "evaluation_only_output_not_passed_to_compressor",
    }
    assert is_completed_forecast_diagnostic(diagnostic)
    assert not is_completed_forecast_diagnostic(
        {**diagnostic, "mean_forecast_regret": float("nan")}
    )
    assert not is_completed_forecast_diagnostic(
        {**diagnostic, "online_selection_isolation": "oracle_used_online"}
    )
    assert not is_completed_forecast_diagnostic({"status": "disabled"})
