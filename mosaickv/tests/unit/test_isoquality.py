"""Synthetic-only tests for the lower-memory/equal-quality claim gate."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

from mosaickv.evaluation.metrics.statistics import PairedMetricComparison

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    from isoquality_gate import (
        QualityDirection,
        choose_candidate_cap_without_quality,
        evaluate_isoquality_pair,
        memory_gate,
        quality_gate,
        validate_isoquality_protocol,
    )
finally:
    sys.path.remove(str(SCRIPTS))

sys.path.insert(0, str(SCRIPTS))
try:
    import prepare_lower_memory_equal_quality as planner
finally:
    sys.path.remove(str(SCRIPTS))


def rows(initial: int, final: int, peak: int):
    return [
        {
            "sample_id": sample_id,
            "initial_active_kv_storage_bytes": initial,
            "final_active_kv_storage_bytes": final,
            "peak_active_kv_storage_bytes": peak,
        }
        for sample_id in ("a", "b", "c", "d")
    ]


def interval(low: float, high: float, *, metric: str = "cider"):
    return PairedMetricComparison(
        metric_name=metric,
        reference_value=100.0,
        candidate_value=101.0,
        candidate_minus_reference=1.0,
        relative_difference=0.01,
        ci_low=low,
        ci_high=high,
        relative_ci_low=low / 100,
        relative_ci_high=high / 100,
        confidence=0.95,
        bootstrap_resamples=10000,
        bootstrap_seed=20271014,
        sample_count=4,
    )


def test_joint_gate_requires_all_three_physical_boundaries_and_quality():
    baseline = rows(40, 48, 48)
    candidate = rows(25, 32, 32)
    result = evaluate_isoquality_pair(
        baseline_method="vl_cache_reimpl",
        baseline_cap_mib=40,
        baseline_rows=baseline,
        candidate_method="mosaickv_exact",
        candidate_cap_mib=25,
        candidate_rows=candidate,
        comparison=interval(0.1, 2.0),
        direction=QualityDirection.HIGHER,
    )
    assert result.memory_pass and result.quality_pass and result.joint_pass
    assert {item.boundary for item in result.reductions} == {"initial", "final", "peak"}
    assert min(item.worst_sample_reduction for item in result.reductions) == pytest.approx(1 / 3)


def test_mean_reduction_cannot_hide_one_sample_failure():
    baseline = rows(100, 100, 100)
    candidate = rows(60, 60, 60)
    candidate[-1] = {
        **candidate[-1],
        "peak_active_kv_storage_bytes": 80,
    }
    passed, reductions = memory_gate(baseline, candidate)
    assert not passed
    peak = next(item for item in reductions if item.boundary == "peak")
    assert peak.aggregate_reduction > 0.30
    assert peak.worst_sample_reduction == pytest.approx(0.20)


def test_direction_aware_quality_gate():
    assert quality_gate(interval(0.0, 2.0), direction=QualityDirection.HIGHER)
    assert not quality_gate(interval(-0.01, 2.0), direction=QualityDirection.HIGHER)
    assert quality_gate(
        interval(-2.0, 0.0, metric="teacher_forced_ppl"),
        direction=QualityDirection.LOWER,
    )
    assert not quality_gate(
        interval(-2.0, 0.01, metric="teacher_forced_ppl"),
        direction=QualityDirection.LOWER,
    )


def test_cap_choice_is_storage_only_and_uses_largest_eligible_candidate():
    baseline = rows(40, 48, 48)
    candidates = {
        20.0: rows(20, 28, 28),
        25.0: rows(25, 32, 32),
        28.0: rows(28, 36, 36),
    }
    assert (
        choose_candidate_cap_without_quality(
            baseline_rows=baseline,
            candidates=candidates,
            baseline_cap_mib=40,
        )
        == 25.0
    )
    assert (
        choose_candidate_cap_without_quality(
            baseline_rows=baseline,
            candidates={28.0: candidates[28.0]},
            baseline_cap_mib=40,
        )
        is None
    )


def test_pair_rejects_unpaired_or_inconsistent_denominators():
    baseline = rows(40, 48, 48)
    candidate = rows(25, 32, 32)
    candidate.pop()
    with pytest.raises(ValueError, match="identical sample IDs"):
        evaluate_isoquality_pair(
            baseline_method="lookm_reimpl",
            baseline_cap_mib=40,
            baseline_rows=baseline,
            candidate_method="mosaickv_exact",
            candidate_cap_mib=25,
            candidate_rows=candidate,
            comparison=interval(0, 1),
            direction=QualityDirection.HIGHER,
        )


def test_paper_protocol_is_registered_but_cannot_submit_from_dirty_source():
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/experiments/lower_memory_equal_quality_v1.yaml"
    )
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    result = validate_isoquality_protocol(protocol)
    assert result["status"] == "registered_execution_disabled"
    assert not result["source_sealed"]
    assert result["lanes"]["vl_cache_coco"]["sample_count"] == 512
    assert result["lanes"]["vl_cache_coco"]["validation_sample_count"] == 32
    assert result["lanes"]["lookm_milebench"]["sample_count"] == 192
    assert result["lanes"]["lookm_milebench"]["validation_sample_count"] == 16
    assert result["lanes"]["vl_cache_coco"]["planned_compressed_method_cap_rows"] == (
        512 * 3 * 5
    )
    assert result["lanes"]["lookm_milebench"]["planned_compressed_method_cap_rows"] == (
        192 * 3 * 5
    )


def test_protocol_rejects_a_weakened_memory_gate():
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/experiments/lower_memory_equal_quality_v1.yaml"
    )
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    protocol["claim"]["minimum_reduction"] = 0.29
    with pytest.raises(ValueError, match="weakened"):
        validate_isoquality_protocol(protocol)


def test_label_blind_paper_cohorts_are_balanced_disjoint_and_order_invariant():
    groups = {
        name: [f"{name}:{index}" for index in range(150)]
        for name in planner.MILEBENCH_SUBTASKS
    }
    excluded = {f"{name}:0" for name in planner.MILEBENCH_SUBTASKS}
    first = planner.select_samples(
        groups,
        excluded=excluded,
        validation_per_group=2,
        paper_per_group=4,
    )
    second = planner.select_samples(
        {name: list(reversed(values)) for name, values in groups.items()},
        excluded=excluded,
        validation_per_group=2,
        paper_per_group=4,
    )
    assert first == second
    assert len(first["validation"]) == 8
    assert len(first["larger"]) == 16
    assert not set(first["validation"]) & set(first["larger"])
    assert not excluded & (set(first["validation"]) | set(first["larger"]))


def test_execution_contract_is_unsealed_and_uses_registered_curve_caps():
    from mosaickv.evaluation.shared_byte_contract import load_contract

    path = (
        Path(__file__).resolve().parents[2]
        / "configs/experiments/shared_byte_isoquality_v1.yaml"
    )
    contract = load_contract(path)
    assert not contract["execution_authorized"]
    assert contract["candidate"]["base_commit"] is None
    assert contract["candidate"]["content_sha256"] is None
    assert contract["lanes"]["vl_cache"]["caps_mib"] == [10, 20, 25, 40, 80]
    assert contract["lanes"]["lookm"]["caps_mib"] == [16, 20, 28, 40, 80]
