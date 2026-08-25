from __future__ import annotations

from pathlib import Path

import yaml


def _seal() -> dict[str, object]:
    package = Path(__file__).parents[2]
    value = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v105_negative_preflight_seal.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    return value


def test_v105_negative_seal_blocks_every_downstream_stage() -> None:
    seal = _seal()
    gate = seal["gate_summary"]
    boundary = seal["scientific_boundary"]
    assert isinstance(gate, dict)
    assert isinstance(boundary, dict)

    assert seal["overall_pass"] is False
    assert seal["gpu_preflight_pass"] is False
    assert seal["calibration_authorized"] is False
    assert seal["heldout_evaluation_authorized"] is False
    assert gate["candidate_samples_completed"] == 0
    assert gate["compressed_target_samples"] == 0
    assert gate["calibration_must_remain_blocked"] is True
    assert boundary["all_v105_planned_identities_consumed"] is True
    assert boundary["future_runtime_validation_requires_fresh_identities"] is True
    assert boundary["thresholds_must_not_be_weakened"] is True


def test_v105_negative_seal_preserves_both_distinct_failure_modes() -> None:
    observed = _seal()["observed_candidate_samples"]
    assert isinstance(observed, list)
    assert len(observed) == 2
    by_task = {row["task"]: row for row in observed if isinstance(row, dict)}

    text = by_task["textvqa_train_calibration"]
    assert text["used_fullkv_fallback"] is True
    assert text["probe_evaluations_used"] == 24
    assert text["required_failure_trace_fields_complete"] is True
    assert text["over_budget_cache_packed"] is False

    info = by_task["infovqa"]
    assert info["used_fullkv_fallback"] is None
    assert info["failure"] == "non-fallback selection exceeds its route capacity"
    assert info["required_failure_trace_fields_complete"] is False
    assert info["required_failure_trace_fields_present"] == 1
    assert len(str(info["trace_sha256"])) == 64
