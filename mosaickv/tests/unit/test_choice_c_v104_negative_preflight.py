from __future__ import annotations

from pathlib import Path

import yaml


def _seal() -> dict[str, object]:
    package = Path(__file__).parents[2]
    value = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v104_negative_preflight_seal.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    return value


def test_v104_negative_seal_blocks_calibration_and_consumes_all_planned_ids() -> None:
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
    assert gate["candidate_samples_required"] == 4
    assert gate["calibration_must_remain_blocked"] is True
    assert boundary["all_v104_planned_identities_consumed"] is True
    assert boundary["thresholds_must_not_be_weakened"] is True


def test_v104_negative_seal_hash_binds_attempted_failure_traces() -> None:
    seal = _seal()
    observed = seal["observed_candidate_samples"]
    assert isinstance(observed, list)
    assert len(observed) == 2
    for row in observed:
        assert isinstance(row, dict)
        assert Path(str(row["trace_path"])).is_absolute()
        assert len(str(row["trace_sha256"])) == 64
        assert row["target_reached"] is False
        assert row["used_fullkv_fallback"] is True
        assert row["required_failure_trace_fields_complete"] is True
        assert row["over_budget_cache_packed"] is False
        assert row["over_budget_cache_decoded"] is False
