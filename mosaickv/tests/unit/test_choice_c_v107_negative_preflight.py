from __future__ import annotations

from pathlib import Path

import yaml


def test_v107_negative_seal_consumes_all_identities_and_records_accounting_failure() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (
            package
            / "configs/ledgers/choice_c_v107_negative_preflight_seal.yaml"
        ).read_text(encoding="utf-8")
    )

    assert seal["classification"] == "sealed_negative_four_sample_gpu_preflight"
    assert seal["overall_pass"] is False
    assert seal["calibration_authorized"] is False
    assert seal["heldout_evaluation_authorized"] is False
    assert seal["gate_summary"]["candidate_samples_completed"] == 0
    assert seal["gate_summary"]["complete_candidate_failure_traces"] == 0
    assert seal["gate_summary"]["byte_accounting_invariant_passed"] is False
    assert seal["scientific_boundary"]["all_v107_planned_identities_consumed"] is True
    assert seal["scientific_boundary"]["future_runtime_validation_requires_fresh_identities"]
    assert seal["root_cause"]["classification"] == (
        "feasible_and_unrestricted_level_namespace_collision"
    )
    planned = seal["planned_samples"]
    assert planned["textvqa_train_calibration"]["sample_ids"] == [
        "textvqa_train_calibration:8160",
        "textvqa_train_calibration:34428",
    ]
    assert planned["infovqa"]["sample_ids"] == [
        "infovqa:93568",
        "infovqa:93362",
    ]
