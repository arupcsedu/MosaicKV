from __future__ import annotations

from pathlib import Path

import yaml


def test_v108_negative_seal_consumes_all_identities_and_records_single_probe() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (
            package
            / "configs/ledgers/choice_c_v108_negative_preflight_seal.yaml"
        ).read_text(encoding="utf-8")
    )

    assert seal["classification"] == "sealed_negative_four_sample_gpu_preflight"
    assert seal["overall_pass"] is False
    assert seal["calibration_authorized"] is False
    assert seal["heldout_evaluation_authorized"] is False
    assert seal["gate_summary"]["candidate_samples_completed"] == 0
    assert seal["gate_summary"]["complete_candidate_failure_traces"] == 2
    assert seal["gate_summary"]["complete_six_field_accounting_traces"] == 0
    assert seal["gate_summary"]["distinct_compressed_joint_probes_per_attempt"] == 1
    assert seal["scientific_boundary"]["all_v108_planned_identities_consumed"] is True
    assert seal["scientific_boundary"][
        "future_runtime_validation_requires_fresh_identities"
    ] is True
    assert seal["root_cause"]["classification"] == (
        "single_position_feasible_joint_candidate_and_incomplete_fallback_accounting"
    )
    assert seal["planned_samples"]["textvqa_train_calibration"]["sample_ids"] == [
        "textvqa_train_calibration:15455",
        "textvqa_train_calibration:31961",
    ]
    assert seal["planned_samples"]["infovqa"]["sample_ids"] == [
        "infovqa:96176",
        "infovqa:91392",
    ]
