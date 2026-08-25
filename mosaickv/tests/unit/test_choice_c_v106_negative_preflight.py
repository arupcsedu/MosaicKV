from __future__ import annotations

from pathlib import Path

import yaml


def test_v106_negative_seal_consumes_every_planned_identity_and_blocks_calibration() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (
            package
            / "configs/ledgers/choice_c_v106_negative_preflight_seal.yaml"
        ).read_text(encoding="utf-8")
    )

    assert seal["classification"] == "sealed_negative_four_sample_gpu_preflight"
    assert seal["overall_pass"] is False
    assert seal["calibration_authorized"] is False
    assert seal["heldout_evaluation_authorized"] is False
    assert seal["gate_summary"]["candidate_samples_completed"] == 0
    assert seal["gate_summary"]["candidate_samples_required"] == 4
    assert seal["gate_summary"]["complete_candidate_failure_traces"] == 2
    assert seal["scientific_boundary"]["all_v106_planned_identities_consumed"] is True
    assert seal["scientific_boundary"]["future_runtime_validation_requires_fresh_identities"]
    planned = seal["planned_samples"]
    assert planned["textvqa_train_calibration"]["sample_ids"] == [
        "textvqa_train_calibration:22833",
        "textvqa_train_calibration:17040",
    ]
    assert planned["infovqa"]["sample_ids"] == [
        "infovqa:95567",
        "infovqa:91783",
    ]
