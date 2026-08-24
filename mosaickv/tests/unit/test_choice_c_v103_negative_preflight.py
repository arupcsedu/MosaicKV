from __future__ import annotations

from pathlib import Path

import yaml


def test_v103_negative_preflight_seal_blocks_every_downstream_stage() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v103_negative_preflight_seal.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert seal["overall_pass"] is False
    assert seal["gpu_preflight_pass"] is False
    assert seal["calibration_authorized"] is False
    assert seal["calibration_ledgers_materialized"] is False
    assert seal["frozen_candidate_created"] is False
    assert seal["heldout_evaluation_authorized"] is False
    assert seal["failure_classification"]["all_four_candidate_samples_completed"] is False
    assert seal["failure_classification"]["result_must_not_authorize_calibration"] is True
    assert seal["scientific_boundary"]["future_runtime_validation_requires_fresh_identities"]


def test_v103_negative_preflight_seal_preserves_trace_and_reference_findings() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v103_negative_preflight_seal.yaml").read_text(
            encoding="utf-8"
        )
    )
    observed = seal["observed_candidate_samples"]

    assert len(observed) == 2
    assert len(seal["unexecuted_candidate_samples"]) == 2
    assert all(row["status"] == "failed" for row in observed)
    assert all(row["used_fullkv_fallback"] is True for row in observed)
    assert all(row["target_reached"] is False for row in observed)
    assert all(row["required_trace_fields_complete"] is True for row in observed)
    assert all(row["cache_accounting_consistent"] is True for row in observed)
    text = next(row for row in observed if row["task"] == "textvqa_train_calibration")
    assert text["textvqa_reference_is_annotation_list"] is True
    assert text["textvqa_reference_count"] == 10
