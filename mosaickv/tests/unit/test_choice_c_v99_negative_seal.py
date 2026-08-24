from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import yaml


def _validator(package: Path) -> ModuleType:
    path = package / "scripts/validate_choice_c_v99_negative_seal.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v99_negative_seal", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v99_seal_is_negative_and_blocks_heldout() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v99_negative_calibration_seal.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert seal["classification"] == "sealed_negative_label_blind_calibration_result"
    assert seal["overall_freeze_gate_pass"] is False
    assert seal["memory_gate_pass"] is True
    assert seal["aggregate_fidelity_gate_pass"] is True
    assert seal["robust_fold_fidelity_gate_pass"] is False
    assert seal["atomic_group_utilization_gate_pass"] is False
    assert seal["progressive_logit_safe_eviction_gate_pass"] is False
    assert seal["frozen_candidate_created"] is False
    assert seal["heldout_evaluation_authorized"] is False
    assert seal["future_protocol"]["heldout_remains_blocked"] is True


def test_v99_seal_pins_artifacts_and_future_scientific_boundary() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v99_negative_calibration_seal.yaml").read_text(
            encoding="utf-8"
        )
    )
    for field in ("protocol", "parent_negative_result_seal"):
        record = seal[field]
        path = package / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    boundary = seal["scientific_boundary"]
    assert boundary["benchmark_labels_accessed_for_selection"] is False
    assert boundary["future_design_must_not_use_v99_answers"] is True
    assert boundary["future_design_must_not_use_v99_failure_identities"] is True
    assert boundary["future_design_must_not_use_v99_per_sample_telemetry"] is True
    future = seal["future_protocol"]
    assert future["design_change"] == "hierarchical_or_head_local_logit_safe_eviction"
    assert future["sequential_full_decode_probe_per_eviction_must_be_replaced"] is True
    assert future["fresh_disjoint_calibration_required"] is True


def test_v99_negative_seal_validator_accepts_immutable_result() -> None:
    package = Path(__file__).parents[2]
    validator = _validator(package)
    result = validator.validate(
        package / "configs/ledgers/choice_c_v99_negative_calibration_seal.yaml"
    )

    assert result["status"] == "valid_negative_seal"
    assert result["measured_rows"] == 192
    assert result["progressive_target_reached_samples"] == 1
    assert result["answer_folds_passed"] == 6
    assert result["token_folds_passed"] == 3
    assert result["frozen_candidate_exists"] is False
    assert result["heldout_evaluation_authorized"] is False
