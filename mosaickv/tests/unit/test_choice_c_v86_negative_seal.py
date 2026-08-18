from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import yaml


def _validator(package: Path) -> ModuleType:
    path = package / "scripts/validate_choice_c_v86_negative_seal.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v86_negative_seal", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v86_seal_is_negative_and_blocks_freeze_and_heldout() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v86_negative_calibration_seal.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert seal["classification"] == "sealed_negative_label_blind_calibration_result"
    assert seal["immutable_result"] is True
    assert seal["overall_freeze_gate_pass"] is False
    assert seal["aggregate_fidelity_gate_pass"] is False
    assert seal["robust_fold_fidelity_gate_pass"] is False
    assert seal["frozen_candidate_created"] is False
    assert seal["heldout_evaluation_authorized"] is False
    assert seal["future_protocol"]["v86_must_remain_negative"] is True
    assert seal["future_protocol"]["heldout_campaign_must_remain_blocked"] is True
    assert seal["future_protocol"]["no_v87_or_large_gpu_campaign_authorized"] is True


def test_v86_seal_pins_protocol_parent_and_scientific_boundary() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v86_negative_calibration_seal.yaml").read_text(
            encoding="utf-8"
        )
    )
    for field in ("protocol", "parent_negative_result_seal"):
        record = seal[field]
        path = package / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    boundary = seal["scientific_boundary"]
    assert boundary["benchmark_labels_accessed_for_selection"] is False
    assert boundary["official_task_scores_accessed_for_selection"] is False
    assert boundary["v85_used_only_as_sample_exclusion_boundary"] is True
    assert boundary["v85_sample_overlap"] == 0
    assert boundary["future_heldout_materialized"] is False


def test_v86_inventory_algorithm_is_deterministic(tmp_path: Path) -> None:
    package = Path(__file__).parents[2]
    validator = _validator(package)
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "a.txt").write_text("alpha\n", encoding="utf-8")

    first = validator._inventory(tmp_path)
    second = validator._inventory(tmp_path)

    assert first == second
    assert first[0] == 2
    assert first[1] == len(b"beta\n") + len(b"alpha\n")
