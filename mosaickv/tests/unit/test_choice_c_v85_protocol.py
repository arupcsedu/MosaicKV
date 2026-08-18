from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import yaml


def _load_validator(package: Path) -> ModuleType:
    path = package / "scripts/validate_choice_c_v85_preregistration.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v85_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _spec(package: Path) -> dict:
    return yaml.safe_load(
        (
            package
            / "configs/experiments/choice_c_v85_v84_frozen_untouched_heldout_quality_memory.yaml"
        ).read_text(encoding="utf-8")
    )


def test_v85_binds_the_immutable_exact_only_v84_candidate() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    frozen = Path(spec["frozen_candidate"])

    assert spec["runtime_source_commit"] == "9e300dda54fb5377aefae06d1f28614e426d42fd"
    assert spec["selected_candidate"] == "hybrid4_exact_risk_moderate_b4_27_5"
    assert spec["candidate_protocol_method"] == "mosaickv_full"
    assert spec["candidate_effective_method"] == "mosaickv_exact"
    assert spec["supports_prototype_or_repair_claim"] is False
    assert len(spec["frozen_candidate_sha256"]) == 64
    if frozen.is_file():
        assert hashlib.sha256(frozen.read_bytes()).hexdigest() == spec["frozen_candidate_sha256"]
        result = _load_validator(package).validate(
            package
            / "configs/experiments/choice_c_v85_v84_frozen_untouched_heldout_quality_memory.yaml",
            frozen,
        )
        assert result["status"] == "valid"


def test_v85_is_untouched_and_keeps_both_memory_and_quality_gates() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    boundary = spec["scientific_boundary"]
    gate = spec["primary_gate"]

    assert spec["frozen_before_sample_materialization"] is True
    assert boundary["v84_candidate_retuning_allowed"] is False
    assert boundary["v84_candidate_overrides_immutable"] is True
    assert boundary["heldout_answers_used_before_execution"] is False
    assert boundary["heldout_references_used_before_execution"] is False
    assert boundary["heldout_task_scores_used_before_execution"] is False
    assert {task["task"] for task in spec["tasks"]} == {"docvqa", "textvqa"}
    assert {task["selection_seed"] for task in spec["tasks"]} == {20280113}
    assert {task["sample_count"] for task in spec["tasks"]} == {128}
    assert gate["minimum_active_memory_reduction_fraction"] == 0.30
    assert gate["minimum_active_plus_residual_memory_reduction_fraction"] == 0.30
    assert gate["task_score_noninferiority_margin"] == 0.05
    assert gate["require_nonnegative_mean_score_difference"] is True
    assert gate["require_bootstrap_lower_bound_above_negative_margin"] is True
    assert gate["require_each_task_to_pass"] is True


def test_v85_materialization_fails_closed_on_v83_v84_exclusions() -> None:
    package = Path(__file__).parents[2]
    materializer = (package / "slurm/choice_c_v85_materialize_heldout.sbatch").read_text(
        encoding="utf-8"
    )

    assert "choice-c-v83-v82-frozen-untouched-heldout-quality-memory-v1/inputs" in materializer
    assert "choice-c-v84-disjoint-risk-gated-hybrid-calibration-v1/inputs" in materializer
    assert "required held-out exclusion directory is missing" in materializer
    assert "--selection-freeze" in materializer
    assert "--seed 20280113" in materializer
    assert "--subset-size 128" in materializer


def test_v85_submission_uses_separate_pinned_runtime_and_protocol_worktrees() -> None:
    package = Path(__file__).parents[2]
    submit = (package / "scripts/submit_choice_c_v85.sh").read_text(encoding="utf-8")
    gpu = (package / "slurm/choice_c_v85_quality_memory_a100.sbatch").read_text(encoding="utf-8")
    analyze = (package / "slurm/choice_c_v85_analyze.sbatch").read_text(encoding="utf-8")

    assert "RUNTIME_COMMIT=9e300dda54fb5377aefae06d1f28614e426d42fd" in submit
    assert "mosaickv_v85_runtime_9e300dda54fb" in submit
    assert "mosaickv_v85_protocol_" in submit
    assert "worktree add --detach" in submit
    assert "refusing to reuse existing v85 output root" in submit
    assert "--candidate-effective-method mosaickv_exact" in analyze
    assert "--array=0-1%2" in gpu
    assert "--gres=gpu:a100:1" in gpu
