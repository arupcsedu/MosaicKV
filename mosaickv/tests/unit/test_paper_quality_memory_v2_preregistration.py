from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from mosaickv.config import load_config
from mosaickv.evaluation.paper_candidate import (
    PAPER_CANDIDATE_TASKS,
    PAPER_CANDIDATE_V2_TASKS,
    PaperCandidateContract,
    PaperCandidateV2Contract,
)


def _package_root() -> Path:
    return Path(__file__).parents[2]


def _load_yaml(relative: str) -> dict[str, Any]:
    return yaml.safe_load((_package_root() / relative).read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((_package_root() / relative).read_bytes()).hexdigest()


def test_frozen_candidate_is_bound_to_existing_immutable_source() -> None:
    frozen = _load_yaml("configs/ledgers/paper_candidate_v1_frozen_99a412b.yaml")
    candidate = frozen["candidate"]
    contract = frozen["freeze_contract"]
    assert candidate["source_git_sha"] == "99a412bf0d2c639a095ff6efc5ca02e435b5e889"
    assert candidate["config_sha256"] == _sha256("configs/paper_candidate_llava_v1.yaml")
    assert contract["algorithm_or_hyperparameter_change_allowed"] is False
    assert contract["candidate_config_change_allowed"] is False
    assert contract["candidate_source_change_allowed"] is False
    assert contract["future_evaluation_may_change_candidate"] is False
    assert frozen["execution_authorization"]["candidate_is_frozen"] is True
    assert frozen["execution_authorization"]["gpu_evaluation_authorized"] is False


def test_preregistration_replaces_mmstar_and_discloses_prior_exposure() -> None:
    protocol = _load_yaml("configs/experiments/paper_quality_memory_v2_preregistered.yaml")
    tasks = {task["task"]: task for task in protocol["tasks"]}
    assert set(tasks) == {"realworldqa", "textvqa", "docvqa"}
    assert "mmstar" not in tasks
    realworldqa = tasks["realworldqa"]
    assert realworldqa["dataset_id"] == "lmms-lab/RealWorldQA"
    assert realworldqa["dataset_revision"] == (
        "907c4e5228fd1703c710ed937601cb5f89ab8d5c"
    )
    assert realworldqa["split"] == "test"
    assert realworldqa["sample_count"] == 128
    assert realworldqa["sample_level_untouched_required"] is True
    assert realworldqa["formal_identity_exclusion_scan_required"] is True
    for task_name in ("textvqa", "docvqa"):
        assert tasks[task_name]["sample_level_untouched_claimed"] is False
        assert tasks[task_name]["paper_role_cannot_be_upgraded_to_untouched"] is True


def test_preregistered_memory_and_quality_gates_are_unchanged() -> None:
    protocol = _load_yaml("configs/experiments/paper_quality_memory_v2_preregistered.yaml")
    assert protocol["nearest_baseline"] == "prompt_attention_topk"
    assert protocol["methods"] == [
        "full_kv",
        "prompt_attention_topk",
        "mosaickv_full__paper_candidate_v1",
    ]
    memory = protocol["memory_gate"]
    assert memory["target_candidate_active_fraction_of_paired_baseline"] == 0.699
    assert memory["minimum_active_memory_reduction_fraction"] == 0.30
    assert memory["strict_inequality_required"] is True
    assert memory["require_every_paired_sample_to_pass"] is True
    quality = protocol["quality_gate"]
    assert quality["paired_score_margin"] == 0.0
    assert quality["require_candidate_mean_official_score_at_least_baseline"] is True
    assert quality["require_every_task_to_pass_separately"] is True
    assert quality["paired_bootstrap_repetitions"] == 10_000


def test_preregistration_is_larger_fixed_and_fully_execution_disabled() -> None:
    protocol = _load_yaml("configs/experiments/paper_quality_memory_v2_preregistered.yaml")
    scope = protocol["scientific_scope"]
    assert scope["samples_per_task"] == 128
    assert scope["total_unique_task_examples"] == 384
    assert scope["total_method_example_rows"] == 1152
    assert protocol["statistical_design"]["increase_over_two_sample_development_gate"] == 64
    authorization = protocol["execution_authorization"]
    assert authorization
    assert all(value is False for value in authorization.values())
    assert protocol["controlled_runtime"]["deterministic_generation"] is True
    assert protocol["controlled_runtime"]["max_new_tokens"] == 64


def test_v2_contract_routes_realworldqa_without_mutating_v1_contract() -> None:
    template = load_config(_package_root() / "configs/paper_candidate_llava_v1.yaml")
    v2 = PaperCandidateV2Contract().resolve_run_config(
        template,
        task="realworldqa",
        baseline_active_bytes=40_000_000,
        storage_quantum_bytes=524_288,
    )

    assert PAPER_CANDIDATE_TASKS == ("mmstar", "textvqa", "docvqa")
    assert PAPER_CANDIDATE_V2_TASKS == ("realworldqa", "textvqa", "docvqa")
    assert v2.dataset.id == "lmms-lab/RealWorldQA"
    assert v2.dataset.revision == "907c4e5228fd1703c710ed937601cb5f89ab8d5c"
    assert v2.dataset.split == "test"
    assert v2.cache.budget_value == PaperCandidateContract().active_budget(
        40_000_000,
        524_288,
    )
    with pytest.raises(ValueError, match="unsupported paper-candidate-v2 task"):
        PaperCandidateV2Contract().resolve_run_config(
            template,
            task="mmstar",
            baseline_active_bytes=40_000_000,
            storage_quantum_bytes=524_288,
        )
