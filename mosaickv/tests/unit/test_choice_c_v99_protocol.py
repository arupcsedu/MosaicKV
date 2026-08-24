from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from mosaickv.config import ConfigurationError, load_config
from mosaickv.types import MosaicKVMethod


def _spec(package: Path) -> dict:
    return yaml.safe_load(
        (
            package
            / "configs/experiments/choice_c_v99_progressive_logit_safe_eviction_calibration.yaml"
        ).read_text(encoding="utf-8")
    )


def _protocol(package: Path) -> ModuleType:
    path = package / "scripts/choice_c_v74_protocol.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v99_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v98_is_sealed_negative_without_failure_identity_inspection() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v98_negative_calibration_seal.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert seal["classification"] == "sealed_negative_label_blind_calibration_result"
    assert seal["overall_freeze_gate_pass"] is False
    assert seal["memory_gate_pass"] is True
    assert seal["aggregate_fidelity_gate_pass"] is True
    assert seal["robust_fold_fidelity_gate_pass"] is False
    assert seal["short_horizon_logit_stability_gate_pass"] is False
    assert seal["frozen_candidate_created"] is False
    assert seal["heldout_evaluation_authorized"] is False
    boundary = seal["scientific_boundary"]
    assert boundary["future_design_must_not_use_v98_answers"] is True
    assert boundary["future_design_must_not_use_v98_failure_identities"] is True
    assert seal["future_protocol"]["design_change"] == (
        "progressive_fullkv_to_target_logit_safe_eviction"
    )


def test_v99_starts_from_fullkv_and_fails_closed_at_the_unchanged_target() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    protocol = _protocol(package)
    base = load_config(package / spec["candidate_config"])
    common = protocol.apply_common_cache_policy(base, spec["common_cache_policy"])
    applied = protocol.apply_candidate(common, spec["candidates"][0])
    selection = applied.selection

    assert spec["prior_result_exclusion_boundary"] == "v98_and_all_prior_choice_c_runs"
    assert spec["selection_rule_frozen_before_calibration"] is True
    assert spec["execution_status"] == "prepared_waiting_for_clean_commit"
    assert {task["selection_seed"] for task in spec["tasks"]} == {20300329}
    assert {task["sample_count"] for task in spec["tasks"]} == {32}
    assert applied.method is MosaicKVMethod.MOSAICKV_EXACT
    assert selection.stable_head_progressive_logit_safe_eviction
    assert selection.stable_head_progressive_eviction_initial_batch_blocks == 16
    assert selection.stable_head_progressive_eviction_maximum_probes == 128
    assert selection.stable_head_progressive_eviction_horizon == 2
    assert selection.stable_head_progressive_eviction_maximum_kl == pytest.approx(0.02)
    assert selection.stable_head_progressive_eviction_maximum_logit_difference == pytest.approx(1.0)
    assert not selection.stable_head_short_horizon_logit_stability
    assert not selection.stable_head_query_coverage_cross_head_stability_gate
    assert not applied.prototypes.enabled
    assert not applied.residual.enabled
    assert not applied.repair.enabled

    with pytest.raises(ConfigurationError, match="controls require the progressive gate"):
        replace(selection, stable_head_progressive_logit_safe_eviction=False)
    with pytest.raises(ConfigurationError, match="cannot use the bounded shortlist gate"):
        replace(
            selection,
            stable_head_short_horizon_logit_stability=True,
            stable_head_logit_stability_shortlist_size=2,
            stable_head_logit_stability_horizon=2,
            stable_head_logit_stability_maximum_kl=0.02,
            stable_head_logit_stability_maximum_logit_difference=1.0,
        )


def test_v99_keeps_every_gate_and_submits_no_heldout() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    objective = spec["calibration_objective"]
    robust = spec["robust_fidelity_analysis"]
    progressive = spec["progressive_logit_safe_eviction_analysis"]

    assert objective["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert objective["require_zero_residual_bytes"] is True
    assert robust["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert robust["fullkv_token_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert robust["fold_count"] == 4
    assert robust["expected_samples_per_fold"] == 8
    assert progressive["labels_accessed"] is False
    assert progressive["require_start_from_fullkv"] is True
    assert progressive["require_every_sample_to_reach_target"] is True
    assert progressive["require_every_accepted_checkpoint_to_pass"] is True
    assert progressive["require_greedy_token_match"] is True

    gpu = (package / "slurm/choice_c_v99_progressive_eviction_a100.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:a100:1" in gpu
    assert "#SBATCH --array=0-1%2" in gpu
    assert "#SBATCH --reservation=bi_fox_dgx" in gpu

    submit = (package / "scripts/submit_choice_c_v99.sh").read_text(encoding="utf-8")
    submitted = "\n".join(line for line in submit.splitlines() if "sbatch" in line)
    assert "heldout" not in submitted
    assert "validate_choice_c_v98_negative_seal.py" in submit
    assert "validate_choice_c_v99_protocol.py" in submit
    assert "choice_c_v99_materialize_progressive_eviction_calibration.sbatch" in submit
    assert "choice_c_v99_progressive_eviction_a100.sbatch" in submit
    assert "choice_c_v99_freeze.sbatch" in submit
    assert "assert_clean_worktree.sh" in submit

    retry = (package / "scripts/submit_choice_c_v99_retry.sh").read_text(encoding="utf-8")
    retry_submissions = "\n".join(line for line in retry.splitlines() if "sbatch" in line)
    assert "heldout" not in retry_submissions
    assert "materialize" not in retry_submissions
    assert retry_submissions.count("JOB=$(sbatch") == 2
    assert "c871e44a176fd62988b1bf8a51c93414200832fe424531b39dcf636e33663a73" in retry
    assert "73cb0b3be2cee3ba69174a5403d48a1549dddfec2cdc436ba6544a5f7b9da59a" in retry
    assert "--partition=bii" in retry
