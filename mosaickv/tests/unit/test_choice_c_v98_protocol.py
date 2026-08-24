from __future__ import annotations

import hashlib
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
            / "configs/experiments/choice_c_v98_short_horizon_logit_calibration.yaml"
        ).read_text(encoding="utf-8")
    )


def _protocol(package: Path) -> ModuleType:
    path = package / "scripts/choice_c_v74_protocol.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v98_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_v97_is_sealed_negative_without_sample_identity_inspection() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v97_negative_calibration_seal.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert seal["classification"] == "sealed_negative_label_blind_calibration_result"
    assert seal["overall_freeze_gate_pass"] is False
    assert seal["memory_gate_pass"] is True
    assert seal["aggregate_fidelity_gate_pass"] is False
    assert seal["robust_fold_fidelity_gate_pass"] is False
    assert seal["atomic_group_utilization_gate_pass"] is False
    assert seal["frozen_candidate_created"] is False
    assert seal["heldout_evaluation_authorized"] is False
    boundary = seal["scientific_boundary"]
    assert boundary["future_design_must_not_use_v97_answers"] is True
    assert boundary["future_design_must_not_use_v97_failure_identities"] is True
    for relation in ("protocol", "parent_negative_result_seal"):
        record = seal[relation]
        path = package / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_v98_replaces_surrogate_admission_with_strict_logit_gate() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    protocol = _protocol(package)
    base = load_config(package / spec["candidate_config"])
    common = protocol.apply_common_cache_policy(base, spec["common_cache_policy"])
    applied = protocol.apply_candidate(common, spec["candidates"][0])
    selection = applied.selection

    assert spec["prior_result_exclusion_boundary"] == "v97_and_all_prior_choice_c_runs"
    assert spec["selection_rule_frozen_before_calibration"] is True
    assert spec["execution_status"] == "prepared_waiting_for_clean_commit"
    assert {task["selection_seed"] for task in spec["tasks"]} == {20290228}
    assert {task["sample_count"] for task in spec["tasks"]} == {32}
    assert applied.method is MosaicKVMethod.MOSAICKV_EXACT
    assert selection.stable_head_short_horizon_logit_stability
    assert selection.stable_head_logit_stability_shortlist_size == 2
    assert selection.stable_head_logit_stability_horizon == 2
    assert selection.stable_head_logit_stability_maximum_kl == pytest.approx(0.02)
    assert selection.stable_head_logit_stability_maximum_logit_difference == pytest.approx(1.0)
    assert not selection.stable_head_query_coverage_cross_head_stability_gate
    assert (
        selection.stable_head_query_coverage_layout_group_atomic_selection_mode
        == "pareto_joint_donor_knapsack"
    )
    assert not applied.prototypes.enabled
    assert not applied.residual.enabled
    assert not applied.repair.enabled

    with pytest.raises(ConfigurationError, match="two-point Pareto shortlist"):
        replace(selection, stable_head_logit_stability_shortlist_size=1)
    with pytest.raises(ConfigurationError, match="controls require the stability gate"):
        replace(selection, stable_head_short_horizon_logit_stability=False)


def test_v98_keeps_all_outer_gates_and_submits_no_heldout() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    objective = spec["calibration_objective"]
    robust = spec["robust_fidelity_analysis"]
    logit = spec["short_horizon_logit_stability_analysis"]

    assert objective["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert objective["require_zero_residual_bytes"] is True
    assert robust["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert robust["fullkv_token_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert robust["fold_count"] == 4
    assert logit["labels_accessed"] is False
    assert logit["require_every_sample_to_pass"] is True
    assert logit["require_fullkv_prefill_reuse"] is True
    assert logit["forbid_second_prefill"] is True
    assert logit["require_probe_state_discard_before_measured_decode"] is True
    assert logit["require_greedy_token_match"] is True

    gpu = (package / "slurm/choice_c_v98_short_horizon_logit_a100.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gres=gpu:a100:1" in gpu
    assert "#SBATCH --array=0-1%2" in gpu
    assert "#SBATCH --reservation=bi_fox_dgx" in gpu

    submit = (package / "scripts/submit_choice_c_v98.sh").read_text(encoding="utf-8")
    submitted = "\n".join(line for line in submit.splitlines() if "sbatch" in line)
    assert "heldout" not in submitted
    assert "validate_choice_c_v97_negative_seal.py" in submit
    assert "choice_c_v98_materialize_short_horizon_logit_calibration.sbatch" in submit
    assert "choice_c_v98_short_horizon_logit_a100.sbatch" in submit
    assert "choice_c_v98_freeze.sbatch" in submit
    assert "assert_clean_worktree.sh" in submit

    retry = (package / "scripts/submit_choice_c_v98_retry.sh").read_text(encoding="utf-8")
    retry_submissions = "\n".join(line for line in retry.splitlines() if "sbatch" in line)
    assert "heldout" not in retry_submissions
    assert "materialize" not in retry_submissions
    assert retry_submissions.count("JOB=$(sbatch") == 2
    assert "24b7088d3ab6ce383aa298e85479e29d4fb9b9895cfc0f9867cb063ee6a6cad3" in retry
    assert "a67c9342a09841e2dc4ed8a0babaf00aa9b40eb8b78f165d0dd64d314d6daf17" in retry
    assert "assert_clean_worktree.sh" in retry
