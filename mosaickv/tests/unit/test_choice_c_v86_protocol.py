from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import yaml

from mosaickv.config import load_config


def _spec(package: Path) -> dict:
    return yaml.safe_load(
        (
            package
            / "configs/experiments/choice_c_v86_disjoint_robust_fidelity_calibration.yaml"
        ).read_text(encoding="utf-8")
    )


def _load_module(package: Path, filename: str, name: str) -> ModuleType:
    scripts = package / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        module_spec = importlib.util.spec_from_file_location(name, scripts / filename)
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_v86_uses_v85_only_as_a_required_exclusion_boundary() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    boundary = spec["scientific_boundary"]

    assert spec["selection_rule_frozen_before_calibration"] is True
    assert spec["parent_negative_result_seal"].endswith(
        "choice_c_v85_negative_heldout_seal.yaml"
    )
    assert boundary["v85_used_only_as_exclusion_boundary"] is True
    for key in (
        "v85_answers_used_for_design",
        "v85_references_used_for_design",
        "v85_task_scores_used_for_design",
        "v85_failure_identities_used_for_design",
        "v85_generated_token_ids_used_for_design",
        "v85_per_sample_telemetry_used_for_design",
        "v85_aggregate_outcomes_used_for_candidate_tuning",
    ):
        assert boundary[key] is False
    assert {task["selection_seed"] for task in spec["tasks"]} == {20280203}
    assert {task["sample_count"] for task in spec["tasks"]} == {128}

    materializer = (
        package / "slurm/choice_c_v86_materialize_robust_calibration.sbatch"
    ).read_text(encoding="utf-8")
    assert "choice-c-v85-v84-frozen-untouched-heldout-quality-memory-v1/inputs" in materializer
    assert "required sealed v85 exclusion directory is missing" in materializer
    assert "/choice-c-v*/inputs" in materializer
    assert '--seed 20280203' in materializer
    assert '--subset-size 128' in materializer


def test_v86_candidate_family_is_same_byte_exact_only_and_label_blind() -> None:
    package = Path(__file__).parents[2]
    spec = _spec(package)
    base = load_config(package / spec["candidate_config"])
    candidates = spec["candidates"]
    protocol = _load_module(
        package,
        "choice_c_v74_protocol.py",
        "mosaickv_test_v86_candidate_protocol",
    )

    assert len(candidates) == 3
    assert {candidate["tier_policy"] for candidate in candidates} == {
        "exact_only_replacement"
    }
    assert {candidate["overrides"]["active_budget_bytes"] for candidate in candidates} == {
        28_500_000
    }
    common = protocol.apply_common_cache_policy(base, spec["common_cache_policy"])
    applied = [protocol.apply_candidate(common, candidate) for candidate in candidates]
    assert all(not config.prototypes.enabled for config in applied)
    assert all(not config.residual.enabled for config in applied)
    assert all(not config.repair.enabled for config in applied)

    objective = spec["calibration_objective"]
    robust = spec["robust_fidelity_analysis"]
    assert objective["labels_accessed"] is False
    assert objective["official_task_scores_accessed"] is False
    assert objective["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert robust["labels_accessed"] is False
    assert robust["official_task_scores_accessed"] is False
    assert robust["require_every_task_and_fold"] is True
    assert robust["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert robust["fullkv_token_noninferiority_margin_vs_prompt_topk"] == 0.05


def test_v86_fold_assignment_is_deterministic_balanced_and_label_independent() -> None:
    package = Path(__file__).parents[2]
    analyzer = _load_module(
        package,
        "analyze_choice_c_v86_robust_fidelity.py",
        "mosaickv_test_v86_robust_analyzer",
    )
    sample_ids = [f"sample-{index:03d}" for index in range(128)]
    first = analyzer._fold_assignments(sample_ids, fold_count=4, salt="v86-test")
    second = analyzer._fold_assignments(list(reversed(sample_ids)), fold_count=4, salt="v86-test")

    assert first == second
    assert {fold: list(first.values()).count(fold) for fold in range(4)} == {
        0: 32,
        1: 32,
        2: 32,
        3: 32,
    }


def test_v86_submission_chain_contains_no_heldout_job() -> None:
    package = Path(__file__).parents[2]
    submit = (package / "scripts/submit_choice_c_v86.sh").read_text(encoding="utf-8")
    submitted = "\n".join(line for line in submit.splitlines() if "sbatch" in line)

    assert "heldout" not in submitted
    assert "choice_c_v86_materialize_robust_calibration.sbatch" in submit
    assert "choice_c_v86_robust_calibration_a100.sbatch" in submit
    assert "choice_c_v86_freeze.sbatch" in submit
    assert "assert_clean_worktree.sh" in submit


def _calibration_row(sample_id: str, answer: str, tokens: list[int], active: int) -> dict:
    return {
        "sample_id": sample_id,
        "answer": answer,
        "generated_token_ids": tokens,
        "active_kv_bytes": active,
        "residual_kv_bytes": 0,
        "compression_time": 0.1,
        "decode_time": 0.2,
        "end_to_end_time": 0.3,
        "generated_tokens": len(tokens),
        "repair_count": 0,
        "status": "completed",
        "error": None,
        "manifest_path": "manifest.json",
        "byte_budget_exact_logical_active_bytes": active,
        "byte_budget_prototype_logical_active_bytes": 0,
        "byte_budget_logical_active_bytes": active,
        "prototype_count": 0,
    }


def test_v86_robust_analyzer_rejects_an_aggregate_only_pass(tmp_path: Path) -> None:
    package = Path(__file__).parents[2]
    analyzer = _load_module(
        package,
        "analyze_choice_c_v86_robust_fidelity.py",
        "mosaickv_test_v86_robust_fail_closed",
    )
    sample_ids = [f"docvqa:{index}" for index in range(8)]
    assignments = analyzer._fold_assignments(
        sample_ids,
        fold_count=2,
        salt="v86-synthetic:docvqa",
    )
    base_config = package / "configs/experiments/choice_c_v72_frozen_candidate.yaml"
    spec = {
        "experiment_id": "synthetic_v86",
        "classification": "preregistered_label_blind_calibration",
        "frozen_candidate_classification": "frozen_synthetic_v86",
        "candidate_config": str(base_config),
        "common_cache_policy": {
            "block_size": 4,
            "physical_budget_block_size": 4,
            "physical_budget_mandatory_policy": "last_token",
        },
        "candidates": [{"id": "aggregate_only", "rationale": "test", "overrides": {}}],
        "tasks": [{"task": "docvqa", "sample_count": 8}],
        "calibration_objective": {
            "labels_accessed": False,
            "official_task_scores_accessed": False,
            "minimum_active_reduction_fraction_vs_prompt_topk": 0.30,
            "minimum_active_plus_residual_reduction_fraction_vs_prompt_topk": 0.30,
            "fullkv_answer_noninferiority_margin_vs_prompt_topk": 0.05,
            "maximum_prototype_logical_fraction": 0.0,
            "require_zero_residual_bytes": True,
        },
        "robust_fidelity_analysis": {
            "labels_accessed": False,
            "official_task_scores_accessed": False,
            "fold_count": 2,
            "fold_assignment": "sha256_sorted_round_robin",
            "fold_assignment_salt": "v86-synthetic",
            "expected_samples_per_fold": 4,
            "require_every_task_and_fold": True,
            "require_base_memory_and_fidelity_gates": True,
            "fullkv_answer_noninferiority_margin_vs_prompt_topk": 0.05,
            "fullkv_token_noninferiority_margin_vs_prompt_topk": 0.05,
        },
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    trials = []
    for arm, active in (
        ("full_kv", 150),
        ("prompt_attention_topk", 100),
        ("aggregate_only", 65),
    ):
        rows = []
        prompt_correct_seen = {0: 0, 1: 0}
        for index, sample_id in enumerate(sample_ids):
            fold = assignments[sample_id]
            full_answer = f"answer-{index}"
            if arm == "full_kv":
                correct = True
            elif arm == "prompt_attention_topk":
                correct = prompt_correct_seen[fold] < 2
                prompt_correct_seen[fold] += 1
            else:
                correct = fold == 1
            answer = full_answer if correct else f"wrong-{index}"
            tokens = [index, 7] if correct else [index, 9]
            rows.append(_calibration_row(sample_id, answer, tokens, active))
        trials.append({"arm": arm, "rows": rows})
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    telemetry = {
        "status": "completed",
        "canonical_eligible": True,
        "labels_accessed_for_selection": False,
        "official_task_scores_accessed_for_selection": False,
        "sample_ids": sample_ids,
        "arm_order": ["full_kv", "prompt_attention_topk", "aggregate_only"],
        "source_snapshot": {"snapshot_sha256": "e" * 64},
        "common_cache_policy": spec["common_cache_policy"],
        "ledger": "calibration.json",
        "ledger_sha256": "f" * 64,
        "trials": trials,
    }
    (telemetry_dir / "v86-docvqa-test.json").write_text(
        json.dumps(telemetry), encoding="utf-8"
    )
    frozen = tmp_path / "analysis/frozen_candidate.yaml"

    result = analyzer.analyze(spec_path, tmp_path, frozen, run_prefix="v86")

    ranking = result["candidate_ranking"][0]
    assert ranking["base_gate_eligible"] is True
    assert ranking["every_fold_answer_gate_eligible"] is False
    assert ranking["every_fold_token_gate_eligible"] is False
    assert result["selected_candidate"] is None
    assert not frozen.exists()
