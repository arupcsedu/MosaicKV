from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import yaml


def _load_script() -> ModuleType:
    scripts = Path(__file__).parents[2] / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "analyze_choice_c_v74_label_blind_calibration.py"
    spec = importlib.util.spec_from_file_location("mosaickv_test_v74_analyzer", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYZER = _load_script()


def _row(sample_id: str, answer: str, tokens: list[int], active: int) -> dict:
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
        "byte_budget_configured_active_byte_cap": active + 5,
        "byte_budget_dense_attainable_budget_bytes": active,
        "byte_budget_dense_quantization_underfill_bytes": 5,
        "byte_budget_capacity_fraction_reduction_bytes": 0,
        "byte_budget_logical_underfill_vs_dense_budget_bytes": 1,
        "byte_budget_final_underfill_vs_dense_budget_bytes": 0,
        "byte_budget_packed_padding_bytes": 1,
        "byte_budget_exact_logical_active_bytes": active - 1,
        "byte_budget_prototype_logical_active_bytes": 1,
        "byte_budget_logical_active_bytes": active,
        "external_stable_route_count": 2,
        "external_stable_requested_positions": 26,
        "external_stable_selected_positions": 26,
        "external_stable_shortfall_positions": 0,
        "external_stable_selected_bytes": active - 1,
        "external_stable_routes_with_shortfall_count": 0,
        "tier_active_cache_length_count": 1,
        "tier_active_cache_length_min": 13,
        "tier_active_cache_length_max": 13,
        "prompt_mass_exact_prompt_mass_fraction": 0.70,
        "prompt_mass_represented_prompt_mass_fraction": 0.75,
        "prototype_count": 2,
    }


def test_label_blind_analyzer_selects_fullkv_fidelity_under_memory_gate(
    tmp_path: Path,
) -> None:
    package_root = Path(__file__).parents[2]
    base_config = package_root / "configs/experiments/choice_c_v72_frozen_candidate.yaml"
    candidates = [
        {"id": "good", "rationale": "test", "overrides": {}},
        {"id": "bad", "rationale": "test", "overrides": {}},
    ]
    spec = {
        "experiment_id": "synthetic_v74",
        "classification": "preregistered_label_blind_calibration",
        "candidate_config": str(base_config),
        "common_cache_policy": {
            "block_size": 8,
            "physical_budget_block_size": 8,
            "physical_budget_mandatory_policy": "last_token",
        },
        "candidates": candidates,
        "tasks": [{"task": "docvqa", "sample_count": 2}],
        "calibration_objective": {
            "labels_accessed": False,
            "official_task_scores_accessed": False,
            "minimum_active_reduction_fraction_vs_prompt_topk": 0.30,
            "minimum_active_plus_residual_reduction_fraction_vs_prompt_topk": 0.30,
            "fullkv_answer_noninferiority_margin_vs_prompt_topk": 0.05,
            "maximum_prototype_logical_fraction": 0.05,
        },
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    arms = ("full_kv", "prompt_attention_topk", "good", "bad")
    answers = {
        "full_kv": ("alpha", "beta"),
        "prompt_attention_topk": ("alpha", "wrong"),
        "good": ("alpha", "beta"),
        "bad": ("wrong", "wrong"),
    }
    active = {"full_kv": 150, "prompt_attention_topk": 100, "good": 65, "bad": 65}
    trials = []
    for arm in arms:
        trials.append(
            {
                "arm": arm,
                "rows": [
                    _row(f"docvqa:{index}", answer, [index, 7], active[arm])
                    for index, answer in enumerate(answers[arm])
                ],
            }
        )
    telemetry = {
        "status": "completed",
        "canonical_eligible": True,
        "labels_accessed_for_selection": False,
        "official_task_scores_accessed_for_selection": False,
        "sample_ids": ["docvqa:0", "docvqa:1"],
        "arm_order": list(arms),
        "source_snapshot": {"snapshot_sha256": "a" * 64},
        "common_cache_policy": spec["common_cache_policy"],
        "ledger": "calibration.json",
        "ledger_sha256": "b" * 64,
        "trials": trials,
    }
    (telemetry_dir / "v74-docvqa-test.json").write_text(json.dumps(telemetry), encoding="utf-8")
    frozen = tmp_path / "analysis/frozen.yaml"

    result = ANALYZER.analyze(spec_path, tmp_path, frozen)

    assert result["selected_candidate"] == "good"
    assert result["labels_accessed"] is False
    assert result["common_cache_policy"] == spec["common_cache_policy"]
    ranking = {row["candidate_id"]: row for row in result["candidate_ranking"]}
    assert ranking["good"]["gate_eligible"] is True
    assert ranking["good"]["exact_dominant_eligible"] is True
    assert ranking["good"]["mean_dense_attainable_budget_bytes"] == 65
    assert ranking["good"]["mean_final_underfill_vs_dense_budget_bytes"] == 0
    assert ranking["good"]["mean_packed_padding_bytes"] == 1
    assert ranking["good"]["mean_external_stable_shortfall_positions"] == 0
    assert ranking["good"]["mean_exact_prompt_mass_fraction"] == 0.70
    assert ranking["good"]["mean_represented_prompt_mass_fraction"] == 0.75
    assert ranking["good"]["mean_prototype_count"] == 2
    assert ranking["bad"]["fidelity_eligible"] is False
    payload = yaml.safe_load(frozen.read_text(encoding="utf-8"))
    assert payload["selected_candidate"] == "good"
    assert payload["labels_accessed"] is False
    assert payload["common_cache_policy"] == spec["common_cache_policy"]

    versioned_root = tmp_path / "versioned"
    versioned_telemetry = versioned_root / "telemetry"
    versioned_telemetry.mkdir(parents=True)
    (versioned_telemetry / "v76-docvqa-test.json").write_text(
        json.dumps(telemetry), encoding="utf-8"
    )
    versioned_frozen = versioned_root / "analysis/frozen.yaml"
    versioned = ANALYZER.analyze(
        spec_path,
        versioned_root,
        versioned_frozen,
        run_prefix="v76",
    )
    assert versioned["selected_candidate"] == "good"
    assert versioned_frozen.is_file()


def test_label_blind_analyzer_fails_closed_on_future_rerank_risk_gates(
    tmp_path: Path,
) -> None:
    package_root = Path(__file__).parents[2]
    base_config = package_root / "configs/experiments/choice_c_v72_frozen_candidate.yaml"
    candidates = [
        {"id": "control", "rationale": "test", "overrides": {}},
        {"id": "inactive", "rationale": "test", "overrides": {}},
        {"id": "safe", "rationale": "test", "overrides": {}},
    ]
    spec = {
        "experiment_id": "synthetic_v84",
        "classification": "preregistered_label_blind_calibration",
        "candidate_config": str(base_config),
        "common_cache_policy": {
            "block_size": 4,
            "physical_budget_block_size": 4,
            "physical_budget_mandatory_policy": "last_token",
        },
        "candidates": candidates,
        "tasks": [{"task": "docvqa", "sample_count": 2}],
        "calibration_objective": {
            "labels_accessed": False,
            "official_task_scores_accessed": False,
            "minimum_active_reduction_fraction_vs_prompt_topk": 0.30,
            "minimum_active_plus_residual_reduction_fraction_vs_prompt_topk": 0.30,
            "fullkv_answer_noninferiority_margin_vs_prompt_topk": 0.05,
            "maximum_prototype_logical_fraction": 0.05,
            "require_zero_residual_bytes": True,
        },
        "future_rerank_risk_analysis": {
            "control_arm": "control",
            "candidate_arms": ["inactive", "safe"],
            "require_positive_accepted_reranks_each_task": True,
            "require_positive_risk_rejections_each_task": True,
            "require_not_below_control_each_task": True,
            "fullkv_answer_noninferiority_margin_vs_control": 0.0,
        },
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    arms = ("full_kv", "prompt_attention_topk", "control", "inactive", "safe")
    trials = []
    for arm in arms:
        rows = [
            _row(
                "docvqa:0",
                "alpha",
                [0, 7],
                65
                if arm not in {"full_kv", "prompt_attention_topk"}
                else (150 if arm == "full_kv" else 100),
            ),
            _row(
                "docvqa:1",
                "beta",
                [1, 7],
                65
                if arm not in {"full_kv", "prompt_attention_topk"}
                else (150 if arm == "full_kv" else 100),
            ),
        ]
        for row in rows:
            row["byte_budget_prototype_logical_active_bytes"] = 0
            row["byte_budget_exact_logical_active_bytes"] = row["active_kv_bytes"]
            row["prototype_count"] = 0
            if arm == "safe":
                row["external_stable_future_rerank_count"] = 1
                row["external_stable_future_rerank_risk_rejected_count"] = 2
            elif arm == "inactive":
                row["external_stable_future_rerank_count"] = 0
                row["external_stable_future_rerank_risk_rejected_count"] = 2
        trials.append({"arm": arm, "rows": rows})
    telemetry = {
        "status": "completed",
        "canonical_eligible": True,
        "labels_accessed_for_selection": False,
        "official_task_scores_accessed_for_selection": False,
        "sample_ids": ["docvqa:0", "docvqa:1"],
        "arm_order": list(arms),
        "source_snapshot": {"snapshot_sha256": "c" * 64},
        "common_cache_policy": spec["common_cache_policy"],
        "ledger": "calibration.json",
        "ledger_sha256": "d" * 64,
        "trials": trials,
    }
    (telemetry_dir / "v84-docvqa-test.json").write_text(json.dumps(telemetry), encoding="utf-8")

    result = ANALYZER.analyze(
        spec_path,
        tmp_path,
        tmp_path / "analysis/frozen.yaml",
        run_prefix="v84",
    )

    ranking = {row["candidate_id"]: row for row in result["candidate_ranking"]}
    assert ranking["inactive"]["future_rerank_risk_activity_eligible"] is False
    assert ranking["inactive"]["gate_eligible"] is False
    assert ranking["safe"]["future_rerank_risk_activity_eligible"] is True
    assert ranking["safe"]["future_rerank_control_fidelity_eligible"] is True
    assert ranking["safe"]["residual_storage_eligible"] is True
    assert ranking["safe"]["gate_eligible"] is True
    # With equal fidelity and memory, the simpler no-rerank control wins the
    # preregistered lexicographic tie-break instead of rewarding complexity.
    assert result["selected_candidate"] == "control"


def test_label_blind_analyzer_requires_every_short_logit_probe_to_pass(
    tmp_path: Path,
) -> None:
    package_root = Path(__file__).parents[2]
    base_config = package_root / "configs/experiments/choice_c_v72_frozen_candidate.yaml"
    candidate_id = "short_logit"
    spec = {
        "experiment_id": "synthetic_v98",
        "classification": "preregistered_label_blind_calibration",
        "candidate_config": str(base_config),
        "common_cache_policy": {
            "block_size": 4,
            "physical_budget_block_size": 4,
            "physical_budget_mandatory_policy": "last_token",
        },
        "candidates": [
            {
                "id": candidate_id,
                "rationale": "test",
                "overrides": {
                    "stable_head_short_horizon_logit_stability": True,
                    "stable_head_logit_stability_shortlist_size": 2,
                    "stable_head_logit_stability_horizon": 2,
                    "stable_head_logit_stability_maximum_kl": 0.02,
                    "stable_head_logit_stability_maximum_logit_difference": 1.0,
                },
            }
        ],
        "tasks": [{"task": "docvqa", "sample_count": 2}],
        "calibration_objective": {
            "labels_accessed": False,
            "official_task_scores_accessed": False,
            "minimum_active_reduction_fraction_vs_prompt_topk": 0.30,
            "minimum_active_plus_residual_reduction_fraction_vs_prompt_topk": 0.30,
            "fullkv_answer_noninferiority_margin_vs_prompt_topk": 0.05,
            "maximum_prototype_logical_fraction": 0.0,
            "require_zero_residual_bytes": True,
        },
        "short_horizon_logit_stability_analysis": {
            "labels_accessed": False,
            "require_every_sample_to_pass": True,
            "require_fullkv_prefill_reuse": True,
            "forbid_second_prefill": True,
            "require_probe_state_discard_before_measured_decode": True,
            "require_greedy_token_match": True,
        },
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    arms = ("full_kv", "prompt_attention_topk", candidate_id)
    trials = []
    for arm, active in (("full_kv", 150), ("prompt_attention_topk", 100), (candidate_id, 65)):
        rows = [
            _row(f"docvqa:{index}", answer, [index, 7], active)
            for index, answer in enumerate(("alpha", "beta"))
        ]
        if arm == candidate_id:
            for row in rows:
                row.update(
                    {
                        "byte_budget_prototype_logical_active_bytes": 0,
                        "byte_budget_exact_logical_active_bytes": active,
                        "prototype_count": 0,
                        "logit_stability_enabled": True,
                        "logit_stability_evaluated_candidate_count": 2,
                        "logit_stability_selected_candidate_passed": True,
                        "logit_stability_selected_greedy_token_match": True,
                        "logit_stability_fullkv_prefill_reused": True,
                        "logit_stability_second_prefill_performed": False,
                        "logit_stability_probe_states_discarded_before_measured_decode": True,
                        "logit_stability_benchmark_labels_accessed": False,
                        "logit_stability_horizon": 2,
                        "logit_stability_maximum_kl_threshold": 0.02,
                        "logit_stability_maximum_logit_difference_threshold": 1.0,
                        "logit_stability_selected_maximum_kl": 0.01,
                        "logit_stability_selected_maximum_logit_difference": 0.5,
                        "logit_stability_used_fail_closed_scalar_anchor": False,
                    }
                )
        trials.append({"arm": arm, "rows": rows})
    telemetry = {
        "status": "completed",
        "canonical_eligible": True,
        "labels_accessed_for_selection": False,
        "official_task_scores_accessed_for_selection": False,
        "sample_ids": ["docvqa:0", "docvqa:1"],
        "arm_order": list(arms),
        "source_snapshot": {"snapshot_sha256": "e" * 64},
        "common_cache_policy": spec["common_cache_policy"],
        "ledger": "calibration.json",
        "ledger_sha256": "f" * 64,
        "trials": trials,
    }
    (telemetry_dir / "v98-docvqa-test.json").write_text(
        json.dumps(telemetry), encoding="utf-8"
    )

    result = ANALYZER.analyze(
        spec_path,
        tmp_path,
        tmp_path / "analysis/frozen.yaml",
        run_prefix="v98",
    )

    ranking = {row["candidate_id"]: row for row in result["candidate_ranking"]}
    assert ranking[candidate_id]["short_horizon_logit_stability_required"] is True
    assert ranking[candidate_id]["short_horizon_logit_stability_eligible"] is True
    assert ranking[candidate_id]["gate_eligible"] is True
    assert result["selected_candidate"] == candidate_id

    missing_root = tmp_path / "missing-boolean"
    missing_telemetry_dir = missing_root / "telemetry"
    missing_telemetry_dir.mkdir(parents=True)
    missing_telemetry = json.loads(json.dumps(telemetry))
    candidate_trial = next(
        trial for trial in missing_telemetry["trials"] if trial["arm"] == candidate_id
    )
    candidate_trial["rows"][0]["logit_stability_selected_candidate_passed"] = None
    (missing_telemetry_dir / "v98-docvqa-test.json").write_text(
        json.dumps(missing_telemetry), encoding="utf-8"
    )
    missing_result = ANALYZER.analyze(
        spec_path,
        missing_root,
        missing_root / "analysis/frozen.yaml",
        run_prefix="v98",
    )
    missing_ranking = {
        row["candidate_id"]: row for row in missing_result["candidate_ranking"]
    }
    assert missing_ranking[candidate_id]["short_horizon_logit_stability_eligible"] is False
    assert missing_ranking[candidate_id]["gate_eligible"] is False
    assert missing_result["selected_candidate"] is None


def test_label_blind_analyzer_requires_progressive_fullkv_eviction_to_reach_target(
    tmp_path: Path,
) -> None:
    package_root = Path(__file__).parents[2]
    candidate_id = "progressive"
    spec = {
        "experiment_id": "synthetic_v99",
        "classification": "preregistered_label_blind_calibration",
        "candidate_config": str(
            package_root / "configs/experiments/choice_c_v72_frozen_candidate.yaml"
        ),
        "common_cache_policy": {
            "block_size": 4,
            "physical_budget_block_size": 4,
            "physical_budget_mandatory_policy": "last_token",
        },
        "candidates": [
            {
                "id": candidate_id,
                "rationale": "test",
                "overrides": {
                    "stable_head_progressive_logit_safe_eviction": True,
                    "stable_head_progressive_eviction_maximum_probes": 8,
                    "stable_head_progressive_eviction_horizon": 2,
                    "stable_head_progressive_eviction_maximum_kl": 0.02,
                    "stable_head_progressive_eviction_maximum_logit_difference": 1.0,
                },
            }
        ],
        "tasks": [{"task": "docvqa", "sample_count": 2}],
        "calibration_objective": {
            "labels_accessed": False,
            "official_task_scores_accessed": False,
            "minimum_active_reduction_fraction_vs_prompt_topk": 0.30,
            "minimum_active_plus_residual_reduction_fraction_vs_prompt_topk": 0.30,
            "fullkv_answer_noninferiority_margin_vs_prompt_topk": 0.05,
            "maximum_prototype_logical_fraction": 0.0,
            "require_zero_residual_bytes": True,
        },
        "progressive_logit_safe_eviction_analysis": {
            "labels_accessed": False,
            "require_start_from_fullkv": True,
            "require_every_sample_to_reach_target": True,
            "require_every_accepted_checkpoint_to_pass": True,
            "require_fullkv_prefill_reuse": True,
            "forbid_second_prefill": True,
            "require_probe_state_discard_before_measured_decode": True,
            "require_greedy_token_match": True,
        },
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    arms = ("full_kv", "prompt_attention_topk", candidate_id)
    trials = []
    for arm, active in (("full_kv", 150), ("prompt_attention_topk", 100), (candidate_id, 65)):
        rows = [
            _row(f"docvqa:{index}", answer, [index, 7], active)
            for index, answer in enumerate(("alpha", "beta"))
        ]
        if arm == candidate_id:
            for row in rows:
                row.update(
                    {
                        "byte_budget_prototype_logical_active_bytes": 0,
                        "byte_budget_exact_logical_active_bytes": active,
                        "prototype_count": 0,
                        "progressive_eviction_enabled": True,
                        "progressive_eviction_started_from_fullkv": True,
                        "progressive_eviction_target_reached": True,
                        "progressive_eviction_selected_candidate_passed": True,
                        "progressive_eviction_used_target_fallback": False,
                        "progressive_eviction_horizon": 2,
                        "progressive_eviction_maximum_kl_threshold": 0.02,
                        "progressive_eviction_maximum_logit_difference_threshold": 1.0,
                        "progressive_eviction_maximum_probes": 8,
                        "progressive_eviction_probes_used": 3,
                        "progressive_eviction_checkpoint_count": 3,
                        "progressive_eviction_accepted_checkpoints": 2,
                        "progressive_eviction_rejected_checkpoints": 1,
                        "progressive_eviction_initial_fullkv_logical_bytes": 150,
                        "progressive_eviction_target_active_bytes": 70,
                        "progressive_eviction_final_logical_active_bytes": 65,
                        "progressive_eviction_accepted_greedy_token_match": True,
                        "progressive_eviction_accepted_bytes_strictly_decrease": True,
                        "progressive_eviction_accepted_maximum_kl": 0.01,
                        "progressive_eviction_accepted_maximum_logit_difference": 0.5,
                        "progressive_eviction_protected_group_bytes": 8,
                        "progressive_eviction_protected_group_target_bytes": 10,
                        "progressive_eviction_protected_group_count": 1,
                        "progressive_eviction_fullkv_prefill_reused": True,
                        "progressive_eviction_second_prefill_performed": False,
                        "progressive_eviction_probe_states_discarded_before_measured_decode": True,
                        "progressive_eviction_benchmark_labels_accessed": False,
                    }
                )
        trials.append({"arm": arm, "rows": rows})
    telemetry = {
        "status": "completed",
        "canonical_eligible": True,
        "labels_accessed_for_selection": False,
        "official_task_scores_accessed_for_selection": False,
        "sample_ids": ["docvqa:0", "docvqa:1"],
        "arm_order": list(arms),
        "source_snapshot": {"snapshot_sha256": "1" * 64},
        "common_cache_policy": spec["common_cache_policy"],
        "ledger": "calibration.json",
        "ledger_sha256": "2" * 64,
        "trials": trials,
    }
    (telemetry_dir / "v99-docvqa-test.json").write_text(
        json.dumps(telemetry), encoding="utf-8"
    )

    result = ANALYZER.analyze(
        spec_path,
        tmp_path,
        tmp_path / "analysis/frozen.yaml",
        run_prefix="v99",
    )
    ranking = {row["candidate_id"]: row for row in result["candidate_ranking"]}
    assert ranking[candidate_id]["progressive_logit_safe_eviction_required"] is True
    assert ranking[candidate_id]["progressive_logit_safe_eviction_eligible"] is True
    assert ranking[candidate_id]["gate_eligible"] is True
    assert result["selected_candidate"] == candidate_id

    failed_root = tmp_path / "target-not-reached"
    failed_telemetry_dir = failed_root / "telemetry"
    failed_telemetry_dir.mkdir(parents=True)
    failed = json.loads(json.dumps(telemetry))
    candidate_trial = next(
        trial for trial in failed["trials"] if trial["arm"] == candidate_id
    )
    candidate_trial["rows"][0]["progressive_eviction_target_reached"] = False
    (failed_telemetry_dir / "v99-docvqa-test.json").write_text(
        json.dumps(failed), encoding="utf-8"
    )
    failed_result = ANALYZER.analyze(
        spec_path,
        failed_root,
        failed_root / "analysis/frozen.yaml",
        run_prefix="v99",
    )
    failed_ranking = {
        row["candidate_id"]: row for row in failed_result["candidate_ranking"]
    }
    assert failed_ranking[candidate_id]["progressive_logit_safe_eviction_eligible"] is False
    assert failed_ranking[candidate_id]["gate_eligible"] is False
    assert failed_result["selected_candidate"] is None


def test_label_blind_analyzer_requires_atomic_group_utilization_per_sample(
    tmp_path: Path,
) -> None:
    package_root = Path(__file__).parents[2]
    spec = {
        "experiment_id": "synthetic_v95",
        "classification": "preregistered_label_blind_calibration",
        "candidate_config": str(
            package_root / "configs/experiments/choice_c_v72_frozen_candidate.yaml"
        ),
        "common_cache_policy": {
            "block_size": 4,
            "physical_budget_block_size": 4,
            "physical_budget_mandatory_policy": "last_token",
        },
        "candidates": [
            {
                "id": "atomic",
                "rationale": "test",
                "overrides": {
                    "stable_head_query_coverage_layout_group_reservation": True,
                    "stable_head_query_coverage_layout_group_atomic_selection": True,
                    "stable_head_query_coverage_layout_group_minimum_utilization_fraction": 0.75,
                },
            }
        ],
        "tasks": [{"task": "docvqa", "sample_count": 2}],
        "calibration_objective": {
            "labels_accessed": False,
            "official_task_scores_accessed": False,
            "minimum_active_reduction_fraction_vs_prompt_topk": 0.30,
            "minimum_active_plus_residual_reduction_fraction_vs_prompt_topk": 0.30,
            "fullkv_answer_noninferiority_margin_vs_prompt_topk": 0.05,
            "maximum_prototype_logical_fraction": 0.05,
        },
    }
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    telemetry_dir = tmp_path / "telemetry"
    telemetry_dir.mkdir()
    trials = []
    for arm, active in (("full_kv", 150), ("prompt_attention_topk", 100), ("atomic", 65)):
        rows = [
            _row(f"docvqa:{index}", answer, [index, 7], active)
            for index, answer in enumerate(("alpha", "beta"))
        ]
        if arm == "atomic":
            prefix = "external_stable_query_coverage_initial_layout_group_"
            for index, row in enumerate(rows):
                row.update(
                    {
                        f"{prefix}candidate_count": 8,
                        f"{prefix}protected_group_count": 2,
                        f"{prefix}protected_block_count": 4,
                        f"{prefix}protected_bytes": 80 if index == 0 else 70,
                        f"{prefix}protected_score": 1.0,
                        f"{prefix}target_bytes": 100,
                        f"{prefix}accepted_candidate_count": 2,
                        f"{prefix}risk_rejected_count": 3,
                        f"{prefix}normalized_query_gain": 0.1,
                        f"{prefix}normalized_prompt_risk": 0.01,
                        f"{prefix}output_impact_risk": 0.02,
                    }
                )
        trials.append({"arm": arm, "rows": rows})
    telemetry = {
        "status": "completed",
        "canonical_eligible": True,
        "labels_accessed_for_selection": False,
        "official_task_scores_accessed_for_selection": False,
        "sample_ids": ["docvqa:0", "docvqa:1"],
        "arm_order": ["full_kv", "prompt_attention_topk", "atomic"],
        "source_snapshot": {"snapshot_sha256": "e" * 64},
        "common_cache_policy": spec["common_cache_policy"],
        "ledger": "calibration.json",
        "ledger_sha256": "f" * 64,
        "trials": trials,
    }
    (telemetry_dir / "v95-docvqa-test.json").write_text(json.dumps(telemetry), encoding="utf-8")

    result = ANALYZER.analyze(
        spec_path,
        tmp_path,
        tmp_path / "analysis/frozen.yaml",
        run_prefix="v95",
    )

    ranking = result["candidate_ranking"][0]
    assert ranking["memory_eligible"] is True
    assert ranking["fidelity_eligible"] is True
    assert ranking["layout_group_activity_eligible"] is True
    assert ranking["layout_group_utilization_eligible"] is False
    assert ranking["minimum_layout_group_reservation_utilization"] == 0.70
    assert ranking["gate_eligible"] is False
    assert result["selected_candidate"] is None
