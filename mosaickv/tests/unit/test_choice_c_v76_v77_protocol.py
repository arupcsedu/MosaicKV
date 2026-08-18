from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v75_negative_result_seal_binds_existing_immutable_artifacts() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v75_negative_heldout_seal.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert seal["classification"] == "sealed_negative_heldout_quality_memory_result"
    assert seal["overall_quality_memory_gate_pass"] is False
    assert seal["candidate_must_not_be_retuned_from_this_result"] is True
    assert seal["labels_must_not_enter_future_calibration"] is True
    protocol = package / seal["protocol"]["path"]
    assert _sha(protocol) == seal["protocol"]["sha256"]
    for record in seal["artifacts"].values():
        assert len(record["sha256"]) == 64
        if Path(record["path"]).is_file():
            assert _sha(Path(record["path"])) == record["sha256"]
    for record in seal["heldout_ledgers"].values():
        assert len(record["sha256"]) == 64
        if Path(record["path"]).is_file():
            assert _sha(Path(record["path"])) == record["sha256"]
    gate_path = Path(seal["artifacts"]["gate"]["path"])
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text())
        assert gate["quality_memory_gate_pass"] is False
        assert gate["task_results"]["docvqa"]["quality_memory_gate_pass"] is False
        assert gate["task_results"]["mmstar"]["quality_memory_gate_pass"] is True
        assert gate["task_results"]["chartqa"]["quality_memory_gate_pass"] is True


def test_v76_uses_new_label_blind_data_and_keeps_both_gates_and_cap() -> None:
    package = Path(__file__).parents[2]
    spec = yaml.safe_load(
        (package / "configs/experiments/choice_c_v76_disjoint_ocr_calibration.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert spec["selection_rule_frozen_before_calibration"] is True
    assert spec["protocol_amendment"]["parent_sample_answers_used_for_candidate_design"] is False
    assert (
        spec["protocol_amendment"]["parent_references_or_task_scores_used_for_candidate_design"]
        is False
    )
    assert spec["protocol_amendment"]["configured_active_byte_cap_frozen_bytes"] == 27_500_000
    assert {candidate["overrides"]["active_budget_bytes"] for candidate in spec["candidates"]} == {
        27_500_000
    }
    assert spec["calibration_objective"]["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert (
        spec["calibration_objective"][
            "minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"
        ]
        == 0.30
    )
    assert (
        spec["calibration_objective"]["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    )
    assert {task["task"] for task in spec["tasks"]} == {"docvqa", "textvqa"}
    assert {task["selection_seed"] for task in spec["tasks"]} == {20270829}


def test_v76_v77_materializers_exclude_sealed_and_calibration_ledgers() -> None:
    package = Path(__file__).parents[2]
    v76 = (package / "slurm/choice_c_v76_materialize_ocr_calibration.sbatch").read_text()
    v77 = (package / "slurm/choice_c_v77_materialize_heldout.sbatch").read_text()
    v75_inputs = "choice-c-v75-frozen-heldout-quality-memory-v6/inputs"

    assert v75_inputs in v76
    assert v75_inputs in v77
    assert '"${V76_ROOT}/inputs"' in v77
    assert "--selection-freeze" not in v76
    assert "--selection-freeze" in v77


def test_v77_remains_blocked_until_a_new_freeze_exists() -> None:
    package = Path(__file__).parents[2]
    spec = yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v77_untouched_heldout_quality_memory.yaml"
        ).read_text(encoding="utf-8")
    )
    submit_v76 = (package / "scripts/submit_choice_c_v76.sh").read_text()
    submit_v77 = (package / "scripts/submit_choice_c_v77_after_freeze.sh").read_text()

    assert spec["execution_status"] == "blocked_until_v76_freeze_and_clean_commit"
    assert spec["frozen_before_sample_materialization"] is True
    assert "choice_c_v77" not in "\n".join(
        line for line in submit_v76.splitlines() if "sbatch" in line
    )
    assert 'if [[ ! -f "${FROZEN}" ]]' in submit_v77
    assert {task["selection_seed"] for task in spec["tasks"]} == {20270831}


def test_v76_negative_seal_binds_gate_and_label_blind_audit() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v76_negative_calibration_seal.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert seal["classification"] == "sealed_negative_label_blind_calibration_result"
    assert seal["memory_gate_pass"] is True
    assert seal["fidelity_gate_pass"] is False
    assert seal["frozen_candidate_created"] is False
    assert seal["future_protocol"]["v77_remains_blocked"] is True
    assert seal["future_protocol"]["gates_must_remain_unchanged"] is True
    assert _sha(package / seal["protocol"]["path"]) == seal["protocol"]["sha256"]
    for record in seal["artifacts"].values():
        path = Path(record["path"])
        if not path.is_absolute():
            path = package / path
        assert len(record["sha256"]) == 64
        if path.is_file():
            assert _sha(path) == record["sha256"]
    audit_path = Path(seal["artifacts"]["selection_audit_json"]["path"])
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit["benchmark_labels_accessed"] is False
        assert audit["official_task_scores_accessed"] is False
        assert audit["reference_fields_accessed"] is False


def test_v78_keeps_gates_and_cap_while_using_new_disjoint_inputs() -> None:
    package = Path(__file__).parents[2]
    spec = yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v78_disjoint_visual_balance_calibration.yaml"
        ).read_text(encoding="utf-8")
    )

    assert spec["selection_rule_frozen_before_calibration"] is True
    assert spec["parent_negative_result_seal"].endswith(
        "choice_c_v76_negative_calibration_seal.yaml"
    )
    assert spec["protocol_amendment"]["parent_benchmark_labels_accessed"] is False
    assert spec["protocol_amendment"]["parent_official_task_scores_accessed"] is False
    assert spec["protocol_amendment"]["configured_active_byte_cap_frozen_bytes"] == 27_500_000
    assert {candidate["overrides"]["active_budget_bytes"] for candidate in spec["candidates"]} == {
        27_500_000
    }
    assert {
        candidate["overrides"]["stable_head_visual_fraction"] for candidate in spec["candidates"]
    } == {0.0, 0.5, 0.6}
    assert spec["calibration_objective"]["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert (
        spec["calibration_objective"][
            "minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"
        ]
        == 0.30
    )
    assert (
        spec["calibration_objective"]["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    )
    assert {task["selection_seed"] for task in spec["tasks"]} == {20270903}

    materializer = (package / "slurm/choice_c_v78_materialize_visual_calibration.sbatch").read_text(
        encoding="utf-8"
    )
    submit = (package / "scripts/submit_choice_c_v78.sh").read_text(encoding="utf-8")
    assert "choice-c-v76-disjoint-ocr-calibration-v1/inputs" in materializer
    assert "--seed 20270903" in materializer
    assert "choice_c_v77" not in "\n".join(line for line in submit.splitlines() if "sbatch" in line)


def test_v78_negative_seal_binds_failed_fidelity_gate() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v78_negative_calibration_seal.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert seal["classification"] == "sealed_negative_label_blind_calibration_result"
    assert seal["memory_gate_pass"] is True
    assert seal["fidelity_gate_pass"] is False
    assert seal["frozen_candidate_created"] is False
    assert seal["label_blind_design_finding"]["stop_increasing_fixed_visual_quotas"] is True
    assert seal["future_protocol"]["heldout_must_not_be_submitted"] is True
    assert _sha(package / seal["protocol"]["path"]) == seal["protocol"]["sha256"]
    for record in seal["artifacts"].values():
        path = Path(record["path"])
        assert len(record["sha256"]) == 64
        if path.is_file():
            assert _sha(path) == record["sha256"]


def test_v79_adaptive_swap_protocol_is_disjoint_and_keeps_both_gates() -> None:
    package = Path(__file__).parents[2]
    spec = yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v79_disjoint_adaptive_swap_calibration.yaml"
        ).read_text(encoding="utf-8")
    )

    assert spec["selection_rule_frozen_before_calibration"] is True
    assert spec["parent_negative_result_seal"].endswith(
        "choice_c_v78_negative_calibration_seal.yaml"
    )
    assert spec["protocol_amendment"]["fixed_visual_quotas_retired"] is True
    assert spec["protocol_amendment"]["parent_benchmark_labels_accessed"] is False
    assert spec["protocol_amendment"]["parent_official_task_scores_accessed"] is False
    assert spec["protocol_amendment"]["configured_active_byte_cap_frozen_bytes"] == 27_500_000
    assert {candidate["overrides"]["active_budget_bytes"] for candidate in spec["candidates"]} == {
        27_500_000
    }
    assert {
        candidate["overrides"]["stable_head_visual_fraction"] for candidate in spec["candidates"]
    } == {0.0}
    adaptive = [
        candidate
        for candidate in spec["candidates"]
        if candidate["overrides"]["stable_head_adaptive_visual_swaps"]
    ]
    assert len(adaptive) == 2
    assert {
        candidate["overrides"]["stable_head_maximum_visual_swaps"] for candidate in adaptive
    } == {2}
    assert spec["calibration_objective"]["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert (
        spec["calibration_objective"][
            "minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"
        ]
        == 0.30
    )
    assert (
        spec["calibration_objective"]["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    )
    assert {task["selection_seed"] for task in spec["tasks"]} == {20270907}
    assert {task["sample_count"] for task in spec["tasks"]} == {32}

    materializer = (
        package / "slurm/choice_c_v79_materialize_adaptive_calibration.sbatch"
    ).read_text(encoding="utf-8")
    submit = (package / "scripts/submit_choice_c_v79.sh").read_text(encoding="utf-8")
    freeze = (package / "slurm/choice_c_v79_freeze.sbatch").read_text(encoding="utf-8")
    assert "choice-c-v78-disjoint-visual-balance-calibration-v1/inputs" in materializer
    assert "--seed 20270907" in materializer
    assert "--subset-size 32" in materializer
    assert "#SBATCH --nodes=2" in freeze
    assert "heldout" not in "\n".join(line for line in submit.splitlines() if "sbatch" in line)


def test_v79_negative_seal_binds_label_blind_swap_audit() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v79_negative_calibration_seal.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert seal["memory_gate_pass"] is True
    assert seal["fidelity_gate_pass"] is False
    assert seal["frozen_candidate_created"] is False
    assert seal["label_blind_design_finding"]["broad_adaptive_swaps_retired"] is True
    assert seal["label_blind_design_finding"]["causal_per_head_claim_permitted"] is False
    assert _sha(package / seal["protocol"]["path"]) == seal["protocol"]["sha256"]
    for record in seal["artifacts"].values():
        path = Path(record["path"])
        assert len(record["sha256"]) == 64
        if path.is_file():
            assert _sha(path) == record["sha256"]


def test_v80_risk_gated_protocol_is_larger_disjoint_and_keeps_gates() -> None:
    package = Path(__file__).parents[2]
    spec = yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v80_disjoint_risk_gated_swap_calibration.yaml"
        ).read_text(encoding="utf-8")
    )

    assert spec["selection_rule_frozen_before_calibration"] is True
    assert spec["parent_negative_result_seal"].endswith(
        "choice_c_v79_negative_calibration_seal.yaml"
    )
    assert spec["protocol_amendment"]["broad_adaptive_swaps_retired"] is True
    assert spec["protocol_amendment"]["exact_per_head_swap_pair_telemetry_required"] is True
    assert {task["sample_count"] for task in spec["tasks"]} == {48}
    assert {task["selection_seed"] for task in spec["tasks"]} == {20271123}
    assert {candidate["overrides"]["active_budget_bytes"] for candidate in spec["candidates"]} == {
        27_500_000
    }
    risk_candidates = [
        candidate
        for candidate in spec["candidates"]
        if candidate["overrides"]["stable_head_swap_risk_gate"]
    ]
    assert len(risk_candidates) == 2
    assert {
        candidate["overrides"]["stable_head_maximum_visual_swaps"] for candidate in risk_candidates
    } == {1}
    assert spec["calibration_objective"]["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert (
        spec["calibration_objective"][
            "minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"
        ]
        == 0.30
    )
    assert (
        spec["calibration_objective"]["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    )

    materializer = (package / "slurm/choice_c_v80_materialize_risk_calibration.sbatch").read_text(
        encoding="utf-8"
    )
    submit = (package / "scripts/submit_choice_c_v80.sh").read_text(encoding="utf-8")
    assert "choice-c-v79-disjoint-adaptive-swap-calibration-v1/inputs" in materializer
    assert "--seed 20271123" in materializer
    assert "--subset-size 48" in materializer
    assert "heldout" not in "\n".join(line for line in submit.splitlines() if "sbatch" in line)


def test_v80_negative_seal_retires_stable_head_swaps() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v80_negative_calibration_seal.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert seal["classification"] == "sealed_negative_label_blind_calibration_result"
    assert seal["memory_gate_pass"] is True
    assert seal["fidelity_gate_pass"] is False
    assert seal["frozen_candidate_created"] is False
    assert seal["label_blind_design_finding"]["stable_head_swaps_retired"] is True
    assert seal["future_protocol"]["gates_must_remain_unchanged"] is True
    assert _sha(package / seal["protocol"]["path"]) == seal["protocol"]["sha256"]
    for record in seal["artifacts"].values():
        path = Path(record["path"])
        assert len(record["sha256"]) == 64
        if path.is_file():
            assert _sha(path) == record["sha256"]


def test_v81_same_byte_decomposition_is_disjoint_and_keeps_gates() -> None:
    package = Path(__file__).parents[2]
    spec = yaml.safe_load(
        (
            package
            / "configs/experiments/choice_c_v81_disjoint_exact_future_rerank_calibration.yaml"
        ).read_text(encoding="utf-8")
    )

    assert spec["selection_rule_frozen_before_calibration"] is True
    assert spec["parent_negative_result_seal"].endswith(
        "choice_c_v80_negative_calibration_seal.yaml"
    )
    amendment = spec["protocol_amendment"]
    assert amendment["stable_head_swaps_retired"] is True
    assert amendment["same_byte_decomposition_required"] is True
    assert amendment["equal_size_replacements_only"] is True
    assert amendment["mandatory_blocks_never_replaced"] is True
    assert {task["sample_count"] for task in spec["tasks"]} == {64}
    assert {task["selection_seed"] for task in spec["tasks"]} == {20271211}
    assert {candidate["overrides"]["active_budget_bytes"] for candidate in spec["candidates"]} == {
        27_500_000
    }
    exact = [
        candidate
        for candidate in spec["candidates"]
        if candidate["tier_policy"] == "exact_only_replacement"
    ]
    assert len(exact) == 3
    reranked = [
        candidate
        for candidate in exact
        if candidate["overrides"]["stable_head_future_query_reranking"]
    ]
    replacement_counts = {
        candidate["overrides"]["stable_head_maximum_future_replacements"] for candidate in reranked
    }
    assert replacement_counts == {
        2,
        4,
    }
    assert all(candidate["overrides"]["maximum_prototypes_per_head"] == 0 for candidate in exact)
    objective = spec["calibration_objective"]
    assert objective["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert spec["decomposition_analysis"]["require_equal_packed_active_bytes_within_task"] is True

    materializer = (
        package / "slurm/choice_c_v81_materialize_exact_future_calibration.sbatch"
    ).read_text(encoding="utf-8")
    submit = (package / "scripts/submit_choice_c_v81.sh").read_text(encoding="utf-8")
    assert "choice-c-v80-disjoint-risk-gated-swap-calibration-v1/inputs" in materializer
    assert "--seed 20271211" in materializer
    assert "--subset-size 64" in materializer
    assert "heldout" not in "\n".join(line for line in submit.splitlines() if "sbatch" in line)


def test_v81_negative_seal_records_degenerate_prompt_forecast() -> None:
    package = Path(__file__).parents[2]
    seal = yaml.safe_load(
        (package / "configs/ledgers/choice_c_v81_negative_calibration_seal.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert seal["classification"] == "sealed_negative_label_blind_calibration_result"
    assert seal["memory_gate_pass"] is True
    assert seal["fidelity_gate_pass"] is False
    assert seal["same_byte_decomposition_pass"] is True
    assert seal["forecast_divergence_observed"] is False
    assert seal["frozen_candidate_created"] is False
    finding = seal["label_blind_design_finding"]
    assert finding["future_rerank_candidates_per_sample"] == 11904
    assert finding["accepted_future_reranks_per_sample"] == 0
    assert finding["next_reranker_must_fail_closed_without_forecast_divergence"] is True
    assert _sha(package / seal["protocol"]["path"]) == seal["protocol"]["sha256"]
    for record in seal["artifacts"].values():
        path = Path(record["path"])
        assert len(record["sha256"]) == 64
        if path.is_file():
            assert _sha(path) == record["sha256"]


def test_v82_hybrid_divergence_protocol_is_disjoint_and_keeps_gates() -> None:
    package = Path(__file__).parents[2]
    spec = yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v82_disjoint_hybrid_divergence_calibration.yaml"
        ).read_text(encoding="utf-8")
    )

    assert spec["selection_rule_frozen_before_calibration"] is True
    assert spec["parent_negative_result_seal"].endswith(
        "choice_c_v81_negative_calibration_seal.yaml"
    )
    amendment = spec["protocol_amendment"]
    assert amendment["draft_cache_must_be_isolated"] is True
    assert amendment["deterministic_greedy_draft"] is True
    assert amendment["forecast_divergence_fail_closed"] is True
    assert amendment["minimum_prompt_forecast_total_variation"] == 0.01
    assert {task["sample_count"] for task in spec["tasks"]} == {64}
    assert {task["selection_seed"] for task in spec["tasks"]} == {20271219}
    assert {candidate["overrides"]["active_budget_bytes"] for candidate in spec["candidates"]} == {
        27_500_000
    }
    reranked = [
        candidate
        for candidate in spec["candidates"]
        if candidate["overrides"]["stable_head_future_query_reranking"]
    ]
    assert len(reranked) == 2
    assert {candidate["overrides"]["forecast_mode"] for candidate in spec["candidates"]} == {
        "hybrid"
    }
    assert {candidate["overrides"]["forecast_draft_steps"] for candidate in reranked} == {2, 4}
    assert all(
        candidate["overrides"]["stable_head_future_divergence_fail_closed"]
        for candidate in reranked
    )
    assert all(
        candidate["overrides"]["stable_head_future_minimum_prompt_forecast_tv"] == 0.01
        for candidate in reranked
    )
    objective = spec["calibration_objective"]
    assert objective["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"] == 0.30
    assert objective["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05

    materializer = (package / "slurm/choice_c_v82_materialize_hybrid_calibration.sbatch").read_text(
        encoding="utf-8"
    )
    freeze = (package / "slurm/choice_c_v82_freeze.sbatch").read_text(encoding="utf-8")
    submit = (package / "scripts/submit_choice_c_v82.sh").read_text(encoding="utf-8")
    assert "choice-c-v81-disjoint-exact-future-rerank-calibration-v1/inputs" in materializer
    assert "--seed 20271219" in materializer
    assert "--subset-size 64" in materializer
    assert "#SBATCH --nodes=2" in freeze
    assert "heldout" not in "\n".join(line for line in submit.splitlines() if "sbatch" in line)
