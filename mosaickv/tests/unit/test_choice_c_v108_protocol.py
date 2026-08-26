from __future__ import annotations

from pathlib import Path

import yaml


def _protocol() -> dict[str, object]:
    package = Path(__file__).parents[2]
    value = yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v108_storage_aligned_preflight.yaml"
        ).read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def test_v108_keeps_memory_and_fidelity_gates_unchanged() -> None:
    protocol = _protocol()
    candidate = protocol["candidate"]
    gates = protocol["unchanged_memory_and_fidelity_gates"]
    assert isinstance(candidate, dict)
    assert isinstance(gates, dict)

    assert candidate["active_budget_bytes"] == 28_500_000
    assert candidate["hierarchical_maximum_forward_kl"] == 0.02
    assert candidate["hierarchical_maximum_absolute_logit_difference"] == 1.0
    assert candidate["hierarchical_greedy_token_match_required"] is True
    assert gates["minimum_active_reduction_fraction_vs_prompt_topk"] == 0.30
    assert gates["minimum_active_plus_residual_reduction_fraction_vs_prompt_topk"] == 0.30
    assert gates["fullkv_answer_noninferiority_margin_vs_prompt_topk"] == 0.05
    assert gates["fullkv_token_noninferiority_margin_vs_prompt_topk"] == 0.05


def test_v108_preregisters_complete_storage_accounting() -> None:
    protocol = _protocol()
    invariants = protocol["storage_boundary_invariants"]
    gate = protocol["preflight_gate"]
    assert isinstance(invariants, dict)
    assert isinstance(gate, dict)

    assert invariants["knapsack_bytes_must_equal_unique_selected_source_bytes"] is True
    assert invariants["unique_selected_source_bytes_must_equal_constructed_active_bytes"] is True
    assert invariants["physical_padding_reported_separately"] is True
    assert invariants["feasible_and_unrestricted_level_zero_artifact_namespaces_separated"]
    assert set(gate["accounting_trace_must_record"]) == {
        "knapsack_bytes",
        "unique_selected_source_bytes",
        "planner_tier_bytes",
        "external_exact_bytes",
        "constructed_active_bytes",
        "physical_padding_bytes",
    }


def test_v108_uses_fresh_identities_and_authorizes_only_the_four_sample_preflight() -> None:
    protocol = _protocol()
    identity = protocol["fresh_identity_plan"]
    downstream = protocol["downstream_actions"]
    source = protocol["candidate_source"]
    assert isinstance(identity, dict)
    assert isinstance(downstream, dict)
    assert isinstance(source, dict)

    assert identity["selection_seed"] == 108_019
    assert identity["sample_count_per_task"] == 2
    assert len(identity["additionally_excluded_v107_preflight_ledgers"]) == 2
    assert identity["ledger_materialization_authorized"] is True
    assert identity["gpu_preflight_authorized"] is True
    assert source["reviewed_implementation_git_sha"] == (
        "clean_authorization_commit_containing_reviewed_runtime"
    )
    assert downstream["preflight_materialization_authorized"] is True
    assert downstream["preflight_execution_authorized"] is True
    assert all(
        value is False
        for key, value in downstream.items()
        if key not in {"preflight_materialization_authorized", "preflight_execution_authorized"}
    )
