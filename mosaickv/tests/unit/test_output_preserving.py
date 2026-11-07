"""Synthetic correctness checks; these numbers are not benchmark results."""

from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mosaickv.adapters.huggingface.types import CachedKeyState  # noqa: E402
from mosaickv.backends.output_preserving import (  # noqa: E402
    _apply_layer_transfer_transactions,
    _build_layer_transfer_context,
    compose_layer_capacity_transfer_candidate,
    compose_route_exchange_candidate,
    layer_capacity_directional_single_candidates,
    layer_capacity_surrogate_candidates,
    layer_capacity_transfer_candidates,
    prompt_draft_route_risk,
    route_exchange_single_candidates,
    select_output_preserving,
    topk_anchor_capture,
    within_route_position_candidates,
)
from mosaickv.cache_state import FullKVState, Modality, ModalitySpan  # noqa: E402
from mosaickv.config import (  # noqa: E402
    ConfigurationError,
    OutputPreservingConfig,
    load_config,
)
from mosaickv.selection.output_preserving import (  # noqa: E402
    corrective_swaps,
    prepare_evidence,
    reconstruct,
)


def test_output_preserving_runner_accepts_registered_v4_classification():
    from run_output_preserving_sweep import validate_plan_classification

    classification = "preregistered_disjoint_development_validation_not_paper_evidence"
    assert validate_plan_classification(classification) == classification


@pytest.mark.parametrize(
    "classification",
    [None, "", "paper_evidence", "preregistered_disjoint_development_validation"],
)
def test_output_preserving_runner_rejects_unregistered_classification(classification):
    from run_output_preserving_sweep import validate_plan_classification

    with pytest.raises(ValueError, match="wrong experiment classification"):
        validate_plan_classification(classification)


def test_output_preserving_runner_uses_v9_registered_trace_counts():
    from run_output_preserving_sweep import registered_surrogate_trace_counts

    v9 = SimpleNamespace(
        logit_candidate_strategy="within_route_position_beam",
        logit_route_candidate_limit=16,
        logit_layer_transfer_limit=8,
    )
    legacy = SimpleNamespace(
        logit_candidate_strategy="layer_capacity_vectorized_surrogate_beam",
        logit_route_candidate_limit=9,
        logit_layer_transfer_limit=8,
    )
    assert registered_surrogate_trace_counts(v9) == (16, 4)
    assert registered_surrogate_trace_counts(legacy) == (8, 2)


def test_frozen_validation_merges_only_reviewed_unexecuted_disjoint_ledgers(tmp_path):
    from prepare_output_preserving_frozen_validation import merge_reviewed_ledgers
    from run_prefixkv_decode_preflight import digest

    route = {
        "schema_version": 1,
        "task": "coco_caption",
        "dataset_id": "lmms-lab/COCO-Caption",
        "dataset_revision": "c3213b54e173ae86517d481de17f31439d6f8c72",
        "split": "val",
    }
    paths = []
    jobs = []
    for index in range(2):
        path = tmp_path / f"{index}.json"
        path.write_text(
            __import__("json").dumps(
                {
                    **route,
                    "samples": [
                        {
                            "sample_id": f"coco_caption:{index}",
                            "prompt": "caption",
                            "references": ["reference"],
                            "media": [],
                        }
                    ],
                }
            )
        )
        paths.append(path)
        jobs.append(
            {
                "ledger": str(path),
                "ledger_sha": digest(path),
                "output": str(tmp_path / f"unused-{index}"),
            }
        )
    merged, bindings = merge_reviewed_ledgers(
        paths, parent_plan={"jobs": jobs}, excluded_ids=set(), expected_samples=2
    )
    assert [sample["sample_id"] for sample in merged["samples"]] == [
        "coco_caption:0",
        "coco_caption:1",
    ]
    assert len(bindings) == 2
    with pytest.raises(ValueError, match="overlap inspected"):
        merge_reviewed_ledgers(
            paths,
            parent_plan={"jobs": jobs},
            excluded_ids={"coco_caption:1"},
            expected_samples=2,
        )
    executed = Path(jobs[0]["output"])
    executed.mkdir()
    (executed / "worker.json").write_text("{}")
    with pytest.raises(ValueError, match="already has execution output"):
        merge_reviewed_ledgers(
            paths, parent_plan={"jobs": jobs}, excluded_ids=set(), expected_samples=2
        )


def test_short_logit_cohort_selection_is_deterministic_and_disjoint():
    from prepare_output_preserving_short_logit_validation import _fresh_ledger

    pool = {
        "task": "coco_caption",
        "dataset_id": "lmms-lab/COCO-Caption",
        "dataset_revision": "revision",
        "split": "val",
        "samples": [
            {"sample_id": f"sample-{index}", "references": ["unused"]} for index in range(4096)
        ],
    }
    excluded = {"sample-0", "sample-1", "sample-2"}
    first = _fresh_ledger(pool, excluded_ids=excluded, samples=128)
    second = _fresh_ledger(pool, excluded_ids=excluded, samples=128)
    ids = [sample["sample_id"] for sample in first["samples"]]
    assert first == second
    assert len(ids) == len(set(ids)) == 128
    assert not set(ids) & excluded


def test_frozen_validation_protocol_cannot_authorize_paper_campaign():
    protocol = (
        Path(__file__).resolve().parents[2]
        / "configs/experiments/output_preserving_draft_frozen_validation_v1.yaml"
    ).read_text()
    assert "sample_count: 128" in protocol
    assert "paper_campaign_submission_authorized: false" in protocol
    assert "minimum_memory_reduction: 0.30" in protocol
    assert "required_ci_lower_bound: 0.0" in protocol


def test_joint_validation_protocol_is_frozen_and_development_only():
    protocol = (
        Path(__file__).resolve().parents[2]
        / "configs/experiments/output_preserving_draft_joint_validation_v2.yaml"
    ).read_text()
    assert "sample_count: 128" in protocol
    assert "draft_ranking_mode: joint_normalized" in protocol
    assert "prompt_objective_weight: 0.5" in protocol
    assert "draft_objective_weight: 0.5" in protocol
    assert "candidate_retuning_authorized: false" in protocol
    assert "paper_campaign_submission_authorized: false" in protocol


def test_short_logit_protocol_is_frozen_and_development_only():
    protocol = (
        Path(__file__).resolve().parents[2]
        / "configs/experiments/output_preserving_short_logit_validation_v3.yaml"
    ).read_text()
    assert "sample_count: 128" in protocol
    assert "reconstruction_prompt_weights: [0.25, 0.50, 0.75]" in protocol
    assert "horizon: 2" in protocol
    assert "maximum_kl: 0.02" in protocol
    assert "benchmark_labels_accessed: false" in protocol
    assert "paper_campaign_submission_authorized: false" in protocol


def test_route_logit_protocol_has_separate_canary_and_calibration_gates():
    protocol = (
        Path(__file__).resolve().parents[2]
        / "configs/experiments/output_preserving_route_logit_validation_v4.yaml"
    ).read_text()
    assert "sample_count: 16" in protocol
    assert "sample_count: 128" in protocol
    assert "route_single_candidates: 8" in protocol
    assert "total_candidate_limit: 16" in protocol
    assert "probe_batch_size: 8" in protocol
    assert "maximum_fail_closed_fraction: 0.20" in protocol
    assert "paper_campaign_submission_authorized: false" in protocol


def test_layer_capacity_protocol_is_dense_label_blind_and_staged():
    protocol = (
        Path(__file__).resolve().parents[2]
        / "configs/experiments/output_preserving_layer_capacity_validation_v5.yaml"
    ).read_text()
    assert "sample_count: 16" in protocol and "sample_count: 128" in protocol
    assert "transaction: complete_layer_slots_across_all_kv_heads" in protocol
    assert "per_head_ragged_lengths: forbidden" in protocol
    assert "short_horizon: 2" in protocol and "extended_horizon: 8" in protocol
    assert "layer_transfer_candidates: 8" in protocol
    assert "benchmark_labels_accessed: false" in protocol
    assert "paper_campaign_submission_authorized: false" in protocol


def test_directional_single_protocol_is_exact_label_blind_and_staged():
    protocol = (
        Path(__file__).resolve().parents[2]
        / "configs/experiments/output_preserving_directional_single_validation_v8.yaml"
    ).read_text()
    assert "sample_count: 16" in protocol and "sample_count: 128" in protocol
    assert "atomic_single_complete_layer_slot_across_all_kv_heads" in protocol
    assert "surrogate_maximum_transaction_width: 1" in protocol
    assert "signed_prompt_draft_output_delta_logit_proxy" in protocol
    assert "maximum_kl: 0.02" in protocol
    assert "maximum_absolute_logit_difference: 1.0" in protocol
    assert "benchmark_labels_accessed: false" in protocol
    assert "paper_campaign_submission_authorized: false" in protocol


def test_route_logit_cohort_is_seeded_and_excludes_consumed_identities():
    from prepare_output_preserving_route_logit_validation import _fresh_ledger

    pool = {
        "task": "coco_caption",
        "dataset_id": "lmms-lab/COCO-Caption",
        "dataset_revision": "revision",
        "split": "val",
        "samples": [
            {"sample_id": f"sample-{index}", "references": ["unused"]} for index in range(4096)
        ],
    }
    excluded = {"sample-0", "sample-1", "sample-2"}
    first = _fresh_ledger(pool, excluded_ids=excluded, samples=16, seed=20271017)
    second = _fresh_ledger(pool, excluded_ids=excluded, samples=16, seed=20271017)
    identities = [sample["sample_id"] for sample in first["samples"]]
    assert first == second
    assert len(identities) == len(set(identities)) == 16
    assert not set(identities) & excluded


def test_capacity_analysis_flattens_point_identity_for_parquet():
    from analyze_output_preserving_capacity_sweep import _parquet_comparison_rows

    actual = _parquet_comparison_rows(
        [
            {
                "reference": ("prompt_attention_topk", 40.0),
                "candidate": ("output_preserving_swaps2", 25.5),
                "candidate_minus_reference": -1.0,
            }
        ]
    )
    assert actual == [
        {
            "candidate_minus_reference": -1.0,
            "reference_label": "prompt_attention_topk",
            "reference_cap_mib": 40.0,
            "candidate_label": "output_preserving_swaps2",
            "candidate_cap_mib": 25.5,
        }
    ]


def test_reconstruction_matches_recomputed_selected_softmax():
    torch.manual_seed(12)
    scores, values = torch.randn(6, 9, dtype=torch.float64), torch.randn(9, 3, dtype=torch.float64)
    scores[0, 3:] = -torch.inf
    attention, values = prepare_evidence(scores.softmax(-1), values)
    selected = (0, 2, 4, 6)
    actual = reconstruct(attention, values, selected)
    expected = scores[:, selected].softmax(-1) @ values[list(selected)]
    torch.testing.assert_close(actual.output, expected)
    assert not torch.allclose(actual.numerator, expected)
    assert torch.equal(attention[0, 3:], torch.zeros(6, dtype=torch.float64))


def test_swaps_correct_value_distortion_and_preserve_capacity():
    # Top mass picks +10 but ignores the -10 cancellation in the full output.
    attention, values = prepare_evidence(
        torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float64),
        torch.tensor([[10.0], [-10.0], [0.0], [0.0]], dtype=torch.float64),
    )
    options = OutputPreservingConfig(shortlist_size=4, max_swaps_per_head=1)
    a_before, v_before = attention.clone(), values.clone()
    result = corrective_swaps(attention, values, (0, 3), (3,), options)
    assert result.positions == (2, 3)
    assert result.final_error < result.initial_error
    assert len(result.swaps) == 1
    torch.testing.assert_close(attention, a_before, rtol=0, atol=0)
    torch.testing.assert_close(values, v_before, rtol=0, atol=0)
    anchor = corrective_swaps(
        attention, values, (0, 3), (3,), replace(options, max_swaps_per_head=0)
    )
    assert anchor.positions == (0, 3) and anchor.swaps == ()


def test_draft_gate_accepts_only_a_swap_that_improves_both_objectives():
    torch.manual_seed(34)
    prompt, values = prepare_evidence(
        torch.rand(3, 6, dtype=torch.float64),
        torch.randn(6, 2, dtype=torch.float64),
    )
    draft, _ = prepare_evidence(torch.rand(2, 6, dtype=torch.float64), values)
    options = OutputPreservingConfig(
        shortlist_size=6,
        max_swaps_per_head=1,
        pair_chunk_size=4,
        minimum_improvement=0,
    )
    prompt_only = corrective_swaps(prompt, values, (0, 1, 5), (5,), options)
    joint = corrective_swaps(
        prompt,
        values,
        (0, 1, 5),
        (5,),
        replace(options, draft_validation=True),
        validation_attention=draft,
    )
    assert prompt_only.positions == (1, 3, 5)
    assert joint.positions == (0, 2, 5)
    assert joint.final_error < joint.initial_error
    assert joint.validation_final_error < joint.validation_initial_error
    assert joint.rejected_validation_pairs > 0


def test_joint_draft_reranker_uses_normalized_prompt_and_draft_gain():
    torch.manual_seed(2)
    prompt, values = prepare_evidence(
        torch.rand(3, 7, dtype=torch.float64),
        torch.randn(7, 3, dtype=torch.float64),
    )
    draft, _ = prepare_evidence(torch.rand(3, 7, dtype=torch.float64), values)
    veto = OutputPreservingConfig(
        shortlist_size=7,
        max_swaps_per_head=1,
        pair_chunk_size=5,
        minimum_improvement=0,
        draft_validation=True,
    )
    prompt_ranked = corrective_swaps(
        prompt, values, (0, 1, 6), (6,), veto, validation_attention=draft
    )
    joint = corrective_swaps(
        prompt,
        values,
        (0, 1, 6),
        (6,),
        replace(
            veto,
            draft_ranking_mode="joint_normalized",
            minimum_joint_improvement=0,
        ),
        validation_attention=draft,
    )
    assert prompt_ranked.positions == (1, 5, 6)
    assert joint.positions == (1, 3, 6)
    assert joint.final_error < joint.initial_error
    assert joint.validation_final_error < joint.validation_initial_error
    assert joint.swaps[0]["joint_normalized_gain"] == pytest.approx(0.8043026943770992)
    assert joint.draft_ranking_mode == "joint_normalized"


def test_joint_draft_reranker_is_not_a_draft_nonregression_veto():
    torch.manual_seed(1)
    prompt, values = prepare_evidence(
        torch.rand(3, 7, dtype=torch.float64),
        torch.randn(7, 3, dtype=torch.float64),
    )
    draft, _ = prepare_evidence(torch.rand(3, 7, dtype=torch.float64), values)
    options = OutputPreservingConfig(
        shortlist_size=7,
        max_swaps_per_head=1,
        pair_chunk_size=5,
        minimum_improvement=0,
        draft_validation=True,
        draft_ranking_mode="joint_normalized",
        minimum_joint_improvement=0,
    )
    result = corrective_swaps(
        prompt,
        values,
        (0, 1, 6),
        (6,),
        options,
        validation_attention=draft,
    )
    swap = result.swaps[0]
    assert result.positions == (0, 3, 6)
    assert swap["normalized_prompt_gain"] < 0 < swap["normalized_draft_gain"]
    assert swap["joint_normalized_gain"] == pytest.approx(0.3917979464594324)


def test_individual_queries_do_not_disappear_into_a_pooled_objective():
    attention = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    values = torch.tensor([[1.0], [-1.0]])
    individual, v = prepare_evidence(attention, values)
    pooled, _ = prepare_evidence(attention, values, pooled=True)
    assert individual.shape == (2, 2) and pooled.shape == (1, 2)
    assert reconstruct(individual, v, (0,)).error != reconstruct(pooled, v, (0,)).error


@pytest.mark.parametrize("seed", range(8))
def test_full_shortlist_matches_best_single_swap_and_is_repeatable(seed):
    torch.manual_seed(seed)
    a, v = prepare_evidence(
        torch.rand(4, 7, dtype=torch.float64), torch.randn(7, 3, dtype=torch.float64)
    )
    options = OutputPreservingConfig(
        shortlist_size=7, max_swaps_per_head=1, pair_chunk_size=2, minimum_improvement=0.0
    )
    selected = (0, 1, 6)
    result = corrective_swaps(a, v, selected, (6,), options)
    feasible = [s for s in combinations(range(7), 3) if 6 in s and len(set(s) - set(selected)) <= 1]
    best = min(reconstruct(a, v, s).error for s in feasible)
    assert result.final_error == pytest.approx(best, abs=1e-12)
    assert result == corrective_swaps(a, v, selected, (6,), options)
    assert len(result.positions) == 3 and 6 in result.positions
    full = corrective_swaps(a, v, tuple(range(7)), (), options)
    assert full.positions == tuple(range(7)) and full.final_error < 1e-25


def test_zero_mass_is_rejected_not_floored_and_mandatory_is_enforced():
    a, v = prepare_evidence(torch.eye(2), torch.ones(2, 1))
    with pytest.raises(ValueError, match="attention mass"):
        reconstruct(a, v, (1,))
    with pytest.raises(ValueError, match="mandatory"):
        corrective_swaps(a, v, (0,), (1,), OutputPreservingConfig())
    with pytest.raises(ValueError, match="unique"):
        reconstruct(a, v, (0, 0))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_bad_attention_fails_closed(bad):
    with pytest.raises(ValueError):
        prepare_evidence(torch.tensor([[bad, 1.0]]), torch.ones(2, 1))


def test_joint_draft_ranking_is_explicitly_validated_and_labeled():
    options = OutputPreservingConfig(
        draft_validation=True,
        draft_ranking_mode="joint_normalized",
    )
    assert options.effective_method == "mosaickv_exact__output_preserving_v1__draft_joint"
    with pytest.raises(ConfigurationError, match="requires draft_validation"):
        OutputPreservingConfig(draft_ranking_mode="joint_normalized")
    with pytest.raises(ConfigurationError, match="weights must sum to one"):
        OutputPreservingConfig(
            draft_validation=True,
            draft_ranking_mode="joint_normalized",
            prompt_objective_weight=0.7,
            draft_objective_weight=0.7,
        )


def test_short_logit_reranking_controls_are_explicit_and_fail_closed():
    options = OutputPreservingConfig(
        draft_validation=True,
        draft_ranking_mode="joint_normalized",
        short_horizon_logit_reranking=True,
        logit_candidate_prompt_weight_low=0.25,
        logit_candidate_prompt_weight_high=0.75,
        logit_stability_horizon=2,
        logit_stability_maximum_kl=0.02,
        logit_stability_maximum_logit_difference=1.0,
    )
    assert (
        options.effective_method == "mosaickv_exact__output_preserving_v1__draft_joint__short_logit"
    )
    with pytest.raises(ConfigurationError, match="bracket"):
        replace(options, logit_candidate_prompt_weight_low=0.6)
    with pytest.raises(ConfigurationError, match="positive horizon"):
        replace(options, logit_stability_horizon=0)


def test_route_exchange_logit_controls_are_bounded_and_distinct():
    options = OutputPreservingConfig(
        draft_validation=True,
        draft_ranking_mode="joint_normalized",
        short_horizon_logit_reranking=True,
        logit_candidate_strategy="route_exchange_beam",
        logit_route_single_candidates=8,
        logit_route_candidate_limit=16,
        logit_route_max_combination_width=3,
        logit_probe_batch_size=8,
        logit_objective_kl_weight=50.0,
        logit_objective_logit_weight=1.0,
        logit_objective_token_weight=1.0,
        logit_stability_horizon=2,
        logit_stability_maximum_kl=0.02,
        logit_stability_maximum_logit_difference=1.0,
    )
    assert options.effective_method.endswith("__draft_joint__route_logit_beam")
    with pytest.raises(ConfigurationError, match="8-16 plan limit"):
        replace(options, logit_route_candidate_limit=7)
    with pytest.raises(ConfigurationError, match="does not use reconstruction weight"):
        replace(options, logit_candidate_prompt_weight_low=0.25)


def test_layer_capacity_controls_register_two_stage_dense_transaction():
    options = OutputPreservingConfig(
        draft_validation=True,
        draft_ranking_mode="joint_normalized",
        short_horizon_logit_reranking=True,
        logit_candidate_strategy="layer_capacity_beam",
        logit_route_single_candidates=6,
        logit_route_candidate_limit=9,
        logit_route_max_combination_width=2,
        logit_probe_batch_size=8,
        logit_objective_kl_weight=50.0,
        logit_objective_logit_weight=1.0,
        logit_objective_token_weight=1.0,
        logit_stability_horizon=2,
        logit_extended_horizon=8,
        logit_extended_candidate_limit=3,
        logit_layer_transfer_limit=8,
        logit_stability_maximum_kl=0.02,
        logit_stability_maximum_logit_difference=1.0,
    )
    assert options.effective_method.endswith("__draft_joint__layer_capacity_logit_beam")
    with pytest.raises(ConfigurationError, match="cover the anchor"):
        replace(options, logit_route_candidate_limit=8)
    with pytest.raises(ConfigurationError, match="extended horizon"):
        replace(options, logit_extended_horizon=2)


def test_surrogate_layer_capacity_controls_probe_only_two_then_one():
    options = OutputPreservingConfig(
        draft_validation=True,
        draft_ranking_mode="joint_normalized",
        short_horizon_logit_reranking=True,
        logit_candidate_strategy="layer_capacity_surrogate_beam",
        logit_route_single_candidates=8,
        logit_route_candidate_limit=9,
        logit_route_max_combination_width=3,
        logit_probe_batch_size=2,
        logit_objective_kl_weight=50.0,
        logit_objective_logit_weight=1.0,
        logit_objective_token_weight=1.0,
        logit_stability_horizon=2,
        logit_extended_horizon=8,
        logit_extended_candidate_limit=1,
        logit_layer_transfer_limit=8,
        logit_stability_maximum_kl=0.02,
        logit_stability_maximum_logit_difference=1.0,
    )
    assert options.effective_method.endswith("__draft_joint__layer_capacity_surrogate_beam")
    with pytest.raises(ConfigurationError, match="extends exactly one"):
        replace(options, logit_extended_candidate_limit=2)
    vectorized = replace(
        options,
        logit_candidate_strategy="layer_capacity_vectorized_surrogate_beam",
    )
    assert vectorized.effective_method.endswith(
        "__draft_joint__layer_capacity_vectorized_surrogate_beam"
    )
    directional = replace(
        vectorized,
        logit_candidate_strategy="layer_capacity_directional_single_beam",
        logit_route_max_combination_width=1,
    )
    assert directional.effective_method.endswith(
        "__draft_joint__layer_capacity_directional_single_beam"
    )
    with pytest.raises(ConfigurationError, match="single transactions"):
        replace(directional, logit_route_max_combination_width=2)
    within_route = replace(
        directional,
        logit_candidate_strategy="within_route_position_beam",
        logit_route_candidate_limit=16,
        logit_extended_candidate_limit=2,
    )
    assert within_route.effective_method.endswith("__draft_joint__within_route_position_beam")
    with pytest.raises(ConfigurationError, match="extends exactly two"):
        replace(within_route, logit_extended_candidate_limit=1)


def test_surrogate_probe_timing_trace_accepts_exact_two_then_one_schedule():
    from analyze_output_preserving_probe_microbenchmark import validate_probe_trace

    timings = [
        {"stage": "surrogate_generation", "candidate_count": 8},
        {
            "stage": "short_two_candidate_probe",
            "realized_batch_size": 2,
            "horizon": 2,
        },
        {
            "stage": "extended_one_candidate_probe",
            "realized_batch_size": 1,
            "horizon": 8,
        },
    ]
    stability = {
        "candidate_strategy": "layer_capacity_vectorized_surrogate_beam",
        "surrogate_candidate_count": 8,
        "model_probed_candidate_count": 2,
        "probe_batch_count": 2,
        "evaluated_candidates": [{}, {}],
        "benchmark_labels_accessed": False,
        "fullkv_prefill_reused": True,
        "second_prefill_performed": False,
        "probe_states_discarded_before_measured_decode": True,
        "probe_phase_timings": timings,
    }
    assert validate_probe_trace(stability) == tuple(timings)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_probed_candidate_count", 3, "registered surrogate schedule"),
        ("probe_phase_timings", [], "omitted phase timings"),
    ],
)
def test_surrogate_probe_timing_trace_rejects_unregistered_execution(field, value, message):
    from analyze_output_preserving_probe_microbenchmark import validate_probe_trace

    stability = {
        "candidate_strategy": "layer_capacity_vectorized_surrogate_beam",
        "surrogate_candidate_count": 8,
        "model_probed_candidate_count": 2,
        "probe_batch_count": 1,
        "evaluated_candidates": [{}, {}],
        "benchmark_labels_accessed": False,
        "fullkv_prefill_reused": True,
        "second_prefill_performed": False,
        "probe_states_discarded_before_measured_decode": True,
        "probe_phase_timings": [
            {"stage": "surrogate_generation", "candidate_count": 8},
            {
                "stage": "short_two_candidate_probe",
                "realized_batch_size": 2,
                "horizon": 2,
            },
        ],
    }
    stability[field] = value
    with pytest.raises(ValueError, match=message):
        validate_probe_trace(stability)


def test_surrogate_layer_plans_are_deterministic_exact_and_multilayer():
    torch.manual_seed(71)
    layers = tuple((torch.randn(1, 2, 8, 3), torch.randn(1, 2, 8, 3)) for _ in range(8))
    full = FullKVState.from_tensors(
        layers,
        block_size=1,
        modality_spans=(ModalitySpan(0, 8, Modality.TEXT),),
        mandatory_logical_positions=(0, 7),
    )
    captures = tuple(torch.rand(1, 4, 2, 8) for _ in range(8))
    captures = tuple(item / item.sum(-1, keepdim=True) for item in captures)
    scores = torch.cat(
        tuple(item.reshape(1, 2, 2, 2, 8).mean((0, 2, 3)).flatten() for item in captures)
    ).numpy()
    config = SimpleNamespace(
        output_preserving=OutputPreservingConfig(max_routed_heads=0, max_swaps_per_head=0),
        forecasting=SimpleNamespace(prompt_window=2),
    )
    limits = {(layer, head): 4 for layer in range(8) for head in range(2)}
    anchor = select_output_preserving(
        full,
        captures,
        config,
        position_limits=limits,
        target_bytes=8 * 2 * 4 * 3 * 4 * 2,
        prompt_attention_scores=scores,
        candidate_id="prompt_topk_anchor",
    )
    risk_routes = []
    for record in anchor.routes:
        risk = 0.1 + int(record["layer"]) / 10
        risk_routes.append(
            {
                **record,
                "initial_error": 1.0,
                "final_error": risk,
                "draft_initial_error": 1.0,
                "draft_final_error": risk,
            }
        )
    proposal = replace(
        anchor,
        routes=tuple(risk_routes),
        prompt_objective_weight=0.5,
        draft_objective_weight=0.5,
    )
    risks = prompt_draft_route_risk(proposal)
    assert sum(item[2] for item in risks) == pytest.approx(1.0)
    assert risks[0][2] < risks[-1][2]
    first = layer_capacity_surrogate_candidates(full, anchor, proposal, limit=8, maximum_width=3)
    second = layer_capacity_surrogate_candidates(full, anchor, proposal, limit=8, maximum_width=3)
    assert first == second
    assert len(first) == 8
    assert {len(candidate.surrogate_transactions) for candidate in first} == {1, 2, 3}
    assert len({candidate.selected_node_ids for candidate in first}) == 8
    for rank, candidate in enumerate(first, start=1):
        assert candidate.surrogate_rank == rank
        assert candidate.selected_bytes == anchor.selected_bytes
        assert candidate.physical_padding_bytes == 0
        assert candidate.surrogate_source_block_visits == len(full.blocks)
        assert candidate.surrogate_candidate_full_block_scans == 0
        assert all(
            not block.mandatory or node_id in candidate.selected_node_ids
            for node_id, block in enumerate(full.blocks)
        )

    context = _build_layer_transfer_context(full, anchor)
    optimized = _apply_layer_transfer_transactions(
        full,
        anchor,
        context,
        ((7, 0), (6, 1)),
        candidate_id="optimized",
    )
    reference = anchor
    for step, (receiver, donor) in enumerate(((7, 0), (6, 1)), start=1):
        reference = compose_layer_capacity_transfer_candidate(
            full,
            reference,
            receiver_layer=receiver,
            donor_layer=donor,
            transfer_slots=1,
            candidate_id=f"reference-{step}",
        )
    assert optimized.selected_node_ids == reference.selected_node_ids
    assert optimized.layer_capacities == reference.layer_capacities
    assert optimized.selected_bytes == reference.selected_bytes
    assert context.source_block_visits == len(full.blocks)


def test_directional_single_plans_cancel_effects_and_preserve_exact_semantics():
    torch.manual_seed(72)
    layers = tuple((torch.randn(1, 2, 8, 3), torch.randn(1, 2, 8, 3)) for _ in range(8))
    full = FullKVState.from_tensors(
        layers,
        block_size=1,
        modality_spans=(ModalitySpan(0, 8, Modality.TEXT),),
        mandatory_logical_positions=(0, 7),
    )
    captures = tuple(torch.rand(1, 4, 2, 8) for _ in range(8))
    captures = tuple(item / item.sum(-1, keepdim=True) for item in captures)
    scores = torch.cat(
        tuple(item.reshape(1, 2, 2, 2, 8).mean((0, 2, 3)).flatten() for item in captures)
    ).numpy()
    config = SimpleNamespace(
        output_preserving=OutputPreservingConfig(max_routed_heads=0, max_swaps_per_head=0),
        forecasting=SimpleNamespace(prompt_window=2),
    )
    limits = {(layer, head): 4 for layer in range(8) for head in range(2)}
    anchor = select_output_preserving(
        full,
        captures,
        config,
        position_limits=limits,
        target_bytes=8 * 2 * 4 * 3 * 4 * 2,
        prompt_attention_scores=scores,
        candidate_id="prompt_topk_anchor",
    )
    directional_routes = []
    for record in anchor.routes:
        layer = int(record["layer"])
        risk = 0.1 + layer / 10
        add_signature = (1.0, 0.0)
        remove_signature = (-1.0, 0.0) if layer == 0 else (1.0, 0.0)
        transfer = {
            "available": True,
            "add_gain": 1.0,
            "remove_cost": 0.25,
            "add_delta_signature": add_signature,
            "remove_delta_signature": remove_signature,
        }
        directional_routes.append(
            {
                **record,
                "initial_error": 1.0,
                "final_error": risk,
                "draft_initial_error": 1.0,
                "draft_final_error": risk,
                "prompt_transfer_effect": transfer,
                "draft_transfer_effect": transfer,
            }
        )
    proposal = replace(
        anchor,
        routes=tuple(directional_routes),
        prompt_objective_weight=0.5,
        draft_objective_weight=0.5,
    )
    first = layer_capacity_directional_single_candidates(full, anchor, proposal, limit=8)
    second = layer_capacity_directional_single_candidates(full, anchor, proposal, limit=8)
    assert first == second
    assert len(first) == len({candidate.selected_node_ids for candidate in first}) == 8
    assert first[0].surrogate_transactions == ((7, 0),)
    assert first[0].directional_prompt_delta_norm == pytest.approx(0.0)
    assert first[0].directional_draft_delta_norm == pytest.approx(0.0)
    for rank, candidate in enumerate(first, start=1):
        assert candidate.surrogate_rank == rank
        assert len(candidate.surrogate_transactions) == 1
        assert candidate.selected_bytes == anchor.selected_bytes
        assert candidate.physical_padding_bytes == 0
        assert candidate.surrogate_source_block_visits == len(full.blocks)
        assert candidate.surrogate_candidate_full_block_scans == 0
        assert all(
            not block.mandatory or node_id in candidate.selected_node_ids
            for node_id, block in enumerate(full.blocks)
        )
    reference = compose_layer_capacity_transfer_candidate(
        full,
        anchor,
        receiver_layer=7,
        donor_layer=0,
        transfer_slots=1,
        candidate_id="reference",
    )
    assert first[0].selected_node_ids == reference.selected_node_ids
    assert first[0].layer_capacities == reference.layer_capacities


def test_within_route_position_candidates_preserve_fixed_shape_and_exact_bytes():
    torch.manual_seed(73)
    layers = tuple((torch.randn(1, 2, 8, 3), torch.randn(1, 2, 8, 3)) for _ in range(8))
    full = FullKVState.from_tensors(
        layers,
        block_size=1,
        modality_spans=(ModalitySpan(0, 8, Modality.TEXT),),
        mandatory_logical_positions=(0, 7),
    )
    captures = tuple(torch.rand(1, 4, 2, 8) for _ in range(8))
    captures = tuple(item / item.sum(-1, keepdim=True) for item in captures)
    scores = torch.cat(
        tuple(item.reshape(1, 2, 2, 2, 8).mean((0, 2, 3)).flatten() for item in captures)
    ).numpy()
    config = SimpleNamespace(
        output_preserving=OutputPreservingConfig(max_routed_heads=0, max_swaps_per_head=0),
        forecasting=SimpleNamespace(prompt_window=2),
    )
    limits = {(layer, head): 4 for layer in range(8) for head in range(2)}
    anchor = select_output_preserving(
        full,
        captures,
        config,
        position_limits=limits,
        target_bytes=8 * 2 * 4 * 3 * 4 * 2,
        prompt_attention_scores=scores,
        candidate_id="prompt_topk_anchor",
    )
    selected = set(anchor.selected_node_ids)
    by_route: dict[tuple[int, int], list[int]] = {}
    for node_id, block in enumerate(full.blocks):
        by_route.setdefault((block.layer, block.kv_head), []).append(node_id)
    records = []
    for record in anchor.routes:
        route = (int(record["layer"]), int(record["kv_head"]))
        nodes = sorted(
            by_route[route], key=lambda node_id: full.blocks[node_id].physical_cache_indices
        )
        removable = next(
            position
            for position, node_id in enumerate(nodes)
            if node_id in selected and not full.blocks[node_id].mandatory
        )
        omitted = next(
            position for position, node_id in enumerate(nodes) if node_id not in selected
        )
        gain = 1.0 + route[0] / 10 + route[1] / 100
        effect = {
            "removed_position": removable,
            "added_position": omitted,
            "prompt": {
                "available": True,
                "gain": gain,
                "initial_error": 2.0,
                "final_error": 2.0 - gain / 2,
                "delta_norm": 0.1 + route[1] / 100,
                "delta_signature": (0.1, -0.1),
            },
            "draft": {
                "available": True,
                "gain": gain,
                "initial_error": 2.0,
                "final_error": 2.0 - gain / 2,
                "delta_norm": 0.1 + route[1] / 100,
                "delta_signature": (-0.1, 0.1),
            },
        }
        records.append(
            {
                **record,
                "draft_initial_error": 2.0,
                "draft_final_error": 2.0,
                "within_route_swap_effects": [effect],
            }
        )
    proposal = replace(
        anchor,
        routes=tuple(records),
        prompt_objective_weight=0.5,
        draft_objective_weight=0.5,
    )
    first = within_route_position_candidates(full, anchor, proposal, limit=16)
    second = within_route_position_candidates(full, anchor, proposal, limit=16)
    assert first == second
    assert len(first) == len({candidate.selected_node_ids for candidate in first}) == 16
    anchor_counts = {
        route: sum(node_id in selected for node_id in nodes) for route, nodes in by_route.items()
    }
    for rank, candidate in enumerate(first, start=1):
        assert candidate.surrogate_rank == rank
        assert candidate.selected_bytes == anchor.selected_bytes
        assert candidate.physical_padding_bytes == 0
        assert len(set(candidate.selected_node_ids) ^ selected) == 2
        assert len(candidate.block_substitutions) == 1
        assert (
            candidate.block_substitutions[0]["added_bytes"]
            == candidate.block_substitutions[0]["removed_bytes"]
        )
        candidate_set = set(candidate.selected_node_ids)
        assert {
            route: sum(node_id in candidate_set for node_id in nodes)
            for route, nodes in by_route.items()
        } == anchor_counts
        assert all(
            not block.mandatory or node_id in candidate_set
            for node_id, block in enumerate(full.blocks)
        )


@pytest.mark.parametrize(
    "mapping",
    [
        {"max_swaps_per_head": -1},
        {"shortlist_size": 65},
        {"max_routed_heads": True},
        {"query_evidence": "draft"},
        {"minimum_query_mass": 0},
        {"minimum_error": float("nan")},
        {"unknown": 1},
    ],
)
def test_strict_candidate_config(mapping):
    with pytest.raises(ConfigurationError):
        OutputPreservingConfig.from_mapping(mapping)


def test_effective_method_distinguishes_corrective_capacity():
    assert (
        OutputPreservingConfig(max_swaps_per_head=2).effective_method
        == "mosaickv_exact__output_preserving_v1"
    )
    assert (
        OutputPreservingConfig(max_swaps_per_head=4).effective_method
        == "mosaickv_exact__output_preserving_v1__swaps4"
    )
    assert (
        OutputPreservingConfig(max_swaps_per_head=8).effective_method
        == "mosaickv_exact__output_preserving_v1__swaps8"
    )
    assert (
        OutputPreservingConfig(draft_validation=True).effective_method
        == "mosaickv_exact__output_preserving_v1__draft_gate"
    )
    with pytest.raises(ConfigurationError, match="individual query evidence"):
        OutputPreservingConfig(query_evidence="pooled", draft_validation=True)


def test_bridge_routes_by_error_preserves_exact_storage_and_positions():
    torch.manual_seed(79)
    keys = torch.randn(1, 2, 8, 3)
    values = torch.randn(1, 2, 8, 3)
    full = FullKVState.from_tensors(
        ((keys, values),),
        block_size=1,
        modality_spans=(ModalitySpan(0, 8, Modality.TEXT),),
        mandatory_logical_positions=(0, 7),
    )
    capture = torch.rand(1, 4, 2, 8)
    capture /= capture.sum(-1, keepdim=True)
    before = (keys.clone(), values.clone(), capture.clone())
    options = OutputPreservingConfig(max_routed_heads=1, shortlist_size=8)
    scores = capture.reshape(1, 2, 2, 2, 8).mean((0, 2, 3)).flatten().numpy()
    config = SimpleNamespace(
        output_preserving=options, forecasting=SimpleNamespace(prompt_window=2)
    )
    plan = select_output_preserving(
        full,
        (capture,),
        config,
        position_limits={(0, 0): 4, (0, 1): 4},
        target_bytes=192,
        prompt_attention_scores=scores,
    )
    worst = max(plan.routes, key=lambda r: r["initial_error"])
    assert worst["routed"]
    assert sum(r["routed"] for r in plan.routes) == 1
    assert plan.selected_bytes == 192
    state = plan.construct(full)
    assert state.statistics.active_kv_bytes == 192
    assert not state.prototypes.source_blocks and not state.residuals.source_blocks
    assert state.next_decode_position == full.next_decode_position
    assert state.original_logical_sequence_length == full.original_logical_sequence_length
    for block, key, value in zip(
        state.exact.blocks, state.exact.key_blocks, state.exact.value_blocks, strict=True
    ):
        h, p = block.kv_head, block.physical_cache_indices[0]
        torch.testing.assert_close(key, keys[:, h : h + 1, p : p + 1], rtol=0, atol=0)
        torch.testing.assert_close(value, values[:, h : h + 1, p : p + 1], rtol=0, atol=0)
    for after, original in zip((keys, values, capture), before, strict=True):
        torch.testing.assert_close(after, original, rtol=0, atol=0)
    for route in plan.routes:
        assert route["final_error"] <= route["initial_error"]
        assert route["evaluated_pairs"] <= options.max_swaps_per_head * options.shortlist_size**2


def test_route_exchange_candidates_are_equal_byte_mandatory_and_deterministic():
    torch.manual_seed(97)
    keys = torch.randn(1, 2, 8, 3)
    values = torch.randn(1, 2, 8, 3)
    full = FullKVState.from_tensors(
        ((keys, values),),
        block_size=1,
        modality_spans=(ModalitySpan(0, 8, Modality.TEXT),),
        mandatory_logical_positions=(0, 7),
    )
    capture = torch.rand(1, 4, 2, 8)
    capture /= capture.sum(-1, keepdim=True)
    scores = capture.reshape(1, 2, 2, 2, 8).mean((0, 2, 3)).flatten().numpy()
    options = OutputPreservingConfig(max_routed_heads=0, max_swaps_per_head=0)
    config = SimpleNamespace(
        output_preserving=options,
        forecasting=SimpleNamespace(prompt_window=2),
    )
    anchor = select_output_preserving(
        full,
        (capture,),
        config,
        position_limits={(0, 0): 4, (0, 1): 4},
        target_bytes=192,
        prompt_attention_scores=scores,
        candidate_id="prompt_topk_anchor",
    )
    route = (0, 0)
    selected = [
        node_id
        for node_id in anchor.selected_node_ids
        if (full.blocks[node_id].layer, full.blocks[node_id].kv_head) == route
    ]
    omitted = [
        node_id
        for node_id, block in enumerate(full.blocks)
        if (block.layer, block.kv_head) == route
        and node_id not in anchor.selected_node_ids
        and not block.mandatory
    ]
    donor = next(node_id for node_id in selected if not full.blocks[node_id].mandatory)
    recipient = omitted[0]
    proposal_ids = tuple(sorted((set(anchor.selected_node_ids) - {donor}) | {recipient}))
    proposal_routes = []
    for record in anchor.routes:
        updated = dict(record)
        if (record["layer"], record["kv_head"]) == route:
            updated.update(
                routed=True,
                final_error=max(0.0, float(record["initial_error"]) * 0.5),
                swaps=[{"removed_position": 1, "added_position": 2}],
            )
        proposal_routes.append(updated)
    proposal = replace(
        anchor,
        selected_node_ids=proposal_ids,
        routes=tuple(proposal_routes),
        candidate_id="route_exchange_source",
    )
    first = route_exchange_single_candidates(full, anchor, proposal, limit=1)
    second = route_exchange_single_candidates(full, anchor, proposal, limit=1)
    assert first == second and len(first) == 1
    candidate = first[0]
    assert candidate.selected_bytes == anchor.selected_bytes == 192
    assert set(candidate.selected_node_ids) == set(proposal_ids)
    assert all(
        not block.mandatory or index in candidate.selected_node_ids
        for index, block in enumerate(full.blocks)
    )
    combined = compose_route_exchange_candidate(
        full,
        anchor,
        proposal,
        (route,),
        candidate_id="repeat",
    )
    assert combined.selected_node_ids == candidate.selected_node_ids
    assert combined.construct(full).statistics.active_kv_bytes == 192


def test_complete_layer_capacity_transfer_is_exact_byte_and_padding_free(monkeypatch):
    torch.manual_seed(105)
    layers = tuple((torch.randn(1, 2, 8, 3), torch.randn(1, 2, 8, 3)) for _ in range(3))
    full = FullKVState.from_tensors(
        layers,
        block_size=1,
        modality_spans=(ModalitySpan(0, 8, Modality.TEXT),),
        mandatory_logical_positions=(0, 7),
    )
    captures = tuple(torch.rand(1, 4, 2, 8) for _ in range(3))
    captures = tuple(item / item.sum(-1, keepdim=True) for item in captures)
    scores = torch.cat(
        tuple(item.reshape(1, 2, 2, 2, 8).mean((0, 2, 3)).flatten() for item in captures)
    ).numpy()
    config = SimpleNamespace(
        output_preserving=OutputPreservingConfig(max_routed_heads=0, max_swaps_per_head=0),
        forecasting=SimpleNamespace(prompt_window=2),
    )
    limits = {(layer, head): 4 for layer in range(3) for head in range(2)}
    anchor = select_output_preserving(
        full,
        captures,
        config,
        position_limits=limits,
        target_bytes=3 * 2 * 4 * 3 * 4 * 2,
        prompt_attention_scores=scores,
        candidate_id="prompt_topk_anchor",
    )
    candidate = compose_layer_capacity_transfer_candidate(
        full,
        anchor,
        receiver_layer=0,
        donor_layer=2,
        transfer_slots=1,
        candidate_id="layer_transfer",
    )
    assert candidate.layer_capacities == (5, 4, 3)
    assert candidate.selected_bytes == anchor.selected_bytes
    assert candidate.physical_padding_bytes == 0
    assert (
        candidate.capacity_transfers[0]["added_bytes"]
        == (candidate.capacity_transfers[0]["removed_bytes"])
    )
    assert all(
        not block.mandatory or node_id in candidate.selected_node_ids
        for node_id, block in enumerate(full.blocks)
    )
    for layer in range(3):
        counts = {
            sum(
                full.blocks[node_id].position_count
                for node_id in candidate.selected_node_ids
                if full.blocks[node_id].layer == layer and full.blocks[node_id].kv_head == head
            )
            for head in range(2)
        }
        assert counts == {candidate.layer_capacities[layer]}
    state = candidate.construct(full)
    assert state.statistics.active_kv_bytes == anchor.selected_bytes
    assert state.original_logical_sequence_length == full.original_logical_sequence_length
    assert state.next_decode_position == full.next_decode_position
    for block in state.exact.blocks:
        assert tuple(block.original_logical_positions) == tuple(
            sorted(block.original_logical_positions)
        )

    import mosaickv.adapters.huggingface.layerwise_exact_cache as transport
    from mosaickv.backends.hf_runtime import _pack_layerwise_exact_runtime_cache

    monkeypatch.setattr(transport, "assert_layerwise_exact_support", lambda _adapter: None)
    adapter = SimpleNamespace(
        capabilities=SimpleNamespace(cached_key_state=CachedKeyState.POST_ROPE)
    )
    packed = _pack_layerwise_exact_runtime_cache(adapter, full, state)
    assert packed.snapshot.independent_layer_lengths
    assert [layer.key.shape[-2] for layer in packed.snapshot.layers] == [5, 4, 3]
    assert packed.layerwise_exact_logical_positions
    assert (
        sum(
            tensor.untyped_storage().nbytes()
            for layer in packed.snapshot.layers
            for tensor in (layer.key, layer.value)
        )
        == candidate.selected_bytes
    )
    assert all(record["padding_bytes"] == 0 for record in packed.slot_records)

    first = layer_capacity_transfer_candidates(
        full,
        anchor,
        ((0, 0, 1.0), (1, 0, 0.25), (2, 0, -0.5)),
        limit=4,
    )
    second = layer_capacity_transfer_candidates(
        full,
        anchor,
        ((0, 0, 1.0), (1, 0, 0.25), (2, 0, -0.5)),
        limit=4,
    )
    assert first == second
    assert len({item.selected_node_ids for item in first}) == 4


def test_fp16_anchor_pooling_matches_baseline_capture_rounding():
    source = SimpleNamespace(layers=(SimpleNamespace(kv_heads=2, sequence_length=8),))
    torch.manual_seed(30)
    raw = torch.rand(1, 8, 3, 8, dtype=torch.float16)
    expected = raw.reshape(1, 2, 4, 3, 8).mean(2)
    actual = topk_anchor_capture(source, (raw,))[0]
    assert actual.dtype == torch.float16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual.float(), raw.float().reshape(1, 2, 4, 3, 8).mean(2))


def test_development_yaml_and_unsupported_backend_fail_closed():
    config = load_config(
        Path(__file__).resolve().parents[2] / "configs/development/output_preserving_exact_v1.yaml"
    )
    assert config.output_preserving.query_evidence == "individual"
    assert config.cache.retention_ratio > 0.99 and config.cache.retention_ratio < 1
    with pytest.raises(ConfigurationError, match="HF eager"):
        replace(config, execution=replace(config.execution, attention_implementation="sdpa"))
    with pytest.raises(ConfigurationError, match="HF eager"):
        replace(config, cache=replace(config.cache, block_size=4))
    with pytest.raises(ConfigurationError, match="HF eager"):
        replace(config, prototypes=replace(config.prototypes, enabled=True))
