"""Synthetic policy invariants are not empirical quality evidence."""

from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
from test_hf_runtime import _capabilities

from mosaickv.cache_state import FullKVState
from mosaickv.config import (
    ConfigurationError,
    EvidencePolicyConfig,
    GraphConfig,
    RunConfig,
    SelectionConfig,
    UtilityConfig,
    canonical_config,
)
from mosaickv.evaluation.fox_campaign import source_config
from mosaickv.graph import build_evidence_graph
from mosaickv.selection import SelectionBudget, compute_block_utilities
from mosaickv.selection.opportunity_demand import (
    DemandWeightedObjective,
    RouteEvidence,
    make_route_evidence,
    normalized_queries,
    opportunity,
    query_floor,
)
from mosaickv.selection.selector import select_exact_subset
from mosaickv.types import BudgetUnit


def source(heads=1, length=6):
    rng = np.random.default_rng(17)
    k, v = [rng.normal(size=(1, heads, length, 4)).astype(np.float32) for _ in range(2)]
    return FullKVState.from_tensors(
        ((k, v),), block_size=1, mandatory_logical_positions=(length - 1,)
    )


@pytest.mark.parametrize("values", [np.ones((0, 3)), [[0, 0]], [[1, -1]], [[np.nan, 1]], [1, 2]])
def test_invalid_query_domains_rejected(values):
    with pytest.raises(ValueError):
        normalized_queries(values)


def test_full_domain_normalization_preserves_zeros_and_does_not_mutate():
    a = np.array([[0.0, 2.0, 2.0], [3.0, 0.0, 0.0]])
    copy = a.copy()
    result = normalized_queries(a)
    np.testing.assert_equal(a, copy)
    np.testing.assert_equal(result, [[0, 0.5, 0.5], [1, 0, 0]])


@pytest.mark.parametrize(
    "payload",
    [
        dict(variant="magic"),
        dict(shortlist_size=True),
        dict(shortlist_size=33),
        dict(minimum_query_mass_fraction=0),
        dict(minimum_query_mass_fraction=float("nan")),
        dict(extra=1),
    ],
)
def test_strict_experimental_configuration(payload):
    with pytest.raises(ConfigurationError):
        EvidencePolicyConfig.from_mapping(payload)


def test_unchanged_config_hash_domain_and_opt_in_round_trip():
    base = RunConfig.from_mapping(source_config("coco"))
    assert "evidence_policy" not in canonical_config(base)
    for variant in ("routing_only", "coverage_only", "both"):
        candidate = replace(base, evidence_policy=EvidencePolicyConfig(variant=variant))
        assert RunConfig.from_mapping(canonical_config(candidate)) == candidate
    with pytest.raises(ConfigurationError):
        replace(
            base, evidence_policy=EvidencePolicyConfig(), cache=replace(base.cache, block_size=4)
        )


def test_query_guard_detects_rare_query_loss_hidden_by_identical_mean():
    a = normalized_queries([[9, 1, 0], [0, 1, 9]])
    assert np.mean(a[:, 0]) == np.mean(a[:, 2])
    passed, ratio = query_floor(a, [0], [2], 0.95)
    assert not passed and ratio == 0


def test_opportunity_prefers_improvable_not_just_diffuse_route():
    improvable = RouteEvidence(
        tuple(range(4)),
        normalized_queries(
            [
                [0.6, 0.35, 0.05, 0],
                [0.05, 0.55, 0.4, 0],
            ]
        ),
        (0, 3),
        (3,),
    )
    diffuse = RouteEvidence(tuple(range(4)), normalized_queries(np.ones((2, 4))), (0, 3), (3,))
    # Relaxed test-only floor makes the known feasible opportunity explicit.
    kwargs = dict(shortlist_size=4, minimum_query_mass_fraction=0.5)
    first = opportunity(improvable, **kwargs)
    assert first["score"] > opportunity(diffuse, **kwargs)["score"]
    assert first == opportunity(improvable, **kwargs)
    assert first["witness_source_ids"] == (0, 1)
    assert 3 not in first["witness_source_ids"]


def objective_fixture(floor=0.95):
    state = source()
    attention = normalized_queries([[0.3, 0.1, 0.2, 0.2, 0.1, 0.1], [0.1, 0.4, 0.1, 0.1, 0.2, 0.1]])
    scores = {i: float(v) for i, v in enumerate(attention.mean(0))}
    evidence = make_route_evidence(state, {(0, 0): attention}, scores, {(0, 0): 3})
    graph = build_evidence_graph(state, GraphConfig(max_neighbors=2))
    utilities = compute_block_utilities(
        graph,
        UtilityConfig(),
        forecast_attention_by_node=scores,
        attention_provenance="synthetic",
        rope_aware=True,
    )
    objective = DemandWeightedObjective(graph, utilities, SelectionConfig(), evidence, floor)
    return state, evidence, objective


def test_weighted_facility_matches_manual_demand_and_incremental_paths():
    _, _, objective = objective_fixture()
    for ids in ((), (0,), (0, 2), tuple(range(6))):
        covered = np.zeros(6)
        for i in ids:
            for j, weighted_similarity in objective._facility[i].items():
                covered[j] = max(covered[j], weighted_similarity)
        assert objective.evaluate(ids).facility_location_coverage == pytest.approx(
            covered.sum() / 6
        )
        incremental = objective.incremental_state()
        for i in ids:
            before = objective.evaluate(incremental._selected).total
            gain = incremental.select(i)
            assert gain == pytest.approx(objective.evaluate(incremental._selected).total - before)
        candidates = tuple(i for i in range(6) if i not in ids)
        assert incremental.marginal_gains(candidates) == pytest.approx(
            [objective.marginal_gain(ids, i) for i in candidates]
        )
    assert objective.evaluate(range(6)).facility_location_coverage == pytest.approx(1)


def test_weighted_facility_diminishing_returns():
    _, _, objective = objective_fixture()
    for size in range(4):
        for small in combinations(range(6), size):
            for extra in set(range(6)) - set(small):
                large = (*small, extra)
                for candidate in set(range(6)) - set(large):
                    assert objective.marginal_gain(
                        small, candidate
                    ) + 1e-12 >= objective.marginal_gain(large, candidate)


def test_query_guard_rolls_back_route_without_changing_bytes_or_mandatory():
    _, evidence, objective = objective_fixture()
    budget = SelectionBudget(96, BudgetUnit.BYTES)
    anchor = evidence[(0, 0)].anchor
    proposed = select_exact_subset(objective, budget, (0, 3, 5))
    result = objective.protect_selection(proposed)
    assert result.active_bytes == proposed.active_bytes == 96
    assert result.selected_node_ids == anchor
    assert objective.guard_report["rolled_back_routes"] == 1
    assert not objective.guard_report["whole_cache_fallback"]


@pytest.mark.parametrize("variant", [None, "routing_only", "coverage_only", "both"])
def test_actual_planner_has_equal_bytes_mandatory_positions_and_variant_label(variant):
    import mosaickv.backends.hf_runtime as runtime

    state = source(heads=4, length=12)
    base = RunConfig.from_mapping(source_config("coco"))
    config = replace(
        base,
        cache=replace(base.cache, budget_value=4 * 6 * 32),
        evidence_policy=None if variant is None else EvidencePolicyConfig(variant=variant),
    )
    rng = np.random.default_rng(22)
    queries = normalized_queries(rng.random((4 * 4, 12))).reshape(1, 4, 4, 12)
    prefill = SimpleNamespace(attention_weights=(queries,))
    token_attention = runtime._prompt_token_attention_by_head(state, prefill, prompt_window=16)
    scores = runtime._block_attention_values_from_token_attention(state, token_attention)
    limits = {(0, h): 6 for h in range(4)}
    plan = runtime.build_compression_plan(
        state,
        None,
        scores,
        _capabilities(safe=False),
        config,
        position_limits=limits,
        prompt_attention_by_node=scores,
        prompt_attentions=prefill.attention_weights,
        graph_head_uncertainty=runtime._attention_uncertainty_by_head(token_attention),
    )
    assert plan.construction.state.statistics.active_kv_bytes == config.cache.budget_value
    assert len(plan.graph_head_routes) == 1
    exact = plan.construction.state.exact.blocks
    for h in range(4):
        blocks = [b for b in exact if b.kv_head == h]
        assert len(blocks) == 6
        assert any(b.mandatory for b in blocks)
    if variant is None:
        assert plan.evidence_policy_statistics is None
    else:
        assert plan.effective_method.endswith(f"__evidence_{variant}_v1")
        assert plan.evidence_policy_statistics["variant"] == variant


def test_evidence_does_not_normalize_an_incomplete_source_domain():
    state = source()
    with pytest.raises(ValueError):
        make_route_evidence(
            state, {(0, 0): np.ones((2, 3))}, dict.fromkeys(range(6), 1.0), {(0, 0): 3}
        )


@pytest.mark.parametrize("variant", ["routing_only", "coverage_only", "both"])
def test_retention_one_bypasses_policy_and_preserves_all_exact_entries(variant):
    import mosaickv.backends.hf_runtime as runtime

    state = source(heads=2)
    base = RunConfig.from_mapping(source_config("coco"))
    config = replace(
        base,
        cache=replace(base.cache, retention_ratio=1.0, budget_value=state.active_bytes),
        evidence_policy=EvidencePolicyConfig(variant=variant),
    )
    scores = dict.fromkeys(range(len(state.blocks)), 1.0)
    plan = runtime.build_compression_plan(
        state,
        None,
        scores,
        _capabilities(safe=False),
        config,
        position_limits={(0, h): 6 for h in range(2)},
        prompt_attention_by_node=scores,
        graph_head_uncertainty={(0, h): 0.5 for h in range(2)},
    )
    assert plan.evidence_policy_statistics is None
    assert plan.selection.selected_node_ids == tuple(range(len(state.blocks)))
    assert plan.construction.state.statistics.active_kv_bytes == state.active_bytes


def test_canary_has_both_caps_all_baselines_and_four_policy_arms():
    from evicastkv_opportunity_study import arms

    matrix = arms()
    assert len(matrix) == len(set(matrix)) == 13
    for cap in (20, 40):
        assert ("prompt_attention_topk", cap, "complete") in matrix
        assert ("vl_cache_reimpl", cap, "complete") in matrix
        for variant in (
            "complete",
            "evidence-routing-v1",
            "evidence-coverage-v1",
            "evidence-both-v1",
        ):
            assert ("mosaickv_exact", cap, variant) in matrix
    assert sum(method == "full_kv" for method, _, _ in matrix) == 1


def test_experimental_arm_requires_explicit_registration():
    from evicastkv_campaign import resolved_config

    from mosaickv.evaluation.fox_campaign import REFERENCE_CONFIG_ROOT

    with pytest.raises(ValueError, match="unregistered"):
        resolved_config(
            {
                "templates": {
                    "coco": {"path": str(REFERENCE_CONFIG_ROOT / "coco/mosaickv_exact.json")}
                }
            },
            "coco",
            "mosaickv_exact",
            20,
            variant="evidence-both-v1",
        )
