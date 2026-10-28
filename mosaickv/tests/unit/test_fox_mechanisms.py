import numpy as np

from mosaickv.baselines.selectors import (
    select_prompt_attention_topk_ids,
    select_prompt_attention_topk_ids_by_head,
)
from mosaickv.cache_state import FullKVState
from mosaickv.evaluation.fox_mechanisms import quota_matched_random_routes
from mosaickv.selection import SelectionBudget
from mosaickv.types import BudgetUnit


def test_random_head_control_preserves_entropy_layer_quota_and_seed():
    routes = [(layer, head) for layer in range(4) for head in range(8)]
    entropy = ((0, 0), (0, 1), (2, 1))
    first = quota_matched_random_routes(routes, entropy, 13, "input")
    assert first == quota_matched_random_routes(routes, entropy, 13, "input")
    assert [sum(layer == k for layer, _ in first) for k in range(4)] == [2, 0, 1, 0]
    assert first != quota_matched_random_routes(routes, entropy, 29, "input")


def test_cross_modal_control_removes_all_edge_sources_without_refill():
    from mosaickv.cache_state import Modality, ModalitySpan
    from mosaickv.config import GraphConfig
    from mosaickv.evaluation.fox_mechanisms import remove_cross_modal_edges
    from mosaickv.graph import build_evidence_graph

    k = np.arange(24, dtype=np.float32).reshape(1, 1, 6, 4)
    state = FullKVState.from_tensors(
        ((k, k.copy()),),
        block_size=1,
        modality_spans=(ModalitySpan(0, 3, Modality.IMAGE), ModalitySpan(3, 6, Modality.TEXT)),
    )
    graph = build_evidence_graph(state, GraphConfig(max_neighbors=4))
    filtered = remove_cross_modal_edges(graph)
    assert filtered.nodes is graph.nodes
    assert len(filtered.weights) < len(graph.weights)
    assert all(
        filtered.nodes[i].block.modality == filtered.nodes[j].block.modality
        for i, j in zip(filtered.row_indices, filtered.column_indices, strict=True)
    )
    original = set(
        zip(graph.row_indices, graph.column_indices, graph.weights, graph.edge_types, strict=True)
    )
    assert set(
        zip(
            filtered.row_indices,
            filtered.column_indices,
            filtered.weights,
            filtered.edge_types,
            strict=True,
        )
    ).issubset(original)


def test_stable_route_zero_topk_equivalence_is_checked_not_assumed():
    rng = np.random.default_rng(17)
    for _ in range(24):
        heads = int(rng.integers(1, 5))
        length = int(rng.integers(4, 24))
        capacity = int(rng.integers(2, length))
        k = np.zeros((1, heads, length, 4), np.float32)
        state = FullKVState.from_tensors(
            ((k, k.copy()),), block_size=1, mandatory_logical_positions=(0, length - 1)
        )
        scores = {i: float(x) for i, x in enumerate(rng.integers(0, 8, size=heads * length))}
        limits = {(0, h): capacity for h in range(heads)}
        budget = SelectionBudget(heads * capacity * 4 * 4 * 2, BudgetUnit.BYTES)
        topk = select_prompt_attention_topk_ids(
            state, budget, prompt_attention_by_node=scores, position_limits=limits
        )
        stable = select_prompt_attention_topk_ids_by_head(
            state,
            budget,
            prompt_attention_by_node=scores,
            head_routes=tuple(limits),
            position_limits=limits,
        )
        assert topk == stable.selected_node_ids


def test_real_planner_activity_retains_fixed_candidates_quotas_and_raw_terms(tmp_path):
    from dataclasses import replace

    import pandas as pd
    from test_hf_runtime import _capabilities

    import mosaickv.backends.hf_runtime as runtime
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.fox_campaign import source_config
    from mosaickv.evaluation.fox_mechanisms import mechanism_audit

    rng = np.random.default_rng(23)
    layers = tuple(
        (
            rng.normal(size=(1, 4, 24, 8)).astype(np.float32),
            rng.normal(size=(1, 4, 24, 8)).astype(np.float32),
        )
        for _ in range(2)
    )
    state = FullKVState.from_tensors(layers, block_size=1, mandatory_logical_positions=(23,))
    base = RunConfig.from_mapping(source_config("coco"))
    base = replace(
        base,
        cache=replace(base.cache, budget_value=6144),
        prototypes=replace(base.prototypes, uncertain_head_fraction=0.25),
        graph=replace(base.graph, uncertain_head_fraction=0.25),
    )
    prompt = {i: float(i % 24 + 1) for i in range(len(state.blocks))}
    limits = {(layer, head): 12 for layer in range(2) for head in range(4)}
    activities = []
    for variant in ("mechanism-complete", "coverage-off", "value-off", "routed-prompt"):
        config = base
        if variant in ("coverage-off", "routed-prompt"):
            config = replace(
                config, selection=replace(config.selection, lambda_g=0.0, lambda_m=0.0)
            )
        if variant in ("value-off", "routed-prompt"):
            config = replace(config, utility=replace(config.utility, lambda_v=0.0, lambda_o=0.0))
        with mechanism_audit(
            variant, "fixture", stable_policy=base, raw_terms_directory=tmp_path / variant
        ) as activity:
            runtime.build_compression_plan(
                state,
                None,
                prompt,
                _capabilities(safe=False),
                config,
                position_limits=limits,
                prompt_attention_by_node=prompt,
                graph_head_uncertainty={(layer, head): 0.2 for layer, head in limits},
            )
        activities.append(activity)
        assert (tmp_path / variant / "raw_marginal_terms.parquet").exists()
    for field in (
        "candidate_universe_sha256",
        "shortlist_sha256",
        "quota_sha256",
        "descriptors_sha256",
        "optimization_node_ids_sha256",
    ):
        assert len({a[field] for a in activities}) == 1
    first = pd.read_parquet(tmp_path / "mechanism-complete/raw_marginal_terms.parquet")
    for variant in ("coverage-off", "value-off", "routed-prompt"):
        other = pd.read_parquet(tmp_path / variant / "raw_marginal_terms.parquet")
        for field in (
            "forecast_attention_probability",
            "value_novelty",
            "expected_attention_output_contribution",
        ):
            assert first[field].equals(other[field])


def test_zero_route_audit_uses_actual_shared_runtime_baseline():
    from dataclasses import replace

    import mosaickv.backends.hf_runtime as runtime
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.fox_campaign import source_config
    from mosaickv.evaluation.fox_mechanisms import mechanism_audit
    from mosaickv.types import MosaicKVMethod

    keys = np.ones((1, 2, 12, 4), np.float32)
    state = FullKVState.from_tensors(
        ((keys, keys.copy()),), block_size=1, mandatory_logical_positions=(11,)
    )
    config = RunConfig.from_mapping(source_config("coco"))
    cache = replace(config.cache, budget_value=2 * 6 * 4 * 4 * 2)
    prompt = {i: float(i % 12 + 1) for i in range(24)}
    limits = {(0, h): 6 for h in range(2)}
    with mechanism_audit("route-0", "fixture", stable_policy=config) as record:
        runtime.build_exact_baseline_plan(
            state,
            MosaicKVMethod.PROMPT_ATTENTION_TOPK,
            cache,
            seed=17,
            prompt_attention_by_node=prompt,
            position_limits=limits,
        )
    assert record["zero_route_equivalence_passed"]
    assert record["graph_nodes"] == 0


def test_mechanism_contract_rejects_different_candidates(tmp_path):
    import copy

    import pytest

    from mosaickv.evaluation.fox_campaign import sha, write_once
    from mosaickv.evaluation.fox_mechanisms import validate_mechanism_contract

    variants = (
        "mechanism-complete",
        "routed-prompt",
        "coverage-off",
        "value-off",
        "cross-modal-off",
        "route-0",
        "route-zero-control",
    )
    rows = []
    for variant in variants:
        activity = dict(
            candidate_universe_sha256="same",
            shortlist_sha256="same",
            quota_sha256="same",
            descriptors_sha256="same",
            optimization_node_ids_sha256="same",
            planner_routes=[[0, 0]],
            selected_source_ids=[0, 1],
            zero_route_equivalence_passed=True,
        )
        if variant == "value-off":
            activity["candidate_universe_sha256"] = "different"
        path = tmp_path / f"{variant}.json"
        write_once(path, activity)
        rows.append(
            dict(
                workload="fixture",
                sample_id="x",
                budget_bytes=2,
                variant=variant,
                mechanism_audit_path=str(path),
                mechanism_audit_sha256=sha(path),
                status="completed",
                token_ids=[1, 2],
                score=None,
                subtask=None,
            )
        )
    with pytest.raises(ValueError, match="domain differs"):
        validate_mechanism_contract(copy.deepcopy(rows))
