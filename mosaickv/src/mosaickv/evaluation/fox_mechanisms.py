"""Evaluation-only activity/equivalence audits; never inspect benchmark labels."""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

from mosaickv.evaluation.fox_campaign import digest


def quota_matched_random_routes(eligible, entropy_routes, seed, sample_id):
    by_layer = defaultdict(list)
    for layer, head in eligible:
        by_layer[layer].append((layer, head))
    counts = Counter(layer for layer, _head in entropy_routes)
    return tuple(
        sorted(
            route
            for layer, count in sorted(counts.items())
            for route in sorted(by_layer[layer], key=lambda r: digest([seed, sample_id, r]))[:count]
        )
    )


@contextmanager
def mechanism_audit(variant, sample_id, *, stable_policy):
    import mosaickv.backends.hf_runtime as runtime
    from mosaickv.selection import SelectionBudget

    record = {}
    original_plan = runtime.build_compression_plan
    original_baseline = runtime.build_exact_baseline_plan
    original_router = runtime._uncertain_head_routes

    def plan(*args, **kwargs):
        result = original_plan(*args, **kwargs)
        nodes = tuple(node.source_id for node in result.graph.nodes)
        limits = kwargs.get("position_limits")
        record.update(
            candidate_universe_sha256=digest(nodes),
            shortlist_sha256=digest(result.similarity_candidate_node_ids),
            quota_sha256=digest(sorted((k[0], k[1], v) for k, v in (limits or {}).items())),
            graph_routes=result.graph_head_routes,
            graph_nodes=len(nodes),
            graph_edges=len(result.graph.weights),
            edge_sources=dict(Counter(t.value for t in result.graph.edge_types)),
            selected_source_sha256=digest(
                tuple(
                    (b.layer, b.kv_head, b.physical_cache_indices)
                    for b in result.construction.state.exact.blocks
                )
            ),
            planner_routes=sorted({(n.block.layer, n.block.kv_head) for n in result.graph.nodes}),
            quota_scope="same prompt-derived dense per-head physical limits",
            optimization_node_ids_sha256=digest(result.optimization_node_ids),
        )
        return result

    def baseline(*args, **kwargs):
        result = original_baseline(*args, **kwargs)
        if variant == "route-0":
            full = result.full_state
            limits = kwargs.get("position_limits")
            if limits is None:
                raise AssertionError("zero-route equivalence requires physical per-head limits")
            stable = runtime.select_prompt_attention_topk_ids_by_head(
                full,
                SelectionBudget(result.active_budget_value, result.selection.budget.unit),
                prompt_attention_by_node=kwargs["prompt_attention_by_node"],
                head_routes=tuple(sorted(limits)),
                position_limits=limits,
                minimum_visual_fraction=stable_policy.selection.stable_head_visual_fraction,
            )
            if stable.selected_node_ids != result.selection.selected_node_ids:
                raise AssertionError("zero-route stable selector is NOT equivalent to shared Top-K")
            record.update(
                zero_route_equivalence_passed=True,
                graph_nodes=0,
                graph_edges=0,
                graph_routes=(),
                actual_dispatch="checked_equivalent_all_stable_prompt_selector",
                selected_source_sha256=digest(
                    tuple(
                        (b.layer, b.kv_head, b.physical_cache_indices)
                        for b in result.state.exact.blocks
                    )
                ),
            )
        return result

    def random_routes(eligible, uncertainty, **kwargs):
        eligible = tuple(eligible)
        reference = original_router(eligible, uncertainty, **kwargs)
        return quota_matched_random_routes(
            eligible, reference, int(variant.split("-")[1]), sample_id
        )

    with ExitStack() as stack:
        stack.enter_context(patch.object(runtime, "build_compression_plan", plan))
        stack.enter_context(patch.object(runtime, "build_exact_baseline_plan", baseline))
        if variant.startswith("random-"):
            stack.enter_context(patch.object(runtime, "_uncertain_head_routes", random_routes))
        yield record
