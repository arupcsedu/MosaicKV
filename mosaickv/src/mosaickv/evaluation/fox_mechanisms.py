"""Evaluation-only activity/equivalence audits; never inspect benchmark labels."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from unittest.mock import patch

from mosaickv.evaluation.fox_campaign import digest


def remove_cross_modal_edges(graph, *, candidate_node_ids=None):
    """Remove every inter-modality edge source, without refilling top-d edges."""
    from mosaickv.graph.builder import _diagnostics
    from mosaickv.graph.types import SparseEvidenceGraph

    keep = tuple(
        i
        for i, (left, right) in enumerate(zip(graph.row_indices, graph.column_indices, strict=True))
        if graph.nodes[left].block.modality == graph.nodes[right].block.modality
    )
    rows = tuple(graph.row_indices[i] for i in keep)
    columns = tuple(graph.column_indices[i] for i in keep)
    weights = tuple(graph.weights[i] for i in keep)
    types = tuple(graph.edge_types[i] for i in keep)
    diagnostics = _diagnostics(
        graph.nodes,
        rows,
        columns,
        types,
        fallback_used=graph.diagnostics.fallback_used,
        candidate_node_ids=None if candidate_node_ids is None else frozenset(candidate_node_ids),
        candidate_graph_mode=graph.diagnostics.candidate_graph_mode,
    )
    return SparseEvidenceGraph(graph.nodes, rows, columns, weights, types, diagnostics)


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
def mechanism_audit(variant, sample_id, *, stable_policy, raw_terms_directory=None):
    import mosaickv.backends.hf_runtime as runtime
    from mosaickv.selection import SelectionBudget

    record = {}
    original_plan = runtime.build_compression_plan
    original_baseline = runtime.build_exact_baseline_plan
    original_router = runtime._uncertain_head_routes
    original_graph = runtime.build_evidence_graph

    def graph(*args, **kwargs):
        if variant == "cross-modal-off":
            # First form the identical complete-policy graph, then delete all
            # cross-modality edges without replacing them with extra neighbors.
            from dataclasses import replace

            args = (
                args[0],
                replace(args[1], cross_modal_weight=stable_policy.graph.cross_modal_weight),
                *args[2:],
            )
        result = original_graph(*args, **kwargs)
        if variant == "cross-modal-off":
            before = len(result.weights)
            requested = kwargs.get("similarity_candidate_node_ids")
            requested_set = None if requested is None else frozenset(requested)
            candidates = (
                None
                if requested_set is None
                else tuple(n.node_id for n in result.nodes if n.source_id in requested_set)
            )
            result = remove_cross_modal_edges(result, candidate_node_ids=candidates)
            record["cross_modal_edges_removed"] = before - len(result.weights)
        return result

    def plan(*args, **kwargs):
        result = original_plan(*args, **kwargs)
        nodes = tuple(node.source_id for node in result.graph.nodes)
        limits = kwargs.get("position_limits")
        import numpy as np

        descriptors = hashlib.sha256()
        for node in result.graph.nodes:
            for vector in (node.pooled_key, node.pooled_value):
                descriptors.update(np.asarray(vector, dtype=np.float64).tobytes())
        if raw_terms_directory is not None:
            from pathlib import Path

            import pandas as pd

            path = Path(raw_terms_directory)
            path.mkdir(parents=True, exist_ok=False)
            terms = []
            for node, utility, decision in zip(
                result.graph.nodes, result.utilities.blocks, result.selection.decisions, strict=True
            ):
                terms.append(
                    dict(
                        source_id=node.source_id,
                        layer=node.block.layer,
                        head=node.block.kv_head,
                        modality=node.block.modality.value,
                        **asdict(utility),
                        **{f"selection_{key}": value for key, value in asdict(decision).items()},
                    )
                )
            pd.DataFrame(terms).to_parquet(path / "raw_marginal_terms.parquet", index=False)
            pd.DataFrame(
                dict(
                    source=result.graph.row_indices,
                    target=result.graph.column_indices,
                    weight=result.graph.weights,
                    edge_type=[t.value for t in result.graph.edge_types],
                )
            ).to_parquet(path / "edges.parquet", index=False)
            from mosaickv.evaluation.fox_campaign import sha

            record["raw_terms"] = {str(p): sha(p) for p in sorted(path.iterdir())}
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
            descriptors_sha256=descriptors.hexdigest(),
            selected_source_ids=tuple(
                sorted(
                    (
                        *result.external_exact_node_ids,
                        *(result.source_node_id(i) for i in result.selection.selected_node_ids),
                    )
                )
            ),
            degree_mean=result.graph.diagnostics.average_degree,
        )
        return result

    def baseline(*args, **kwargs):
        result = original_baseline(*args, **kwargs)
        record.update(
            selected_source_ids=result.selection.selected_node_ids,
            selected_source_sha256=digest(result.selection.selected_node_ids),
        )
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
        stack.enter_context(patch.object(runtime, "build_evidence_graph", graph))
        stack.enter_context(patch.object(runtime, "build_exact_baseline_plan", baseline))
        if variant.startswith("random-"):
            stack.enter_context(patch.object(runtime, "_uncertain_head_routes", random_routes))
        yield record


def validate_mechanism_contract(rows):
    """Fail closed on treatment-dependent candidate/quota domains, not outcomes."""
    import json
    from pathlib import Path

    from mosaickv.evaluation.fox_campaign import sha

    by_input = defaultdict(dict)
    for row in rows:
        if row["status"] != "completed":
            raise ValueError("incomplete mechanism experiment")
        key = (row["workload"], row["sample_id"], row["budget_bytes"])
        variant = row["variant"]
        if variant in by_input[key]:
            raise ValueError(f"duplicate mechanism experiment: {key}/{variant}")
        path = Path(row["mechanism_audit_path"])
        if sha(path) != row.get("mechanism_audit_sha256"):
            raise ValueError("mechanism activity artifact changed or is not hash-bound")
        activity = json.loads(path.read_text())
        for path, expected in activity.get("raw_terms", {}).items():
            if sha(Path(path)) != expected:
                raise ValueError("mechanism term artifacts changed")
        by_input[key][variant] = (row, activity)
    if not by_input:
        raise ValueError("empty mechanism experiment")
    report = []
    for key, variants in sorted(by_input.items()):
        fixed = (
            "mechanism-complete",
            "routed-prompt",
            "coverage-off",
            "value-off",
            "cross-modal-off",
        )
        required = (*fixed, "route-0", "route-zero-control")
        if not set(required).issubset(variants):
            raise ValueError("missing prespecified mechanism condition")
        # Bind the actual input/processor/runtime/scorer identity exported by
        # the worker, not merely its display sample ID. These fields are not a
        # claim that unrecorded query tensors or original KV were fingerprinted.
        for field in ("input_sha", "processor_sha", "source_sha", "manifest_sha256", "scorer_sha"):
            expected_length = 40 if field == "source_sha" else 64
            values = [variants[v][0].get(field) for v in variants]
            if any(
                not isinstance(value, str)
                or len(value) != expected_length
                or any(char not in "0123456789abcdef" for char in value)
                for value in values
            ):
                raise ValueError(f"missing or invalid mechanism source binding: {key}/{field}")
            if len(set(values)) != 1:
                raise ValueError(f"controlled mechanism source differs: {key}/{field}")
        for field in (
            "candidate_universe_sha256",
            "shortlist_sha256",
            "quota_sha256",
            "descriptors_sha256",
            "optimization_node_ids_sha256",
            "planner_routes",
            "graph_routes",
        ):
            if any(field not in variants[v][1] for v in fixed):
                raise ValueError(f"missing controlled mechanism domain: {key}/{field}")
            fingerprints = {digest(variants[v][1][field]) for v in fixed}
            if len(fingerprints) != 1:
                raise ValueError(f"controlled mechanism domain differs: {key}/{field}")
        zero, checked = variants["route-0"]
        topk, _ = variants["route-zero-control"]
        if (
            not checked.get("zero_route_equivalence_passed")
            or zero["token_ids"] != topk["token_ids"]
        ):
            raise ValueError("zero routing is not equivalent to shared Top-K")
        complete = variants["mechanism-complete"][1]
        anchor = set(complete["selected_source_ids"])
        for variant, (row, activity) in sorted(variants.items()):
            chosen = set(activity["selected_source_ids"])
            if not chosen or len(chosen) != len(activity["selected_source_ids"]):
                raise ValueError(f"empty or duplicate selected source IDs: {key}/{variant}")
            report.append(
                dict(
                    workload=key[0],
                    sample_id=key[1],
                    budget_bytes=key[2],
                    variant=variant,
                    selected_jaccard=len(chosen & anchor) / len(chosen | anchor),
                    graph_nodes=activity.get("graph_nodes", 0),
                    graph_edges=activity.get("graph_edges", 0),
                    cross_modal_edges_removed=activity.get("cross_modal_edges_removed"),
                    edge_sources=activity.get("edge_sources", {}),
                    fixed_candidate_quota_passed=True,
                    quality=row["score"],
                    metric_scope=row["subtask"] or "COCO_corpus_not_sample",
                )
            )
    return report
