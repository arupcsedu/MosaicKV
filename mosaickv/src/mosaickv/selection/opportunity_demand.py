"""Experimental prompt-only evidence policy. No labels, probes or K/V changes.

Weighted facility location is pooled demand; the separate per-query floor is
what prevents this variant from silently collapsing to mean-only protection.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from mosaickv.selection.objective import BudgetedObjective
from mosaickv.selection.selector import select_exact_subset
from mosaickv.selection.types import ObjectiveBreakdown


@dataclass(frozen=True)
class RouteEvidence:
    source_ids: tuple[int, ...]
    attention: np.ndarray
    anchor: tuple[int, ...]
    mandatory: tuple[int, ...]


def normalized_queries(attention):
    a = np.asarray(attention, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 1 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("prompt evidence must be a finite nonnegative [query,position] matrix")
    totals = a.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("every prompt query needs positive full-domain attention mass")
    return a / totals


def make_route_evidence(full_state, queries_by_head, scalar_scores, position_limits):
    """Build the exact scalar Top-K anchor, including mandatory unit blocks."""
    grouped = defaultdict(list)
    for source_id, block in enumerate(full_state.blocks):
        identity = (block.layer, block.kv_head)
        if identity in queries_by_head:
            if block.position_count != 1:
                raise ValueError("evidence policy requires one-token source blocks")
            grouped[identity].append((source_id, block))
    result = {}
    for identity, entries in sorted(grouped.items()):
        raw = queries_by_head[identity]
        positions = [block.physical_cache_indices[0] for _, block in entries]
        if sorted(positions) != list(range(raw.shape[1])):
            raise ValueError("query evidence must cover the complete source route, not a shortlist")
        a = normalized_queries(raw[:, positions])
        ids = tuple(source_id for source_id, _ in entries)
        mandatory = tuple(i for i, (_, block) in enumerate(entries) if block.mandatory)
        capacity = position_limits[identity]
        if type(capacity) is not int or not len(mandatory) <= capacity <= len(entries):
            raise ValueError("infeasible evidence route capacity")
        protected = set(mandatory)
        optional = sorted(
            (i for i in range(len(ids)) if i not in protected),
            key=lambda i: (-float(scalar_scores[ids[i]]), ids[i]),
        )
        if any(
            not math.isfinite(float(scalar_scores[sid])) or scalar_scores[sid] < 0 for sid in ids
        ):
            raise ValueError("anchor scores must be finite and nonnegative")
        anchor = tuple(sorted((*mandatory, *optional[: capacity - len(mandatory)])))
        result[identity] = RouteEvidence(ids, a, anchor, mandatory)
    if set(result) != set(queries_by_head):
        raise ValueError("query routes do not match source KV routes")
    return result


def query_floor(attention, anchor, selected, fraction):
    """Relative to each query's own anchor mass, never a pooled mean."""
    before = attention[:, list(anchor)].sum(axis=1)
    after = attention[:, list(selected)].sum(axis=1)
    ratio = np.divide(after, before, out=np.ones_like(after), where=before > 0)
    return bool(np.all(after + 1e-12 >= fraction * before)), float(ratio.min())


def opportunity(route, *, shortlist_size, minimum_query_mass_fraction):
    """Best bounded feasible single exchange's capped-query gain.

    Common target is the anchor's mean retained mass, not one target per query:
    per-query anchor targets would all be saturated and allow no positive gain.
    Scoring changes no cache and allocates no capacity across routes.
    """
    a = route.attention
    anchor = np.asarray(route.anchor, dtype=np.int64)
    mass = a[:, anchor].sum(axis=1)
    target = max(float(mass.mean()), 1e-12)
    base = float(np.minimum(mass / target, 1).mean())
    protected = set(route.mandatory)
    selected = set(route.anchor)
    donors = [i for i in route.anchor if i not in protected]
    receivers = [i for i in range(a.shape[1]) if i not in selected]
    derivative = a.T @ ((mass < target).astype(np.float64) / (target * len(mass)))
    donors = sorted(donors, key=lambda i: (derivative[i], route.source_ids[i]))[:shortlist_size]
    receivers = sorted(receivers, key=lambda i: (-derivative[i], route.source_ids[i]))[
        :shortlist_size
    ]
    best_gain, best_pair, feasible = 0.0, None, 0
    for donor in donors:
        if not receivers:
            break
        new_mass = mass[:, None] - a[:, donor, None] + a[:, receivers]
        valid = (new_mass + 1e-12 >= minimum_query_mass_fraction * mass[:, None]).all(axis=0)
        gains = np.minimum(new_mass / target, 1).mean(axis=0) - base
        feasible += int(valid.sum())
        for receiver, gain, accepted in zip(receivers, gains, valid, strict=True):
            pair = (route.source_ids[donor], route.source_ids[receiver])
            if accepted and (
                gain > best_gain + 1e-12
                or (
                    gain > 1e-12
                    and abs(gain - best_gain) <= 1e-12
                    and (best_pair is None or pair < best_pair)
                )
            ):
                best_gain, best_pair = float(gain), pair
    score = min(1.0, max(0.0, best_gain / max(1 - base, 1e-12)))
    return dict(
        score=score,
        capped_query_gain=best_gain,
        anchor_coverage=base,
        candidate_pairs=len(donors) * len(receivers),
        feasible_pairs=feasible,
        witness_source_ids=best_pair,
    )


class DemandWeightedObjective(BudgetedObjective):
    """Nonnegative demand-weighted facility rewards; same local/modal terms."""

    def __init__(self, graph, utilities, config, evidence, floor):
        self.evidence = {
            identity: evidence[identity]
            for identity in {(node.block.layer, node.block.kv_head) for node in graph.nodes}
        }
        self.floor = floor
        source_location = {
            sid: (identity, index)
            for identity, route in self.evidence.items()
            for index, sid in enumerate(route.source_ids)
        }
        self.local_sources = {}
        weights = []
        for node in graph.nodes:
            identity, index = source_location[node.source_id]
            self.local_sources[node.node_id] = (identity, index)
            weights.append(float(self.evidence[identity].attention[:, index].mean()))
        if len(source_location) != len(graph.nodes):
            raise ValueError(
                "weighted objective requires all routed source positions, "
                "not shortlist normalization"
            )
        self.demand = np.asarray(weights, dtype=np.float64)
        self.demand /= self.demand.sum()
        self.guard_report = {}
        super().__init__(graph, utilities, config)

    def _build_facility(self):
        original = super()._build_facility()
        n = len(original)
        return tuple(
            {target: value * n * self.demand[target] for target, value in row.items()}
            for row in original
        )

    def evaluate(self, selected_node_ids):
        selected = self._selected(selected_node_ids)
        covered = np.zeros(len(self.graph.nodes), dtype=np.float64)
        for node_id in selected:
            for target, value in self._facility[node_id].items():
                covered[target] = max(covered[target], value)
        coverage = min(1.0, max(0.0, math.fsum(covered) / len(covered)))
        modalities = {self.graph.nodes[i].block.modality for i in selected}
        modal_coverage = len(modalities) / len(self._modalities)
        local = math.fsum(self.utilities.for_node(i).local_utility for i in selected)
        return ObjectiveBreakdown(
            selected,
            local,
            coverage,
            1 - coverage,
            modal_coverage,
            1 - modal_coverage,
            local - self.config.lambda_g * coverage - self.config.lambda_m * modal_coverage,
        )

    def protect_selection(self, selection):
        chosen = set(selection.selected_node_ids)
        by_route = defaultdict(list)
        source_to_node = {node.source_id: node.node_id for node in self.graph.nodes}
        for node_id in chosen:
            identity, index = self.local_sources[node_id]
            by_route[identity].append(index)
        checks = []
        for identity, route in sorted(self.evidence.items()):
            selected = by_route[identity]
            if len(selected) != len(route.anchor):
                raise ValueError("query protection must not change physical route capacity")
            valid, ratio = query_floor(route.attention, route.anchor, selected, self.floor)
            checks.append(
                dict(
                    layer=identity[0],
                    kv_head=identity[1],
                    proposed_minimum_query_mass_ratio=ratio,
                    final_minimum_query_mass_ratio=ratio if valid else 1.0,
                    prompt_queries=int(route.attention.shape[0]),
                    attainable_positions=len(route.anchor),
                    rolled_back=not valid,
                )
            )
            if not valid:
                chosen.difference_update(source_to_node[route.source_ids[i]] for i in selected)
                chosen.update(source_to_node[route.source_ids[i]] for i in route.anchor)
        result = select_exact_subset(self, selection.budget, chosen)
        if result.active_bytes != selection.active_bytes:
            raise ValueError("query guard changed the exact selected-byte budget")
        for identity, route in self.evidence.items():
            installed = [
                self.local_sources[i][1]
                for i in result.selected_node_ids
                if self.local_sources[i][0] == identity
            ]
            passed, _ = query_floor(route.attention, route.anchor, installed, self.floor)
            if not passed or len(installed) != len(route.anchor):
                raise ValueError("final query guard invariant failed after route rollback")
        self.guard_report = dict(
            minimum_query_mass_fraction=self.floor,
            routes=checks,
            rolled_back_routes=sum(c["rolled_back"] for c in checks),
            whole_cache_fallback=False,
        )
        return result
