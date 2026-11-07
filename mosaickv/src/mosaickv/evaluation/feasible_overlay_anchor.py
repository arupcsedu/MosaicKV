"""Choose a complete actual prototype group, never a padded or mandatory one."""

import math
from collections import Counter
from dataclasses import replace

from mosaickv.prototypes.construction import plan_prototype_assignments


def choose(graph, selection, config, anchors, scores):
    anchors = tuple(sorted(set(anchors)))
    if not anchors or any(not math.isfinite(float(scores[a])) for a in anchors):
        raise RuntimeError("finite compatible selected prototype anchors required")
    selected = set(selection.selected_node_ids)
    if not set(anchors) <= selected:
        raise ValueError("prototype anchors must already be selected exact")

    def rank(a):
        return float(scores[a]), -a

    def plan(a):
        return plan_prototype_assignments(graph, selection, config, prototype_anchor_node_ids=(a,))

    def valid(p, a):
        expected = config.group_size_for(graph.nodes[a].block.layer, graph.nodes[a].block.kv_head)
        return (
            p.incompatibility is None
            and len(p.assignment_rows) == expected
            and all(
                anchor == a and node not in selected and not graph.nodes[node].block.mandatory
                for node, anchor, _ in p.assignment_rows
            )
        )

    first = max(anchors, key=rank)
    initial = plan(first)
    if valid(initial, first):
        return (
            first,
            initial,
            dict(
                original_best_anchor_preserved=True,
                initial_anchor=first,
                initial_member_count=len(initial.assignment_rows),
                selected_anchor=first,
                assignment_plans_computed=1,
                no_optional_reservation_or_budget_changed=True,
            ),
        )
    # Descriptor-only discovery; do not materialize multiple prototypes or
    # loosen the original construction cap. Fully populated size-(m+1) segments
    # have exactly one selected anchor, so their assignment is unchanged when
    # the final allowlist is reduced back to that anchor alone.
    discovery = plan_prototype_assignments(
        graph,
        selection,
        replace(config, maximum_prototypes_per_head=None),
        prototype_anchor_node_ids=anchors,
    )
    if discovery.incompatibility is not None:
        raise RuntimeError(f"prototype feasibility discovery failed: {discovery.incompatibility}")
    counts = Counter(a for _, a, _ in discovery.assignment_rows)
    feasible = [
        a
        for a in anchors
        if counts[a]
        == config.group_size_for(graph.nodes[a].block.layer, graph.nodes[a].block.kv_head)
    ]
    if not feasible:
        raise RuntimeError("no selected image anchor has a complete nonmandatory prototype group")
    winner = max(feasible, key=rank)
    final = plan(winner)
    if not valid(final, winner):
        raise RuntimeError("final single-anchor prototype does not satisfy exact membership")
    return (
        winner,
        final,
        dict(
            original_best_anchor_preserved=False,
            initial_anchor=first,
            initial_member_count=len(initial.assignment_rows),
            initial_incompatibility=initial.incompatibility,
            selected_anchor=winner,
            feasible_anchor_count=len(feasible),
            assignment_plans_computed=3,
            no_optional_reservation_or_budget_changed=True,
        ),
    )
