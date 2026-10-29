"""Conservative installed-plan guard; no draft, labels, fitting, or new swaps.

The graph's actual subset is compared to its same-capacity Top-K anchor using
each query-head/prompt-position separately. This is a prompt-reconstruction
constraint, not a guarantee of autoregressive task quality.
"""

from __future__ import annotations

import numpy as np

from mosaickv.selection.selector import select_exact_subset


def output_errors(attention, values, selected):
    a, v = np.asarray(attention, dtype=np.float64), np.asarray(values, dtype=np.float64)
    if a.ndim != 2 or v.ndim != 2 or a.shape[1] != v.shape[0]:
        raise ValueError("expected individual [queries,positions] and [positions,value_dim]")
    if not np.isfinite(a).all() or not np.isfinite(v).all() or (a < 0).any():
        raise ValueError("nonfinite or negative prompt evidence")
    if (
        not selected
        or len(selected) != len(set(selected))
        or any(i < 0 or i >= a.shape[1] for i in selected)
    ):
        raise ValueError("invalid selected source positions")
    totals = a.sum(axis=1, keepdims=True)
    if (totals <= 0).any():
        raise ValueError("zero full-domain prompt mass")
    a = a / totals
    mass = a[:, list(selected)].sum(axis=1)
    if (mass <= 0).any():
        raise ValueError("zero selected query mass")
    target = a @ v
    kept = (a[:, list(selected)] @ v[list(selected)]) / mass[:, None]
    errors = ((target - kept) ** 2).sum(axis=1)
    return mass, errors


def decide(attention, values, anchor, proposal, mandatory, *, minimum_query_mass_fraction):
    if len(anchor) != len(proposal) or not set(mandatory) <= set(proposal):
        raise ValueError("proposal changed physical capacity or removed mandatory positions")
    before_mass, before = output_errors(attention, values, anchor)
    after_mass, after = output_errors(attention, values, proposal)
    mass_ok = bool(np.all(after_mass + 1e-12 >= minimum_query_mass_fraction * before_mass))
    # Roundoff slack is fixed for FP64 arithmetic, never calibrated to answers.
    output_ok = bool(np.all(after <= before + 1e-10 * np.maximum(1.0, before)))
    accepted = mass_ok and output_ok
    installed = proposal if accepted else anchor
    _, final = output_errors(attention, values, installed)
    return tuple(installed), dict(
        proposal_accepted=accepted,
        rolled_back=not accepted,
        changed_positions=len(set(installed) - set(anchor)),
        anchor_error=float(before.mean()),
        proposed_error=float(after.mean()),
        installed_error=float(final.mean()),
        maximum_per_query_error_increase=float(np.max(final - before)),
        minimum_mass_ratio=float(np.min(after_mass / before_mass)),
        individual_queries=len(before),
        mass_guard_passed=mass_ok,
        output_guard_passed=output_ok,
    )


def guard_installed_selection(
    full_state, graph, objective, selection, evidence, *, minimum_query_mass_fraction
):
    chosen = set(selection.selected_node_ids)
    source_to_node = {n.source_id: n.node_id for n in graph.nodes}
    checks = []
    for identity, route in sorted(evidence.items()):
        selected = tuple(
            i
            for i, sid in enumerate(route.source_ids)
            if sid in source_to_node and source_to_node[sid] in chosen
        )
        layer = full_state.layers[identity[0]]
        if layer.value_head_dimension != 1 or layer.value_sequence_dimension != 2:
            raise NotImplementedError("installed guard requires explicit B,H,N,D value layout")
        v = layer.value
        if v.ndim != 4 or v.shape[0] != 1:
            raise ValueError("only single-request value evidence is supported")
        values = v[0, identity[1]]
        if hasattr(values, "detach"):
            values = values.detach().float().cpu().numpy()
        positions = [full_state.blocks[sid].physical_cache_indices[0] for sid in route.source_ids]
        installed, record = decide(
            route.attention,
            np.asarray(values)[positions],
            route.anchor,
            selected,
            route.mandatory,
            minimum_query_mass_fraction=minimum_query_mass_fraction,
        )
        chosen.difference_update(source_to_node[sid] for sid in route.source_ids)
        chosen.update(source_to_node[route.source_ids[i]] for i in installed)
        checks.append(dict(layer=identity[0], kv_head=identity[1], **record))
    result = select_exact_subset(objective, selection.budget, chosen)
    if result.active_bytes != selection.active_bytes:
        raise ValueError("installed output guard changed physical bytes")
    return result, dict(
        version="installed_output_guard_v2",
        routes=checks,
        accepted_routes=sum(r["proposal_accepted"] for r in checks),
        changed_positions=sum(r["changed_positions"] for r in checks),
        no_new_proposals=True,
        no_label_access=True,
        whole_cache_fallback=False,
    )
