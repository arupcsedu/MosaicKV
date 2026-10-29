"""Pinned layer-profile shape under a lifetime cap; no quality guarantee.

Profile weights come from the
attributed PrefixKV offline allocation, not evaluation answers. The allocator
preserves mandatory minima, equal lengths within each layer and one layer's
growth-copy reserve; it never hides padding behind logical byte accounting.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileLifetimeCapacity:
    prompt_positions: tuple[int, ...]
    prompt_bytes: int
    history_bytes: int
    resident_ceiling_bytes: int
    growth_scratch_bytes: int
    charged_ceiling_bytes: int
    cap_bytes: int
    capacity_shortfall_bytes: int
    largest_single_increment_feasible: bool
    history_positions: tuple[int, ...] = ()


def allocate_profile_history(forget_ratios, mean_capacity, minimum):
    """Same total slots, explicit sink/recent floor; no target-length input."""
    forget_ratios = tuple(forget_ratios)
    if (
        type(mean_capacity) is not int
        or type(minimum) is not int
        or not 0 <= minimum <= mean_capacity
    ):
        raise ValueError("invalid history mean/floor")
    if any(
        isinstance(r, bool)
        or not isinstance(r, (int, float))
        or not math.isfinite(r)
        or not 0 <= r < 1
        for r in forget_ratios
    ):
        raise ValueError("invalid history profile weights")
    weights = tuple(1 - r for r in forget_ratios)
    if not weights or any(not math.isfinite(w) or not 0 < w <= 1 for w in weights):
        raise ValueError("invalid history profile weights")
    result = [minimum] * len(weights)
    queue = [((minimum + 1) / w, i) for i, w in enumerate(weights)]
    heapq.heapify(queue)
    for _ in range((mean_capacity - minimum) * len(weights)):
        _level, i = heapq.heappop(queue)
        result[i] += 1
        heapq.heappush(queue, ((result[i] + 1) / weights[i], i))
    return tuple(result)


def allocate_profile_lifetime(
    forget_ratios,
    source_lengths,
    mandatory_counts,
    layer_position_bytes,
    *,
    cap_bytes,
    generated_capacity,
    history_allocation="uniform",
    minimum_history_capacity=0,
):
    """Weighted water filling with a physical, maximum-layer scratch cost.

    Deterministic proportional fairness, not claimed globally optimal for a
    heterogeneous-cost knapsack. Every rejected single increment is infeasible;
    report remaining capacity explicitly. Equal-geometry controls use the same
    allocator, so any lattice shortfall is identical, not an algorithm advantage.
    """
    ratios, lengths, mandatory, costs = map(
        tuple,
        (
            forget_ratios,
            source_lengths,
            mandatory_counts,
            layer_position_bytes,
        ),
    )
    n = len(ratios)
    if (
        not n
        or any(len(x) != n for x in (lengths, mandatory, costs))
        or type(cap_bytes) is not int
        or cap_bytes < 1
        or type(generated_capacity) is not int
        or generated_capacity < 0
        or any(type(x) is not int or x < 1 for x in (*lengths, *costs))
        or any(type(x) is not int or x < 1 or x > lengths[i] for i, x in enumerate(mandatory))
        or any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or not 0 <= x < 1
            for x in ratios
        )
    ):
        raise ValueError("invalid label-blind profile/lifetime geometry")
    if not isinstance(history_allocation, str) or history_allocation not in {
        "uniform",
        "profile_weighted",
    }:
        raise ValueError("unsupported history allocation")
    generated = (
        (generated_capacity,) * n
        if history_allocation == "uniform"
        else allocate_profile_history(ratios, generated_capacity, minimum_history_capacity)
    )
    selected = list(mandatory)
    history = sum(c * g for c, g in zip(costs, generated, strict=True))
    prompt = sum(p * c for p, c in zip(selected, costs, strict=True))
    scratch = max((p + g) * c for p, g, c in zip(selected, generated, costs, strict=True))
    if prompt + history + scratch > cap_bytes:
        raise ValueError("mandatory prompt plus history/growth cannot fit the hard cap")
    weights = tuple(1 - r for r in ratios)
    queue = [((selected[i] + 1) / weights[i], i) for i in range(n) if selected[i] < lengths[i]]
    heapq.heapify(queue)
    while queue:
        _level, layer = heapq.heappop(queue)
        new_scratch = max(scratch, (selected[layer] + 1 + generated[layer]) * costs[layer])
        if prompt + costs[layer] + history + new_scratch > cap_bytes:
            continue
        selected[layer] += 1
        prompt += costs[layer]
        scratch = new_scratch
        if selected[layer] < lengths[layer]:
            heapq.heappush(queue, ((selected[layer] + 1) / weights[layer], layer))
    feasible = any(
        selected[i] < lengths[i]
        and prompt + costs[i] + history + max(scratch, (selected[i] + 1 + generated[i]) * costs[i])
        <= cap_bytes
        for i in range(n)
    )
    return ProfileLifetimeCapacity(
        tuple(selected),
        prompt,
        history,
        prompt + history,
        scratch,
        prompt + history + scratch,
        cap_bytes,
        cap_bytes - prompt - history - scratch,
        feasible,
        generated,
    )


@dataclass(frozen=True)
class ProfiledExactSelection:
    positions: tuple
    capacities: tuple[int, ...]
    prompt_bytes: int
    profile_sha256: str
    capacity: ProfileLifetimeCapacity

    def trace(self):
        import hashlib
        import json
        from dataclasses import asdict

        return {
            "variant": "pinned_profile_shape_exact_anchor_resize_v3",
            "benchmark_labels_accessed": False,
            "profile_sha256": self.profile_sha256,
            "profile_source": "PrefixKV_pinned_offline_layer_forget_ratios",
            "profile_calibration_identity_provenance": "not_available_in_upstream_numeric_list",
            "not_official_PrefixKV": True,
            "not_independent_quality_evidence": True,
            "capacity": asdict(self.capacity),
            "layer_capacities": list(self.capacities),
            "prompt_bytes": self.prompt_bytes,
            "padding_bytes": 0,
            "selected_positions_sha256": hashlib.sha256(
                json.dumps(self.positions, separators=(",", ":")).encode()
            ).hexdigest(),
        }


def build_profiled_exact_selection(
    full_state, token_attention, exact_blocks, capacity, profile_sha
):
    """Resize each frozen exact anchor; keep mandatory and original positions.

    Shrinking prioritizes prompt mass inside the exact anchor, expansion adds
    omitted positions by prompt mass. No K/V read/transform or model probe is
    involved. The original graph/utility-selected anchor remains separately
    auditable, not mislabeled as the final resized selection.
    """
    import numpy as np

    if full_state.block_size != 1:
        raise ValueError("profiled exact resizing requires common one-token blocks")
    anchors = {}
    for block in exact_blocks:
        anchors.setdefault((block.layer, block.kv_head), set()).update(block.physical_cache_indices)
    mandatory = {
        i
        for i, p in enumerate(full_state.logical_positions.physical_to_logical)
        if p in full_state.mandatory_logical_positions
    }
    positions = []
    for layer_index, layer in enumerate(full_state.layers):
        heads = []
        count = capacity.prompt_positions[layer_index]
        if count < len(mandatory):
            raise ValueError("profile capacity lost a mandatory prompt token")
        for head in range(layer.kv_heads):
            identity = (layer_index, head)
            scores = np.asarray(token_attention[identity], dtype=np.float64)
            if scores.shape != (layer.sequence_length,) or not np.isfinite(scores).all():
                raise ValueError("profile selection needs finite prompt scores for every route")
            order = np.lexsort((np.arange(len(scores)), -scores))
            anchor = anchors.get(identity, set())
            ranked = [int(p) for p in order if int(p) in anchor and int(p) not in mandatory]
            ranked.extend(int(p) for p in order if int(p) not in anchor and int(p) not in mandatory)
            selected = tuple(sorted(mandatory | set(ranked[: count - len(mandatory)])))
            if len(selected) != count:
                raise ValueError("profile selection failed exact route capacity")
            heads.append(selected)
        positions.append(tuple(heads))
    return ProfiledExactSelection(
        tuple(positions), capacity.prompt_positions, capacity.prompt_bytes, profile_sha, capacity
    )
