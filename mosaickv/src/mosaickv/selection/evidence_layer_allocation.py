"""Label-blind, exact-byte layer allocation around MosaicKV's prompt anchor.

Development variant: prompt attention/value output error is a surrogate, NOT
proof of generated-answer fidelity. Inputs contain tensors and cache metadata
only, never benchmark answers, references, correctness scores or future labels.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

# Fixed before the successor cohort is materialized. These are numerical
# tolerances for FP32 attention/value arithmetic, not quality non-inferiority.
RECONSTRUCTION_ATOL = 1e-7
RECONSTRUCTION_RTOL = 1e-6


def reconstruction_mse(mass: Any, values: Any, positions: tuple[tuple[int, ...], ...]) -> Any:
    """Per-head channel-mean squared error after retained-softmax renormalization.

    The target is the full prompt-derived attention output, not an answer or a
    future decode target. Reductions use FP64 on the already FP32 working values.
    Invalid/non-finite or zero-mass subsets cannot pass a guard.
    """
    import torch

    if mass.ndim != 2 or values.ndim != 3 or mass.shape != values.shape[:2]:
        raise ValueError("reconstruction requires HN attention and HND values")
    if len(positions) != mass.shape[0]:
        raise ValueError("reconstruction requires positions for every head")
    if (
        not bool(torch.isfinite(mass).all())
        or not bool(torch.isfinite(values).all())
        or bool((mass < 0).any())
        or bool((mass.sum(-1) <= 0).any())
    ):
        raise ValueError("reconstruction requires finite values and positive attention mass")
    mask = torch.zeros_like(mass, dtype=torch.bool)
    for head, row in enumerate(positions):
        if not row or len(set(row)) != len(row) or any(p < 0 or p >= mass.shape[1] for p in row):
            raise ValueError("invalid reconstruction subset positions")
        mask[head, list(row)] = True
    full_mass = mass.sum(-1, dtype=torch.float64).unsqueeze(-1)
    kept = mass * mask
    retained_mass = kept.sum(-1, dtype=torch.float64).unsqueeze(-1)
    if bool((retained_mass <= 0).any()):
        raise ValueError("reconstruction subset has zero retained attention mass")
    full = (mass.unsqueeze(-1) * values).sum(1, dtype=torch.float64) / full_mass
    retained = (kept.unsqueeze(-1) * values).sum(1, dtype=torch.float64) / retained_mass
    error = (retained - full).square().mean(-1)
    if not bool(torch.isfinite(error).all()):
        raise ValueError("non-finite reconstruction error")
    return error


def reconstruction_guard(
    anchor: tuple[tuple[float, ...], ...], proposed: tuple[tuple[float, ...], ...]
) -> dict[str, Any]:
    """Atomic rollback unless EVERY layer/head is non-worse; never average a loss away."""
    if not anchor or len(anchor) != len(proposed):
        raise ValueError("reconstruction guard requires matching nonempty layers")
    rejected = []
    for layer, (before, after) in enumerate(zip(anchor, proposed, strict=True)):
        if not before or len(before) != len(after):
            raise ValueError("reconstruction guard requires matching nonempty heads")
        for head, (a, b) in enumerate(zip(before, after, strict=True)):
            if not math.isfinite(a) or not math.isfinite(b) or min(a, b) < 0:
                raise ValueError("reconstruction guard requires finite nonnegative errors")
            if b > a + RECONSTRUCTION_ATOL + RECONSTRUCTION_RTOL * a:
                rejected.append([layer, head])
    return {
        "policy": "atomic_transaction_every_head_nonregression_v1",
        "metric": "renormalized_prompt_attention_output_channel_mean_squared_error",
        "absolute_tolerance": RECONSTRUCTION_ATOL,
        "relative_tolerance": RECONSTRUCTION_RTOL,
        "accepted": not rejected,
        "rejected_layer_heads": rejected,
        "rejected_head_count": len(rejected),
        "decision": "accept_proposal" if not rejected else "retain_same_byte_compressed_anchor",
        "used_fullkv_fallback": False,
        "anchor_mse_by_layer_head": anchor,
        "proposed_mse_by_layer_head": proposed,
        "final_mse_by_layer_head": anchor if rejected else proposed,
    }


def validate_reconstruction_trace(allocation: dict[str, Any]) -> dict[str, Any]:
    """Validate serialized guard decisions; this is NOT a model-quality gate."""
    guard = allocation["reconstruction_guard"]
    expected = reconstruction_guard(
        guard["anchor_mse_by_layer_head"], guard["proposed_mse_by_layer_head"]
    )
    for key, value in expected.items():
        if guard.get(key) != value:
            raise ValueError(f"reconstruction guard trace inconsistent: {key}")
    selected_hash = (
        guard["proposed_positions_sha256"]
        if guard["accepted"]
        else guard["anchor_positions_sha256"]
    )
    if allocation["selected_positions_sha256"] != selected_hash:
        raise ValueError("reconstruction guard final positions do not match decision")
    seconds = guard["host_wall_seconds_including_device_transfers"]
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("reconstruction guard timing missing or non-finite")
    return {
        "accepted": guard["accepted"],
        "rejected_head_count": guard["rejected_head_count"],
        "measurement_seconds": seconds,
        "anchor_mse_mean": float(
            np.mean([x for row in guard["anchor_mse_by_layer_head"] for x in row])
        ),
        "proposed_mse_mean": float(
            np.mean([x for row in guard["proposed_mse_by_layer_head"] for x in row])
        ),
        "final_mse_mean": float(
            np.mean([x for row in guard["final_mse_by_layer_head"] for x in row])
        ),
    }


def allocate_layer_slots(
    curves: tuple[tuple[float, ...], ...],
    lower: tuple[int, ...],
    upper: tuple[int, ...],
    total_slots: int,
) -> tuple[int, ...]:
    """Largest feasible allocation; deterministic ties, exact physical slots."""
    if not curves or len(curves) != len(lower) or len(curves) != len(upper):
        raise ValueError("layer allocation vectors must be nonempty and have equal lengths")
    if type(total_slots) is not int or not sum(lower) <= total_slots <= sum(upper):
        raise ValueError("layer allocation cannot satisfy the exact global capacity")
    queue: list[tuple[float, int]] = []
    counts = list(lower)
    envelopes = []
    for layer, (curve, low, high) in enumerate(zip(curves, lower, upper, strict=True)):
        if not 1 <= low <= high <= len(curve):
            raise ValueError("layer capacity violates source or mandatory bounds")
        values = np.asarray(curve, dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("layer marginal weights must be finite and nonnegative")
        # Reserved evidence may reorder early slots. The suffix upper envelope
        # is concave and keeps allocation deterministic without dropping groups.
        envelope = np.maximum.accumulate(values[::-1])[::-1]
        envelopes.append(envelope)
        if low < high:
            heapq.heappush(queue, (-float(envelope[low]), layer))
    for _ in range(total_slots - sum(counts)):
        if not queue:
            raise ValueError("layer allocation exhausted feasible capacity")
        _, layer = heapq.heappop(queue)
        counts[layer] += 1
        if counts[layer] < upper[layer]:
            heapq.heappush(queue, (-float(envelopes[layer][counts[layer]]), layer))
    return tuple(counts)


def evidence_ranking(
    score: np.ndarray,
    mandatory: tuple[int, ...],
    text: tuple[int, ...],
    visual_groups: tuple[tuple[int, ...], ...],
    reservation_capacity: int,
) -> tuple[int, ...]:
    """Reserve boundaries, salient prompt text and spatial visual proxies."""
    if score.ndim != 1 or not np.isfinite(score).all() or (score < 0).any():
        raise ValueError("evidence scores must be a finite nonnegative vector")
    size = len(score)
    if len(set(mandatory)) != len(mandatory) or any(p < 0 or p >= size for p in mandatory):
        raise ValueError("invalid mandatory evidence positions")
    if len(mandatory) > reservation_capacity:
        raise ValueError("mandatory evidence exceeds attainable layer capacity")
    ranked = tuple(int(p) for p in np.argsort(-score, kind="stable"))
    selected = list(sorted(mandatory))
    room = max(0, reservation_capacity - len(selected))
    optional_limit = min(room, max(1, reservation_capacity // 4))
    text_set = set(text)
    text_rank = [p for p in ranked if p in text_set and p not in selected]
    visual = [min(group, key=lambda p: (-float(score[p]), p)) for group in visual_groups if group]
    # Interleave text and spatial evidence rather than swallowing the complete
    # small budget with a long instruction or a single high-attention region.
    optional = []
    for i in range(max(len(text_rank), len(visual))):
        for group in (text_rank, visual):
            if i < len(group) and group[i] not in selected and group[i] not in optional:
                optional.append(group[i])
    selected.extend(optional[:optional_limit])
    used = set(selected)
    selected.extend(p for p in ranked if p not in used)
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class EvidenceLayerPlan:
    positions: tuple[tuple[tuple[int, ...], ...], ...]
    capacities: tuple[int, ...]
    prompt_bytes: int
    anchor_errors: tuple[float, ...]
    selected_prompt_masses: tuple[float, ...]
    mandatory_counts: tuple[int, ...]
    reconstruction: dict[str, Any] | None = None

    def trace(self) -> dict[str, Any]:
        result = {
            "variant": (
                "output_impact_layer_allocation_v1"
                if self.reconstruction is None
                else "guarded_output_impact_layer_allocation_v1"
            ),
            "evidence": "prompt_forecast_attention_and_source_values_only",
            "benchmark_labels_accessed": False,
            "output_error_is_a_surrogate_not_answer_fidelity": True,
            "allocation_bounds": [0.5, 1.5],
            "evidence_reservation_fraction": 0.25,
            "ocr_source": "visual_patch_spatial_proxy_no_ocr_annotations",
            "layer_capacities": list(self.capacities),
            "anchor_output_errors": list(self.anchor_errors),
            "selected_prompt_mass_by_layer": list(self.selected_prompt_masses),
            "mandatory_counts": list(self.mandatory_counts),
            "prompt_bytes": self.prompt_bytes,
            "padding_bytes": 0,
            "selected_positions_sha256": hashlib.sha256(
                json.dumps(self.positions, separators=(",", ":")).encode()
            ).hexdigest(),
            **(
                {} if self.reconstruction is None else {"reconstruction_guard": self.reconstruction}
            ),
        }
        if self.reconstruction is not None and "exchange_search" in self.reconstruction:
            result["variant"] = "direct_error_single_slot_layer_exchange_v1"
            result["evidence_reservation_fraction"] = "unchanged_anchor_only_not_reapplied"
            result["ocr_source"] = "unchanged_anchor_policy_no_new_OCR_or_spatial_reservation"
        if self.reconstruction is not None and "single_exchange" in self.reconstruction:
            result["variant"] = "V6_single_exchange_shortlist_not_final_selection"
            result["allocation_bounds"] = "one_layer_slot_donor_to_receiver_per_proposal"
            result["evidence_reservation_fraction"] = "unchanged_anchor_only_not_reapplied"
            result["ocr_source"] = "unchanged_anchor_policy_no_new_OCR_or_spatial_reservation"
            result["role"] = "actual_full_horizon_draft_guard_required_before_promotion"
        return result


def build_evidence_layer_plan(
    full_state: Any,
    token_attention: dict[tuple[int, int], Any],
    anchor_records: tuple[dict[str, Any], ...],
    *,
    prompt_budget_bytes: int,
    uniform_capacity: int,
    guard_reconstruction: bool = False,
) -> EvidenceLayerPlan:
    """Allocate complete layer slots; retain original K/V and logical indices."""
    import torch

    from mosaickv.cache_state import Modality

    layers = full_state.layers
    costs = tuple(layer.byte_size // layer.sequence_length for layer in layers)
    if len(set(costs)) != 1 or prompt_budget_bytes % costs[0]:
        raise ValueError("evidence allocation requires uniform exact per-layer slot bytes")
    total = prompt_budget_bytes // costs[0]
    physical = {
        logical: position
        for position, logical in enumerate(full_state.logical_positions.physical_to_logical)
    }
    anchor: dict[tuple[int, int], set[int]] = {
        (i, h): set() for i, layer in enumerate(layers) for h in range(layer.kv_heads)
    }
    for row in anchor_records:
        if row.get("tier") != "exact":
            raise ValueError("evidence allocation requires an exact-only MosaicKV anchor")
        anchor[(int(row["layer"]), int(row["kv_head"]))].add(physical[int(row["logical_position"])])
    mandatory: dict[tuple[int, int], list[int]] = {
        (i, h): [] for i, layer in enumerate(layers) for h in range(layer.kv_heads)
    }
    for block in full_state.blocks:
        if block.mandatory:
            mandatory[(block.layer, block.kv_head)].extend(block.physical_cache_indices)
    text = tuple(
        physical[p]
        for span in full_state.modality_spans
        if span.modality is Modality.TEXT
        for p in range(span.start, span.end)
        if p in physical
    )
    visual_groups: list[tuple[int, ...]] = []
    for span in full_state.modality_spans:
        if span.modality not in {Modality.IMAGE, Modality.VIDEO}:
            continue
        visual_positions = [physical[p] for p in range(span.start, span.end) if p in physical]
        visual_groups.extend(
            tuple(int(p) for p in group) for group in np.array_split(visual_positions, 4)
        )
    rankings, curves, errors, masses, bounds_low, bounds_high, mandatory_counts = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i, layer in enumerate(layers):
        if (
            layer.key_head_dimension,
            layer.key_sequence_dimension,
            layer.value_head_dimension,
            layer.value_sequence_dimension,
        ) != (1, 2, 1, 2):
            raise ValueError("evidence allocation supports audited BHSD source tensors only")
        values = layer.value.detach()[0].float()
        mass = torch.as_tensor(
            np.stack([token_attention[(i, h)] for h in range(layer.kv_heads)]),
            device=values.device,
            dtype=torch.float32,
        )
        if (
            mass.shape != values.shape[:2]
            or not bool(torch.isfinite(mass).all())
            or bool((mass < 0).any())
        ):
            raise ValueError("invalid source attention for evidence allocation")
        mass = mass / mass.sum(-1, keepdim=True).clamp_min(1e-20)
        output = (mass.unsqueeze(-1) * values).sum(1)
        deviation = (values - output.unsqueeze(1)).square().mean(-1).sqrt()
        impact = mass * deviation
        anchor_mask = torch.zeros_like(mass)
        for h in range(layer.kv_heads):
            anchor_mask[h, sorted(anchor[(i, h)])] = 1
        kept = mass * anchor_mask
        approximation = (kept.unsqueeze(-1) * values).sum(1) / kept.sum(-1, keepdim=True).clamp_min(
            1e-20
        )
        error = (approximation - output).square().mean(-1).sqrt().mean()
        # Value magnitude is retained across layers; unlike per-route L1 risk,
        # sensitive layers do not all collapse to the same normalized weight.
        raw = impact + 0.1 * mass * output.square().mean(-1).sqrt().unsqueeze(-1)
        scores = raw + 0.1 * raw.mean(-1, keepdim=True) * anchor_mask
        host = torch.stack((scores, raw, mass)).detach().cpu().numpy()
        errors.append(float(error.item()))
        must = max(len(set(mandatory[(i, h)])) for h in range(layer.kv_heads))
        low = max(1, must, math.floor(uniform_capacity * 0.5))
        high = min(layer.sequence_length, max(low, math.ceil(uniform_capacity * 1.5)))
        bounds_low.append(low)
        bounds_high.append(high)
        mandatory_counts.append(must)
        rank = tuple(
            evidence_ranking(
                host[0, h], tuple(sorted(set(mandatory[(i, h)]))), text, tuple(visual_groups), low
            )
            for h in range(layer.kv_heads)
        )
        rankings.append(rank)
        curves.append(
            tuple(
                float(v)
                for v in np.mean(
                    np.stack([host[1, h, list(rank[h])] for h in range(layer.kv_heads)]), axis=0
                )
            )
        )
        masses.append(host[2])
    counts = allocate_layer_slots(tuple(curves), tuple(bounds_low), tuple(bounds_high), total)
    positions = tuple(
        tuple(tuple(sorted(row[:count])) for row in rank)
        for rank, count in zip(rankings, counts, strict=True)
    )
    reconstruction = None
    if guard_reconstruction:
        guard_started = time.perf_counter()
        anchor_positions = tuple(
            tuple(tuple(sorted(anchor[(i, h)])) for h in range(layer.kv_heads))
            for i, layer in enumerate(layers)
        )
        # A rollback must preserve the SAME measured physical budget, not quietly
        # select FullKV or use a padded/ragged head layout.
        if (
            any(
                len(row) != uniform_capacity or not set(mandatory[(i, h)]) <= set(row)
                for i, heads in enumerate(anchor_positions)
                for h, row in enumerate(heads)
            )
            or uniform_capacity * sum(costs) != prompt_budget_bytes
        ):
            raise ValueError(
                "reconstruction anchor must satisfy mandatory and exact physical bytes"
            )
        before, proposed = [], []
        for i, layer in enumerate(layers):
            values = layer.value.detach()[0].float()
            mass = torch.as_tensor(masses[i], device=values.device, dtype=torch.float32)
            before.append(
                tuple(
                    float(x) for x in reconstruction_mse(mass, values, anchor_positions[i]).tolist()
                )
            )
            proposed.append(
                tuple(float(x) for x in reconstruction_mse(mass, values, positions[i]).tolist())
            )
        reconstruction = reconstruction_guard(tuple(before), tuple(proposed))
        reconstruction["proposed_layer_capacities"] = counts
        reconstruction["proposed_positions_sha256"] = hashlib.sha256(
            json.dumps(positions, separators=(",", ":")).encode()
        ).hexdigest()
        reconstruction["anchor_positions_sha256"] = hashlib.sha256(
            json.dumps(anchor_positions, separators=(",", ":")).encode()
        ).hexdigest()
        if not reconstruction["accepted"]:
            positions = anchor_positions
            counts = tuple(uniform_capacity for _ in layers)
        reconstruction["host_wall_seconds_including_device_transfers"] = (
            time.perf_counter() - guard_started
        )
    selected_mass = tuple(
        float(np.mean([mass[h, list(row)].sum() for h, row in enumerate(heads)]))
        for mass, heads in zip(masses, positions, strict=True)
    )
    return EvidenceLayerPlan(
        positions,
        counts,
        sum(n * c for n, c in zip(counts, costs, strict=True)),
        tuple(errors),
        selected_mass,
        tuple(mandatory_counts),
        reconstruction,
    )
