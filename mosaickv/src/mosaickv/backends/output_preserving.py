"""HF bridge for the explicitly labeled output-preserving exact-selection variant."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import combinations
from typing import TYPE_CHECKING, Any

from mosaickv.baselines.selectors import select_prompt_attention_topk_ids_by_head
from mosaickv.cache_state import FullKVState, MosaicKVState
from mosaickv.selection.output_preserving import corrective_swaps, prepare_evidence, reconstruct
from mosaickv.selection.types import SelectionBudget
from mosaickv.types import BudgetUnit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mosaickv.config import RunConfig


def topk_anchor_capture(source: FullKVState, attention_layers: Sequence[Any]) -> tuple[Any, ...]:
    """Reproduce the baseline capture reduction, including its source-dtype rounding.

    The objective still uses unpooled attention. Pooling only after promotion to
    FP32 would change the historical FP16 GQA Top-K anchor near tied scores.
    """
    if len(attention_layers) != len(source.layers):
        raise ValueError("attention capture must cover every cache layer")
    result = []
    for layer, attention in zip(source.layers, attention_layers, strict=True):
        if attention.ndim != 4 or attention.shape[0] != 1:
            raise ValueError("Top-K anchor capture requires [1, query_heads, window, positions]")
        batch, heads, window, positions = attention.shape
        if heads % layer.kv_heads or positions != layer.sequence_length:
            raise ValueError("Top-K anchor query/cache geometry is incompatible")
        result.append(
            attention.reshape(
                batch, layer.kv_heads, heads // layer.kv_heads, window, positions
            ).mean(2)
        )
    return tuple(result)


@dataclass(frozen=True)
class OutputPreservingSelection:
    selected_node_ids: tuple[int, ...]
    anchor_node_ids: tuple[int, ...]
    routes: tuple[dict[str, Any], ...]
    target_bytes: int
    selected_bytes: int
    query_evidence: str
    max_swaps_per_head: int
    effective_method: str
    draft_ranking_mode: str = "veto"
    prompt_objective_weight: float = 0.5
    draft_objective_weight: float = 0.5
    draft_steps: int = 0
    draft_cache_isolated: bool = True
    candidate_id: str = "configured_reconstruction"
    short_horizon_logit_reranking: bool = False
    exchange_routes: tuple[tuple[int, int], ...] = ()
    source_scores: tuple[float, ...] = ()
    layer_capacities: tuple[int, ...] = ()
    capacity_transfers: tuple[dict[str, Any], ...] = ()
    physical_padding_bytes: int = 0
    surrogate_score: float = 0.0
    surrogate_rank: int = 0
    surrogate_transactions: tuple[tuple[int, int], ...] = ()
    surrogate_source_block_visits: int = 0
    surrogate_candidate_full_block_scans: int = 0
    directional_prompt_delta_norm: float = 0.0
    directional_draft_delta_norm: float = 0.0
    directional_predicted_gain: float = 0.0
    directional_score: float = 0.0
    block_substitutions: tuple[dict[str, Any], ...] = ()

    def construct(self, source: FullKVState) -> MosaicKVState:
        exact = source.gather_exact_blocks(tuple(source.blocks[i] for i in self.selected_node_ids))
        state = MosaicKVState.create(source, exact=exact)
        if state.statistics.active_kv_bytes != self.selected_bytes:
            raise ValueError("output-preserving exact-tier storage differs from selected bytes")
        return state

    def trace(self) -> dict[str, Any]:
        return {
            "requested_method": "mosaickv_exact",
            "effective_method": self.effective_method,
            "tier_mode": "output_preserving_exact_only_development_variant",
            "tier_reason": "original source K/V only; graph/prototype/residual/repair disabled",
            "selection_provenance": (
                "fixed_shape_within_route_substitution_bounded_logit_verification_v1"
                if self.short_horizon_logit_reranking
                and "__within_route_position_beam" in self.effective_method
                else (
                    "directional_single_transfer_bounded_logit_verification_over_layer_slots_v1"
                    if self.short_horizon_logit_reranking
                    and "__layer_capacity_directional_single_beam" in self.effective_method
                    else (
                        "vectorized_magnitude_ranked_bounded_logit_verification_over_layer_slots_v2"
                        if self.short_horizon_logit_reranking
                        and "__layer_capacity_vectorized_surrogate_beam" in self.effective_method
                        else (
                            "surrogate_ranked_bounded_fullkv_logit_verification_over_layer_slots_v1"
                            if self.short_horizon_logit_reranking
                            and "__layer_capacity_surrogate_beam" in self.effective_method
                            else (
                                "bounded_two_stage_fullkv_logit_reranking_over_complete_layer_slots_v1"
                                if self.short_horizon_logit_reranking
                                and "__layer_capacity_logit_beam" in self.effective_method
                                else (
                                    "bounded_batched_fullkv_logit_reranking_over_route_local_equal_byte_beam_v1"
                                    if self.short_horizon_logit_reranking
                                    and "__route_logit_beam" in self.effective_method
                                    else (
                                        "bounded_fullkv_short_horizon_logit_reranking_over_equal_byte_plans_v1"
                                        if self.short_horizon_logit_reranking
                                        else (
                                            "prompt_topk_anchor_then_joint_normalized_prompt_draft_swaps_v2"
                                            if self.draft_steps
                                            and self.draft_ranking_mode == "joint_normalized"
                                            else (
                                                "prompt_topk_anchor_then_prompt_and_draft_gated_output_swaps_v1"
                                                if self.draft_steps
                                                else (
                                                    "prompt_topk_anchor_then_bounded_"
                                                    "renormalized_output_swaps_v1"
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            "active_budget_value": self.target_bytes,
            "selected_source_bytes": self.selected_bytes,
            "budget_unit": "bytes",
            "selected_blocks": [{"node_id": i} for i in self.selected_node_ids],
            "anchor_selected_blocks": [{"node_id": i} for i in self.anchor_node_ids],
            "prototypes": [],
            "graph_edges": [],
            "repair_events": [],
            "graph_diagnostics": {"status": "disabled_by_explicit_variant"},
            "forecast_statistics": {
                "source": (
                    "prompt_attention_and_isolated_deterministic_draft_attention"
                    if self.draft_steps
                    else "prompt_attention_only"
                ),
                "draft_steps": self.draft_steps,
                "draft_cache_isolated": self.draft_cache_isolated,
                "query_evidence": self.query_evidence,
            },
            "output_reconstruction": {
                "objective": "sum_query_squared_renormalized_attention_output_error",
                "accumulation_dtype": "float32",
                "routing": "descending_initial_reconstruction_error_then_layer_head",
                "maximum_swaps_per_head": self.max_swaps_per_head,
                "initial_error": sum(r["initial_error"] for r in self.routes),
                "final_error": sum(r["final_error"] for r in self.routes),
                "routed_heads": sum(r["routed"] for r in self.routes),
                "accepted_swaps": sum(len(r["swaps"]) for r in self.routes),
                "draft_validation": bool(self.draft_steps),
                "draft_ranking_mode": self.draft_ranking_mode,
                "prompt_objective_weight": self.prompt_objective_weight,
                "draft_objective_weight": self.draft_objective_weight,
                "selected_candidate_id": self.candidate_id,
                "short_horizon_logit_reranking": self.short_horizon_logit_reranking,
                "exchange_routes": [list(route) for route in self.exchange_routes],
                "layer_capacities": list(self.layer_capacities),
                "capacity_transfers": list(self.capacity_transfers),
                "physical_padding_bytes": self.physical_padding_bytes,
                "surrogate_score": self.surrogate_score,
                "surrogate_rank": self.surrogate_rank,
                "surrogate_transactions": [
                    [receiver, donor] for receiver, donor in self.surrogate_transactions
                ],
                "surrogate_source_block_visits": self.surrogate_source_block_visits,
                "surrogate_candidate_full_block_scans": (self.surrogate_candidate_full_block_scans),
                "directional_prompt_delta_norm": self.directional_prompt_delta_norm,
                "directional_draft_delta_norm": self.directional_draft_delta_norm,
                "directional_predicted_gain": self.directional_predicted_gain,
                "directional_score": self.directional_score,
                "block_substitutions": list(self.block_substitutions),
                "draft_initial_error": sum(
                    r["draft_initial_error"]
                    for r in self.routes
                    if r["draft_initial_error"] is not None
                ),
                "draft_final_error": sum(
                    r["draft_final_error"]
                    for r in self.routes
                    if r["draft_final_error"] is not None
                ),
                "rejected_validation_pairs": sum(
                    r["rejected_validation_pairs"] for r in self.routes
                ),
                "rejected_joint_pairs": sum(r["rejected_joint_pairs"] for r in self.routes),
                "routes": list(self.routes),
            },
        }


def _compact_directional_signature(delta: Any, *, bins: int = 8) -> tuple[float, ...]:
    """Compress a signed output perturbation without discarding its direction.

    The fixed contiguous-bin reducer is deterministic, has no learned state,
    and never reads task labels.  It is a label-blind proxy used only to rank
    exact-byte plans before the real short-horizon logit check.
    """

    import torch

    if delta.ndim != 2 or delta.shape[0] < 1 or delta.shape[1] < 1:
        raise ValueError("directional signature requires [queries, value_dim]")
    if bins < 1:
        raise ValueError("directional signature bin count must be positive")
    vector = delta.detach().to(torch.float32).mean(0)
    chunks = torch.tensor_split(vector, min(bins, int(vector.numel())))
    signature = tuple(float(chunk.mean().item()) for chunk in chunks)
    if not signature or not all(math.isfinite(value) for value in signature):
        raise ValueError("directional signature must be finite and nonempty")
    return signature


def _route_transfer_effects(
    attention: Any,
    values: Any,
    selected_positions: Sequence[int],
    mandatory_positions: Sequence[int],
    node_ids: Sequence[int],
    source_scores: Sequence[float],
    *,
    minimum_mass: float,
) -> dict[str, Any]:
    """Predict signed add/remove effects for the next dense layer slot.

    The added and removed positions exactly match the immutable rankings used
    by ``_LayerTransferContext``.  The resulting signatures approximate the
    direction of the downstream logit change; actual logits remain protected
    by the registered KL/token/logit probe.
    """

    chosen = tuple(sorted(selected_positions))
    chosen_set = set(chosen)
    protected = set(mandatory_positions)
    omitted = sorted(
        (position for position in range(len(node_ids)) if position not in chosen_set),
        key=lambda position: (-float(source_scores[node_ids[position]]), node_ids[position]),
    )
    removable = sorted(
        (position for position in chosen if position not in protected),
        key=lambda position: (float(source_scores[node_ids[position]]), node_ids[position]),
    )
    if not omitted or not removable:
        return {
            "available": False,
            "add_gain": 0.0,
            "remove_cost": 0.0,
            "add_delta_signature": (),
            "remove_delta_signature": (),
        }
    initial = reconstruct(attention, values, chosen, minimum_mass=minimum_mass)
    added = reconstruct(
        attention,
        values,
        tuple(sorted((*chosen, omitted[0]))),
        minimum_mass=minimum_mass,
    )
    removed = reconstruct(
        attention,
        values,
        tuple(position for position in chosen if position != removable[0]),
        minimum_mass=minimum_mass,
    )
    return {
        "available": True,
        "add_gain": float(initial.error - added.error),
        "remove_cost": float(removed.error - initial.error),
        "add_delta_signature": _compact_directional_signature(added.output - initial.output),
        "remove_delta_signature": _compact_directional_signature(removed.output - initial.output),
    }


def _single_swap_effect(
    attention: Any,
    values: Any,
    selected_positions: Sequence[int],
    *,
    removed_position: int,
    added_position: int,
    minimum_mass: float,
) -> dict[str, Any]:
    """Measure one fixed-cardinality position substitution without labels."""

    chosen = tuple(sorted(selected_positions))
    chosen_set = set(chosen)
    if removed_position not in chosen_set or added_position in chosen_set:
        return {"available": False}
    after = tuple(sorted((chosen_set - {removed_position}) | {added_position}))
    initial = reconstruct(attention, values, chosen, minimum_mass=minimum_mass)
    final = reconstruct(attention, values, after, minimum_mass=minimum_mass)
    signature = _compact_directional_signature(final.output - initial.output)
    return {
        "available": True,
        "removed_position": removed_position,
        "added_position": added_position,
        "gain": float(initial.error - final.error),
        "initial_error": float(initial.error),
        "final_error": float(final.error),
        "delta_norm": math.sqrt(sum(value * value for value in signature)),
        "delta_signature": signature,
    }


def select_output_preserving(
    source: FullKVState,
    attention_layers: Sequence[Any],
    config: RunConfig,
    *,
    position_limits: Mapping[tuple[int, int], int],
    target_bytes: int,
    prompt_attention_scores: Any,
    draft_attention_steps: Sequence[Sequence[Any]] = (),
    draft_cache_isolated: bool = True,
    candidate_id: str = "configured_reconstruction",
) -> OutputPreservingSelection:
    """Route by initial error; retain every GQA query head and prompt position.

    The common Top-K implementation remains the initial selector. No new
    physical capacity allocation or mandatory policy is introduced here.
    """
    options = config.output_preserving
    if options is None or source.block_size != 1:
        raise ValueError(
            "output-preserving selection requires explicit config and one-token blocks"
        )
    if options.draft_validation != bool(draft_attention_steps):
        raise ValueError(
            "draft-gated selection requires captured draft attention, and prompt-only "
            "selection must not receive it"
        )
    if options.draft_validation and not draft_cache_isolated:
        raise ValueError("draft-gated selection requires an isolated rollout cache")
    routes = tuple(sorted(position_limits))
    expected_routes = tuple(
        (li, h) for li, layer in enumerate(source.layers) for h in range(layer.kv_heads)
    )
    if routes != expected_routes or len(attention_layers) != len(source.layers):
        raise ValueError("route capacities and attention layers must cover the complete cache")
    by_route: dict[tuple[int, int], list[int]] = {route: [] for route in routes}
    for node_id, block in enumerate(source.blocks):
        if block.position_count != 1:
            raise ValueError("output-preserving swaps require one-token source blocks")
        by_route[(block.layer, block.kv_head)].append(node_id)
    anchor = select_prompt_attention_topk_ids_by_head(
        source,
        SelectionBudget(target_bytes, BudgetUnit.BYTES),
        prompt_attention_by_node={},
        prompt_attention_scores=prompt_attention_scores,
        head_routes=routes,
        position_limits=position_limits,
    )
    selected_ids = set(anchor.selected_node_ids)
    positions: dict[tuple[int, int], tuple[int, ...]] = {}
    mandatory: dict[tuple[int, int], tuple[int, ...]] = {}
    telemetry: dict[tuple[int, int], dict[str, Any]] = {}

    def evidence(route: tuple[int, int]) -> tuple[Any, Any, int, int]:
        li, head = route
        layer = source.layers[li]
        attention = attention_layers[li]
        if (
            attention.ndim != 4
            or attention.shape[0] != 1
            or attention.shape[1] % layer.kv_heads
            or attention.shape[-1] != layer.sequence_length
        ):
            raise ValueError("captured attention must be unpooled [1, query_heads, window, cache]")
        group = int(attention.shape[1]) // layer.kv_heads
        window = min(config.forecasting.prompt_window, int(attention.shape[-2]))
        raw = attention[0, head * group : (head + 1) * group, -window:, :]
        value = layer.value.movedim(
            (layer.value_head_dimension, layer.value_sequence_dimension), (0, 1)
        )[head]
        # Single-request HF layout; remaining singleton batch axis is not a feature.
        if value.numel() != layer.sequence_length * value.shape[-1]:
            raise ValueError("output-preserving runtime supports batch size one only")
        a, v = prepare_evidence(
            raw.reshape(-1, layer.sequence_length),
            value.reshape(layer.sequence_length, -1),
            pooled=options.query_evidence == "pooled",
        )
        return a, v, group, window

    def draft_evidence(route: tuple[int, int], values: Any) -> Any:
        import torch

        li, head = route
        layer = source.layers[li]
        rows = []
        for step in draft_attention_steps:
            if len(step) != len(source.layers):
                raise ValueError("draft attention step must cover every cache layer")
            attention = step[li]
            if (
                attention.ndim != 4
                or attention.shape[0] != 1
                or attention.shape[-2] != 1
                or attention.shape[1] % layer.kv_heads
                or attention.shape[-1] < layer.sequence_length
            ):
                raise ValueError("draft attention must be [1, query_heads, 1, growing_cache]")
            group = int(attention.shape[1]) // layer.kv_heads
            rows.append(
                attention[
                    0,
                    head * group : (head + 1) * group,
                    0,
                    : layer.sequence_length,
                ]
            )
        if not rows:
            raise ValueError("draft-gated selection has no draft attention rows")
        prepared, _ = prepare_evidence(
            torch.cat(rows, dim=0),
            values,
            pooled=False,
        )
        return prepared

    for route in routes:
        node_ids = sorted(by_route[route], key=lambda i: source.blocks[i].physical_cache_indices[0])
        by_route[route] = node_ids
        if [source.blocks[i].physical_cache_indices[0] for i in node_ids] != list(
            range(len(node_ids))
        ):
            raise ValueError("source physical positions must be complete and unique per route")
        if len({source.blocks[i].byte_size for i in node_ids}) != 1:
            raise ValueError("equal-byte swaps require homogeneous source-position storage")
        chosen = tuple(p for p, i in enumerate(node_ids) if i in selected_ids)
        if len(chosen) != min(position_limits[route], len(node_ids)):
            raise ValueError("Top-K anchor did not attain the complete physical route capacity")
        positions[route] = chosen
        mandatory[route] = tuple(p for p, i in enumerate(node_ids) if source.blocks[i].mandatory)
        a, v, group, window = evidence(route)
        initial = reconstruct(a, v, chosen, minimum_mass=options.minimum_query_mass)
        prompt_transfer = _route_transfer_effects(
            a,
            v,
            chosen,
            mandatory[route],
            node_ids,
            prompt_attention_scores,
            minimum_mass=options.minimum_query_mass,
        )
        telemetry[route] = {
            "layer": route[0],
            "kv_head": route[1],
            "query_heads": group,
            "prompt_positions": window,
            "objective_query_rows": int(a.shape[0]),
            "capacity": len(chosen),
            "initial_error": initial.error,
            "final_error": initial.error,
            "routed": False,
            "swaps": [],
            "evaluated_pairs": 0,
            "rejected_zero_mass_pairs": 0,
            "draft_initial_error": None,
            "draft_final_error": None,
            "rejected_validation_pairs": 0,
            "rejected_joint_pairs": 0,
            "draft_ranking_mode": options.draft_ranking_mode,
            "prompt_transfer_effect": prompt_transfer,
            "draft_transfer_effect": None,
        }
    ranked = sorted(routes, key=lambda r: (-telemetry[r]["initial_error"], r))
    for route in ranked[: options.max_routed_heads]:
        if (
            telemetry[route]["initial_error"] <= options.minimum_error
            or options.max_swaps_per_head == 0
        ):
            continue
        a, v, _group, _window = evidence(route)
        validation = draft_evidence(route, v) if options.draft_validation else None
        draft_transfer = (
            None
            if validation is None
            else _route_transfer_effects(
                validation,
                v,
                positions[route],
                mandatory[route],
                by_route[route],
                prompt_attention_scores,
                minimum_mass=options.minimum_query_mass,
            )
        )
        result = corrective_swaps(
            a,
            v,
            positions[route],
            mandatory[route],
            options,
            validation_attention=validation,
        )
        swap_effects: list[dict[str, Any]] = []
        for swap in result.swaps:
            removed_position = int(swap["removed_position"])
            added_position = int(swap["added_position"])
            prompt_effect = _single_swap_effect(
                a,
                v,
                positions[route],
                removed_position=removed_position,
                added_position=added_position,
                minimum_mass=options.minimum_query_mass,
            )
            draft_effect = (
                None
                if validation is None
                else _single_swap_effect(
                    validation,
                    v,
                    positions[route],
                    removed_position=removed_position,
                    added_position=added_position,
                    minimum_mass=options.minimum_query_mass,
                )
            )
            if prompt_effect.get("available") is True and (
                draft_effect is None or draft_effect.get("available") is True
            ):
                swap_effects.append(
                    {
                        "removed_position": removed_position,
                        "added_position": added_position,
                        "swap_record": dict(swap),
                        "prompt": prompt_effect,
                        "draft": draft_effect,
                    }
                )
        telemetry[route].update(
            final_error=result.final_error,
            routed=True,
            swaps=list(result.swaps),
            evaluated_pairs=result.evaluated_pairs,
            rejected_zero_mass_pairs=result.rejected_zero_mass_pairs,
            draft_initial_error=result.validation_initial_error,
            draft_final_error=result.validation_final_error,
            rejected_validation_pairs=result.rejected_validation_pairs,
            rejected_joint_pairs=result.rejected_joint_pairs,
            draft_ranking_mode=result.draft_ranking_mode,
            draft_transfer_effect=draft_transfer,
            within_route_swap_effects=swap_effects,
        )
        positions[route] = result.positions
    final_ids = tuple(sorted(by_route[r][p] for r in routes for p in positions[r]))
    active_bytes = sum(source.blocks[i].byte_size for i in final_ids)
    anchor_bytes = sum(source.blocks[i].byte_size for i in anchor.selected_node_ids)
    if active_bytes != anchor_bytes or active_bytes > target_bytes:
        raise ValueError("output-preserving swaps changed physical capacity or exceeded the cap")
    return OutputPreservingSelection(
        selected_node_ids=final_ids,
        anchor_node_ids=anchor.selected_node_ids,
        routes=tuple(telemetry[r] for r in routes),
        target_bytes=target_bytes,
        selected_bytes=active_bytes,
        query_evidence=options.query_evidence,
        max_swaps_per_head=options.max_swaps_per_head,
        effective_method=options.effective_method,
        draft_ranking_mode=options.draft_ranking_mode,
        prompt_objective_weight=options.prompt_objective_weight,
        draft_objective_weight=options.draft_objective_weight,
        draft_steps=len(draft_attention_steps),
        draft_cache_isolated=draft_cache_isolated,
        candidate_id=candidate_id,
        short_horizon_logit_reranking=options.short_horizon_logit_reranking,
        source_scores=tuple(float(value) for value in prompt_attention_scores),
    )


def compose_layer_capacity_transfer_candidate(
    source: FullKVState,
    anchor: OutputPreservingSelection,
    *,
    receiver_layer: int,
    donor_layer: int,
    transfer_slots: int,
    candidate_id: str,
) -> OutputPreservingSelection:
    """Move complete dense layer slots at an exactly conserved byte boundary.

    A slot is added or removed for every KV head in the affected layer.  This
    preserves a dense tensor inside each layer while allowing independent
    layer lengths.  Per-head ragged lengths are rejected because they would
    allocate padding and therefore would not be a physical-capacity transfer.
    """

    if not candidate_id or receiver_layer == donor_layer or transfer_slots < 1:
        raise ValueError("layer-capacity transfer needs distinct layers, slots, and an ID")
    if not anchor.source_scores or len(anchor.source_scores) != len(source.blocks):
        raise ValueError("layer-capacity transfer requires immutable source scores")
    layer_count = len(source.layers)
    if not 0 <= receiver_layer < layer_count or not 0 <= donor_layer < layer_count:
        raise ValueError("layer-capacity transfer layer is out of range")

    routes: dict[tuple[int, int], list[int]] = {}
    selected: dict[tuple[int, int], set[int]] = {}
    for node_id, block in enumerate(source.blocks):
        route = (block.layer, block.kv_head)
        routes.setdefault(route, []).append(node_id)
    for route, node_ids in routes.items():
        node_ids.sort(key=lambda node_id: source.blocks[node_id].physical_cache_indices)
        selected[route] = {node_id for node_id in anchor.selected_node_ids if node_id in node_ids}
        if not selected[route]:
            raise ValueError(f"anchor removed every block from route {route}")

    layer_capacities: list[int] = []
    for layer_index, layer in enumerate(source.layers):
        counts = {len(selected[(layer_index, kv_head)]) for kv_head in range(layer.kv_heads)}
        if len(counts) != 1:
            raise ValueError("anchor has per-head ragged capacity inside a layer")
        layer_capacities.append(next(iter(counts)))

    added: list[int] = []
    removed: list[int] = []
    receiver = source.layers[receiver_layer]
    donor = source.layers[donor_layer]
    if receiver.kv_heads != donor.kv_heads:
        raise ValueError("layer-capacity transfer requires matching KV-head counts")
    for kv_head in range(receiver.kv_heads):
        receiver_route = (receiver_layer, kv_head)
        donor_route = (donor_layer, kv_head)
        receiver_selected = selected[receiver_route]
        donor_selected = selected[donor_route]
        omitted = [
            node_id for node_id in routes[receiver_route] if node_id not in receiver_selected
        ]
        removable = [node_id for node_id in donor_selected if not source.blocks[node_id].mandatory]
        omitted.sort(key=lambda node_id: (-anchor.source_scores[node_id], node_id))
        removable.sort(key=lambda node_id: (anchor.source_scores[node_id], node_id))
        if len(omitted) < transfer_slots or len(removable) < transfer_slots:
            raise ValueError("layer-capacity transfer has insufficient receiver/donor blocks")
        receiver_add = omitted[:transfer_slots]
        donor_remove = removable[:transfer_slots]
        receiver_selected.update(receiver_add)
        donor_selected.difference_update(donor_remove)
        added.extend(receiver_add)
        removed.extend(donor_remove)

    added_bytes = sum(source.blocks[node_id].byte_size for node_id in added)
    removed_bytes = sum(source.blocks[node_id].byte_size for node_id in removed)
    if added_bytes != removed_bytes:
        raise ValueError("layer-capacity transfer did not conserve physical bytes")
    layer_capacities[receiver_layer] += transfer_slots
    layer_capacities[donor_layer] -= transfer_slots
    if layer_capacities[donor_layer] < 1:
        raise ValueError("layer-capacity transfer emptied the donor layer")

    final_ids = tuple(sorted(node_id for values in selected.values() for node_id in values))
    mandatory = {node_id for node_id, block in enumerate(source.blocks) if block.mandatory}
    if not mandatory.issubset(final_ids):
        raise ValueError("layer-capacity transfer removed a mandatory block")
    selected_bytes = sum(source.blocks[node_id].byte_size for node_id in final_ids)
    if selected_bytes != anchor.selected_bytes or selected_bytes > anchor.target_bytes:
        raise ValueError("layer-capacity transfer changed bytes or exceeded the hard cap")
    for layer_index, layer in enumerate(source.layers):
        if {len(selected[(layer_index, kv_head)]) for kv_head in range(layer.kv_heads)} != {
            layer_capacities[layer_index]
        }:
            raise ValueError("layer-capacity transfer produced per-head padding")

    receiver_routes = tuple((receiver_layer, head) for head in range(receiver.kv_heads))
    donor_routes = tuple((donor_layer, head) for head in range(donor.kv_heads))
    transfer = {
        "receiver_layer": receiver_layer,
        "donor_layer": donor_layer,
        "transfer_slots": transfer_slots,
        "added_node_ids": added,
        "removed_node_ids": removed,
        "added_bytes": added_bytes,
        "removed_bytes": removed_bytes,
        "padding_bytes": 0,
    }
    return replace(
        anchor,
        selected_node_ids=final_ids,
        selected_bytes=selected_bytes,
        candidate_id=candidate_id,
        exchange_routes=tuple(sorted((*receiver_routes, *donor_routes))),
        layer_capacities=tuple(layer_capacities),
        capacity_transfers=(*anchor.capacity_transfers, transfer),
        physical_padding_bytes=0,
    )


def layer_capacity_transfer_candidates(
    source: FullKVState,
    anchor: OutputPreservingSelection,
    route_contributions: Sequence[tuple[int, int, float]],
    *,
    limit: int,
) -> tuple[OutputPreservingSelection, ...]:
    """Build deterministic, distinct layer-capacity plans from logit impact."""

    if not 1 <= limit <= 15:
        raise ValueError("layer-capacity candidate limit must be in [1, 15]")
    impact = {layer: 0.0 for layer in range(len(source.layers))}
    observed = {layer: 0 for layer in range(len(source.layers))}
    for layer, _head, contribution in route_contributions:
        impact[layer] += float(contribution)
        observed[layer] += 1
    for layer in impact:
        if observed[layer]:
            impact[layer] /= observed[layer]
    receivers = sorted(impact, key=lambda layer: (-impact[layer], layer))
    donors = sorted(impact, key=lambda layer: (impact[layer], layer))
    candidates: list[OutputPreservingSelection] = []
    seen: set[tuple[int, ...]] = set()
    # One- and two-slot transactions are deliberately interleaved so the
    # shortlist spans several meaningful physical allocations.
    for transfer_slots in (1, 2):
        for receiver_layer in receivers:
            for donor_layer in donors:
                if receiver_layer == donor_layer:
                    continue
                try:
                    candidate = compose_layer_capacity_transfer_candidate(
                        source,
                        anchor,
                        receiver_layer=receiver_layer,
                        donor_layer=donor_layer,
                        transfer_slots=transfer_slots,
                        candidate_id=(
                            f"layer_restore_l{receiver_layer:02d}_"
                            f"donor_l{donor_layer:02d}_s{transfer_slots}"
                        ),
                    )
                except ValueError:
                    continue
                if candidate.selected_node_ids in seen:
                    continue
                candidates.append(candidate)
                seen.add(candidate.selected_node_ids)
                if len(candidates) == limit:
                    return tuple(candidates)
    return tuple(candidates)


def prompt_draft_route_risk(
    proposal: OutputPreservingSelection,
) -> tuple[tuple[int, int, float], ...]:
    """Return magnitude-preserving, label-blind residual risk per route.

    Prompt and draft residual errors are normalized across routes, not divided
    by each route's own initial error.  Consequently, untouched routes retain
    their relative error magnitudes instead of all collapsing to one.  The
    function never inspects generated answers, references, benchmark scores,
    or FullKV logits.
    """

    raw: list[tuple[int, int, float, float]] = []
    for record in proposal.routes:
        draft_final_value = record.get("draft_final_error")
        prompt_residual = max(float(record["final_error"]), 0.0)
        draft_residual = (
            prompt_residual if draft_final_value is None else max(float(draft_final_value), 0.0)
        )
        raw.append(
            (
                int(record["layer"]),
                int(record["kv_head"]),
                prompt_residual,
                draft_residual,
            )
        )
    prompt_total = sum(item[2] for item in raw)
    draft_total = sum(item[3] for item in raw)
    if prompt_total <= 1e-12 or draft_total <= 1e-12:
        raise ValueError("surrogate route risk requires positive residual evidence")
    risks: list[tuple[int, int, float]] = []
    for layer, head, prompt_residual, draft_residual in raw:
        risk = (
            proposal.prompt_objective_weight * prompt_residual / prompt_total
            + proposal.draft_objective_weight * draft_residual / draft_total
        )
        risks.append((layer, head, float(risk)))
    risks.sort(key=lambda item: (item[0], item[1]))
    return tuple(risks)


@dataclass(frozen=True)
class _LayerTransferContext:
    route_nodes: Mapping[tuple[int, int], tuple[int, ...]]
    selected_node_ids: frozenset[int]
    omitted_by_route: Mapping[tuple[int, int], tuple[int, ...]]
    removable_by_route: Mapping[tuple[int, int], tuple[int, ...]]
    layer_capacities: tuple[int, ...]
    mandatory_node_ids: frozenset[int]
    source_block_visits: int


def _build_layer_transfer_context(
    source: FullKVState,
    anchor: OutputPreservingSelection,
) -> _LayerTransferContext:
    """Index one source cache once for all bounded transfer candidates."""

    if not anchor.source_scores or len(anchor.source_scores) != len(source.blocks):
        raise ValueError("layer-capacity transfer requires immutable source scores")
    selected_ids = frozenset(anchor.selected_node_ids)
    route_lists: dict[tuple[int, int], list[int]] = {
        (layer_index, head): []
        for layer_index, layer in enumerate(source.layers)
        for head in range(layer.kv_heads)
    }
    mandatory = set()
    visits = 0
    for node_id, block in enumerate(source.blocks):
        visits += 1
        route_lists[(block.layer, block.kv_head)].append(node_id)
        if block.mandatory:
            mandatory.add(node_id)
    route_nodes: dict[tuple[int, int], tuple[int, ...]] = {}
    omitted: dict[tuple[int, int], tuple[int, ...]] = {}
    removable: dict[tuple[int, int], tuple[int, ...]] = {}
    capacities: list[int] = []
    for layer_index, layer in enumerate(source.layers):
        layer_counts = set()
        for head in range(layer.kv_heads):
            route = (layer_index, head)
            ordered = tuple(
                sorted(
                    route_lists[route],
                    key=lambda node_id: source.blocks[node_id].physical_cache_indices,
                )
            )
            chosen = tuple(node_id for node_id in ordered if node_id in selected_ids)
            if not chosen:
                raise ValueError(f"anchor removed every block from route {route}")
            layer_counts.add(len(chosen))
            route_nodes[route] = ordered
            omitted[route] = tuple(
                sorted(
                    (node_id for node_id in ordered if node_id not in selected_ids),
                    key=lambda node_id: (-anchor.source_scores[node_id], node_id),
                )
            )
            removable[route] = tuple(
                sorted(
                    (node_id for node_id in chosen if not source.blocks[node_id].mandatory),
                    key=lambda node_id: (anchor.source_scores[node_id], node_id),
                )
            )
        if len(layer_counts) != 1:
            raise ValueError("anchor has per-head ragged capacity inside a layer")
        capacities.append(next(iter(layer_counts)))
    if sum(source.blocks[node_id].byte_size for node_id in selected_ids) != anchor.selected_bytes:
        raise ValueError("anchor selected-byte accounting is inconsistent")
    return _LayerTransferContext(
        route_nodes=route_nodes,
        selected_node_ids=selected_ids,
        omitted_by_route=omitted,
        removable_by_route=removable,
        layer_capacities=tuple(capacities),
        mandatory_node_ids=frozenset(mandatory),
        source_block_visits=visits,
    )


def _apply_layer_transfer_transactions(
    source: FullKVState,
    anchor: OutputPreservingSelection,
    context: _LayerTransferContext,
    transaction_set: tuple[tuple[int, int], ...],
    *,
    candidate_id: str,
) -> OutputPreservingSelection:
    """Apply one multi-layer transaction without rescanning source blocks."""

    if not transaction_set or not candidate_id:
        raise ValueError("vectorized transfer requires transactions and an ID")
    receiver_counts = Counter(receiver for receiver, _donor in transaction_set)
    donor_counts = Counter(donor for _receiver, donor in transaction_set)
    if set(receiver_counts) & set(donor_counts):
        raise ValueError("a layer cannot receive and donate in one atomic plan")
    capacities = list(context.layer_capacities)
    for layer, count in receiver_counts.items():
        capacities[layer] += count
        if capacities[layer] > source.layers[layer].sequence_length:
            raise ValueError("receiver layer exceeds its physical sequence length")
    for layer, count in donor_counts.items():
        capacities[layer] -= count
        if capacities[layer] < 1:
            raise ValueError("layer-capacity transfer emptied the donor layer")

    receiver_offsets: Counter[int] = Counter()
    donor_offsets: Counter[int] = Counter()
    added: set[int] = set()
    removed: set[int] = set()
    transfers: list[dict[str, Any]] = []
    exchanged_routes: set[tuple[int, int]] = set()
    for receiver_layer, donor_layer in transaction_set:
        receiver = source.layers[receiver_layer]
        donor = source.layers[donor_layer]
        if receiver.kv_heads != donor.kv_heads:
            raise ValueError("layer-capacity transfer requires matching KV-head counts")
        transaction_added: list[int] = []
        transaction_removed: list[int] = []
        for head in range(receiver.kv_heads):
            receiver_route = (receiver_layer, head)
            donor_route = (donor_layer, head)
            receiver_rank = receiver_offsets[receiver_layer]
            donor_rank = donor_offsets[donor_layer]
            try:
                added_node = context.omitted_by_route[receiver_route][receiver_rank]
                removed_node = context.removable_by_route[donor_route][donor_rank]
            except IndexError as error:
                raise ValueError(
                    "layer-capacity transfer has insufficient receiver/donor blocks"
                ) from error
            transaction_added.append(added_node)
            transaction_removed.append(removed_node)
            exchanged_routes.update((receiver_route, donor_route))
        receiver_offsets[receiver_layer] += 1
        donor_offsets[donor_layer] += 1
        if added.intersection(transaction_added) or removed.intersection(transaction_removed):
            raise ValueError("atomic layer transfer reused a source block")
        added.update(transaction_added)
        removed.update(transaction_removed)
        added_bytes = sum(source.blocks[node_id].byte_size for node_id in transaction_added)
        removed_bytes = sum(source.blocks[node_id].byte_size for node_id in transaction_removed)
        if added_bytes != removed_bytes:
            raise ValueError("layer-capacity transfer did not conserve physical bytes")
        transfers.append(
            {
                "receiver_layer": receiver_layer,
                "donor_layer": donor_layer,
                "transfer_slots": 1,
                "added_node_ids": transaction_added,
                "removed_node_ids": transaction_removed,
                "added_bytes": added_bytes,
                "removed_bytes": removed_bytes,
                "padding_bytes": 0,
            }
        )
    if added & context.selected_node_ids or not removed <= context.selected_node_ids:
        raise ValueError("atomic layer transfer violates source membership")
    final_set = (context.selected_node_ids - removed) | added
    if not context.mandatory_node_ids <= final_set:
        raise ValueError("layer-capacity transfer removed a mandatory block")
    selected_bytes = (
        anchor.selected_bytes
        + sum(source.blocks[node_id].byte_size for node_id in added)
        - sum(source.blocks[node_id].byte_size for node_id in removed)
    )
    if selected_bytes != anchor.selected_bytes or selected_bytes > anchor.target_bytes:
        raise ValueError("layer-capacity transfer changed bytes or exceeded the hard cap")
    return replace(
        anchor,
        selected_node_ids=tuple(sorted(final_set)),
        selected_bytes=selected_bytes,
        candidate_id=candidate_id,
        exchange_routes=tuple(sorted(exchanged_routes)),
        layer_capacities=tuple(capacities),
        capacity_transfers=(*anchor.capacity_transfers, *transfers),
        physical_padding_bytes=0,
        surrogate_source_block_visits=context.source_block_visits,
        surrogate_candidate_full_block_scans=0,
    )


def layer_capacity_surrogate_candidates(
    source: FullKVState,
    anchor: OutputPreservingSelection,
    proposal: OutputPreservingSelection,
    *,
    limit: int,
    maximum_width: int,
) -> tuple[OutputPreservingSelection, ...]:
    """Build 8--15 exact-byte multi-layer plans without model probing.

    Complete layer slots are transferred from low-risk donor layers to
    high-risk receiver layers.  Candidate construction is bounded: the six
    best primitive transactions define width-one, width-two, and width-three
    specifications, which are interleaved to retain diversity.  Only the
    returned shortlist is intended for subsequent model-logit verification.
    """

    if not 8 <= limit <= 15:
        raise ValueError("surrogate layer-capacity limit must be in [8, 15]")
    if not 1 <= maximum_width <= 3:
        raise ValueError("surrogate layer-capacity width must be in [1, 3]")
    if anchor.selected_bytes != proposal.selected_bytes or (
        anchor.target_bytes != proposal.target_bytes
    ):
        raise ValueError("surrogate plans require one exact byte boundary")

    route_risk = prompt_draft_route_risk(proposal)
    layer_sum = {layer: 0.0 for layer in range(len(source.layers))}
    layer_count = {layer: 0 for layer in range(len(source.layers))}
    for layer, _head, risk in route_risk:
        layer_sum[layer] += risk
        layer_count[layer] += 1
    layer_risk = {layer: layer_sum[layer] / max(layer_count[layer], 1) for layer in layer_sum}
    transactions = sorted(
        (
            (layer_risk[receiver] - layer_risk[donor], receiver, donor)
            for receiver in layer_risk
            for donor in layer_risk
            if receiver != donor and layer_risk[receiver] > layer_risk[donor]
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )[:6]
    if len(transactions) < 3:
        raise ValueError("surrogate layer-capacity planning lacks distinct risk levels")

    specifications: dict[int, list[tuple[float, tuple[tuple[int, int], ...]]]] = {
        width: [] for width in range(1, maximum_width + 1)
    }
    primitives = tuple((receiver, donor) for _gain, receiver, donor in transactions)
    gains = {(receiver, donor): gain for gain, receiver, donor in transactions}
    for width in specifications:
        for transaction_set in combinations(primitives, width):
            receivers = {receiver for receiver, _donor in transaction_set}
            donors = {donor for _receiver, donor in transaction_set}
            if receivers & donors:
                continue
            score = sum(gains[transaction] for transaction in transaction_set)
            specifications[width].append((score, transaction_set))
        specifications[width].sort(key=lambda item: (-item[0], item[1]))

    ordered_specs: list[tuple[float, tuple[tuple[int, int], ...]]] = []
    depth = 0
    while any(depth < len(specifications[width]) for width in specifications):
        for width in range(1, maximum_width + 1):
            if depth < len(specifications[width]):
                ordered_specs.append(specifications[width][depth])
        depth += 1

    context = _build_layer_transfer_context(source, anchor)
    candidates: list[OutputPreservingSelection] = []
    seen: set[tuple[int, ...]] = set()
    for score, transaction_set in ordered_specs:
        try:
            candidate = _apply_layer_transfer_transactions(
                source,
                anchor,
                context,
                transaction_set,
                candidate_id=(
                    "surrogate_"
                    + "_".join(
                        f"r{receiver:02d}d{donor:02d}" for receiver, donor in transaction_set
                    )
                ),
            )
        except ValueError:
            continue
        if candidate.selected_node_ids in seen:
            continue
        seen.add(candidate.selected_node_ids)
        candidates.append(
            replace(
                candidate,
                surrogate_score=float(score),
                surrogate_transactions=transaction_set,
            )
        )
        if len(candidates) == limit:
            break
    if len(candidates) < 8:
        raise ValueError("surrogate planning did not produce 8 distinct feasible plans")
    candidates.sort(
        key=lambda candidate: (
            -candidate.surrogate_score,
            candidate.surrogate_transactions,
            candidate.candidate_id,
        )
    )
    return tuple(
        replace(candidate, surrogate_rank=rank)
        for rank, candidate in enumerate(candidates, start=1)
    )


def layer_capacity_directional_single_candidates(
    source: FullKVState,
    anchor: OutputPreservingSelection,
    proposal: OutputPreservingSelection,
    *,
    limit: int,
) -> tuple[OutputPreservingSelection, ...]:
    """Rank deterministic one-slot layer exchanges by signed output effects.

    Each proposal contains exactly one receiver/donor transaction.  Prompt and
    draft add/remove perturbations are combined before their norm is measured,
    so opposing effects are preferred over equally large reinforcing effects.
    The compact signed perturbation is a label-blind logit proxy; the returned
    top two plans still require the unchanged real-logit verification.
    """

    if not 8 <= limit <= 15:
        raise ValueError("directional layer-capacity limit must be in [8, 15]")
    if anchor.selected_bytes != proposal.selected_bytes or (
        anchor.target_bytes != proposal.target_bytes
    ):
        raise ValueError("directional plans require one exact byte boundary")

    route_risk = prompt_draft_route_risk(proposal)
    layer_risk = {layer: 0.0 for layer in range(len(source.layers))}
    layer_risk_count = {layer: 0 for layer in range(len(source.layers))}
    records = {(int(record["layer"]), int(record["kv_head"])): record for record in proposal.routes}
    for layer, head, risk in route_risk:
        layer_risk[layer] += risk
        layer_risk_count[layer] += 1
        if (layer, head) not in records:
            raise ValueError("directional telemetry does not cover every route")
    for layer in layer_risk:
        layer_risk[layer] /= max(layer_risk_count[layer], 1)

    def effect(record: Mapping[str, Any], evidence: str) -> Mapping[str, Any]:
        value = record.get(f"{evidence}_transfer_effect")
        if evidence == "draft" and not isinstance(value, Mapping):
            value = record.get("prompt_transfer_effect")
        if not isinstance(value, Mapping) or value.get("available") is not True:
            raise ValueError("directional transfer telemetry is unavailable")
        return value

    def layer_effect(layer: int, evidence: str, operation: str) -> tuple[float, tuple[float, ...]]:
        scalar_name = "add_gain" if operation == "add" else "remove_cost"
        signature_name = f"{operation}_delta_signature"
        scalar = 0.0
        signature: list[float] | None = None
        for head in range(source.layers[layer].kv_heads):
            item = effect(records[(layer, head)], evidence)
            values = tuple(float(value) for value in item[signature_name])
            if not values:
                raise ValueError("directional signature is empty")
            if signature is None:
                signature = [0.0] * len(values)
            if len(values) != len(signature):
                raise ValueError("directional signatures have inconsistent dimensions")
            scalar += float(item[scalar_name])
            for index, value in enumerate(values):
                signature[index] += value
        assert signature is not None
        return scalar, tuple(signature)

    layer_effects = {
        (layer, evidence, operation): layer_effect(layer, evidence, operation)
        for layer in range(len(source.layers))
        for evidence in ("prompt", "draft")
        for operation in ("add", "remove")
    }

    raw: list[dict[str, Any]] = []
    for receiver in range(len(source.layers)):
        for donor in range(len(source.layers)):
            risk_gain = layer_risk[receiver] - layer_risk[donor]
            if receiver == donor or risk_gain <= 0:
                continue
            prompt_add_gain, prompt_add = layer_effects[(receiver, "prompt", "add")]
            prompt_remove_cost, prompt_remove = layer_effects[(donor, "prompt", "remove")]
            draft_add_gain, draft_add = layer_effects[(receiver, "draft", "add")]
            draft_remove_cost, draft_remove = layer_effects[(donor, "draft", "remove")]
            prompt_delta = tuple(
                add + remove for add, remove in zip(prompt_add, prompt_remove, strict=True)
            )
            draft_delta = tuple(
                add + remove for add, remove in zip(draft_add, draft_remove, strict=True)
            )
            raw.append(
                {
                    "receiver": receiver,
                    "donor": donor,
                    "risk_gain": risk_gain,
                    "prompt_gain": prompt_add_gain - prompt_remove_cost,
                    "draft_gain": draft_add_gain - draft_remove_cost,
                    "prompt_norm": math.sqrt(sum(value * value for value in prompt_delta)),
                    "draft_norm": math.sqrt(sum(value * value for value in draft_delta)),
                }
            )
    if len(raw) < limit:
        raise ValueError("directional planning lacks enough distinct layer pairs")

    def positive_scale(field: str) -> float:
        return max(max(abs(float(item[field])) for item in raw), 1e-30)

    prompt_gain_scale = positive_scale("prompt_gain")
    draft_gain_scale = positive_scale("draft_gain")
    prompt_norm_scale = positive_scale("prompt_norm")
    draft_norm_scale = positive_scale("draft_norm")
    risk_scale = positive_scale("risk_gain")
    for item in raw:
        predicted_gain = (
            proposal.prompt_objective_weight * float(item["prompt_gain"]) / prompt_gain_scale
            + proposal.draft_objective_weight * float(item["draft_gain"]) / draft_gain_scale
        )
        directional_cost = (
            proposal.prompt_objective_weight * float(item["prompt_norm"]) / prompt_norm_scale
            + proposal.draft_objective_weight * float(item["draft_norm"]) / draft_norm_scale
        )
        item["predicted_gain"] = predicted_gain
        item["score"] = predicted_gain + float(item["risk_gain"]) / risk_scale - directional_cost
    raw.sort(
        key=lambda item: (
            -float(item["score"]),
            float(item["prompt_norm"]) + float(item["draft_norm"]),
            -float(item["risk_gain"]),
            int(item["receiver"]),
            int(item["donor"]),
        )
    )

    context = _build_layer_transfer_context(source, anchor)
    candidates: list[OutputPreservingSelection] = []
    seen: set[tuple[int, ...]] = set()
    for item in raw:
        receiver, donor = int(item["receiver"]), int(item["donor"])
        try:
            candidate = _apply_layer_transfer_transactions(
                source,
                anchor,
                context,
                ((receiver, donor),),
                candidate_id=f"directional_r{receiver:02d}d{donor:02d}",
            )
        except ValueError:
            continue
        if candidate.selected_node_ids in seen:
            continue
        seen.add(candidate.selected_node_ids)
        candidates.append(
            replace(
                candidate,
                surrogate_score=float(item["score"]),
                surrogate_transactions=((receiver, donor),),
                directional_prompt_delta_norm=float(item["prompt_norm"]),
                directional_draft_delta_norm=float(item["draft_norm"]),
                directional_predicted_gain=float(item["predicted_gain"]),
                directional_score=float(item["score"]),
            )
        )
        if len(candidates) == limit:
            break
    if len(candidates) != limit:
        raise ValueError("directional planning did not produce the registered plan count")
    return tuple(
        replace(candidate, surrogate_rank=rank)
        for rank, candidate in enumerate(candidates, start=1)
    )


def within_route_position_candidates(
    source: FullKVState,
    anchor: OutputPreservingSelection,
    proposal: OutputPreservingSelection,
    *,
    limit: int,
) -> tuple[OutputPreservingSelection, ...]:
    """Build deterministic one-block substitutions with fixed route lengths.

    Every candidate removes one nonmandatory anchor block and restores one
    omitted block in the *same* layer/KV-head route.  Consequently the dense
    HF tensor shape, physical byte count, and every route capacity remain
    unchanged.  Prompt/draft reconstruction telemetry ranks candidates before
    the unchanged real-logit verification; benchmark labels are never read.
    """

    if not 8 <= limit <= 16:
        raise ValueError("within-route position candidate limit must be in [8, 16]")
    if anchor.selected_bytes != proposal.selected_bytes or (
        anchor.target_bytes != proposal.target_bytes
    ):
        raise ValueError("within-route substitutions require one exact byte boundary")
    if anchor.anchor_node_ids != proposal.anchor_node_ids:
        raise ValueError("within-route substitutions require one immutable anchor")

    route_nodes: dict[tuple[int, int], list[int]] = {}
    for node_id, block in enumerate(source.blocks):
        route_nodes.setdefault((block.layer, block.kv_head), []).append(node_id)
    for node_ids in route_nodes.values():
        node_ids.sort(key=lambda node_id: source.blocks[node_id].physical_cache_indices)
    anchor_selected = frozenset(anchor.selected_node_ids)
    mandatory = frozenset(node_id for node_id, block in enumerate(source.blocks) if block.mandatory)
    anchor_records = {
        (int(record["layer"]), int(record["kv_head"])): record for record in anchor.routes
    }
    proposal_records = {
        (int(record["layer"]), int(record["kv_head"])): record for record in proposal.routes
    }
    if set(anchor_records) != set(route_nodes) or set(proposal_records) != set(route_nodes):
        raise ValueError("within-route telemetry does not cover every cache route")

    raw: list[dict[str, Any]] = []
    for route in sorted(route_nodes):
        record = proposal_records[route]
        for effect in record.get("within_route_swap_effects", ()):
            prompt = effect.get("prompt")
            draft = effect.get("draft")
            if not isinstance(prompt, Mapping) or prompt.get("available") is not True:
                continue
            if not isinstance(draft, Mapping) or draft.get("available") is not True:
                continue
            removed_position = int(effect["removed_position"])
            added_position = int(effect["added_position"])
            nodes = route_nodes[route]
            if not 0 <= removed_position < len(nodes) or not 0 <= added_position < len(nodes):
                raise ValueError("within-route substitution position is out of range")
            removed_node = nodes[removed_position]
            added_node = nodes[added_position]
            if (
                removed_node not in anchor_selected
                or removed_node in mandatory
                or added_node in anchor_selected
            ):
                continue
            if source.blocks[removed_node].byte_size != source.blocks[added_node].byte_size:
                raise ValueError("within-route substitution is not byte neutral")
            prompt_signature = tuple(float(value) for value in prompt["delta_signature"])
            draft_signature = tuple(float(value) for value in draft["delta_signature"])
            if len(prompt_signature) != len(draft_signature) or not prompt_signature:
                raise ValueError("within-route directional signatures are incompatible")
            combined = tuple(
                proposal.prompt_objective_weight * p + proposal.draft_objective_weight * d
                for p, d in zip(prompt_signature, draft_signature, strict=True)
            )
            raw.append(
                {
                    "route": route,
                    "removed_position": removed_position,
                    "added_position": added_position,
                    "removed_node": removed_node,
                    "added_node": added_node,
                    "prompt_gain": float(prompt["gain"]),
                    "draft_gain": float(draft["gain"]),
                    "prompt_norm": float(prompt["delta_norm"]),
                    "draft_norm": float(draft["delta_norm"]),
                    "combined_norm": math.sqrt(sum(value * value for value in combined)),
                    "prompt_final_error": float(prompt["final_error"]),
                    "draft_final_error": float(draft["final_error"]),
                    "swap_record": dict(effect.get("swap_record", {})),
                }
            )
    if len(raw) < limit:
        raise ValueError("within-route planning lacks enough distinct feasible substitutions")

    def scale(field: str) -> float:
        return max(max(abs(float(item[field])) for item in raw), 1e-30)

    prompt_gain_scale = scale("prompt_gain")
    draft_gain_scale = scale("draft_gain")
    prompt_norm_scale = scale("prompt_norm")
    draft_norm_scale = scale("draft_norm")
    combined_norm_scale = scale("combined_norm")
    for item in raw:
        gain = (
            proposal.prompt_objective_weight * float(item["prompt_gain"]) / prompt_gain_scale
            + proposal.draft_objective_weight * float(item["draft_gain"]) / draft_gain_scale
        )
        perturbation = (
            0.5
            * (
                proposal.prompt_objective_weight * float(item["prompt_norm"]) / prompt_norm_scale
                + proposal.draft_objective_weight * float(item["draft_norm"]) / draft_norm_scale
            )
            + 0.5 * float(item["combined_norm"]) / combined_norm_scale
        )
        item["predicted_gain"] = gain
        item["score"] = gain - perturbation
    raw.sort(
        key=lambda item: (
            -float(item["score"]),
            float(item["combined_norm"]),
            -float(item["predicted_gain"]),
            item["route"],
            int(item["removed_position"]),
            int(item["added_position"]),
        )
    )

    candidates: list[OutputPreservingSelection] = []
    seen: set[tuple[int, ...]] = set()
    for item in raw:
        selected = set(anchor_selected)
        selected.remove(int(item["removed_node"]))
        selected.add(int(item["added_node"]))
        if not mandatory <= selected:
            raise ValueError("within-route substitution removed a mandatory block")
        selected_ids = tuple(sorted(selected))
        if selected_ids in seen:
            continue
        selected_bytes = sum(source.blocks[node_id].byte_size for node_id in selected_ids)
        if selected_bytes != anchor.selected_bytes or selected_bytes > anchor.target_bytes:
            raise ValueError("within-route substitution changed the physical byte boundary")
        route = item["route"]
        telemetry = []
        for telemetry_route in sorted(anchor_records):
            record = dict(anchor_records[telemetry_route])
            if telemetry_route == route:
                swap_record = dict(item["swap_record"])
                swap_record.update(
                    removed_position=int(item["removed_position"]),
                    added_position=int(item["added_position"]),
                )
                record.update(
                    routed=True,
                    swaps=[swap_record],
                    final_error=float(item["prompt_final_error"]),
                    draft_final_error=float(item["draft_final_error"]),
                )
            telemetry.append(record)
        substitution = {
            "layer": int(route[0]),
            "kv_head": int(route[1]),
            "removed_position": int(item["removed_position"]),
            "added_position": int(item["added_position"]),
            "removed_node_id": int(item["removed_node"]),
            "added_node_id": int(item["added_node"]),
            "removed_bytes": source.blocks[int(item["removed_node"])].byte_size,
            "added_bytes": source.blocks[int(item["added_node"])].byte_size,
        }
        candidate = replace(
            anchor,
            selected_node_ids=selected_ids,
            routes=tuple(telemetry),
            selected_bytes=selected_bytes,
            candidate_id=(
                f"within_l{route[0]:02d}_h{route[1]:02d}_"
                f"r{int(item['removed_position']):04d}_a{int(item['added_position']):04d}"
            ),
            exchange_routes=(route,),
            block_substitutions=(substitution,),
            physical_padding_bytes=0,
            surrogate_score=float(item["score"]),
            directional_prompt_delta_norm=float(item["prompt_norm"]),
            directional_draft_delta_norm=float(item["draft_norm"]),
            directional_predicted_gain=float(item["predicted_gain"]),
            directional_score=float(item["score"]),
        )
        seen.add(selected_ids)
        candidates.append(candidate)
        if len(candidates) == limit:
            break
    if len(candidates) != limit:
        raise ValueError("within-route planning did not produce the registered plan count")
    return tuple(
        replace(candidate, surrogate_rank=rank)
        for rank, candidate in enumerate(candidates, start=1)
    )


def compose_route_exchange_candidate(
    source: FullKVState,
    anchor: OutputPreservingSelection,
    proposal: OutputPreservingSelection,
    routes: Sequence[tuple[int, int]],
    *,
    candidate_id: str,
) -> OutputPreservingSelection:
    """Apply complete same-head proposal swaps for a deterministic route subset.

    Every selected route replaces the anchor's positions with the proposal's
    positions. Route cardinality, mandatory membership, source bytes, and the
    global hard cap are checked again instead of inferred from the generating
    selector.
    """

    chosen_routes = tuple(sorted(set(routes)))
    if not candidate_id or not chosen_routes or len(chosen_routes) != len(tuple(routes)):
        raise ValueError("route-exchange candidates need unique routes and an ID")
    if (
        anchor.target_bytes != proposal.target_bytes
        or anchor.selected_bytes != proposal.selected_bytes
        or anchor.anchor_node_ids != proposal.anchor_node_ids
    ):
        raise ValueError("route-exchange plans do not share one anchor and byte boundary")

    def by_route(node_ids: Sequence[int]) -> dict[tuple[int, int], tuple[int, ...]]:
        grouped: dict[tuple[int, int], list[int]] = {}
        for node_id in node_ids:
            block = source.blocks[node_id]
            grouped.setdefault((block.layer, block.kv_head), []).append(node_id)
        return {route: tuple(sorted(values)) for route, values in grouped.items()}

    anchor_nodes = by_route(anchor.selected_node_ids)
    proposal_nodes = by_route(proposal.selected_node_ids)
    expected_routes = set(anchor_nodes)
    if set(proposal_nodes) != expected_routes or not set(chosen_routes).issubset(expected_routes):
        raise ValueError("route-exchange plans do not cover the same cache routes")
    anchor_telemetry = {(int(item["layer"]), int(item["kv_head"])): item for item in anchor.routes}
    proposal_telemetry = {
        (int(item["layer"]), int(item["kv_head"])): item for item in proposal.routes
    }
    if set(anchor_telemetry) != expected_routes or set(proposal_telemetry) != expected_routes:
        raise ValueError("route-exchange telemetry does not cover the selected cache")
    mandatory_by_route: dict[tuple[int, int], set[int]] = {}
    for node_id, block in enumerate(source.blocks):
        if block.mandatory:
            mandatory_by_route.setdefault((block.layer, block.kv_head), set()).add(node_id)

    selected: list[int] = []
    telemetry: list[dict[str, Any]] = []
    chosen = set(chosen_routes)
    for route in sorted(expected_routes):
        before, after = anchor_nodes[route], proposal_nodes[route]
        if len(before) != len(after):
            raise ValueError("route-local exchanges changed physical capacity")
        route_nodes = after if route in chosen else before
        mandatory = mandatory_by_route.get(route, set())
        if not mandatory.issubset(route_nodes):
            raise ValueError("route-local exchange removed a mandatory block")
        selected.extend(route_nodes)
        record = proposal_telemetry[route] if route in chosen else anchor_telemetry[route]
        telemetry.append(dict(record))

    selected_ids = tuple(sorted(selected))
    selected_bytes = sum(source.blocks[node_id].byte_size for node_id in selected_ids)
    if selected_bytes != anchor.selected_bytes or selected_bytes > anchor.target_bytes:
        raise ValueError("route-local exchanges changed bytes or exceeded the hard cap")
    return replace(
        anchor,
        selected_node_ids=selected_ids,
        routes=tuple(telemetry),
        candidate_id=candidate_id,
        exchange_routes=chosen_routes,
    )


def route_exchange_single_candidates(
    source: FullKVState,
    anchor: OutputPreservingSelection,
    proposal: OutputPreservingSelection,
    *,
    limit: int,
) -> tuple[OutputPreservingSelection, ...]:
    """Return deterministic one-route plans prefiltered by label-blind evidence gain."""

    if not 1 <= limit <= 8:
        raise ValueError("single-route candidate limit must be in [1, 8]")
    anchor_routes = {(int(item["layer"]), int(item["kv_head"])): item for item in anchor.routes}
    proposal_routes = {(int(item["layer"]), int(item["kv_head"])): item for item in proposal.routes}
    ranked: list[tuple[float, float, tuple[int, int]]] = []
    for route in sorted(anchor_routes):
        record = proposal_routes[route]
        if not record.get("swaps"):
            continue
        initial = float(record["initial_error"])
        final = float(record["final_error"])
        prompt_gain = (initial - final) / max(initial, 1e-30)
        draft_initial = record.get("draft_initial_error")
        draft_final = record.get("draft_final_error")
        draft_gain = 0.0
        if isinstance(draft_initial, (float, int)) and isinstance(draft_final, (float, int)):
            draft_gain = (float(draft_initial) - float(draft_final)) / max(
                float(draft_initial), 1e-30
            )
        ranked.append((prompt_gain + draft_gain, prompt_gain, route))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return tuple(
        compose_route_exchange_candidate(
            source,
            anchor,
            proposal,
            (route,),
            candidate_id=f"route_l{route[0]:02d}_h{route[1]:02d}",
        )
        for _joint_gain, _prompt_gain, route in ranked[:limit]
    )
