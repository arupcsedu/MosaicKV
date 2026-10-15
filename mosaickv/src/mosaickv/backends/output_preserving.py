"""HF bridge for the explicitly labeled output-preserving exact-selection variant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mosaickv.baselines.selectors import select_prompt_attention_topk_ids_by_head
from mosaickv.cache_state import FullKVState, MosaicKVState
from mosaickv.selection.output_preserving import corrective_swaps, prepare_evidence, reconstruct
from mosaickv.selection.types import SelectionBudget
from mosaickv.types import BudgetUnit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

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
    draft_steps: int = 0
    draft_cache_isolated: bool = True

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
                "prompt_topk_anchor_then_prompt_and_draft_gated_output_swaps_v1"
                if self.draft_steps
                else "prompt_topk_anchor_then_bounded_renormalized_output_swaps_v1"
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
                "routes": list(self.routes),
            },
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
        result = corrective_swaps(
            a,
            v,
            positions[route],
            mandatory[route],
            options,
            validation_attention=validation,
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
        )
        positions[route] = result.positions
    final_ids = tuple(sorted(by_route[r][p] for r in routes for p in positions[r]))
    active_bytes = sum(source.blocks[i].byte_size for i in final_ids)
    anchor_bytes = sum(source.blocks[i].byte_size for i in anchor.selected_node_ids)
    if active_bytes != anchor_bytes or active_bytes > target_bytes:
        raise ValueError("output-preserving swaps changed physical capacity or exceeded the cap")
    return OutputPreservingSelection(
        final_ids,
        anchor.selected_node_ids,
        tuple(telemetry[r] for r in routes),
        target_bytes,
        active_bytes,
        options.query_evidence,
        options.max_swaps_per_head,
        options.effective_method,
        len(draft_attention_steps),
        draft_cache_isolated,
    )
