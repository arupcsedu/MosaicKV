"""Experimental six-mechanism precision overlay ON the validated storage reader.

Not the old dense-cache compressor and not a reproduction of its compaction:
all parent prompt positions remain. The bounded planner chooses extra exact
protection; a shared, phase-safe prototype overrides a few unprotected rows.
Only original input observations enter planning. CPU residuals can promote the
prototype's members at the first provisional decode step. Parent source codes,
exact protection, lossless history and execution optimizations are unchanged.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from unittest.mock import patch

import torch

COMPONENT_MODES = (
    "full",
    "parent_reserved",
    "no_forecast",
    "no_graph",
    "no_value",
    "modular_selection",
    "no_extra_exact",
    "exact_members",
    "no_repair",
    "forced_repair",
)


@dataclass(frozen=True)
class OverlayPolicy:
    layers: tuple[int, ...] = (31,)
    extra_exact_positions: int = 4
    prototype_members: int = 2
    maximum_extra_gpu_bytes: int = 16384
    entropy_threshold: float = 0.5
    force_repair: bool = False
    enabled: bool = True
    provisional_tokens: int = 2
    component_mode: str = "full"
    record_phase_timings: bool = False
    forecast_distribution: str = "mean_query"
    prototype_representation: str = "absolute_kv"
    utility_signal: str = "original_value"
    demand_weighted_coverage: bool = False
    routing_signal: str = "value_energy"
    graph_accumulation: str = "preselected"
    replace_source_codes: bool = False
    repair_policy: str = "entropy"
    prototype_risk_threshold: float = 0.01
    precision_transaction: str = "none"
    readout_control: bool = False
    readout_selection: str = "none"
    readout_target: str = "conditional_head"
    inline_repair: bool = False
    graph_positional_search: str = "reference"

    def __post_init__(self):
        if not self.layers or tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError("unique sorted routed layers required")
        if any(type(x) is not int or not 0 <= x < 32 for x in self.layers):
            raise ValueError("registered LLaVA layer indices required")
        if type(self.extra_exact_positions) is not int or not 1 <= self.extra_exact_positions <= 16:
            raise ValueError("extra exact count must be in [1,16]")
        if type(self.prototype_members) is not int or not 2 <= self.prototype_members <= 8:
            raise ValueError("prototype must have 2--8 real members")
        if type(self.maximum_extra_gpu_bytes) is not int or self.maximum_extra_gpu_bytes < 1:
            raise ValueError("positive hard auxiliary byte allowance required")
        if not 0 <= self.entropy_threshold <= 1:
            raise ValueError("entropy threshold must be in [0,1]")
        if type(self.force_repair) is not bool or type(self.enabled) is not bool:
            raise ValueError("explicit activation booleans required")
        if type(self.provisional_tokens) is not int or self.provisional_tokens not in (2, 128):
            raise ValueError("registered two-token probe or 128-token parity control required")
        if self.component_mode not in COMPONENT_MODES:
            raise ValueError("explicit registered component mode required")
        if type(self.record_phase_timings) is not bool:
            raise ValueError("explicit phase timing flag required")
        if self.component_mode != "full" and not self.enabled:
            raise ValueError("component ablations require allocated, enabled overlay state")
        if self.forecast_distribution not in ("mean_query", "individual_queries"):
            raise ValueError("registered mean-query or individual-query forecast required")
        if self.prototype_representation not in ("absolute_kv", "parent_residual_delta"):
            raise ValueError("registered absolute or parent-residual prototype required")
        if self.utility_signal not in (
            "original_value",
            "parent_reconstruction_error",
            "exact_restoration_gain",
        ):
            raise ValueError("registered original-value or parent-error utility required")
        if self.routing_signal not in ("value_energy", "parent_error_opportunity"):
            raise ValueError("registered routing signal required")
        if type(self.demand_weighted_coverage) is not bool:
            raise ValueError("explicit demand-weighted coverage flag required")
        if self.graph_accumulation not in ("preselected", "segmented_tensor_topk"):
            raise ValueError("registered sparse edge implementation required")
        if type(self.replace_source_codes) is not bool:
            raise ValueError("explicit source replacement flag required")
        if self.repair_policy not in ("entropy", "entropy_or_prototype_risk"):
            raise ValueError("registered repair policy required")
        import math

        if not math.isfinite(self.prototype_risk_threshold) or self.prototype_risk_threshold <= 0:
            raise ValueError("positive finite prototype risk threshold required")
        if self.precision_transaction not in ("none", "exact", "prototype", "joint_guarded"):
            raise ValueError("registered precision transaction required")
        if self.precision_transaction != "none" and (
            self.prototype_representation != "parent_residual_delta" or not self.enabled
        ):
            raise ValueError("precision transaction requires active parent-residual prototypes")
        if type(self.readout_control) is not bool:
            raise ValueError("explicit readout control flag required")
        if self.readout_selection not in ("none", "exact", "repair", "joint"):
            raise ValueError("registered conditional-readout selection variant required")
        if self.readout_target not in ("conditional_head", "fullkv_draft"):
            raise ValueError("registered conditional-head or actual FullKV draft target required")
        if self.readout_target == "fullkv_draft" and self.readout_selection == "none":
            raise ValueError("FullKV draft target requires active readout selection")
        if type(self.inline_repair) is not bool:
            raise ValueError("explicit inline repair activation required")
        if self.graph_positional_search not in ("reference", "indexed"):
            raise ValueError("registered positional-graph execution policy required")
        if self.inline_repair and (
            self.layers != (31,)
            or not self.enabled
            or self.provisional_tokens != 2
            or self.readout_selection != "none"
            or self.readout_control
            or self.repair_policy != "entropy"
        ):
            raise ValueError("inline repair supports only the registered last-layer entropy policy")
        if (self.readout_control or self.readout_selection != "none") and (
            self.layers != (31,) or self.provisional_tokens != 2 or not self.enabled
        ):
            raise ValueError("readout controls require one final layer and one provisional update")
        if self.readout_control and self.component_mode != "full":
            raise ValueError("engineering readout control requires the complete overlay")
        if self.readout_selection != "none" and self.component_mode not in (
            "full",
            "no_graph",
            "no_value",
            "modular_selection",
            "no_repair",
        ):
            raise ValueError("registered conditional-readout policy ablation required")
        if self.readout_control and self.readout_selection != "none":
            raise ValueError("separate control and active readout selection arms required")


class _Phases:
    """Exclusive synchronized construction phases; opt-in diagnostic only."""

    def __init__(self, active, device):
        self.active, self.device, self.values = active, device, {}
        self.last = self._now()

    def _now(self):
        from time import perf_counter

        if self.active and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return perf_counter()

    def mark(self, name):
        if self.active:
            now = self._now()
            self.values[name] = now - self.last
            self.last = now


def _bytes(values):
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    return backing_bytes(tuple(values))


class LayerOverlay:
    """No original full-cache/input owner may survive this local constructor."""

    def __init__(
        self,
        *,
        head,
        positions,
        exact_ids,
        exact_pair,
        prototype_pair,
        phases,
        residual_pair,
        anchor,
        dispersion,
        evidence,
    ):
        self.head, self.anchor = head, anchor
        device = prototype_pair[0].device
        self.positions = torch.tensor(positions, dtype=torch.long, device=device)
        self.exact_ids = torch.tensor(exact_ids, dtype=torch.long, device=device)
        self.exact_pair = tuple(t.detach().clone().reshape(-1, t.shape[-1]) for t in exact_pair)
        self.prototype_pair = tuple(t.detach().clone().reshape(1, -1) for t in prototype_pair)
        self.cos, self.sin = tuple(t.detach().clone() for t in phases)
        c, s = self.cos[-1:].float(), self.sin[-1:].float()
        key = self.prototype_pair[0].float()
        half = key.shape[-1] // 2

        def rotate(x):
            return torch.cat((-x[..., half:], x[..., :half]), -1)

        neutral = (key * c - rotate(key) * s) / (c.square() + s.square()).clamp_min(1e-12)
        self.member_keys = (
            neutral * self.cos[:-1].float() + rotate(neutral) * self.sin[:-1].float()
        ).to(prototype_pair[0].dtype)
        self.mixed = tuple(
            torch.empty((len(positions), t.shape[-1]), dtype=t.dtype, device=device)
            for t in prototype_pair
        )
        # Actual, charged reservation. Contains no original residual values
        # until promotion; stable addresses also support the parent's replay.
        self.promotion = tuple(
            torch.zeros((len(positions), t.shape[-1]), dtype=t.dtype, device=device)
            for t in prototype_pair
        )
        self.promoted = torch.zeros((), dtype=torch.bool, device=device)
        self.actual_fills = torch.zeros((), dtype=torch.int64, device=device)
        self.residual_pair = tuple(t.detach().cpu().contiguous() for t in residual_pair)
        if device.type == "cuda":
            self.residual_pair = tuple(t.pin_memory() for t in self.residual_pair)
        self.dispersion, self.evidence = dispersion, evidence
        self.repair_count = 0
        self.attention_mass = 0.0
        self.component_mode = "full"
        self.prototype_representation = "absolute_kv"

    def gpu_tensors(self):
        return (
            self.positions,
            self.exact_ids,
            *self.exact_pair,
            *self.prototype_pair,
            self.cos,
            self.sin,
            self.member_keys,
            *self.mixed,
            *self.promotion,
            self.promoted,
            self.actual_fills,
        )

    @property
    def cpu_bytes(self):
        return _bytes(self.residual_pair)

    def fill(self, component, destination):
        if destination.ndim != 4 or destination.shape[0] != 1:
            raise ValueError("batch-one dense reader lease required")
        if self.component_mode == "parent_reserved":
            # Same REAL allocated storage, intentionally inactive data path.
            # This is a reservation control, not an efficient parent benchmark.
            self.actual_fills.add_(1)
            return
        # Original exact rows and their original logical positions, no donors.
        if self.component_mode != "no_extra_exact":
            destination[0, self.head].index_copy_(0, self.exact_ids, self.exact_pair[component])
        proto = (
            self.member_keys
            if component == 0
            else self.prototype_pair[1].expand(len(self.positions), -1)
        )
        # All outputs/reservations are charged before graph capture. Promotion
        # changes data and one device flag, never a captured Python branch.
        if self.prototype_representation == "parent_residual_delta":
            torch.index_select(
                destination[0, self.head], 0, self.positions, out=self.mixed[component]
            )
            self.mixed[component].add_(proto)
            proto = self.mixed[component]
        torch.where(self.promoted, self.promotion[component], proto, out=self.mixed[component])
        destination[0, self.head].index_copy_(0, self.positions, self.mixed[component])
        self.actual_fills.add_(1)

    def promote(self):
        if self.repair_count:
            raise RuntimeError("bounded diagnostic permits one promotion transaction")
        cuda = self.positions.device.type == "cuda"
        begin = torch.cuda.Event(enable_timing=True) if cuda else None
        end = torch.cuda.Event(enable_timing=True) if cuda else None
        if cuda:
            if not all(t.is_pinned() for t in self.residual_pair):
                raise RuntimeError("real pinned CPU residual payloads required")
            begin.record()
        for target, source in zip(self.promotion, self.residual_pair, strict=True):
            target.copy_(source.reshape_as(target), non_blocking=cuda)
        self.promoted.fill_(True)
        if cuda:
            end.record()
            end.synchronize()
        if not all(
            torch.equal(a.cpu(), b.reshape_as(a))
            for a, b in zip(self.promotion, self.residual_pair, strict=True)
        ):
            raise RuntimeError("promoted K/V do not equal original residuals bitwise")
        self.repair_count += 1
        return dict(
            restored_bytes=self.cpu_bytes,
            restored_positions=self.positions.cpu().tolist(),
            head=self.head,
            transfer_asynchronous=cuda,
            promoted_bitwise_original=True,
            transfer_ms=begin.elapsed_time(end) if cuda else 0.0,
        )


def plan_layer(
    adapter,
    prepared,
    attention,
    original,
    hidden,
    phases,
    probabilities,
    memberships,
    policy,
    *,
    pin_residuals=True,
    parent_pair=None,
):
    """Reuse the original forecast/graph/utility/selector/prototype functions."""
    from mosaickv.adapters.huggingface.types import CachedKeyState
    from mosaickv.cache_state import FullKVState
    from mosaickv.config import (
        ForecastingConfig,
        GraphConfig,
        PrototypeConfig,
        ResidualConfig,
        SelectionConfig,
        UtilityConfig,
    )
    from mosaickv.evaluation.protected_projection_source import protected_positions
    from mosaickv.forecasting import build_query_forecast
    from mosaickv.graph import build_evidence_graph
    from mosaickv.prototypes.construction import construct_three_tier_cache
    from mosaickv.prototypes.rope import RotaryPrototypeKeyTransform
    from mosaickv.selection import (
        BudgetedObjective,
        SelectionBudget,
        compute_block_utilities,
        lazy_greedy_select,
    )
    from mosaickv.types import BudgetUnit, ForecastMode

    n, h, d = original[0].shape[-2], original[0].shape[1], original[0].shape[-1]
    timer = _Phases(policy.record_phase_timings, original[0].device)
    text = tuple(
        p for s in prepared.modality_map if s.modality == "text" for p in range(s.start, s.end)
    )
    protected = protected_positions(memberships, text, length=n)
    # Label-blind route choice: prompt-demand-weighted value energy. Fixed
    # source-index tie break. No benchmark answers or reference logits exist.
    demand = probabilities[0, :, -16:].float().mean(1)
    energy = original[1][0].float().square().sum(-1)
    if policy.routing_signal == "parent_error_opportunity":
        if parent_pair is None:
            raise ValueError("actual parent required for opportunity routing")
        error = (original[1][0].float() - parent_pair[1][0].float()).square().sum(-1)
        available = (demand * error).masked_fill(protected.to(error.device), 0.0)
        # Optimistic, feasible-cardinality V-error surrogate, not a logit
        # certificate. Deterministic lower-head tie break via argmax.
        opportunities = available.topk(min(policy.extra_exact_positions, n), dim=-1).values.sum(-1)
        head = int(opportunities.argmax())
    else:
        head = int((demand * energy).sum(-1).argmax())
    required = tuple(protected[head].nonzero().flatten().tolist())
    pair = tuple(t[:, head : head + 1] for t in original)
    full = FullKVState.from_tensors(
        (pair,),
        block_size=1,
        modality_spans=prepared.modality_map,
        mandatory_logical_positions=required,
        cached_key_state=CachedKeyState.POST_ROPE,
    )
    timer.mark("route_and_descriptors")
    window = min(16, n)
    query = attention.q_proj(hidden[:, -window:]).view(1, window, h, d).transpose(1, 2)
    query = query[:, head : head + 1]
    c, s = phases[0][:, None, -window:], phases[1][:, None, -window:]
    rotated = torch.cat((-query[..., d // 2 :], query[..., : d // 2]), -1)
    query = query * c + rotated * s
    individual_probabilities = None
    if policy.forecast_distribution == "individual_queries" or policy.demand_weighted_coverage:
        individual_probabilities = torch.softmax(
            (query[0, 0].float() @ pair[0][0, 0].float().T) / d**0.5, -1
        )
    if policy.component_mode == "no_forecast":
        prediction = demand[head]
        individual_probabilities = probabilities[0, head, -window:].float()
        forecast_mode = "observed_prompt_attention_no_forecast"
    else:
        forecast = build_query_forecast(
            (query,),
            (),
            (1,),
            ForecastingConfig(
                mode=ForecastMode.PROMPT_WINDOW,
                prompt_window=window,
                draft_steps=0,
                centroid_count=2,
            ),
            original_logical_sequence_length=n,
            draft_cache_isolated=True,
        )
        if policy.forecast_distribution == "individual_queries":
            # Forecast attention from the distribution, not attention of its
            # mean query. Individual observations survive the nonlinearity.
            prediction = individual_probabilities.mean(0)
        else:
            prediction = torch.softmax(
                (forecast.layers[0][0].prompt_mean.float().reshape(1, d) @ pair[0][0, 0].float().T)
                / d**0.5,
                -1,
            ).reshape(-1)
        forecast_mode = forecast.provenance.mode.value
    timer.mark("prompt_queries_and_forecast")
    from mosaickv.graph.builder import positional_search

    with positional_search(policy.graph_positional_search):
        graph = build_evidence_graph(
            full,
            GraphConfig(
                enabled=policy.component_mode != "no_graph",
                max_neighbors=4,
                similarity_chunk_size=128,
                edge_accumulation_mode=policy.graph_accumulation,
            ),
            prompt_attentions=(probabilities[:, head : head + 1],),
            prompt_attention_query_window=window,
        )
    timer.mark("sparse_graph")
    utility_graph = graph
    if policy.utility_signal == "parent_reconstruction_error":
        if parent_pair is None:
            raise ValueError("actual parent required for error-aware utility")
        errors = (original[1][0, head].float() - parent_pair[1][0, head].float()).cpu().numpy()
        utility_graph = replace(
            graph,
            nodes=tuple(replace(node, pooled_value=errors[node.node_id]) for node in graph.nodes),
        )
    utilities = compute_block_utilities(
        utility_graph,
        UtilityConfig(
            lambda_v=0.0 if policy.component_mode == "no_value" else -0.25,
            lambda_o=0.0 if policy.component_mode == "no_value" else -0.25,
        ),
        forecast_attention_by_node=prediction.detach().cpu().numpy(),
        attention_provenance="actual_post_RoPE_prompt_mean_forecast_not_labels",
        rope_aware=True,
    )
    restoration_report = None
    if policy.utility_signal == "exact_restoration_gain":
        from mosaickv.evaluation.precision_gain import exact_restoration_gain

        if parent_pair is None:
            raise ValueError("actual parent needed for signed restoration gain")
        gain, anchor_error = exact_restoration_gain(
            query[0, 0], tuple(t[0, head] for t in original), tuple(t[0, head] for t in parent_pair)
        )
        normalized = (gain / max(anchor_error, 1e-20)).clamp(-1, 1).cpu().tolist()

        def gain_utility(u):
            contribution = u.value_contribution + 1.0 + normalized[u.node_id]
            local = (
                utilities.weights.lambda_q * u.forecast_attention_probability
                - utilities.weights.lambda_v * contribution
                - utilities.weights.lambda_o * u.uniqueness
            )
            return replace(
                u, expected_attention_output_contribution=contribution, local_utility=local
            )

        utilities = replace(utilities, blocks=tuple(gain_utility(u) for u in utilities.blocks))
        restoration_report = dict(
            anchor_attention_output_mse=anchor_error,
            maximum_relative_gain=max(normalized),
            minimum_relative_gain=min(normalized),
            weight=0.25,
            common_gain_offset=1.0,
            offset_is_constant_at_fixed_cardinality=True,
            actual_key_change_and_softmax_normalization=True,
            benchmark_quality_certificate=False,
        )
    timer.mark("value_utility")
    selection_config = SelectionConfig(
        lambda_g=0.0 if policy.component_mode == "modular_selection" else -0.1,
        lambda_m=0.0 if policy.component_mode == "modular_selection" else -0.1,
        stop_on_nonpositive_gain=False,
        minimum_stratum_fraction=0.0,
    )
    count = min(n - policy.prototype_members, len(required) + policy.extra_exact_positions)
    if count <= len(required):
        raise RuntimeError("no feasible optional protection/prototype capacity")
    objective = BudgetedObjective(
        graph,
        utilities,
        selection_config,
    )
    if policy.demand_weighted_coverage:
        from mosaickv.selection.opportunity_demand import (
            DemandWeightedObjective,
            make_route_evidence,
        )

        evidence = make_route_evidence(
            full,
            {(0, 0): individual_probabilities.detach().cpu().numpy()},
            prediction.detach().cpu().numpy(),
            {(0, 0): count},
        )
        objective = DemandWeightedObjective(
            graph, utilities, selection_config, evidence, floor=0.95
        )
    selected = lazy_greedy_select(objective, SelectionBudget(count * d * 4, BudgetUnit.BYTES))
    if policy.demand_weighted_coverage:
        selected = objective.protect_selection(selected)
    timer.mark("budgeted_selection")
    anchors = tuple(
        node for node in selected.selected_node_ids if str(full.blocks[node].modality) == "image"
    )
    if policy.record_phase_timings or policy.component_mode != "full":
        # Common matched-allocation rule: the bounded constructor partitions
        # image nodes into member_count+1 segments. Require a full real group,
        # rather than padding a short prototype or silently changing bytes.
        ordered = [i for i, b in enumerate(full.blocks) if str(b.modality) == "image"]
        size = policy.prototype_members + 1
        selected_set = set(selected.selected_node_ids)
        feasible = set()
        for start in range(0, len(ordered), size):
            segment = ordered[start : start + size]
            if len(set(segment) - selected_set) == policy.prototype_members:
                feasible.update(set(segment) & selected_set)
        anchors = tuple(a for a in anchors if a in feasible)
    if not anchors:
        raise RuntimeError("real image prototype requires a compatible selected image anchor")
    anchor = max(anchors, key=lambda node: (float(prediction[node].detach()), -node))
    transform = RotaryPrototypeKeyTransform(
        phases[0][:, None], phases[1][:, None], "actual_pinned_LLaVA_Llama_phase_v1"
    )
    constructed = construct_three_tier_cache(
        full,
        graph,
        selected,
        adapter.capabilities,
        prototype_config=PrototypeConfig(
            group_size=policy.prototype_members,
            max_position_span=1024,
            maximum_prototypes_per_head=1,
            budget_planning_mode="one_pass_bounded_anchor_cover",
            source_modalities=("image",),
        ),
        residual_config=ResidualConfig(require_pinned_memory=pin_residuals),
        active_byte_budget=(count + 1) * d * 4,
        retention_ratio=0.5,
        active_budget=SelectionBudget((count + 1) * d * 4, BudgetUnit.BYTES),
        prototype_anchor_node_ids=(anchor,),
        key_transform=transform,
        exact_gather_mode="batched_layer",
    )
    timer.mark("prototype_and_residual_construction")
    if len(constructed.prototypes) != 1:
        raise RuntimeError(f"real prototype required, not {constructed.mode}: {constructed.reason}")
    record = constructed.prototypes[0]
    ids = tuple(
        sorted(
            p
            for node in record.assigned_node_ids
            for p in full.blocks[node].original_logical_positions
        )
    )
    if len(ids) < 2 or set(ids) & set(required):
        raise RuntimeError("prototype must combine unprotected, nonmandatory members")
    extras = tuple(sorted(set(selected.selected_node_ids) - set(required)))
    index = torch.tensor(extras, device=pair[0].device)
    member_index = torch.tensor(ids, device=pair[0].device)
    proto_pair = (
        constructed.state.prototypes.prototype_keys[0],
        constructed.state.prototypes.prototype_values[0],
    )
    dispersion = record.diagnostics.dispersion
    if policy.prototype_representation == "parent_residual_delta":
        from mosaickv.evaluation.residual_delta_prototype import construct

        if parent_pair is None:
            raise ValueError("actual parent reconstruction required for residual prototypes")
        proto_pair, dispersion = construct(
            tuple(t[0, head] for t in original),
            tuple(t[0, head] for t in parent_pair),
            tuple(t[0] for t in phases),
            ids,
            record.anchor_logical_position,
            (1.0,) * len(ids),
        )
    phase_ids = torch.tensor((*ids, record.anchor_logical_position), device=pair[0].device)
    phase_subset = tuple(t[0].index_select(0, phase_ids).contiguous() for t in phases)
    evidence = dict(
        forecast_mode=forecast_mode,
        graph_nodes=len(graph.nodes),
        graph_edges=len(graph.weights),
        graph_positional_search=policy.graph_positional_search,
        cross_modal_edges=sum(
            graph.nodes[a].block.modality != graph.nodes[b].block.modality
            for a, b in zip(graph.row_indices, graph.column_indices, strict=True)
        ),
        value_contribution_max=max(
            u.expected_attention_output_contribution for u in utilities.blocks
        ),
        selection="lazy_greedy_submodular_extra_exact_protection",
        selected_extra=list(extras),
        prototype_members=list(ids),
        original_protected_positions=list(required),
        prototype_phase_safe=constructed.safety.safe,
        original_parent_positions_preserved=True,
        benchmark_labels_accessed=False,
        forecast_is_prompt_only=True,
        prototype_policy="shared_KV_overlay_at_original_member_phases_not_token_compaction",
        component_mode=policy.component_mode,
        forecast_distribution=policy.forecast_distribution,
        prototype_representation=policy.prototype_representation,
        prototype_anchor_position=record.anchor_logical_position,
        delta_prototype_uniform_weights=policy.prototype_representation == "parent_residual_delta",
        parent_source_storage_retired=False,
        utility_signal=policy.utility_signal,
        signed_restoration_gain=restoration_report,
        routing_signal=policy.routing_signal,
        demand_weighted_coverage=policy.demand_weighted_coverage,
        graph_accumulation=policy.graph_accumulation,
        per_query_mass_guard=getattr(objective, "guard_report", None),
        disabled_components={
            "parent_reserved": ["all_overlay_reads"],
            "no_forecast": ["future_query_forecast"],
            "no_graph": ["graph_edges"],
            "no_value": ["value_and_novelty_weights"],
            "modular_selection": ["facility_and_modality_coverage_rewards"],
            "no_extra_exact": ["additional_exact_protection_reads"],
            "exact_members": ["prototype_reads_replaced_with_original_members"],
            "no_repair": ["uncertainty_promotion"],
        }.get(policy.component_mode, []),
    )
    if policy.record_phase_timings:
        from mosaickv.evaluation.fox_campaign import digest

        evidence["graph_coo_sha256"] = digest(
            dict(
                rows=graph.row_indices,
                columns=graph.column_indices,
                weights=graph.weights,
                types=tuple(e.value for e in graph.edge_types),
            )
        )
    if policy.readout_selection != "none":
        # The original graph/forecast/value/submodular seed remains operative.
        # Only its bounded neighboring receiver pool enters draft reranking.
        neighbors = {
            int(b)
            for a, b in zip(graph.row_indices, graph.column_indices, strict=True)
            if int(a) in extras
        }
        available = set(range(n)) - set(required) - set(ids) - set(extras)
        scores = {u.node_id: u.local_utility for u in utilities.blocks}
        ordered = sorted(available, key=lambda i: (i not in neighbors, -scores[i], i))
        evidence["readout_receiver_shortlist"] = ordered[:16]
        evidence["readout_receiver_pool_uses_graph_then_utility"] = True
    overlay = LayerOverlay(
        head=head,
        positions=ids,
        exact_ids=extras,
        exact_pair=tuple(t[0, 0].index_select(0, index) for t in pair),
        prototype_pair=proto_pair,
        phases=phase_subset,
        residual_pair=tuple(t[0, 0].index_select(0, member_index) for t in pair),
        anchor=record.anchor_logical_position,
        dispersion=dispersion,
        evidence=evidence,
    )
    if _bytes(overlay.gpu_tensors()) > policy.maximum_extra_gpu_bytes:
        raise RuntimeError("overlay including repair reservation exceeds hard auxiliary cap")
    overlay.component_mode = policy.component_mode
    overlay.prototype_representation = policy.prototype_representation
    if policy.precision_transaction != "none" and policy.component_mode != "parent_reserved":
        from mosaickv.evaluation.precision_transaction_study import plan_precision_transaction

        if parent_pair is None:
            raise ValueError("precision transaction requires original parent observations")
        payload = dict(
            original_pair=tuple(t[0, head].detach().cpu().clone() for t in original),
            parent_pair=tuple(t[0, head].detach().cpu().clone() for t in parent_pair),
            prompt_queries=query[0, 0].detach().cpu().clone(),
            phases=tuple(t[0].detach().cpu().clone() for t in phases),
            evidence={
                k: evidence[k]
                for k in (
                    "selected_extra",
                    "prototype_members",
                    "original_protected_positions",
                    "prototype_anchor_position",
                    "prototype_representation",
                )
            },
            overlay=dict(
                member_keys=overlay.member_keys.detach().cpu().clone(),
                prototype_value=overlay.prototype_pair[1].detach().cpu().clone(),
            ),
        )
        proof = plan_precision_transaction(payload, mode=policy.precision_transaction)
        replacement = proof.pop("replacement")
        revised = replacement["exact_ids"]
        if len(revised) != len(extras) or set(revised) & (set(required) | set(ids)):
            raise RuntimeError("precision transaction changed capacity or protected memberships")
        old = overlay.exact_ids.cpu().tolist()
        overlay.exact_ids.copy_(torch.tensor(revised, dtype=torch.long, device="cpu"))
        for target, source in zip(overlay.exact_pair, original, strict=True):
            target.copy_(source[0, head].index_select(0, overlay.exact_ids))
        if proof["accepted_prototype"]:
            for target, source in zip(
                overlay.prototype_pair, replacement["prototype_pair"], strict=True
            ):
                target.copy_(source)
            overlay.member_keys.copy_(replacement["member_keys"])
            overlay.dispersion = replacement["dispersion"]
        evidence.update(
            selected_extra=list(revised),
            submodular_seed_extra=old,
            precision_transaction=policy.precision_transaction,
            precision_transaction_evidence=proof,
            original_query_holdout_not_independent_examples=True,
        )
        timer.mark("precision_transaction_cpu_and_buffer_updates")
    if policy.component_mode == "exact_members":
        # Reuse already charged promotion buffers. A preparation control,
        # explicitly NOT an uncertainty-triggered repair event.
        for target, source in zip(overlay.promotion, overlay.residual_pair, strict=True):
            target.copy_(source.reshape_as(target), non_blocking=False)
        overlay.promoted.fill_(True)
        evidence["exact_member_control_preloaded"] = True
    timer.mark("overlay_buffers_and_control_preload")
    evidence["construction_phase_seconds"] = timer.values
    return overlay


def _private_probe(
    original_run, args, kwargs, options, attention_observer=None, readout_capture=None
):
    """New thread => new request-local ContextVars, NOT nested cache bindings.

    A registered one-update scope retains exact per-layer engagement checks.
    The separate 128-prediction path remains the bitwise GPU parity control.
    """
    from concurrent.futures import ThreadPoolExecutor

    from mosaickv.evaluation import common_streaming_timing as common

    @torch.inference_mode()
    def task():
        from contextlib import nullcontext

        from mosaickv.evaluation import prepared_patch_dispatch as patch_dispatch
        from mosaickv.evaluation.bounded_probe_horizon import two_prediction_probe
        from mosaickv.evaluation.graph_accounting import use as accounting
        from mosaickv.evaluation.graph_lookahead import use as lookahead
        from mosaickv.evaluation.native_execution_plan import use as native
        from mosaickv.evaluation.paper_geometry import registered_geometry
        from mosaickv.evaluation.prompt_prefetch import use as prefetch
        from mosaickv.evaluation.reconstruction_policy import use as readers
        from mosaickv.evaluation.replay_descriptor_policy import use as descriptors
        from mosaickv.evaluation.request_kernel_dispatch import use_dispatch
        from mosaickv.evaluation.source_hotpath import use as hotpath

        def body(**opts):
            with (
                registered_geometry(603),
                lookahead(False),
                patch_dispatch.use(opts.get("reuse_patch_dispatch", False)),
                accounting(opts.get("graph_accounting", "original")),
                readers(opts.get("reconstruction_readers", "original")),
                use_dispatch(opts.get("reuse_kernel_dispatch", False)),
                descriptors(opts.get("reuse_code_descriptors", False)),
                hotpath(opts.get("source_hotpath", "original")),
                native(False),
                prefetch(False),
            ):
                with ExitStack() as hooks:
                    if readout_capture is not None:
                        hooks.enter_context(readout_capture.use())
                    if attention_observer is not None:
                        adapter = args[0]
                        original_decode = adapter.decode_one_token

                        def captured(*a, **kw):
                            result = original_decode(*a, **dict(kw, capture_attentions=True))
                            attention_observer(result.attention_weights)
                            return result

                        hooks.enter_context(patch.object(adapter, "decode_one_token", captured))
                    values, detail = original_run(*args, **kwargs)
                return values, (), detail

        wrapped = body
        for decorator in reversed(
            (
                common.joint_projection_scope,
                common.uniform_core_scope,
                common.native_history_scope,
                common.input_norm_scope,
                common.post_attention_scope,
                common.exact_rotary_scope,
            )
        ):
            wrapped = decorator(wrapped)
        scope = two_prediction_probe() if kwargs["tokens"] == 2 else nullcontext()
        with scope:
            values, _, detail = wrapped(**options)
        torch.cuda.synchronize()
        return values, detail

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="six-source-probe") as pool:
        return pool.submit(task).result()


@contextmanager
def integrated_storage(
    adapter, policy: OverlayPolicy, *, reader_options=None, diagnostic_capture=None
):
    """Private patches only. Disabled scope must be bitwise parent-equivalent."""
    from mosaickv.evaluation import storage_runs
    from mosaickv.evaluation import streaming_source_constructor as constructor
    from mosaickv.evaluation.sequential_source_history import SequentialSourceHistoryLayer
    from mosaickv.repair.core import normalized_next_token_entropy

    overlays, report = (
        {},
        dict(
            variant="storage_six_precision_overlay_v1",
            parent_storage_source_sha="a6f689fcdfae2b9023045aa412c6d90d8ce79012",
            enabled=policy.enabled,
            layers={},
            repair_events=[],
            original_compaction_design_reproduction=False,
            benchmark_labels_accessed=False,
            component_mode=policy.component_mode,
        ),
    )
    fullkv_snapshots = {}
    original_constructor = constructor.StreamingDirectSource
    original_fill = SequentialSourceHistoryLayer.materialize_component
    original_cache, original_run = storage_runs.cache_from_formats, storage_runs.run

    class Encoder(original_constructor):
        def __init__(self, model_adapter, prepared, *args, **kwargs):
            super().__init__(model_adapter, prepared, *args, **kwargs)
            self.prepared_for_overlay, self.model_adapter = prepared, model_adapter

        def __call__(self, index, attention, original, args, kwargs, output):
            packet = super().__call__(index, attention, original, args, kwargs, output)
            if policy.readout_target == "fullkv_draft":
                from time import perf_counter

                from mosaickv.evaluation.offloaded_fullkv_draft import snapshot

                if index in fullkv_snapshots:
                    raise RuntimeError("one original prefill snapshot per layer required")
                torch.cuda.synchronize()
                snapshot_started = perf_counter()
                fullkv_snapshots[index] = snapshot(original)
                report["fullkv_cpu_snapshot_construction_seconds"] = (
                    report.get("fullkv_cpu_snapshot_construction_seconds", 0.0)
                    + perf_counter()
                    - snapshot_started
                )
            if policy.enabled and index in policy.layers:
                approximation = None
                if (
                    policy.prototype_representation == "parent_residual_delta"
                    or policy.utility_signal
                    in ("parent_reconstruction_error", "exact_restoration_gain")
                    or policy.routing_signal == "parent_error_opportunity"
                ):
                    from time import perf_counter

                    torch.cuda.synchronize()
                    started = perf_counter()
                    approximation = packet.materialize(attention)
                    torch.cuda.synchronize()
                    report["parent_reference_materialization_seconds"] = perf_counter() - started
                overlays[index] = plan_layer(
                    self.model_adapter,
                    self.prepared_for_overlay,
                    attention,
                    original,
                    kwargs.get("hidden_states", args[0] if args else None),
                    kwargs["position_embeddings"],
                    output[1],
                    self.memberships[index],
                    policy,
                    parent_pair=approximation,
                )
                del approximation
                report["layers"][str(index)] = overlays[index].evidence
                if policy.readout_control or policy.readout_selection != "none":
                    head = overlays[index].head
                    overlays[index].readout_original = tuple(
                        t[0, head].detach().to(device="cpu", copy=True).contiguous()
                        for t in original
                    )
                if diagnostic_capture is not None:
                    # Evaluation-only export, never queried by the planner.
                    # Scratch reconstruction is released BEFORE decode. Store
                    # only this routed head, not original full-cache owners.
                    head = overlays[index].head
                    hidden = kwargs.get("hidden_states", args[0] if args else None)
                    n, h, d = original[0].shape[-2], original[0].shape[1], original[0].shape[-1]
                    w = min(16, n)
                    query = attention.q_proj(hidden[:, -w:]).view(1, w, h, d).transpose(1, 2)
                    query = query[:, head : head + 1]
                    cos, sin = kwargs["position_embeddings"]
                    rotated = torch.cat((-query[..., d // 2 :], query[..., : d // 2]), -1)
                    query = query * cos[:, None, -w:] + rotated * sin[:, None, -w:]
                    approximate = packet.materialize(attention)
                    diagnostic_capture(
                        dict(
                            layer=index,
                            head=head,
                            original_pair=tuple(
                                t[0, head].detach().cpu().clone() for t in original
                            ),
                            parent_pair=tuple(
                                t[0, head].detach().cpu().clone() for t in approximate
                            ),
                            prompt_queries=query[0, 0].detach().cpu().clone(),
                            original_prompt_probabilities=output[1][0, head, -w:]
                            .detach()
                            .cpu()
                            .clone(),
                            phases=tuple(t[0].detach().cpu().clone() for t in (cos, sin)),
                            evidence=overlays[index].evidence,
                            overlay=dict(
                                exact_pair=tuple(
                                    t.detach().cpu().clone() for t in overlays[index].exact_pair
                                ),
                                member_keys=overlays[index].member_keys.detach().cpu().clone(),
                                prototype_value=overlays[index]
                                .prototype_pair[1]
                                .detach()
                                .cpu()
                                .clone(),
                            ),
                            evaluation_only=True,
                            labels_accessed=False,
                            materialization="constructor_packet_not_a_decode_reader_timing",
                        )
                    )
                    del approximate, query, rotated
            return packet

    def fill(layer, component):
        original_fill(layer, component)
        if policy.readout_control or policy.readout_selection != "none":
            from mosaickv.evaluation.exact_last_layer_readout import record_component

            record_component(layer, component, parent=True)
        overlay = overlays.get(layer.attention.layer_idx)
        if overlay is not None:
            overlay.fill(component, layer.leased[0])
        if policy.readout_control or policy.readout_selection != "none":
            record_component(layer, component, parent=False)

    def cache(*args, **kwargs):
        result = original_cache(*args, **kwargs)
        extra = tuple(t for overlay in overlays.values() for t in overlay.gpu_tensors())
        result.shared_tensors = (*result.shared_tensors, *extra)
        result.shared_bytes = _bytes(result.shared_tensors)
        result.initial_bytes = _bytes(
            (*result.shared_tensors, *(t for layer in result.layers for t in layer.owned_tensors()))
        )
        result.observe()
        return result

    def run(*args, **kwargs):
        from mosaickv.evaluation.source_replacement import audit_retirement

        audit_retirement(report["source_replacement"])
        if policy.enabled and set(overlays) != set(policy.layers):
            raise RuntimeError("every configured mechanism layer must have a real overlay")
        report["prompt_length"] = args[1].state.logical_sequence_length
        inline_result = None
        if overlays and policy.inline_repair:
            from mosaickv.evaluation.inline_first_step_repair import InlineRepair

            with InlineRepair(adapter, overlays[31], policy, report) as inline:
                inline_result = original_run(*args, **kwargs)
            if not inline.done:
                raise RuntimeError("inline first-step repair transport did not engage")
            report.update(
                provisional_tokens=2,
                provisional_model_updates=0,
                probe_decode_wall_seconds=0.0,
                probe_storage=None,
                separate_compressed_cache_probe_performed=False,
            )
        if overlays and not policy.inline_repair:
            # A separate, disposable two-prediction cache uses the SAME prefill
            # and source packets. No nested live cache or second prefill.
            if reader_options is None:
                raise ValueError("frozen parent reader options required for the private probe")
            probe_kwargs = dict(kwargs, tokens=policy.provisional_tokens)
            fullkv_draft_logits = None
            if policy.readout_target == "fullkv_draft":
                import gc
                import weakref

                from mosaickv.evaluation.offloaded_fullkv_draft import run as exact_draft

                if set(fullkv_snapshots) != set(range(32)):
                    raise RuntimeError("complete original FullKV prefill snapshots required")
                owners = [weakref.ref(t) for pair in fullkv_snapshots.values() for t in pair]
                fullkv_draft_logits, draft_evidence = exact_draft(
                    adapter, args[1], tuple(fullkv_snapshots[i] for i in range(32))
                )
                fullkv_snapshots.clear()
                gc.collect()
                draft_evidence["cpu_original_tensor_objects_alive_after_draft"] = sum(
                    ref() is not None for ref in owners
                )
                if draft_evidence["cpu_original_tensor_objects_alive_after_draft"]:
                    raise RuntimeError("temporary CPU FullKV snapshots survived forecast")
                report["fullkv_draft_forecast"] = draft_evidence
                if diagnostic_capture is not None:
                    diagnostic_capture(
                        dict(
                            capture_kind="fullkv_draft",
                            layer=31,
                            logits=fullkv_draft_logits,
                            evidence=draft_evidence,
                            evaluation_only=True,
                            labels_accessed=False,
                        )
                    )

            def observe(weights):
                for layer, overlay in overlays.items():
                    if len(weights) <= layer or weights[layer] is None:
                        raise RuntimeError("actual decode attention unavailable for prototype risk")
                    overlay.attention_mass = float(
                        weights[layer][0, overlay.head, -1].index_select(0, overlay.positions).sum()
                    )

            risk_policy = policy.repair_policy == "entropy_or_prototype_risk"
            capture = None
            if policy.readout_control or policy.readout_selection != "none":
                from mosaickv.evaluation.exact_last_layer_readout import ReadoutCapture

                capture = ReadoutCapture(overlays[31].head)
            before, probe_details = _private_probe(
                original_run,
                args,
                probe_kwargs,
                reader_options,
                attention_observer=observe if risk_policy else None,
                readout_capture=capture,
            )
            predicted_selected = None
            readout_trigger = False
            if capture is not None:
                from mosaickv.evaluation.exact_last_layer_readout import verify_controls

                def probe():
                    return _private_probe(original_run, args, probe_kwargs, reader_options)[0]

                if policy.readout_control:
                    report["exact_last_layer_readout"] = verify_controls(
                        adapter,
                        capture,
                        overlays[31],
                        overlays[31].readout_original,
                        before[1],
                        probe,
                    )
                    proof = report["exact_last_layer_readout"]
                else:
                    from mosaickv.evaluation.conditional_readout_selection import select

                    proof, predicted_selected = select(
                        adapter,
                        capture,
                        overlays[31],
                        overlays[31].readout_original,
                        before[1],
                        mode=policy.readout_selection,
                        target_logits=None
                        if fullkv_draft_logits is None
                        else fullkv_draft_logits[1],
                    )
                    report["conditional_readout_selection"] = proof
                    readout_trigger = proof["requested_residual_restoration"]
                if diagnostic_capture is not None:
                    diagnostic_capture(
                        dict(
                            capture_kind="last_layer_readout",
                            layer=31,
                            data=capture.data,
                            original_prompt=overlays[31].readout_original,
                            reference_logits=before[1].detach().cpu(),
                            evidence=proof,
                            evaluation_only=True,
                            labels_accessed=False,
                        )
                    )
                overlays[31].readout_original = None
                capture.data.clear()
            entropy = normalized_next_token_entropy(before[1])
            report["provisional_entropy"] = entropy
            report["probe_storage"] = probe_details["storage"]
            report["probe_decode_wall_seconds"] = probe_details["decode_wall_seconds"]
            report["provisional_tokens"] = policy.provisional_tokens
            forced = policy.force_repair or policy.component_mode == "forced_repair"
            from mosaickv.evaluation.precision_gain import repair_risk

            risks = (
                {
                    str(i): repair_risk(
                        o.attention_mass,
                        o.dispersion,
                        max(float(o.residual_pair[1].float().norm(dim=-1).mean()), 1e-12),
                    )
                    for i, o in overlays.items()
                }
                if risk_policy
                else {}
            )
            report["actual_decode_prototype_risk"] = risks
            report["actual_decode_prototype_mass"] = (
                {str(i): o.attention_mass for i, o in overlays.items()} if risk_policy else {}
            )
            report["prototype_risk_threshold"] = policy.prototype_risk_threshold
            risk_trigger = bool(risks) and max(risks.values()) >= policy.prototype_risk_threshold
            if policy.component_mode not in ("no_repair", "exact_members", "parent_reserved") and (
                forced or entropy >= policy.entropy_threshold or risk_trigger or readout_trigger
            ):
                index = min(overlays)
                event = overlays[index].promote()
                report["repair_events"].append(
                    dict(
                        event,
                        layer=index,
                        trigger_reason=(
                            "controlled_readout_plan_enumeration"
                            if report.get("conditional_readout_selection", {})
                            .get("selected", {})
                            .get("evaluation_only_forced_plan")
                            else "controlled_entropy_zero"
                            if forced
                            else "conditional_readout_uncertainty_reduction"
                            if readout_trigger
                            else "prototype_risk"
                            if risk_trigger
                            else "entropy"
                        ),
                        current_token_recomputed_once=True,
                        stress_control=forced
                        or bool(
                            report.get("conditional_readout_selection", {})
                            .get("selected", {})
                            .get("evaluation_only_forced_plan")
                        ),
                    )
                )
            report["provisional_first_decode_logit_sha"] = None
        after, details = original_run(*args, **kwargs) if inline_result is None else inline_result
        if overlays and not policy.inline_repair and predicted_selected is not None:
            from mosaickv.evaluation.exact_last_layer_readout import compare_logits

            check = compare_logits(predicted_selected, after[1])
            report["conditional_readout_selection"]["actual_selected_logit_control"] = check
            if not check["logits_bitwise"]:
                raise RuntimeError("selected conditional suffix differs from actual final decode")
        if overlays and not policy.inline_repair:
            report["repair_logit_max_change"] = float((before[1] - after[1]).abs().max())
            report["repair_token_changed"] = int(before[1].argmax()) != int(after[1].argmax())
            if policy.readout_selection != "none":
                report["allocation_and_optional_repair_joint_effect"] = True
                report["repair_quality_benefit_isolated"] = False
        report["extra_gpu_bytes"] = _bytes(
            tuple(t for o in overlays.values() for t in o.gpu_tensors())
        )
        report["residual_cpu_bytes"] = sum(o.cpu_bytes for o in overlays.values())
        report["all_residuals_pinned"] = all(
            t.is_pinned() for o in overlays.values() for t in o.residual_pair
        )
        report["storage"] = details["storage"]
        report["parent_reader_and_lossless_history_used"] = True
        report["actual_overlay_fills"] = {str(i): int(o.actual_fills) for i, o in overlays.items()}
        if overlays and any(
            int(o.actual_fills)
            < 2
            * (
                kwargs["tokens"]
                - 1
                + (0 if policy.inline_repair else policy.provisional_tokens - 1)
            )
            for o in overlays.values()
        ):
            raise RuntimeError("prototype reader did not engage for every provisional/final step")
        return after, details

    with ExitStack() as stack:
        from mosaickv.evaluation.source_replacement import use as source_replacement

        report["source_replacement"] = stack.enter_context(
            source_replacement(policy.replace_source_codes)
        )
        stack.callback(fullkv_snapshots.clear)
        stack.enter_context(patch.object(constructor, "StreamingDirectSource", Encoder))
        stack.enter_context(
            patch.object(SequentialSourceHistoryLayer, "materialize_component", fill)
        )
        stack.enter_context(patch.object(storage_runs, "cache_from_formats", cache))
        stack.enter_context(patch.object(storage_runs, "run", run))
        yield report
