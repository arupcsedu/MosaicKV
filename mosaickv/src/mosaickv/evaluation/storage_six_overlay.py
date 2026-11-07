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
from dataclasses import dataclass
from unittest.mock import patch

import torch


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
        # Original exact rows and their original logical positions, no donors.
        destination[0, self.head].index_copy_(0, self.exact_ids, self.exact_pair[component])
        proto = (
            self.member_keys
            if component == 0
            else self.prototype_pair[1].expand(len(self.positions), -1)
        )
        # All outputs/reservations are charged before graph capture. Promotion
        # changes data and one device flag, never a captured Python branch.
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
    text = tuple(
        p for s in prepared.modality_map if s.modality == "text" for p in range(s.start, s.end)
    )
    protected = protected_positions(memberships, text, length=n)
    # Label-blind route choice: prompt-demand-weighted value energy. Fixed
    # source-index tie break. No benchmark answers or reference logits exist.
    demand = probabilities[0, :, -16:].float().mean(1)
    energy = original[1][0].float().square().sum(-1)
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
    window = min(16, n)
    query = attention.q_proj(hidden[:, -window:]).view(1, window, h, d).transpose(1, 2)
    query = query[:, head : head + 1]
    c, s = phases[0][:, None, -window:], phases[1][:, None, -window:]
    rotated = torch.cat((-query[..., d // 2 :], query[..., : d // 2]), -1)
    query = query * c + rotated * s
    forecast = build_query_forecast(
        (query,),
        (),
        (1,),
        ForecastingConfig(
            mode=ForecastMode.PROMPT_WINDOW, prompt_window=window, draft_steps=0, centroid_count=2
        ),
        original_logical_sequence_length=n,
        draft_cache_isolated=True,
    )
    prediction = torch.softmax(
        (forecast.layers[0][0].prompt_mean.float().reshape(1, d) @ pair[0][0, 0].float().T)
        / d**0.5,
        -1,
    ).reshape(-1)
    graph = build_evidence_graph(
        full,
        GraphConfig(max_neighbors=4, similarity_chunk_size=128),
        prompt_attentions=(probabilities[:, head : head + 1],),
        prompt_attention_query_window=window,
    )
    utilities = compute_block_utilities(
        graph,
        UtilityConfig(lambda_v=-0.25, lambda_o=-0.25),
        forecast_attention_by_node=prediction.detach().cpu().numpy(),
        attention_provenance="actual_post_RoPE_prompt_mean_forecast_not_labels",
        rope_aware=True,
    )
    objective = BudgetedObjective(
        graph,
        utilities,
        SelectionConfig(
            lambda_g=-0.1,
            lambda_m=-0.1,
            stop_on_nonpositive_gain=False,
            minimum_stratum_fraction=0.0,
        ),
    )
    count = min(n - policy.prototype_members, len(required) + policy.extra_exact_positions)
    if count <= len(required):
        raise RuntimeError("no feasible optional protection/prototype capacity")
    selected = lazy_greedy_select(objective, SelectionBudget(count * d * 4, BudgetUnit.BYTES))
    anchors = tuple(
        node for node in selected.selected_node_ids if str(full.blocks[node].modality) == "image"
    )
    if not anchors:
        raise RuntimeError("real image prototype requires a compatible selected image anchor")
    anchor = max(anchors, key=lambda node: (float(prediction[node]), -node))
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
    phase_ids = torch.tensor((*ids, record.anchor_logical_position), device=pair[0].device)
    phase_subset = tuple(t[0].index_select(0, phase_ids).contiguous() for t in phases)
    evidence = dict(
        forecast_mode=forecast.provenance.mode.value,
        graph_nodes=len(graph.nodes),
        graph_edges=len(graph.weights),
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
    )
    overlay = LayerOverlay(
        head=head,
        positions=ids,
        exact_ids=extras,
        exact_pair=tuple(t[0, 0].index_select(0, index) for t in pair),
        prototype_pair=proto_pair,
        phases=phase_subset,
        residual_pair=tuple(t[0, 0].index_select(0, member_index) for t in pair),
        anchor=record.anchor_logical_position,
        dispersion=record.diagnostics.dispersion,
        evidence=evidence,
    )
    if _bytes(overlay.gpu_tensors()) > policy.maximum_extra_gpu_bytes:
        raise RuntimeError("overlay including repair reservation exceeds hard auxiliary cap")
    return overlay


def _private_probe(original_run, args, kwargs, options):
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
def integrated_storage(adapter, policy: OverlayPolicy, *, reader_options=None):
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
        ),
    )
    original_constructor = constructor.StreamingDirectSource
    original_fill = SequentialSourceHistoryLayer.materialize_component
    original_cache, original_run = storage_runs.cache_from_formats, storage_runs.run

    class Encoder(original_constructor):
        def __init__(self, model_adapter, prepared, *args, **kwargs):
            super().__init__(model_adapter, prepared, *args, **kwargs)
            self.prepared_for_overlay, self.model_adapter = prepared, model_adapter

        def __call__(self, index, attention, original, args, kwargs, output):
            packet = super().__call__(index, attention, original, args, kwargs, output)
            if policy.enabled and index in policy.layers:
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
                )
                report["layers"][str(index)] = overlays[index].evidence
            return packet

    def fill(layer, component):
        original_fill(layer, component)
        overlay = overlays.get(layer.attention.layer_idx)
        if overlay is not None:
            overlay.fill(component, layer.leased[0])

    def cache(*args, **kwargs):
        result = original_cache(*args, **kwargs)
        extra = tuple(t for overlay in overlays.values() for t in overlay.gpu_tensors())
        result.shared_tensors = (*result.shared_tensors, *extra)
        result.shared_bytes = _bytes(result.shared_tensors)
        result.initial_bytes = _bytes(
            (*extra, *(t for layer in result.layers for t in layer.owned_tensors()))
        )
        result.observe()
        return result

    def run(*args, **kwargs):
        if policy.enabled and set(overlays) != set(policy.layers):
            raise RuntimeError("every configured mechanism layer must have a real overlay")
        report["prompt_length"] = args[1].state.logical_sequence_length
        if overlays:
            # A separate, disposable two-prediction cache uses the SAME prefill
            # and source packets. No nested live cache or second prefill.
            if reader_options is None:
                raise ValueError("frozen parent reader options required for the private probe")
            probe_kwargs = dict(kwargs, tokens=policy.provisional_tokens)
            before, probe_details = _private_probe(original_run, args, probe_kwargs, reader_options)
            entropy = normalized_next_token_entropy(before[1])
            report["provisional_entropy"] = entropy
            report["probe_storage"] = probe_details["storage"]
            report["probe_decode_wall_seconds"] = probe_details["decode_wall_seconds"]
            report["provisional_tokens"] = policy.provisional_tokens
            if policy.force_repair or entropy >= policy.entropy_threshold:
                index = min(overlays)
                event = overlays[index].promote()
                report["repair_events"].append(
                    dict(
                        event,
                        layer=index,
                        trigger_reason="controlled_entropy_zero"
                        if policy.force_repair
                        else "entropy",
                        current_token_recomputed_once=True,
                        stress_control=policy.force_repair,
                    )
                )
            report["provisional_first_decode_logit_sha"] = None
        after, details = original_run(*args, **kwargs)
        if overlays:
            report["repair_logit_max_change"] = float((before[1] - after[1]).abs().max())
            report["repair_token_changed"] = int(before[1].argmax()) != int(after[1].argmax())
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
            int(o.actual_fills) < 2 * (kwargs["tokens"] + policy.provisional_tokens - 2)
            for o in overlays.values()
        ):
            raise RuntimeError("prototype reader did not engage for every provisional/final step")
        return after, details

    with ExitStack() as stack:
        stack.enter_context(patch.object(constructor, "StreamingDirectSource", Encoder))
        stack.enter_context(
            patch.object(SequentialSourceHistoryLayer, "materialize_component", fill)
        )
        stack.enter_context(patch.object(storage_runs, "cache_from_formats", cache))
        stack.enter_context(patch.object(storage_runs, "run", run))
        yield report
