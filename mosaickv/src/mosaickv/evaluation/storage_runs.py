"""Shared isolated runtime for consumed-input storage/quality/latency diagnostics."""

from typing import Any, cast

import torch

from mosaickv.evaluation.exact_storage import ExactCache, ExactFormat, ExactLayer
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


def cache_from_formats(
    adapter: Any,
    formats: Any,
    *,
    updates: int,
    audit: bool,
    sealed: bool = True,
    batched_history: bool = False,
    source_history: bool = False,
    source_history_constructor: str = "reference",
    shared_workspace: bool = False,
    workspace_graph: bool = False,
    consolidated_history: bool = False,
    segmented_history: bool = False,
    sequential_lease: bool = False,
    sequential_source_history: bool = False,
    prompt_graph_reader: bool = False,
    prompt_graph_policy=None,
) -> ExactCache:
    from mosaickv.evaluation.sealed_history import SealedHistoryLayer

    if source_history_constructor not in ("reference", "direct", "direct_batched_checks") or (
        not source_history and source_history_constructor != "reference"
    ):
        raise ValueError("explicit source history required for its constructor policy")
    if type(shared_workspace) is not bool or type(workspace_graph) is not bool:
        raise ValueError("shared workspace and graph policies must be explicit booleans")
    if type(consolidated_history) is not bool or (
        consolidated_history
        and (
            not source_history
            or shared_workspace
            or source_history_constructor not in ("direct", "direct_batched_checks")
        )
    ):
        raise ValueError(
            "consolidation requires explicit direct source history without workspace graphs"
        )
    if (workspace_graph and not shared_workspace) or (shared_workspace and not source_history):
        raise ValueError("workspace graphs require an explicit shared source-history workspace")
    if type(segmented_history) is not bool or (
        segmented_history
        and (
            not source_history
            or source_history_constructor not in ("direct", "direct_batched_checks")
            or shared_workspace
            or consolidated_history
        )
    ):
        raise ValueError("segmented history requires only direct source history")
    if type(sequential_source_history) is not bool or (
        sequential_source_history
        and (
            not sequential_lease
            or not source_history
            or not sealed
            or not consolidated_history
            or source_history_constructor not in ("direct", "direct_batched_checks")
            or shared_workspace
            or workspace_graph
            or segmented_history
            or batched_history
            or type(updates) is not int
            or not 1 <= updates <= 1024
        )
    ):
        raise ValueError(
            "sequential source history requires explicit isolated consolidated direct history"
        )
    if type(sequential_lease) is not bool or (
        sequential_lease
        and not sequential_source_history
        and (
            sealed
            or batched_history
            or source_history
            or shared_workspace
            or consolidated_history
            or segmented_history
        )
    ):
        raise ValueError("sequential lease requires explicit raw append-only history")
    cache = ExactCache(audit_workspace=audit)
    if prompt_graph_policy is not None and not prompt_graph_reader:
        raise ValueError("graph optimization policy requires the explicit graph reader")
    if type(prompt_graph_reader) is not bool or (
        prompt_graph_reader and (not sequential_source_history or updates < 2)
    ):
        raise ValueError("prompt graph requires explicit bounded sequential source history")
    cls = SealedHistoryLayer if sealed else ExactLayer
    if sequential_lease:
        from mosaickv.evaluation.sequential_kv_lease import SequentialLayer

        cls = SequentialLayer
    if batched_history:
        if not sealed:
            raise ValueError("batched history requires lossless sealed storage")
        from mosaickv.evaluation.batched_history import BatchedHistoryLayer

        cls = BatchedHistoryLayer
    if source_history:
        if not sealed or batched_history or (updates != 128 and not sequential_source_history):
            raise ValueError("source history requires its explicit sealed 128-token diagnostic")
        from mosaickv.evaluation.source_history import SourceHistoryCache, SourceHistoryLayer

        cache = SourceHistoryCache(audit_workspace=audit, constructor=source_history_constructor)
        cls = SourceHistoryLayer
        if sequential_source_history:
            from mosaickv.evaluation.sequential_source_history import SequentialSourceHistoryLayer

            cls = SequentialSourceHistoryLayer
            cache.sequential_source_history = True
        if segmented_history:
            from mosaickv.evaluation.segmented_history import SegmentedSourceLayer

            cls = SegmentedSourceLayer
        if shared_workspace:
            from mosaickv.evaluation.graphed_storage import SharedSourceCache, SharedSourceLayer

            cache = SharedSourceCache(
                audit_workspace=audit,
                constructor=source_history_constructor,
                graphs=workspace_graph,
            )
            cls = SharedSourceLayer
    if prompt_graph_reader:
        from mosaickv.evaluation.graph_accounting import layer_class
        from mosaickv.evaluation.prompt_graph_reader import PromptGraphCache

        cache = PromptGraphCache(
            audit_workspace=audit,
            constructor=source_history_constructor,
            policy=prompt_graph_policy,
        )
        cache.sequential_source_history = True
        cls = layer_class()
    shared: list[Any] = []
    from mosaickv.evaluation.layer_code_delta import bind

    formats = bind(formats)
    for i, format in enumerate(formats):
        if hasattr(format, "fork_for_decode"):
            format = format.fork_for_decode(updates)
        shared.extend(format.shared())
        layer = cls(
            format,
            adapter._language_layers()[i].self_attn,
            observe=cache.observe,
            direct=True,
            updates=updates,
            audit=audit,
            **({"cache": cache} if shared_workspace else {}),
        )
        cache.layers.append(layer)
        if prompt_graph_reader:
            layer.prompt_graph_pool = cache.prompt_graph_pool
        cache.build_records.append(
            dict(layer=i, format=format.kind, component_bytes=format.component_bytes)
        )
    cache.shared_tensors = tuple(shared)
    cache.shared_bytes = backing_bytes(cache.shared_tensors)
    cache.initial_bytes = backing_bytes(
        tuple(t for x in cache.layers for t in cast(ExactLayer, x).owned_tensors())
    )
    cache.observe()
    from mosaickv.evaluation.graph_lookahead import attach as attach_lookahead

    attach_lookahead(cache)
    from mosaickv.evaluation.prompt_prefetch import attach

    attach(cache, updates)
    return cache


def run(
    adapter: Any,
    prefill: Any,
    formats: tuple[ExactFormat, ...],
    *,
    tokens: int = 32,
    forced: tuple[int, ...] | None = None,
    audit: bool = True,
    sealed: bool = True,
    batched_history: bool = False,
    source_history: bool = False,
    source_history_constructor: str = "reference",
    shared_workspace: bool = False,
    workspace_graph: bool = False,
    consolidated_history: bool = False,
    segmented_history: bool = False,
    history_reference: Any = None,
    history_repair_enabled: bool = True,
    sequential_lease: bool = False,
    sequential_source_history: bool = False,
    prompt_graph_reader: bool = False,
    prompt_graph_policy=None,
) -> tuple[Any, dict[str, Any]]:
    import time

    from mosaickv.adapters.huggingface.types import DecodeState
    from mosaickv.evaluation.lossless_prompt_cache import IsolatedStorageTransport

    if (
        type(tokens) is not int
        or not 2 <= tokens <= 128
        or (forced is not None and len(forced) != tokens)
    ):
        raise ValueError(
            "bounded 2--128 tokens and complete optional FullKV-generated prefix required"
        )
    if type(history_repair_enabled) is not bool or (
        history_reference is not None and (sealed or source_history or batched_history)
    ):
        raise ValueError("draft history diagnostic requires explicit raw append-only history")
    cache: Any = cache_from_formats(
        adapter,
        formats,
        updates=tokens,
        audit=audit,
        sealed=sealed,
        batched_history=batched_history,
        source_history=source_history,
        source_history_constructor=source_history_constructor,
        shared_workspace=shared_workspace,
        workspace_graph=workspace_graph,
        consolidated_history=consolidated_history,
        segmented_history=segmented_history,
        sequential_lease=sequential_lease,
        sequential_source_history=sequential_source_history,
        prompt_graph_reader=prompt_graph_reader,
        prompt_graph_policy=prompt_graph_policy,
    )
    length = max(f.shape[-2] for f in formats)
    state = DecodeState(
        cache,
        torch.ones(1, length, dtype=torch.long, device=adapter.device),
        length,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
        prefill.state.modality_map,
        dict(
            prefill.state.model_state, mosaickv_packed_all_valid=True, mosaickv_cache_all_valid=True
        ),
    )
    values = [prefill.logits.detach().reshape(-1).float()]
    token = prefill.next_token_id
    event: Any = torch.cuda.Event
    starts, ends = event(enable_timing=True), event(enable_timing=True)
    torch.cuda.synchronize()
    started = time.monotonic()
    from contextlib import ExitStack

    with ExitStack() as stack:
        if prompt_graph_reader:
            stack.callback(cache.prompt_graph_pool.close)
        history_repair = None
        if history_reference is not None:
            from mosaickv.evaluation.draft_history_causal import HistoryRepairTransport

            history_repair = stack.enter_context(
                HistoryRepairTransport(
                    adapter, cache, history_reference, enabled=history_repair_enabled
                )
            )
        if shared_workspace:
            stack.callback(cache.release_workspace)
        stack.enter_context(IsolatedStorageTransport(adapter, cache))
        if sequential_lease:
            from mosaickv.evaluation.sequential_kv_lease import SequentialAttentionTransport

            stack.enter_context(SequentialAttentionTransport(adapter, cache))
        from mosaickv.evaluation.exact_head_bypass import ExactHeadTransport, HeadBypassFormat

        bypass = None
        if any(isinstance(layer.representation, HeadBypassFormat) for layer in cache.layers):
            bypass = stack.enter_context(ExactHeadTransport(adapter, cache))
        if any(hasattr(f, "valid_prompt_counts") for f in formats):
            from mosaickv.evaluation.completion_masks import CompletionMaskTransport

            stack.enter_context(CompletionMaskTransport(adapter, cache))
        if source_history:
            from mosaickv.evaluation.source_history import SourceInputTransport

            stack.enter_context(SourceInputTransport(adapter, cache))
        from mosaickv.evaluation.joint_incoming_projection import transport as joint_transport

        stack.enter_context(joint_transport(adapter, cache))
        from mosaickv.evaluation.post_attention_segment import transport as segment_transport

        segment = stack.enter_context(segment_transport(adapter, cache))
        starts.record()
        for step in range(tokens - 1):
            if source_history:
                from mosaickv.evaluation.source_history import seal_source_full_tails

                if shared_workspace and any(layer.tail_used == 16 for layer in cache.layers):
                    cache.release_workspace()
                sealed_now = seal_source_full_tails(cache)
                if consolidated_history and sealed_now:
                    from mosaickv.evaluation.consolidated_history import consolidate

                    consolidate(cache)
                if segmented_history and sealed_now:
                    from mosaickv.evaluation.segmented_history import prepare_tables

                    prepare_tables(cache)
            if batched_history:
                from mosaickv.evaluation.batched_history import seal_full_tails

                seal_full_tails(cache)
            if forced is not None:
                token = torch.tensor([[forced[step]]], device=adapter.device, dtype=torch.long)
            output = adapter.decode_one_token(token, state, capture_queries=False)
            values.append(output.logits.detach().reshape(-1).float())
            token = output.next_token_id
        if shared_workspace:
            cache.release_workspace()
        if prompt_graph_reader:
            if segment is not None:
                segment.close()
            cache.prompt_graph_pool.close()
        ends.record()
    torch.cuda.synchronize()
    wall = time.monotonic() - started
    milliseconds = starts.elapsed_time(ends)
    logits = torch.stack(values).cpu()
    if (
        not torch.isfinite(logits).all()
        or state.next_decode_position != prefill.state.next_decode_position + tokens - 1
    ):
        raise RuntimeError("nonfinite output or changed logical decode positions")
    record = cache.storage_record()
    if history_repair is not None:
        record["draft_history_repair"] = history_repair.record()
    if bypass is not None:
        record.update(
            exact_head_bypass_recomputed_heads=bypass.recomputed_heads,
            exact_head_bypass_private_state=True,
            full_head_history_reservation_charged_from_initial_boundary=True,
            main_selected_KV_unchanged_virtual_head_context_added=True,
            inactive_duplicate_main_head_storage_still_charged=True,
            exact_head_native_MHA_batch_shape_preserved=True,
            exact_head_numerical_differences=getattr(cache, "exact_head_numerical_differences", []),
        )
    if any(hasattr(f, "valid_prompt_counts") for f in formats):
        record.update(
            unchanged_original_selected_memberships=False,
            padded_completion_membership_variant=True,
            original_selected_values_preserved_but_membership_extended=True,
        )
    record.update(
        sequential_kv_lease=sequential_lease,
        sequential_component_fills=[getattr(layer, "component_fills", 0) for layer in cache.layers],
        native_full_source_decodes=sum(
            getattr(layer, "native_full_source_decodes", 0) for layer in cache.layers
        ),
        native_full_component_fills=sum(
            getattr(layer, "native_full_component_fills", 0) for layer in cache.layers
        ),
        native_fused_restore_fills=sum(
            getattr(layer, "native_fused_restore_fills", 0) for layer in cache.layers
        ),
        segmented_history=segmented_history,
        sequential_source_history=sequential_source_history,
        segment_table_builds=getattr(cache, "segment_table_builds", 0),
        source_check_flags_checked=getattr(cache, "source_check_flags_checked", 0),
        source_check_syncs=getattr(cache, "source_check_syncs", 0),
    )
    if record["outstanding_layer_leases"] or record["history_updates_per_layer"] != [
        tokens - 1
    ] * len(formats):
        raise RuntimeError("cache state/lease isolation failure")
    return logits, dict(
        storage=record,
        decode_wall_seconds=wall,
        decode_cuda_ms=milliseconds,
        audit_enabled=audit,
        sealed_history=sealed,
        batched_history=batched_history,
        source_history=source_history,
        source_history_constructor=source_history_constructor,
        shared_workspace=shared_workspace,
        workspace_graph=workspace_graph,
        consolidated_history=consolidated_history,
        segmented_history=segmented_history,
        batched_history_seals=getattr(cache, "batched_history_seals", 0),
        batched_history_encode_batches=getattr(cache, "batched_history_encode_batches", 0),
        output_tokens=tokens,
        fixed_reference_prefix=forced is not None,
        used_fullkv_fallback=False,
        original_logical_length=prefill.state.logical_sequence_length,
        final_next_position=state.next_decode_position,
        timing_excludes_prefill_selection_constructor=True,
    )


def source_topup(
    positions: Any, probabilities: Any, *, slots: int, delimiter: tuple[int, ...] = ()
) -> tuple[Any, dict[str, Any]]:
    """Strict supersets using only source rows already archived for that layer.

    No donor removal and no benchmark/reference-answer access. Uniform per-head
    growth is feasible only to the smallest available omitted-source count.
    """
    from mosaickv.evaluation.exact_storage import token_partition

    if type(slots) is not int or slots < 1:
        raise ValueError("positive integer source-sharing topup required")
    if len(positions) != len(probabilities):
        raise ValueError("complete prompt-derived layer attention required")
    result, events = [], []
    for layer, (heads, attention) in enumerate(zip(positions, probabilities, strict=True)):
        source, _ = token_partition(heads)
        source_set = set(source)
        maximum = min(len(source_set - set(h)) for h in heads)
        count = min(slots, maximum)
        mass = attention.float().mean((0, 2)).cpu()
        if mass.shape[0] != len(heads) or not torch.isfinite(mass).all():
            raise ValueError("prompt attention does not match MHA routes")
        rows = []
        for head, selected in enumerate(heads):
            candidates = source_set - set(selected)
            ranked = sorted(
                candidates, key=lambda p: (p not in delimiter, -float(mass[head, p]), p)
            )
            added = tuple(ranked[:count])
            row = tuple(sorted((*selected, *added)))
            if len(row) != len(selected) + count or not set(selected).issubset(row):
                raise RuntimeError("source-sharing topup removed or duplicated evidence")
            rows.append(row)
            events.append(
                dict(
                    layer=layer,
                    head=head,
                    added_positions=list(added),
                    added_count=count,
                    removed_count=0,
                    already_archived_source=True,
                )
            )
        result.append(tuple(rows))
    return tuple(result), dict(
        requested_extra_slots=slots,
        events=events,
        no_donor_evictions=True,
        no_benchmark_label_access=True,
    )
