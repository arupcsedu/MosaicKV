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
) -> ExactCache:
    from mosaickv.evaluation.sealed_history import SealedHistoryLayer

    if source_history_constructor not in ("reference", "direct") or (
        not source_history and source_history_constructor != "reference"
    ):
        raise ValueError("explicit source history required for its constructor policy")
    cache = ExactCache(audit_workspace=audit)
    cls = SealedHistoryLayer if sealed else ExactLayer
    if batched_history:
        if not sealed:
            raise ValueError("batched history requires lossless sealed storage")
        from mosaickv.evaluation.batched_history import BatchedHistoryLayer

        cls = BatchedHistoryLayer
    if source_history:
        if not sealed or batched_history or updates != 128:
            raise ValueError("source history requires its explicit sealed 128-token diagnostic")
        from mosaickv.evaluation.source_history import SourceHistoryCache, SourceHistoryLayer

        cache = SourceHistoryCache(audit_workspace=audit, constructor=source_history_constructor)
        cls = SourceHistoryLayer
    shared: list[Any] = []
    for i, format in enumerate(formats):
        shared.extend(format.shared())
        layer = cls(
            format,
            adapter._language_layers()[i].self_attn,
            observe=cache.observe,
            direct=True,
            updates=updates,
            audit=audit,
        )
        cache.layers.append(layer)
        cache.build_records.append(
            dict(layer=i, format=format.kind, component_bytes=format.component_bytes)
        )
    cache.shared_tensors = tuple(shared)
    cache.shared_bytes = backing_bytes(cache.shared_tensors)
    cache.initial_bytes = backing_bytes(
        tuple(t for x in cache.layers for t in cast(ExactLayer, x).owned_tensors())
    )
    cache.observe()
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
    cache = cache_from_formats(
        adapter,
        formats,
        updates=tokens,
        audit=audit,
        sealed=sealed,
        batched_history=batched_history,
        source_history=source_history,
        source_history_constructor=source_history_constructor,
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
        stack.enter_context(IsolatedStorageTransport(adapter, cache))
        if source_history:
            from mosaickv.evaluation.source_history import SourceInputTransport

            stack.enter_context(SourceInputTransport(adapter, cache))
        starts.record()
        for step in range(tokens - 1):
            if source_history:
                from mosaickv.evaluation.source_history import seal_source_full_tails

                seal_source_full_tails(cache)
            if batched_history:
                from mosaickv.evaluation.batched_history import seal_full_tails

                seal_full_tails(cache)
            if forced is not None:
                token = torch.tensor([[forced[step]]], device=adapter.device, dtype=torch.long)
            output = adapter.decode_one_token(token, state, capture_queries=False)
            values.append(output.logits.detach().reshape(-1).float())
            token = output.next_token_id
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
