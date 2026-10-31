"""Consumed-input controls, not a production backend or paper promotion.

Run ordinary selected BF16 HF caches, and an unchanged precision prompt with
raw generated history, through the same complete-request measurement boundary.
Scoped function substitution is single-request-only and restored on failure.
No model weights, attention formulas, source positions, or selectors are edited.
"""

import json
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from mosaickv.evaluation import request_precision, storage_runs
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

_LOCK = threading.Lock()


@dataclass
class NativePayload:
    snapshot: Any
    head_positions: tuple[Any, ...] = ()


def construct_native(
    adapter: Any,
    prepared: Any,
    prefill: Any,
    capture: Any,
    cfg: Any,
    method: str,
    profile_path: str,
    ratio: float,
    precision: str,
) -> tuple[Any, dict[str, Any]]:
    from mosaickv.adapters.huggingface.types import CacheLayerSnapshot
    from mosaickv.evaluation.storage_baselines import baseline_payload

    if precision != "lossless":
        raise ValueError("ordinary BF16 control cannot quantize prompt values")
    profile = tuple(json.loads(Path(profile_path).read_text()))
    frozen, online = request_precision.online_frozen(
        prefill,
        profile,
        ratio,
        method=method,
        priors=capture.priors if capture is not None and capture.summaries else None,
    )
    positions, transformed, proof = baseline_payload(
        adapter, prepared, prefill, cfg, method, frozen, profile_path, borrow_source=True
    )
    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    pairs = []
    for i, (key, value) in enumerate(source):
        if transformed is not None:
            pairs.append(transformed[i])
        else:
            heads = positions[i]
            if len({len(h) for h in heads}) != 1:
                raise ValueError("ordinary control requires equal head lengths within each layer")
            index = torch.tensor(heads, device=key.device)[None, :, :, None].expand(
                1, key.shape[1], len(heads[0]), key.shape[-1]
            )
            pairs.append((key.gather(2, index), value.gather(2, index)))
    template = adapter.borrow_past_key_values(prefill.state.past_key_values)
    snapshot = replace(
        template,
        layers=tuple(CacheLayerSnapshot(k, v, 2) for k, v in pairs),
        active_sequence_length=max(k.shape[-2] for k, _ in pairs),
        independent_layer_lengths=True,
    )
    actual = backing_bytes(tuple(t for pair in pairs for t in pair))
    if actual != proof["source_slot_byte_cap"]:
        raise ValueError("ordinary selected cache did not spend the exact raw source capacity")
    memberships = (
        positions if positions is not None else proof["selected_source_positions_by_layer"]
    )
    return (NativePayload(snapshot, tuple(memberships)),), dict(
        selection=proof,
        online=online,
        precision="ordinary_BF16",
        source_codec="none",
        generated_history="ordinary_BF16_append_only",
        no_compressed_history_reconstruction=True,
    )


def run_native(adapter: Any, prefill: Any, formats: Any, **kwargs: Any) -> Any:
    from mosaickv.backends.hf_runtime import PackedRuntimeCache, _state_from_packed

    if kwargs.get("tokens") != 128 or len(formats) != 1:
        raise ValueError("bounded 128-prediction ordinary control required")
    payload = formats[0]
    snapshot = payload.snapshot
    if snapshot is None:
        raise ValueError("native payload can be consumed only once")
    packed = PackedRuntimeCache(
        snapshot,
        (),
        snapshot.active_sequence_length,
        True,
        (),
        (),
        (),
        tuple(tuple(heads[0]) for heads in payload.head_positions),
    )
    state = _state_from_packed(adapter, prefill.state, packed)
    state.model_state["native_prompt_positions_by_layer_head"] = payload.head_positions
    # Do not retain an immutable prompt copy after DynamicCache concatenates:
    # that would artificially double-count prompt storage throughout decoding.
    payload.snapshot = None
    del snapshot, packed

    def allocated() -> int:
        pairs, _ = adapter._legacy_layers(state.past_key_values)
        return backing_bytes(tuple(t for pair in pairs for t in pair))

    initial = peak = allocated()
    values = [prefill.logits.detach().reshape(-1).float()]
    token = prefill.next_token_id
    for _ in range(127):
        out = adapter.decode_one_token(token, state, capture_queries=False)
        values.append(out.logits.detach().reshape(-1).float())
        token = out.next_token_id
        peak = max(peak, allocated())
    return torch.stack(values).cpu(), dict(
        audit_enabled=False,
        native_cache_class=type(state.past_key_values).__qualname__,
        storage=dict(
            initial_active_kv_storage_bytes=initial,
            final_active_kv_storage_bytes=allocated(),
            peak_active_kv_storage_bytes=peak,
            boundary="native_live_KV_at_completed_decode_boundaries",
            KV_transaction_peak_certified=False,
            transient_cat_allocations_in_whole_request_allocator_peak=True,
        ),
    )


def complete_control(
    adapter: Any,
    messages: Any,
    cfg: Any,
    *,
    method: str,
    profile_path: str,
    ratio: float,
    mode: str,
) -> Any:
    if mode not in ("ordinary_selected_BF16", "query3_raw_history", "query3_source_history"):
        raise ValueError("unregistered storage-control mode")
    if method not in request_precision.METHODS:
        raise ValueError("registered compressed method required")
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("diagnostic substitutions do not support concurrent requests")
    try:
        arguments: dict[str, Any] = dict(
            method=method, profile_path=profile_path, ratio=ratio, audit=False
        )
        if mode == "ordinary_selected_BF16":
            with (
                patch.object(request_precision, "construct_request_formats", construct_native),
                patch.object(storage_runs, "run", run_native),
            ):
                logits, record = request_precision.complete_request(
                    adapter, messages, cfg, precision="lossless", **arguments
                )
        elif mode == "query3_raw_history":
            original_run = storage_runs.run

            def raw(*args: Any, **kwargs: Any) -> Any:
                kwargs.update(
                    source_history=False,
                    source_history_constructor="reference",
                    consolidated_history=False,
                    sealed=False,
                )
                return original_run(*args, **kwargs)

            with patch.object(storage_runs, "run", raw):
                logits, record = request_precision.complete_request(
                    adapter, messages, cfg, precision="query3", **arguments
                )
            record["online_variant"]["generated_history"] = "raw_BF16_append_only"
        else:
            logits, record = request_precision.complete_request(
                adapter, messages, cfg, precision="query3", **arguments
            )
        record.update(
            storage_control_mode=mode,
            diagnostic_only=True,
            same_consumed_inputs_not_independent_quality_evidence=True,
            performance_superiority_established=False,
            effective_history_mode=record["online_variant"]["generated_history"],
        )
        return logits, record
    finally:
        _LOCK.release()
