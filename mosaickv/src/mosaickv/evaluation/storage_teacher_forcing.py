"""Evaluation-only, complete teacher forcing through isolated storage formats.

Targets must never be passed into storage construction or online selection.
The one-target path performs no decode or speculative cache reservation.
"""

import math
import time
from contextlib import ExitStack
from dataclasses import replace

import torch

from mosaickv.adapters.huggingface.types import DecodeState
from mosaickv.evaluation.lossless_prompt_cache import IsolatedStorageTransport, backing_bytes
from mosaickv.evaluation.storage_runs import cache_from_formats


def validate_targets(target_ids, *, maximum=1024):
    if type(maximum) is not int or not 1 <= maximum <= 4097:
        raise ValueError("explicit teacher-forcing limit in [1,4097] required")
    if (
        not isinstance(target_ids, tuple)
        or not 1 <= len(target_ids) <= maximum
        or any(type(t) is not int or t < 0 for t in target_ids)
    ):
        raise ValueError("complete nonempty integer target tuple within registered limit required")
    return target_ids


def _score(
    adapter, prefill, state, target_ids, *, cache=None, sequential=True, return_logits=False
):
    device = prefill.logits.device
    gpu = device.type == "cuda"
    event0 = torch.cuda.Event(enable_timing=True) if gpu else None
    event1 = torch.cuda.Event(enable_timing=True) if gpu else None
    if gpu:
        torch.cuda.synchronize(device)
    begun = time.monotonic()
    losses, argmaxes, logits = [], [], []
    current = prefill.logits.detach().reshape(-1).float()
    with torch.inference_mode(), ExitStack() as stack:
        if cache is not None:
            stack.enter_context(IsolatedStorageTransport(adapter, cache))
            if sequential:
                from mosaickv.evaluation.sequential_kv_lease import SequentialAttentionTransport

                stack.enter_context(SequentialAttentionTransport(adapter, cache))
        if event0 is not None:
            event0.record()
        for index, target in enumerate(target_ids):
            if not torch.isfinite(current).all() or target >= current.numel():
                raise ValueError("nonfinite logits or target outside vocabulary")
            losses.append(-torch.log_softmax(current, dim=-1)[target])
            argmaxes.append(current.argmax())
            if return_logits:
                logits.append(current.cpu())
            if index + 1 < len(target_ids):
                token = torch.tensor([[target]], dtype=torch.long, device=adapter.device)
                output = adapter.decode_one_token(token, state, capture_queries=False)
                current = output.logits.detach().reshape(-1).float()
        if event1 is not None:
            event1.record()
    if gpu:
        torch.cuda.synchronize(device)
    wall = time.monotonic() - begun
    per_token = torch.stack(losses).cpu().tolist()
    nll = math.fsum(per_token)
    if not math.isfinite(nll) or nll < 0:
        raise RuntimeError("invalid teacher-forced likelihood")
    record = dict(
        nll_sum=nll,
        evaluated_token_count=len(target_ids),
        per_token_nll=per_token,
        conditional_greedy_tokens=torch.stack(argmaxes).cpu().tolist(),
        target_ids=list(target_ids),
        target_mask="all_annotation_tokens_no_prompt_no_added_EOS",
        no_EOS_stop=True,
        no_truncation=True,
        evaluation_only=True,
        wall_seconds=wall,
        cuda_ms=event0.elapsed_time(event1) if event0 is not None else None,
        updates=len(target_ids) - 1,
    )
    if (
        state is not None
        and state.next_decode_position != prefill.state.next_decode_position + len(target_ids) - 1
    ):
        raise RuntimeError("teacher-forcing logical position drift")
    if cache is not None:
        storage = cache.storage_record()
        if storage["outstanding_layer_leases"] or storage["history_updates_per_layer"] != [
            len(target_ids) - 1
        ] * len(cache.layers):
            raise RuntimeError("teacher-forcing storage/lease isolation failure")
        record["storage"] = storage
    if return_logits:
        return record, torch.stack(logits)
    return record


def score_formats(
    adapter, prefill, formats, target_ids, *, maximum=1024, sequential=True, return_logits=False
):
    targets = validate_targets(target_ids, maximum=maximum)
    if not formats or type(sequential) is not bool:
        raise ValueError("complete storage formats and explicit transport policy required")
    if len(targets) == 1:
        result = _score(adapter, prefill, None, targets, return_logits=return_logits)
        record = result[0] if return_logits else result
        count = backing_bytes(tuple(t for f in formats for t in f.tensors()))
        record["storage"] = dict(
            initial_active_kv_storage_bytes=count,
            final_active_kv_storage_bytes=count,
            peak_active_kv_storage_bytes=count,
            outstanding_layer_leases=0,
            history_updates_per_layer=[0] * len(formats),
            no_decode_cache_allocated=True,
            whole_request_peak_certified=False,
        )
        return result
    cache = cache_from_formats(
        adapter,
        formats,
        updates=len(targets) - 1,
        audit=False,
        sealed=False,
        sequential_lease=sequential,
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
    return _score(
        adapter,
        prefill,
        state,
        targets,
        cache=cache,
        sequential=sequential,
        return_logits=return_logits,
    )


def score_ordinary(adapter, prefill, target_ids, *, maximum=1024, return_logits=False):
    targets = validate_targets(target_ids, maximum=maximum)
    state = None
    if len(targets) > 1:
        snapshot = adapter.extract_past_key_values(prefill.state.past_key_values, clone=True)
        state = replace(
            prefill.state,
            past_key_values=adapter.inject_past_key_values(snapshot),
            attention_mask=prefill.state.attention_mask.clone(),
            model_state=dict(prefill.state.model_state),
        )
    return _score(adapter, prefill, state, targets, return_logits=return_logits)
