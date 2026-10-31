"""Label-blind short-horizon prompt demand from an isolated FullKV draft.

This successor diagnostic is not part of the frozen balanced validation. It
reuses the original prefill, copies its cache with the audited forecasting
bridge, and discards draft tokens/cache. No reference artifact is accepted.
"""

from typing import Any

import torch


def reduce_prompt_mass(
    attention_steps: Any, *, prompt_length: int, heads: tuple[int, ...]
) -> tuple[Any, ...]:
    """Preserve individual head demand, normalize away draft-history mass."""
    if len(attention_steps) != 4 or prompt_length < 1 or not heads:
        raise ValueError("registered four-step draft and positive prompt/head layout required")
    if any(len(step) != len(heads) for step in attention_steps):
        raise ValueError("draft attention layer count changed")
    layers = []
    for layer, count in enumerate(heads):
        demands = []
        for step, attentions in enumerate(attention_steps):
            p = attentions[layer]
            if (
                tuple(p.shape) != (1, count, 1, prompt_length + step + 1)
                or not bool(torch.isfinite(p).all())
                or bool((p < 0).any())
            ):
                raise ValueError("finite batch-one MHA draft attention/layout required")
            prompt = p[0, :, 0, :prompt_length].detach().float()
            mass = prompt.sum(-1, keepdim=True)
            if bool((mass <= 0).any()):
                raise ValueError("draft head has no positive original-prompt demand")
            demands.append(prompt / mass)
        layers.append(torch.stack(demands).mean(0))
    return tuple(layers)


def collect(adapter: Any, prefill: Any) -> tuple[Any, dict[str, Any]]:
    from mosaickv.forecasting.huggingface import run_isolated_greedy_query_rollout

    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    length = int(source[0][0].shape[-2])
    heads = tuple(int(pair[0].shape[1]) for pair in source)
    if any(int(pair[0].shape[-2]) != length for pair in source):
        raise ValueError("full uniform prompt lengths required before forecasting")
    metadata = (
        prefill.state.active_cache_length,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
    )
    rollout = run_isolated_greedy_query_rollout(
        adapter, prefill, steps=4, capture_attentions=True, capture_distributions=False
    )
    demand = reduce_prompt_mass(rollout.attention_steps, prompt_length=length, heads=heads)
    if metadata != (
        prefill.state.active_cache_length,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
    ):
        raise RuntimeError("draft changed final-run logical/physical positions")
    return demand, dict(
        mode="four_step_fullkv_draft_prompt_demand",
        steps=4,
        original_prefill_reused=True,
        second_prefill=False,
        original_cache_unchanged=rollout.original_cache_unchanged,
        clone_and_integrity_seconds=rollout.cache_clone_seconds,
        draft_decode_seconds=rollout.draft_decode_seconds,
        synchronization_calls=rollout.synchronization_calls,
        draft_token_ids=rollout.token_ids,
        benchmark_labels_or_references_used=False,
        query_heads_preserved=True,
        history_mass_removed_before_normalization=True,
        included_in_selection_and_complete_request_timing=True,
    )
