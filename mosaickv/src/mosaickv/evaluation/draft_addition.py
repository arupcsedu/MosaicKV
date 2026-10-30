"""Prospective restore-only layer slots from short FullKV draft attention.

No reference answers or evaluation-prefix logits are accepted by these APIs.
Original members are never removed. This is a separately labeled diagnostic.
"""

from typing import Any

import torch


def per_head_addition_gains(probabilities: Any, values: Any, heads: Any, source_length: int) -> Any:
    """Signed output-error reduction from one extra original prompt token.

    Keep all already generated draft history, account for softmax
    renormalization, and normalize each query/head by full output magnitude.
    Return independent per-head/per-position gains; never pool queries first.
    """
    if (
        probabilities.ndim != 4
        or probabilities.shape[0] != 1
        or probabilities.shape[1] != len(heads)
        or values.ndim != 4
        or values.shape[:2] != probabilities.shape[:2]
        or values.shape[-2] != probabilities.shape[-1]
        or not 0 < source_length <= values.shape[-2]
    ):
        raise ValueError("batch-one MHA attention/value/original-source geometry required")
    p = probabilities[0].float()
    v = values[0].float()
    if not torch.isfinite(p).all() or not torch.isfinite(v).all():
        raise ValueError("finite model-derived attention and values required")
    mask = torch.zeros((len(heads), v.shape[1]), dtype=torch.bool, device=v.device)
    for head, selected in enumerate(heads):
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(not 0 <= i < source_length for i in selected)
        ):
            raise ValueError("complete unique original prompt membership required")
        mask[head, list(selected)] = True
    mask[:, source_length:] = True
    full = torch.einsum("hqn,hnd->hqd", p, v)
    retained_p = p * mask[:, None, :]
    mass = retained_p.sum(-1)
    if not (mass > 0).all():
        raise ValueError("retained prompt/history mass must be positive")
    retained = torch.einsum("hqn,hnd->hqd", retained_p, v) / mass[..., None]
    residual = full - retained
    baseline = residual.square().sum(-1)
    scale = full.square().sum(-1).clamp_min(1e-12)
    gains = torch.zeros((len(heads), source_length), device=v.device)
    # Bounded query chunks avoid H*Q*S*D for long prompt windows.
    for query in range(p.shape[1]):
        a = p[:, query, :source_length]
        fraction = a / (mass[:, query, None] + a).clamp_min(1e-12)
        delta = (v[:, :source_length] - retained[:, query, None, :]) * fraction[..., None]
        error = (residual[:, query, None, :] - delta).square().sum(-1)
        gains += (baseline[:, query, None] - error) / scale[:, query, None] / p.shape[1]
    gains.masked_fill_(mask[:, :source_length], -torch.inf)
    return gains


def collect_draft_gains(
    adapter: Any, prefill: Any, positions: Any, *, steps: int = 4
) -> tuple[Any, dict[str, Any]]:
    from mosaickv.backends.hf_runtime import _clone_decode_state

    if type(steps) is not int or not 1 <= steps <= 4:
        raise ValueError("preregistered one-to-four-step temporary draft required")
    state = _clone_decode_state(adapter, prefill.state)
    token = prefill.next_token_id
    sums: list[Any] = [None] * len(positions)
    token_ids = []
    for _ in range(steps):
        token_ids.append(int(token.item()))
        output = adapter.decode_one_token(
            token, state, capture_queries=False, capture_attentions=True
        )
        pairs, _ = adapter._legacy_layers(output.state.past_key_values)
        for layer, (attention, (_, values), selected) in enumerate(
            zip(output.attention_weights, pairs, positions, strict=True)
        ):
            gain = per_head_addition_gains(
                attention, values, selected, prefill.state.logical_sequence_length
            )
            sums[layer] = gain if sums[layer] is None else sums[layer] + gain
        token = output.next_token_id
    return tuple(g / steps for g in sums), dict(
        draft_steps=steps,
        draft_token_ids=token_ids,
        original_prefill_reused=True,
        second_prefill=False,
        model_weights_updated=False,
        benchmark_label_access=False,
        future_evaluation_logits_used=False,
    )


def layer_addition_plans(positions: Any, gains: Any, *, maximum: int = 8) -> tuple[Any, ...]:
    if len(positions) != len(gains) or type(maximum) is not int or not 1 <= maximum <= 8:
        raise ValueError("complete gains and bounded one-to-eight plans required")
    proposals: list[dict[str, Any]] = []
    for layer, (heads, values) in enumerate(zip(positions, gains, strict=True)):
        scores = values.detach().double().cpu()
        if (
            scores.ndim != 2
            or scores.shape[0] != len(heads)
            or torch.isnan(scores).any()
            or torch.isposinf(scores).any()
        ):
            raise ValueError("independent finite-or-negative-infinite per-head gains required")
        added: list[int] = []
        score = 0.0
        for head, selected in enumerate(heads):
            row = scores[head].clone()
            row[list(selected)] = -torch.inf
            position = int(row.argmax())
            if not torch.isfinite(row[position]):
                added = []
                break
            added.append(position)
            score += float(row[position])
        if not added or score <= 0:
            continue
        revised = tuple(
            tuple(tuple(h) for h in old)
            if i != layer
            else tuple(tuple(sorted((*h, p))) for h, p in zip(old, added, strict=True))
            for i, old in enumerate(positions)
        )
        proposals.append(
            dict(
                layer=layer,
                prospective_normalized_gain=score,
                added_positions=added,
                positions=revised,
                original_evidence_removed=False,
                one_layer_slot_only=True,
            )
        )
    return tuple(
        sorted(proposals, key=lambda p: (-p["prospective_normalized_gain"], p["layer"]))[:maximum]
    )
