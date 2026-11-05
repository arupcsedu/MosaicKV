"""Label-blind, storage-funded exact exceptions for a lossy prompt variant."""

import math
from dataclasses import replace
from typing import Any

import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import PrecisionFormat


def stable_largest(score: Any, limit: int) -> Any:
    if (
        type(limit) is not int
        or limit < 0
        or score.ndim != 1
        or not bool(torch.isfinite(score).all())
        or bool((score < 0).any())
    ):
        raise ValueError("finite nonnegative flat scores and nonnegative integer limit required")
    limit = min(limit, score.numel())
    if not limit:
        return torch.empty(0, dtype=torch.int32, device=score.device)
    cutoff = torch.topk(score, limit, sorted=False).values.min()
    higher = (score > cutoff).nonzero().flatten()
    equal = (score == cutoff).nonzero().flatten()
    selected = torch.cat((higher, equal[: limit - higher.numel()]))
    selected = selected[score[selected] > 0].sort().values
    return selected.to(torch.int32)


def fund_corrections(
    fmt: PrecisionFormat,
    original: tuple[Any, ...],
    *,
    fraction: float,
    key_weight: Any = None,
    value_weight: Any = None,
) -> tuple[PrecisionFormat, dict[str, Any]]:
    from mosaickv.evaluation.deferred_exact_encoding import finalize

    # Storage-funded choices must see the original encoded bytes, not raw
    # constructor intermediates. Never estimate or change the exception budget.
    fmt = finalize(fmt)
    if (
        not fmt.encoded_exact
        or fmt.corrections
        or len(original) != 2
        or any(
            tuple(t.shape) != fmt.shape
            or t.dtype != torch.bfloat16
            or t.device != fmt.pair[0].codes.device
            for t in original
        )
        or original[0].numel() >= 2**31
    ):
        raise ValueError("uncorrected encoded-exact precision packet and original pair required")
    if (
        type(fraction) not in (float, int)
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 0.005
    ):
        raise ValueError("fixed correction fraction must be in [0, 0.005]")
    h, n, d = fmt.shape[1:]
    p = fmt.exact_indices.numel()
    saved = (
        h * p * d * 4 + n * 4 - backing_bytes(tuple(t for b in fmt.exact_values for t in b.buffers))
    )
    # Six bytes per original scalar (int32 location + unchanged BF16 value).
    limit = min(math.floor(original[0].numel() * fraction), max(0, saved) // 12)
    reconstructed = fmt.materialize()
    patches, records = [], []
    for i, (reference, approximate, weight) in enumerate(
        zip(original, reconstructed, (key_weight, value_weight), strict=True)
    ):
        error = (reference.float() - approximate.float()).abs()
        before = float(error.square().sum())
        if weight is not None:
            if (
                weight.shape != ((1, h, 1, d) if i == 0 else (1, h, n, 1))
                or not bool(torch.isfinite(weight).all())
                or bool((weight < 0).any())
            ):
                raise ValueError("finite nonnegative prompt-query/attention weights required")
            score = error * weight
        else:
            score = error
        indices = stable_largest(score.flatten(), limit)
        values = reference.contiguous().flatten().index_select(0, indices.long()).clone()
        patches.append(SparseBitPatch(indices, values))
        remaining = error.square().flatten()
        remaining[indices.long()] = 0
        records.append(
            dict(
                component="key" if i == 0 else "value",
                exceptions=indices.numel(),
                squared_error_before=before,
                squared_error_after=float(remaining.sum()),
            )
        )
    corrected = replace(fmt, corrections=tuple(patches))
    paid = backing_bytes(tuple(t for patch in patches for t in patch.tensors))
    if paid > saved:
        raise RuntimeError("correction exceptions exceeded saved exact-text storage")
    return corrected, dict(
        saved_text_and_map_bytes=saved,
        exception_bytes=paid,
        raw_text_equivalent_bytes=fmt.component_bytes + saved,
        corrected_component_bytes=corrected.component_bytes,
        no_budget_increase=True,
        fraction=fraction,
        components=records,
    )


def prompt_weights(
    attention: Any, record: Any, probabilities: Any, positions: Any, *, window: int = 16
) -> tuple[Any, Any]:
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    if type(window) is not int or window < 1:
        raise ValueError("positive integer prompt window required")

    hidden = record["normalized"][:, -window:]
    heads, dimension = len(positions), attention.head_dim
    with torch.inference_mode():
        query = attention.q_proj(hidden).view(1, hidden.shape[1], heads, dimension).transpose(1, 2)
        query, _ = apply_rotary_pos_emb(
            query, query, record["cos"][:, -hidden.shape[1] :], record["sin"][:, -hidden.shape[1] :]
        )
        key_weight = query.float().square().mean(-2, keepdim=True).sqrt() / math.sqrt(dimension)
        index = torch.tensor(positions, dtype=torch.long, device=probabilities.device)[None]
        demand = probabilities[..., -window:, :].float().mean(-2).gather(-1, index)
    return key_weight, demand[..., None]
