"""Same-input causal diagnostic; only the pure union rule is online-eligible.

Future-prefix logits and the unchanged guard are evaluation-only and never
enter selection, storage formatting or the union-expansion rule.
"""

from __future__ import annotations

import hashlib
from typing import Any


def union_positions(positions: Any) -> tuple[Any, ...]:
    """Restore already-archived token rows to other heads; never remove a row."""
    result = []
    if not positions:
        raise ValueError("complete nonempty source selections required")
    for layer in positions:
        if (
            not layer
            or len({len(head) for head in layer}) != 1
            or any(
                not head
                or list(sorted(set(head))) != list(head)
                or any(type(p) is not int or p < 0 for p in head)
                for head in layer
            )
        ):
            raise ValueError("uniform, sorted unique nonnegative source positions required")
        shared = tuple(sorted({p for head in layer for p in head}))
        result.append(tuple(shared for _ in layer))
    return tuple(result)


def forced_prefix(adapter: Any, prefill: Any, cache: Any, targets: list[int]) -> tuple[Any, Any]:
    """Evaluation-only fixed model-generated prefix; no benchmark annotation."""
    import torch

    from mosaickv.adapters.huggingface.types import DecodeState
    from mosaickv.evaluation.lossless_prompt_cache import IsolatedStorageTransport

    if len(targets) != 32 or any(type(t) is not int or t < 0 for t in targets):
        raise ValueError("exactly 32 frozen model-generated target tokens required")
    length = max(layer.get_seq_length() for layer in cache.layers)
    state = DecodeState(
        cache,
        torch.ones(1, length, device=adapter.device, dtype=torch.long),
        length,
        prefill.state.logical_sequence_length,
        prefill.state.next_decode_position,
        prefill.state.modality_map,
        dict(
            prefill.state.model_state, mosaickv_packed_all_valid=True, mosaickv_cache_all_valid=True
        ),
    )
    values = [prefill.logits.detach().reshape(-1).float()]
    with IsolatedStorageTransport(adapter, cache):
        for token in targets[:-1]:
            output = adapter.decode_one_token(
                torch.tensor([[token]], device=adapter.device), state, capture_queries=False
            )
            values.append(output.logits.detach().reshape(-1).float())
    logits = torch.stack(values).cpu()
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("nonfinite union-restoration logits")
    if state.next_decode_position != prefill.state.next_decode_position + 31:
        raise RuntimeError("union restoration changed logical positions")
    return logits, cache.storage_record()


def evaluate_union(
    adapter: Any,
    prefill: Any,
    capture: Any,
    original_positions: Any,
    targets: list[int],
    *,
    full_logits: Any,
    anchor_logits: Any,
    guard_config: Any,
    maximum_reference_kl: float,
    maximum_logit_difference: float,
) -> dict[str, Any]:
    import torch

    from mosaickv.backends.hf_allocation_guard import decide_draft_stability
    from mosaickv.evaluation.delimiter_control import compare_prefix_logits
    from mosaickv.evaluation.lossless_prompt_cache import construct_cache
    from mosaickv.evaluation.projection_cache import construct_projection_cache
    from mosaickv.evaluation.worker_support import settle

    expanded = union_positions(original_positions)
    # Independent reference: gather original post-RoPE K/V, without a source
    # archive or a prediction-based replacement.
    raw = construct_cache(
        adapter, prefill.state.past_key_values, expanded, encode=False, fused_unpack=False
    )
    reference, _ = forced_prefix(adapter, prefill, raw, targets)
    del raw
    settle(adapter)
    cache = construct_projection_cache(
        adapter, prefill, capture, expanded, policy="source_all", audit_workspace=True
    )
    candidate, storage = forced_prefix(adapter, prefill, cache, targets)
    if not torch.equal(candidate, reference):
        raise RuntimeError("source union differs from independently gathered original K/V")
    rows = compare_prefix_logits(full_logits, anchor_logits, candidate)
    guard = decide_draft_stability(rows, guard_config)
    absolute = all(
        r["candidate"]["logit_kl"] <= maximum_reference_kl
        and r["candidate"]["maximum_logit_difference"] <= maximum_logit_difference
        for r in rows
    )
    result = dict(
        representation_bitwise_equal_original_expanded_KV=True,
        representation_maximum_logit_difference=0.0,
        logits_sha256=hashlib.sha256(candidate.numpy().tobytes()).hexdigest(),
        forced_prefix_prediction_ids=candidate.argmax(-1).tolist(),
        targets_are_saved_model_generated_not_benchmark_answers=True,
        no_donor_eviction=True,
        no_original_selected_position_removed=True,
        used_fullkv_fallback=False,
        unchanged_relative_guard=guard,
        absolute_fullkv_limits_passed=absolute,
        prefix_comparisons=rows,
        storage=storage,
        original_per_layer_head_lengths=[len(layer[0]) for layer in original_positions],
        expanded_per_layer_head_lengths=[len(layer[0]) for layer in expanded],
        expanded_head_positions=sum(
            len(new) - len(old)
            for before, after in zip(original_positions, expanded, strict=True)
            for old, new in zip(before, after, strict=True)
        ),
        no_benchmark_quality_claim=True,
        same_realized_byte_budget_claim=False,
        whole_request_peak_certified=False,
    )
    del cache, candidate
    settle(adapter)
    return result
