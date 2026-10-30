"""All paper-baseline consumed diagnostics; never replace LOOK-M merged values.

Common source-slot capacity, append-only decode, and shared lossless transport.
Encoded backing varies: these rows are NOT called exact-byte-matched curves.
"""

from dataclasses import asdict, replace
from typing import Any

import torch


def baseline_payload(
    adapter: Any, prepared: Any, prefill: Any, cfg: Any, method: str, frozen: Any, profile_path: str
) -> tuple[Any, Any, dict[str, Any]]:
    """Return original per-head membership or actual transformed pairs and provenance."""
    from mosaickv.cache_state import FullKVState
    from mosaickv.config import LookMConfig, PrefixKVConfig, VLCacheConfig, canonical_config
    from mosaickv.types import BudgetUnit

    original, _ = adapter._legacy_layers(prefill.state.past_key_values)
    capacities = [len(heads[0]) for heads in frozen["uniform_0.5_control"]["anchor"]["positions"]]
    position_bytes = sum(t.shape[1] * t.shape[-1] * t.element_size() for t in original[0])
    cap = sum(capacities) * position_bytes
    source_bytes = sum(t.numel() * t.element_size() for pair in original for t in pair)
    cache = replace(
        cfg.cache,
        budget_unit=BudgetUnit.BYTES,
        budget_value=cap,
        retention_ratio=cap / source_bytes,
        physical_budget_mandatory_policy="last_token",
    )
    proof = dict(
        method=method,
        source_slot_byte_cap=cap,
        common_mandatory_policy="last_token",
        append_only=True,
        block_size=1,
        paper_native_reproduction=False,
        common_source_capacity_not_encoded_byte_matching=True,
        cache_config=canonical_config(replace(cfg, cache=cache, bounded_decode=None))["cache"],
    )
    if method in ("mosaickv_exact", "prompt_attention_topk"):
        role = "uniform_0.5_control" if method == "mosaickv_exact" else "window_topk_control"
        positions = frozen[role]["anchor"]["positions"]
        proof.update(selection="frozen_consumed_membership", selection_role=role)
        return positions, None, proof
    if method == "full_kv":
        positions = tuple(
            tuple(tuple(range(key.shape[-2])) for _ in range(key.shape[1])) for key, _ in original
        )
        proof.update(
            quality_reference_only=True,
            excluded_from_competitor_latency=True,
            common_source_capacity_not_encoded_byte_matching=False,
        )
        return positions, None, proof
    snapshot = adapter.extract_past_key_values(prefill.state.past_key_values)
    full = FullKVState.from_cache_snapshot(
        snapshot,
        modality_spans=prepared.modality_map,
        token_ids=prepared.model_inputs["input_ids"],
        block_size=1,
        original_logical_sequence_length=prefill.state.logical_sequence_length,
        next_decode_position=prefill.state.next_decode_position,
        mandatory_logical_positions=(prefill.state.logical_sequence_length - 1,),
    )
    if method == "prefixkv_reimpl":
        from mosaickv.baselines.prefixkv import build_prefixkv_reimpl_plan

        prefix = PrefixKVConfig(
            enabled=True,
            profile_path=profile_path,
            rescale_offline_profile_to_byte_budget=True,
            profile_reference_retention_ratio=0.5,
            hf_cache_transport="ragged_llama_eager_v1",
            decode_policy="append_only",
        )
        plan = build_prefixkv_reimpl_plan(
            full,
            prefill.attention_weights,
            prefix,
            cache,
            model_id=cfg.model.id,
            model_revision=cfg.model.revision,
        )
        positions = tuple(
            tuple(
                layer.selected_physical_positions for _ in range(full.layers[layer.layer].kv_heads)
            )
            for layer in plan.layers
        )
        proof.update(
            variant="prefixkv_byte_rescaled_append_only_reimpl",
            baseline_commit=prefix.official_repository_sha,
            profile_path=profile_path,
            baseline_parameters=asdict(prefix),
        )
    elif method == "vl_cache_reimpl":
        from mosaickv.baselines.vl_cache import build_vl_cache_reimpl_plan

        vl = VLCacheConfig(enabled=True)
        vl_plan = build_vl_cache_reimpl_plan(
            full, prefill.attention_weights, vl, cache, layer_local_physical_budget=True
        )
        positions = tuple(
            tuple(head.selected_physical_positions for head in layer.heads)
            for layer in vl_plan.layers
        )
        proof.update(
            variant="generalized_LLaVA15_vl_cache_reimpl_append_only",
            not_Mistral_paper_reproduction=True,
        )
    elif method == "lookm_reimpl":
        from mosaickv.baselines.lookm import build_lookm_reimpl_plan

        # Apply the same paper head selection/merge independently per layer.
        # Deterministic balanced 297/298-slot allocation spends the exact global
        # source capacity, without adding padding or discarding LOOK-M merges.
        layer_count = len(original)
        lower, extra = divmod(sum(capacities), layer_count)
        look = LookMConfig(enabled=True)
        low = build_lookm_reimpl_plan(
            full, prefill.attention_weights, look, cache, physical_prompt_capacity=lower
        )
        high = (
            build_lookm_reimpl_plan(
                full, prefill.attention_weights, look, cache, physical_prompt_capacity=lower + 1
            )
            if extra
            else low
        )
        selected = [
            head
            for layer in range(layer_count)
            for head in (high.heads if layer < extra else low.heads)
            if head.layer == layer
        ]
        pairs = tuple(
            (
                torch.cat([h.key for h in selected if h.layer == layer], dim=1),
                torch.cat([h.value for h in selected if h.layer == layer], dim=1),
            )
            for layer in range(layer_count)
        )
        actual = sum(t.numel() * t.element_size() for pair in pairs for t in pair)
        if actual != cap:
            raise RuntimeError("balanced LOOK-M transformed cache did not spend the source cap")
        proof.update(
            variant="lookm_reimpl_balanced_layer_capacity",
            baseline_commit=look.official_repository_sha,
            merged_KV_preserved=True,
            source_regeneration_eligible=False,
            ineligibility_reason="merged_KV_are_not_original_projection_outputs",
            layer_capacities=[lower + int(layer < extra) for layer in range(layer_count)],
            actual_source_slot_bytes=actual,
        )
        return None, pairs, proof
    else:
        raise ValueError("only the registered six consumed diagnostic methods are permitted")
    actual = sum(
        len(h[0]) * sum(t.shape[1] * t.shape[-1] * t.element_size() for t in pair)
        for h, pair in zip(positions, original, strict=True)
    )
    if actual != cap:
        raise RuntimeError("baseline allocation did not spend the common source-slot cap")
    proof["actual_source_slot_bytes"] = actual
    return positions, None, proof


def ordinary_pairs_reference(adapter: Any, prefill: Any, pairs: Any, *, tokens: int = 32) -> Any:
    """Independent ordinary HF cache holding the baseline's actual merged K/V."""
    from mosaickv.adapters.huggingface.types import CacheLayerSnapshot
    from mosaickv.backends.hf_runtime import PackedRuntimeCache, _state_from_packed

    template = adapter.extract_past_key_values(prefill.state.past_key_values)
    layers = tuple(CacheLayerSnapshot(k.detach().clone(), v.detach().clone(), 2) for k, v in pairs)
    snapshot = replace(
        template,
        layers=layers,
        active_sequence_length=max(k.shape[-2] for k, v in pairs),
        independent_layer_lengths=True,
    )
    packed = PackedRuntimeCache(snapshot, (), snapshot.active_sequence_length, True, (), (), ())
    state = _state_from_packed(adapter, prefill.state, packed)
    values, token = [prefill.logits.detach().reshape(-1).float().cpu()], prefill.next_token_id
    for _ in range(tokens - 1):
        output = adapter.decode_one_token(token, state, capture_queries=False)
        values.append(output.logits.detach().reshape(-1).float().cpu())
        token = output.next_token_id
    return torch.stack(values)
