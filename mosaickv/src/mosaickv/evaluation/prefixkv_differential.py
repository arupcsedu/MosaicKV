"""Evaluation-only differential oracle against pinned, unmodified PrefixKV.

Only upstream profile I/O is redirected to an in-memory, recorded profile.
Selection and eviction execute the actual upstream class, not a copied formula.
An effective byte-rescaled profile is an adaptation, NOT a published point.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
from pathlib import Path
from typing import Any, cast

from mosaickv.baselines.prefixkv import PrefixKVCompressionPlan

PIN = "597f1ab032704951550f93bcc8a23f1454b80aa4"


def upstream_oracle(
    repository: Path, forget_ratios: tuple[float, ...], *, start_size: int = 1, distance: int = -25
) -> tuple[Any, dict[str, Any]]:
    source = repository / "third_party/PrefixKV"
    sha = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if sha != PIN or dirty:
        raise RuntimeError("upstream PrefixKV must be clean at its pinned commit")
    path = source / "prefixkv.py"
    spec = importlib.util.spec_from_file_location("_prefixkv_differential_upstream", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = json.dumps(list(forget_ratios), allow_nan=False)

    def profile_read(name: str, mode: str) -> io.StringIO:
        if name != "confs/prefixkv_differential_0.0.json" or mode != "r":
            raise RuntimeError("unexpected upstream I/O; writes are forbidden")
        return io.StringIO(payload)

    module.__dict__["open"] = profile_read
    oracle = module.PrefixKV(
        model_name="differential",
        start_size=start_size,
        distance=distance,
        layer_num=len(forget_ratios),
        batch_size=1,
        profile=False,
    )
    return oracle, {
        "upstream_sha": sha,
        "upstream_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "oracle_profile_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "oracle_forget_ratios": list(forget_ratios),
        "io_adaptation": "read-only in-memory profile; unmodified upstream selector/evictor",
    }


def compare_upstream_decode(
    adapter: Any,
    prefill: Any,
    plan: PrefixKVCompressionPlan,
    repository: Path,
    *,
    steps: int = 64,
    atol: float = 1e-4,
) -> dict[str, Any]:
    """Compare actual selections, KV values, logits and evictions for every step.

    Uses a shared FullKV prefill and independent owned caches. No labels, scores,
    second prefill, copied eviction formula, or model.generate enter this oracle.
    """
    import torch
    from transformers.cache_utils import DynamicCache

    from mosaickv.backends.hf_runtime import (
        _pack_prefixkv_ragged,
        _state_from_packed,
        maintain_prefixkv_decode_cache,
    )

    if plan.config.decode_policy not in {"profile_ratio", "effective_profile_ratio"}:
        raise ValueError("differential gate requires an explicit upstream-compatible schedule")
    ratios = tuple(
        layer.decode_forget_ratio
        if layer.decode_forget_ratio is not None
        else 1.0 - layer.decode_retention_ratio
        for layer in plan.layers
    )
    oracle, provenance = upstream_oracle(
        repository,
        ratios,
        start_size=plan.config.start_size,
        distance=plan.config.eviction_distance,
    )
    legacy = adapter._legacy_layers(prefill.state.past_key_values)[0]
    source_copies = tuple((k.clone(), v.clone()) for k, v in legacy)
    # Position-marker tensors make selection agreement independent of numerical KV coincidences.
    markers = tuple(
        (
            torch.arange(k.shape[-2], device=k.device).reshape(1, 1, -1, 1),
            torch.arange(k.shape[-2], device=k.device).reshape(1, 1, -1, 1),
        )
        for k, _ in legacy
    )
    marker_oracle, _ = upstream_oracle(
        repository,
        ratios,
        start_size=plan.config.start_size,
        distance=plan.config.eviction_distance,
    )
    selected = marker_oracle(
        markers,
        num_of_token=prefill.state.logical_sequence_length,
        attentions=prefill.attention_weights,
    )
    selected_ids = tuple(tuple(k.flatten().tolist()) for k, _ in selected)
    if selected_ids != tuple(layer.selected_physical_positions for layer in plan.layers):
        raise RuntimeError("upstream/local selected positions differ")
    expected_layers = oracle(
        legacy,
        num_of_token=prefill.state.logical_sequence_length,
        attentions=prefill.attention_weights,
    )
    packed = _pack_prefixkv_ragged(adapter, plan, prefill.state.past_key_values)
    state = _state_from_packed(adapter, prefill.state, packed)
    for actual, (k, v) in zip(packed.snapshot.layers, expected_layers, strict=True):
        torch.testing.assert_close(actual.key, k, rtol=0, atol=0)
        torch.testing.assert_close(actual.value, v, rtol=0, atol=0)
    expected_cache = cast(
        DynamicCache,
        DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in expected_layers)),
    )
    language = adapter.model.model.language_model
    token = prefill.next_token_id
    checks: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        logical = state.next_decode_position
        with torch.inference_mode():
            hidden = language.embed_tokens(token)
            position = torch.tensor([[logical]], device=token.device)
            rotary = language.rotary_emb(hidden, position)
            for layer in language.layers:
                hidden = layer(
                    hidden,
                    attention_mask=None,
                    position_ids=position,
                    past_key_value=expected_cache,
                    cache_position=position[0],
                    position_embeddings=rotary,
                )
            expected = adapter.model.lm_head(language.norm(hidden))[:, -1, :]
        actual = adapter.decode_one_token(token, state, capture_queries=False)
        torch.testing.assert_close(actual.logits, expected, rtol=0, atol=atol)
        if not torch.equal(actual.next_token_id, expected.argmax(-1, keepdim=True)):
            raise RuntimeError("upstream/local greedy token differs")
        state, events = maintain_prefixkv_decode_cache(adapter, actual.state, plan, step_index=step)
        # Decode attentions are intentionally trivial: upstream uses only Q=1
        # to dispatch its fixed-offset eviction, not their numeric values.
        dummy = tuple(torch.zeros((1, 1, 1, 1), device=token.device) for _ in plan.layers)
        expected_layers = oracle(
            expected_cache.to_legacy_cache(), num_of_token=logical + 1, attentions=dummy
        )
        for (k, v), (ek, ev) in zip(
            adapter._legacy_layers(state.past_key_values)[0], expected_layers, strict=True
        ):
            torch.testing.assert_close(k, ek, rtol=0, atol=atol)
            torch.testing.assert_close(v, ev, rtol=0, atol=atol)
        # A separate marker trajectory checks original logical positions after eviction.
        selected = tuple(
            (
                torch.cat((k, torch.full_like(k[..., :1, :], logical)), -2),
                torch.cat((v, torch.full_like(v[..., :1, :], logical)), -2),
            )
            for k, v in selected
        )
        selected = marker_oracle(selected, num_of_token=logical + 1, attentions=dummy)
        logical_ids = tuple(tuple(k.flatten().tolist()) for k, _ in selected)
        if logical_ids != state.model_state["prefixkv_logical_positions"]:
            raise RuntimeError("upstream/local eviction or logical positions differ")
        expected_cache = cast(DynamicCache, DynamicCache.from_legacy_cache(tuple(expected_layers)))
        tensors = [t for pair in adapter._legacy_layers(state.past_key_values)[0] for t in pair]
        actual_bytes = sum(t.untyped_storage().nbytes() for t in tensors)
        if actual_bytes != sum(t.numel() * t.element_size() for t in tensors):
            raise RuntimeError("physical storage includes unreported padding")
        checks.append(
            {
                "step": step,
                "maximum_logit_difference": float(
                    (actual.logits.float() - expected.float()).abs().max()
                ),
                "token": int(actual.next_token_id.item()),
                "active_bytes": actual_bytes,
                "events": events,
                "selected_positions_and_cache_match": True,
            }
        )
        token = actual.next_token_id
    for (k, v), (saved_k, saved_v) in zip(legacy, source_copies, strict=True):
        if not torch.equal(k, saved_k) or not torch.equal(v, saved_v):
            raise RuntimeError("prefill source was mutated")
    return {
        **provenance,
        "passed": True,
        "steps": checks,
        "token_agreement": 1.0,
        "maximum_logit_difference": max(row["maximum_logit_difference"] for row in checks),
        "initial_active_bytes": plan.retained_bytes,
        "layer_cache_sizes": plan.layer_cache_sizes,
        "source_isolated": True,
        "plan": plan.trace(),
        "claim_scope": "upstream algorithm on common HF cache, not legacy-checkpoint reproduction",
    }
