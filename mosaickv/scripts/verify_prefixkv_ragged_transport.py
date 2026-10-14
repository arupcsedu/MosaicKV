#!/usr/bin/env python3
"""Pinned-checkpoint ragged PrefixKV transport gate, not a quality benchmark."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

import torch
from run_byte_capped_published_baseline_pilot import _require_clean_source, byte_capped_configs
from transformers.cache_utils import DynamicCache
from verify_hf_adapters import REVISIONS, checkpoint_messages, provenance

from mosaickv.adapters.huggingface import (
    compare_cache_reinjection,
    compare_with_generate,
    load_hf_adapter,
)
from mosaickv.backends.hf_runtime import (
    _pack_prefixkv_ragged,
    _state_from_packed,
    maintain_prefixkv_decode_cache,
)
from mosaickv.baselines.prefixkv import build_prefixkv_reimpl_plan
from mosaickv.cache_state import FullKVState
from mosaickv.config import load_config


def validate(args):
    root = Path(__file__).resolve().parents[2]
    _require_clean_source(root, args.expected_git_sha)
    if not torch.cuda.is_available():
        raise RuntimeError("this checkpoint gate requires a Slurm GPU allocation")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model_id = "llava-hf/llava-1.5-7b-hf"
    revision = REVISIONS[model_id]
    cfg = {
        "model_id": model_id,
        "model_revision": revision,
        "max_new_tokens": 16,
        "logit_atol": 1e-4,
        "transport": "ragged_llama_eager_v1",
        "byte_cap": 40 * 2**20,
        "attention_implementation": "eager",
        "precision": "fp16",
    }
    report = {
        "status": "running",
        "measurement_type": "validation_smoke",
        "source": provenance(cfg),
        "configuration": cfg,
        "dataset": "mosaickv/adapter-validation-image",
        "dataset_revision": "synthetic_transport_fixture_v1",
        "quality_claim": False,
    }
    try:
        adapter = load_hf_adapter(
            model_id,
            revision=revision,
            model_kwargs={
                "torch_dtype": torch.float16,
                "device_map": "cuda:0",
                "attn_implementation": "eager",
                "local_files_only": True,
            },
            processor_kwargs={"local_files_only": True},
        )
        messages, media = checkpoint_messages(model_id, None)
        prepared = adapter.prepare_inputs(messages)
        report["media_sha256"] = sha256(media).hexdigest()
        for name, function in (
            ("ordinary_generate", compare_with_generate),
            ("retention_one_reinjection", compare_cache_reinjection),
        ):
            parity = function(adapter, prepared, max_new_tokens=16)
            report[name] = asdict(parity)
            if parity.token_agreement != 1.0 or parity.maximum_logit_difference > cfg["logit_atol"]:
                raise RuntimeError(f"{name} parity failed")
        config = byte_capped_configs(
            load_config(root / "mosaickv/configs/paper_candidate_llava_v1.yaml"),
            repository=root,
            family="prefixkv",
            task="textvqa",
            model_id=model_id,
            model_revision=revision,
            seed=0,
            output_length=16,
            byte_cap=cfg["byte_cap"],
            profile_forget_ratio="0.8",
            profile_reference_retention_ratio=0.2,
            prefixkv_transport="ragged_llama_eager_v1",
        )["prefixkv_reimpl"]
        prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=True)
        full = FullKVState.from_cache_snapshot(
            adapter.extract_past_key_values(prefill.state.past_key_values),
            modality_spans=prepared.modality_map,
            token_ids=prepared.model_inputs["input_ids"],
            block_size=1,
            mandatory_logical_positions=(prepared.logical_sequence_length - 1,),
        )
        dense_width = full.active_bytes // prepared.logical_sequence_length
        target = cfg["byte_cap"] // dense_width * dense_width
        plan = build_prefixkv_reimpl_plan(
            full,
            prefill.attention_weights,
            config.prefixkv,
            replace(config.cache, budget_value=target),
            model_id=model_id,
            model_revision=revision,
            physical_prompt_capacity=target // dense_width,
        )
        packed = _pack_prefixkv_ragged(adapter, plan, prefill.state.past_key_values)
        state = _state_from_packed(adapter, prefill.state, packed)
        oracle = DynamicCache.from_legacy_cache(
            tuple((layer.key.clone(), layer.value.clone()) for layer in packed.snapshot.layers)
        )
        token = prefill.next_token_id
        language = adapter.model.model.language_model
        maximum = 0.0
        checks = []
        for step in range(1, 17):
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
                        past_key_value=oracle,
                        cache_position=position[0],
                        position_embeddings=rotary,
                    )
                expected = adapter.model.lm_head(language.norm(hidden))[:, -1, :]
            actual = adapter.decode_one_token(token, state, capture_queries=False)
            delta = float((actual.logits.float() - expected.float()).abs().max())
            maximum = max(maximum, delta)
            if not torch.isfinite(actual.logits).all() or delta > cfg["logit_atol"]:
                raise RuntimeError("compressed ragged transport logit parity failed")
            if not torch.equal(actual.next_token_id, expected.argmax(-1, keepdim=True)):
                raise RuntimeError("compressed ragged transport token parity failed")
            state, events = maintain_prefixkv_decode_cache(
                adapter, actual.state, plan, step_index=step
            )
            oracle_layers = []
            for (key, value), layer in zip(oracle.to_legacy_cache(), plan.layers, strict=True):
                if (
                    math.trunc(key.shape[-2] - (logical + 1) * layer.retention_ratio) > 0
                    and key.shape[-2] > 1
                ):
                    offset = min(layer.eviction_offset, key.shape[-2] - 1)
                    key = torch.cat((key[..., :offset, :], key[..., offset + 1 :, :]), dim=-2)
                    value = torch.cat((value[..., :offset, :], value[..., offset + 1 :, :]), dim=-2)
                oracle_layers.append((key, value))
            oracle = DynamicCache.from_legacy_cache(tuple(oracle_layers))
            checks.append({"step": step, "maximum_logit_difference": delta, "evictions": events})
            token = actual.next_token_id
        if plan.retained_bytes != target or len(set(plan.layer_cache_sizes)) < 2:
            raise RuntimeError("gate did not exercise byte-matched unequal cache layers")
        report.update(
            {
                "status": "passed",
                "active_bytes": plan.retained_bytes,
                "target_bytes": target,
                "padding_bytes": 0,
                "layer_cache_sizes": list(plan.layer_cache_sizes),
                "maximum_compressed_logit_difference": maximum,
                "steps": checks,
            }
        )
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    validate(parser.parse_args())


if __name__ == "__main__":
    main()
