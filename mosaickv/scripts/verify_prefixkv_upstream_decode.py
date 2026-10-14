#!/usr/bin/env python3
"""Clean GPU gate against actual upstream PrefixKV at native 50% then 40 MiB."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import torch
from run_byte_capped_published_baseline_pilot import _require_clean_source, byte_capped_configs
from verify_hf_adapters import REVISIONS, checkpoint_messages, provenance

from mosaickv.adapters.huggingface import (
    compare_cache_reinjection,
    compare_with_generate,
    load_hf_adapter,
)
from mosaickv.baselines.prefixkv import build_prefixkv_reimpl_plan
from mosaickv.cache_state import FullKVState
from mosaickv.config import CacheConfig, PrefixKVConfig, load_config
from mosaickv.evaluation.prefixkv_differential import compare_upstream_decode
from mosaickv.types import BudgetUnit


def validate(args):
    root = Path(__file__).resolve().parents[2]
    _require_clean_source(root, args.expected_git_sha)
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("requires one allocated GPU")
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = "llava-hf/llava-1.5-7b-hf"
    revision = REVISIONS[model]
    cfg = {
        "model_id": model,
        "model_revision": revision,
        "precision": "fp16",
        "attention_backend": "eager",
        "steps": 64,
        "atol": 1e-4,
        "dataset": "mosaickv/adapter-validation-image",
        "dataset_revision": "synthetic_transport_fixture_v1",
    }
    report = {
        "status": "running",
        "source": provenance(cfg),
        "configuration": cfg,
        "measurement_type": "validation_smoke",
        "quality_claim": False,
        "points": {},
    }
    try:
        adapter = load_hf_adapter(
            model,
            revision=revision,
            model_kwargs={
                "torch_dtype": torch.float16,
                "device_map": "cuda:0",
                "attn_implementation": "eager",
                "local_files_only": True,
            },
            processor_kwargs={"local_files_only": True},
        )
        messages, media = checkpoint_messages(model, None)
        prepared = adapter.prepare_inputs(messages)
        report["media_sha256"] = sha256(media).hexdigest()
        for name, function in (
            ("ordinary_generate", compare_with_generate),
            ("retention_one", compare_cache_reinjection),
        ):
            from dataclasses import asdict

            result = function(adapter, prepared, max_new_tokens=16)
            report[name] = asdict(result)
            if result.token_agreement != 1 or result.maximum_logit_difference > cfg["atol"]:
                raise RuntimeError(f"{name} parity failed")
        prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=True)
        full = FullKVState.from_cache_snapshot(
            adapter.extract_past_key_values(prefill.state.past_key_values),
            modality_spans=prepared.modality_map,
            token_ids=prepared.model_inputs["input_ids"],
            block_size=1,
            mandatory_logical_positions=(0, prepared.logical_sequence_length - 1),
        )
        native = PrefixKVConfig(
            enabled=True,
            profile_path=str(root / "third_party/PrefixKV/confs/prefixkv_llava-v1.5-7b_0.5.json"),
            hf_cache_transport="ragged_llama_eager_v1",
            decode_boundary_policy="protected",
            allocation_policy="upstream_round",
            decode_policy="profile_ratio",
        )
        config = byte_capped_configs(
            load_config(root / "mosaickv/configs/paper_candidate_llava_v1.yaml"),
            repository=root,
            family="prefixkv",
            task="textvqa",
            model_id=model,
            model_revision=revision,
            seed=0,
            output_length=64,
            byte_cap=40 * 2**20,
            profile_forget_ratio="0.8",
            profile_reference_retention_ratio=0.2,
            prefixkv_transport="ragged_llama_eager_v1",
            mandatory_policy="first_and_last",
        )["prefixkv_reimpl"]
        points = (
            (
                "native_profile_r050",
                native,
                CacheConfig(full.active_bytes, BudgetUnit.BYTES, 0.5, 1),
            ),
            (
                "byte_rescaled_40mib",
                replace(config.prefixkv, decode_policy="effective_profile_ratio"),
                config.cache,
            ),
        )
        for name, policy, cache in points:
            plan = build_prefixkv_reimpl_plan(
                full,
                prefill.attention_weights,
                policy,
                cache,
                model_id=model,
                model_revision=revision,
            )
            result = compare_upstream_decode(
                adapter, prefill, plan, root, steps=64, atol=cfg["atol"]
            )
            if name == "byte_rescaled_40mib" and plan.retained_bytes != 40 * 2**20:
                raise RuntimeError("40 MiB gate did not fill its exact shared cap")
            report["points"][name] = result
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    validate(parser.parse_args())
