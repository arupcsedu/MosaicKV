#!/usr/bin/env python3
"""Validate retention-one identity for every audited published KV policy."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from verify_hf_adapters import REVISIONS, checkpoint_messages, provenance

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.backends import compare_runtime_retention_one
from mosaickv.config import CacheConfig, load_config
from mosaickv.types import BudgetUnit, MosaicKVMethod

_METHODS = (
    MosaicKVMethod.ELASTIC_CACHE_REIMPL,
    MosaicKVMethod.H2O_REIMPL,
    MosaicKVMethod.STREAMINGLLM_REIMPL,
    MosaicKVMethod.PYRAMIDKV_REIMPL,
    MosaicKVMethod.SNAPKV_REIMPL,
    MosaicKVMethod.ROCO_REIMPL,
)


def _model_kwargs(hub_cache: Path) -> dict[str, Any]:
    return {
        "attn_implementation": "eager",
        "cache_dir": str(hub_cache),
        "device_map": "auto",
        "torch_dtype": "auto",
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    """Run all audited local policies at the exact FullKV operating point."""

    if not torch.cuda.is_available():
        raise RuntimeError("real-checkpoint published-baseline parity requires CUDA")
    model_id = str(args.model_id)
    if model_id not in REVISIONS:
        raise ValueError(f"model is not audited: {model_id}")
    revision = str(args.revision or REVISIONS[model_id])
    if revision != REVISIONS[model_id]:
        raise ValueError("parity requires the audited immutable model revision")

    cache_root = Path(args.cache_root)
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", cache_root / "huggingface" / "hub"))
    adapter = load_hf_adapter(
        model_id,
        revision=revision,
        model_kwargs=_model_kwargs(hub_cache),
        processor_kwargs={"cache_dir": str(hub_cache), "local_files_only": True},
    )
    messages, media_bytes = checkpoint_messages(model_id, None)
    prepared = adapter.prepare_inputs(messages)
    template = load_config(args.config)
    cache = CacheConfig(
        budget_value=2_147_483_647,
        budget_unit=BudgetUnit.BLOCKS,
        retention_ratio=1.0,
        block_size=1,
        accounting_spec_sha=template.cache.accounting_spec_sha,
        packing_mode=template.cache.packing_mode,
    )
    reports: list[dict[str, Any]] = []
    for method in _METHODS:
        config = replace(
            template,
            method=method,
            model=replace(template.model, id=model_id, revision=revision),
            generation=replace(
                template.generation,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
            ),
            cache=cache,
        )
        parity = compare_runtime_retention_one(adapter, prepared, config)
        result = asdict(parity)
        if parity.generated_tokens != args.max_new_tokens:
            raise RuntimeError(
                f"{method.value} generated {parity.generated_tokens} tokens, "
                f"expected {args.max_new_tokens}"
            )
        if parity.token_agreement != 1.0:
            raise RuntimeError(f"{method.value} token identity failed: {parity}")
        if parity.maximum_logit_difference > args.logit_atol:
            raise RuntimeError(f"{method.value} logit parity failed: {parity}")
        reports.append({"method": method.value, **result})

    input_ids = prepared.model_inputs["input_ids"].detach().cpu().numpy().tobytes()
    report_config = {
        "suite": "published_baseline_checkpoint_retention_one_v1",
        "model_id": model_id,
        "model_revision": revision,
        "methods": [method.value for method in _METHODS],
        "attention_implementation": "eager",
        "retention_ratio": 1.0,
        "block_size": 1,
        "max_new_tokens": args.max_new_tokens,
        "logit_absolute_tolerance": args.logit_atol,
    }
    return {
        "schema_version": 1,
        "status": "passed",
        "synthetic": False,
        "source": provenance(report_config),
        "model_id": model_id,
        "model_revision": revision,
        "dataset": "mosaickv/published-baseline-parity-input",
        "dataset_revision": sha256(b"published-baseline-parity-input-v1").hexdigest(),
        "prompt_set_sha": sha256(b"Describe the visual input briefly.").hexdigest(),
        "media_set_sha": sha256(media_bytes).hexdigest(),
        "tokenization_sha": sha256(input_ids).hexdigest(),
        "precision": str(adapter.model.dtype),
        "retention_ratio": 1.0,
        "methods": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--revision")
    parser.add_argument(
        "--cache-root",
        default="/scratch/djy8hg/cache/mosaickv",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/hf_elastic_cache_reimpl.yaml"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--logit-atol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_new_tokens < 2:
        parser.error("--max-new-tokens must be at least two")
    if args.logit_atol < 0:
        parser.error("--logit-atol must be non-negative")
    report = validate(args)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite parity report: {args.output}")
    args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
