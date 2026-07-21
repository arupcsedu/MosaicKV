#!/usr/bin/env python3
"""No-download validation gates for the unified MosaicKV HF runtime."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from scripts.verify_hf_adapters import (
    REVISIONS,
    checkpoint_messages,
    provenance,
    tiny_adapters,
)

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.backends import HuggingFaceMosaicKVModel, compare_runtime_retention_one
from mosaickv.config import (
    CacheConfig,
    DatasetConfig,
    ExecutionConfig,
    ForecastingConfig,
    GenerationConfig,
    ModelConfig,
    ResidualConfig,
    RunConfig,
)
from mosaickv.evaluation.messages import build_multimodal_messages
from mosaickv.evaluation.model import EvaluationRequest
from mosaickv.types import (
    Backend,
    BudgetUnit,
    ForecastMode,
    MosaicKVMethod,
    Precision,
)


def _config(
    model_id: str,
    method: MosaicKVMethod,
    ratio: float,
    *,
    max_new_tokens: int,
    revision: str = "a" * 40,
    precision: Precision = Precision.FP32,
    block_size: int = 1,
) -> RunConfig:
    return RunConfig(
        model=ModelConfig(model_id, revision, precision),
        dataset=DatasetConfig("mosaickv/tiny-runtime", "schema-v1", "test"),
        execution=ExecutionConfig(Backend.HUGGINGFACE, "eager", 0, True),
        generation=GenerationConfig(max_new_tokens=max_new_tokens),
        cache=CacheConfig(2_147_483_647, BudgetUnit.BLOCKS, ratio, block_size),
        method=method,
        forecasting=ForecastingConfig(
            enabled=method is not MosaicKVMethod.FULLKV,
            mode=ForecastMode.HYBRID,
            prompt_window=2,
            draft_steps=2,
            centroid_count=2,
        ),
        residual=ResidualConfig(require_pinned_memory=False),
    )


def _request(run_id: str) -> EvaluationRequest:
    return EvaluationRequest(
        run_id,
        "tiny-sample",
        "synthetic_smoke",
        build_multimodal_messages("tiny prompt"),
        {},
    )


def _trace(root: Path, run_id: str) -> dict[str, Any]:
    paths = tuple((root / run_id).glob("*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one trace for {run_id}, found {len(paths)}")
    result = json.loads(paths[0].read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("runtime trace is not an object")
    return result


def validate(max_new_tokens: int) -> dict[str, Any]:
    """Run every method on three real randomly initialized HF architectures."""

    if max_new_tokens < 2:
        raise ValueError("max_new_tokens must be at least two")
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_reports: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="mosaickv-hf-runtime-") as directory:
        trace_root = Path(directory)
        for model_id, adapter in tiny_adapters():
            method_reports: list[dict[str, Any]] = []
            prepared = adapter.prepare_inputs(build_multimodal_messages("tiny prompt"))
            parity = compare_runtime_retention_one(
                adapter,
                prepared,
                _config(
                    model_id,
                    MosaicKVMethod.MOSAICKV_EXACT,
                    1.0,
                    max_new_tokens=max_new_tokens,
                ),
            )
            if parity.token_agreement != 1.0 or parity.maximum_logit_difference > 1e-6:
                raise RuntimeError(f"{model_id} retention-1 numerical parity failed: {parity}")
            full_runtime = HuggingFaceMosaicKVModel(
                adapter,
                _config(
                    model_id,
                    MosaicKVMethod.FULLKV,
                    1.0,
                    max_new_tokens=max_new_tokens,
                ),
                trace_directory=trace_root,
            )
            full = full_runtime.generate(_request(f"{model_id}-fullkv"))
            full_trace = _trace(trace_root, f"{model_id}-fullkv")
            active_by_ratio: list[int] = []
            exact_one_trace: dict[str, Any] | None = None
            for ratio in (0.5, 0.75, 1.0):
                run_id = f"{model_id}-exact-{ratio}"
                generated = HuggingFaceMosaicKVModel(
                    adapter,
                    _config(
                        model_id,
                        MosaicKVMethod.MOSAICKV_EXACT,
                        ratio,
                        max_new_tokens=max_new_tokens,
                    ),
                    trace_directory=trace_root,
                ).generate(_request(run_id))
                trace = _trace(trace_root, run_id)
                if not generated.answer.strip():
                    raise RuntimeError(f"{model_id} exact ratio {ratio} decoded empty text")
                if generated.metrics.active_kv_bytes is None:
                    raise RuntimeError("runtime did not report active cache bytes")
                active_by_ratio.append(generated.metrics.active_kv_bytes)
                if ratio == 1.0:
                    exact_one_trace = trace
                    if generated.answer != full.answer:
                        raise RuntimeError(f"{model_id} retention 1 output differs from FullKV")
            if active_by_ratio != sorted(active_by_ratio):
                raise RuntimeError(f"{model_id} active cache bytes are not monotonic")
            if exact_one_trace is None or (
                exact_one_trace["generated_token_ids"] != full_trace["generated_token_ids"]
            ):
                raise RuntimeError(f"{model_id} retention 1 token IDs differ from FullKV")
            for method in (MosaicKVMethod.MOSAICKV_PROTO, MosaicKVMethod.MOSAICKV_FULL):
                run_id = f"{model_id}-{method.value}"
                generated = HuggingFaceMosaicKVModel(
                    adapter,
                    _config(model_id, method, 0.5, max_new_tokens=max_new_tokens),
                    trace_directory=trace_root,
                ).generate(_request(run_id))
                trace = _trace(trace_root, run_id)
                required = {
                    "selected_blocks",
                    "prototypes",
                    "graph_edges",
                    "forecast_statistics",
                    "repair_events",
                    "timing_breakdown",
                }
                if not required <= trace.keys() or not generated.answer.strip():
                    raise RuntimeError(f"{model_id} {method.value} trace/output is incomplete")
                if not str(trace["effective_method"]).endswith("mosaickv_exact_safety_fallback"):
                    raise RuntimeError(f"{model_id} performed an unsafe post-RoPE prototype merge")
                method_reports.append(
                    {
                        "method": method.value,
                        "effective_method": trace["effective_method"],
                        "valid_text": True,
                        "trace_complete": True,
                    }
                )
            model_reports.append(
                {
                    "model": model_id,
                    "retention_one_token_agreement": parity.token_agreement,
                    "retention_one_maximum_logit_difference": (parity.maximum_logit_difference),
                    "active_bytes_monotonic": True,
                    "active_bytes_by_ratio": dict(
                        zip(("0.5", "0.75", "1.0"), active_by_ratio, strict=True)
                    ),
                    "methods": method_reports,
                }
            )
    source = provenance(
        {
            "suite": "hf_runtime_tiny_v1",
            "max_new_tokens": max_new_tokens,
            "seed": 0,
        }
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "measurement_type": "validation_smoke",
        "synthetic": True,
        "device": device,
        "max_new_tokens": max_new_tokens,
        "source": source,
        "models": model_reports,
    }


def validate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    """Run the unified retention-1 path against FullKV on one pinned checkpoint."""

    model_id = str(args.model_id)
    if model_id not in REVISIONS:
        raise ValueError(f"model is not audited: {model_id}")
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint runtime parity requires CUDA")
    revision = str(args.revision or REVISIONS[model_id])
    if revision != REVISIONS[model_id]:
        raise ValueError("checkpoint runtime parity requires the audited immutable revision")
    cache_root = Path(args.cache_root)
    hub_cache = Path(os.environ.get("HF_HUB_CACHE", cache_root / "huggingface" / "hub"))
    adapter = load_hf_adapter(
        model_id,
        revision=revision,
        model_kwargs={
            "attn_implementation": "eager",
            "cache_dir": str(hub_cache),
            "device_map": "auto",
            "dtype": "auto",
            "local_files_only": not args.allow_download,
            "low_cpu_mem_usage": True,
        },
        processor_kwargs={
            "cache_dir": str(hub_cache),
            "local_files_only": not args.allow_download,
        },
    )
    messages, media_bytes = checkpoint_messages(model_id, args.internvl_pixel_values)
    prepared = adapter.prepare_inputs(messages)
    model_dtype = str(adapter.model.dtype)
    if "bfloat16" in model_dtype:
        precision = Precision.BF16
    elif "float32" in model_dtype:
        precision = Precision.FP32
    else:
        precision = Precision.FP16
    config = _config(
        model_id,
        MosaicKVMethod.MOSAICKV_EXACT,
        1.0,
        max_new_tokens=args.max_new_tokens,
        revision=revision,
        precision=precision,
        block_size=args.block_size,
    )
    parity = compare_runtime_retention_one(adapter, prepared, config)
    tolerance = float(args.logit_atol)
    if parity.token_agreement != 1.0 or parity.maximum_logit_difference > tolerance:
        raise RuntimeError(f"unified runtime retention-1 parity failed: {parity}")
    input_ids = prepared.model_inputs["input_ids"].detach().cpu().numpy().tobytes()
    report_config = {
        "suite": "hf_runtime_checkpoint_retention_one_v1",
        "model_id": model_id,
        "model_revision": revision,
        "attention_implementation": "eager",
        "block_size": args.block_size,
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "retention_ratio": 1.0,
        "logit_absolute_tolerance": tolerance,
    }
    return {
        "schema_version": 1,
        "status": "passed",
        "measurement_type": "validation_smoke",
        "synthetic": False,
        "source": provenance(report_config),
        "model_id": model_id,
        "model_revision": revision,
        "dataset": "mosaickv/runtime-validation-input",
        "dataset_revision": sha256(b"runtime-validation-input-v1").hexdigest(),
        "prompt_set_sha": sha256(b"Describe the visual input briefly.").hexdigest(),
        "media_set_sha": sha256(media_bytes).hexdigest(),
        "tokenization_sha": sha256(input_ids).hexdigest(),
        "precision": model_dtype,
        "cache_budget": {"retention_ratio": 1.0, "unit": "blocks"},
        "block_size": args.block_size,
        "parity": asdict(parity),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument(
        "--cache-root",
        default="/scratch/djy8hg/cache/mosaickv",
    )
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--internvl-pixel-values", type=Path)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--logit-atol", type=float, default=1e-4)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.max_new_tokens < 2:
        parser.error("--max-new-tokens must be at least two")
    if args.block_size <= 0:
        parser.error("--block-size must be positive")
    report = validate_checkpoint(args) if args.model_id else validate(args.max_new_tokens)
    serialized = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite validation report: {destination}")
        destination.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
