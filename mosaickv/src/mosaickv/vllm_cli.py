"""CLI orchestration for the measured vLLM FullKV backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import cast

from mosaickv.backends.vllm_runtime import (
    VLLMFullKVModel,
    VLLMRuntimeOptions,
    native_integration_capability,
    require_native_mosaickv_support,
)
from mosaickv.config import RunConfig, config_sha256, load_config
from mosaickv.evaluation.harness import EvaluationHarness
from mosaickv.evaluation.lmms_adapter import run_lmms_development_evaluation
from mosaickv.evaluation.storage import JsonlResultStore
from mosaickv.evaluation.tasks import default_task_registry, load_synthetic_samples
from mosaickv.hf_cli import _direct_config, _input_provenance, _pil_synthetic_samples
from mosaickv.manifest import (
    ArtifactProvenance,
    ManifestError,
    RunManifestWriter,
    sha256_bytes,
)
from mosaickv.types import Backend, JsonObject, MeasurementType


def _external_cache() -> Path:
    configured = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not configured:
        raise ValueError(
            "set HF_HOME or HUGGINGFACE_HUB_CACHE to a cache outside the home directory"
        )
    cache = Path(configured).expanduser().resolve()
    home = Path.home().resolve()
    if cache == home or home in cache.parents:
        raise ValueError(f"Hugging Face cache must be outside the home directory: {cache}")
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def resolve_vllm_config(args: argparse.Namespace) -> RunConfig | None:
    """Resolve vLLM YAML/TOML or direct flags without importing vLLM."""

    if args.config is not None:
        config = load_config(str(args.config))
        return config if config.execution.backend is Backend.VLLM else None
    if args.model is None or str(args.backend) != "vllm":
        return None
    return _direct_config(args)


def _trace_sha(trace_root: Path, run_id: str, fallback: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((trace_root / run_id).glob("*.json"))
    if not paths:
        return sha256_bytes(fallback.read_bytes())
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _task_name(config: RunConfig, explicit: str | None) -> str:
    name = explicit or config.dataset.id
    registry = default_task_registry()
    try:
        registry.resolve(name)
        return name
    except LookupError:
        matches = [
            candidate
            for candidate in registry.names()
            if registry.resolve(candidate).dataset_id == name
        ]
        if len(matches) != 1:
            raise
        return matches[0]


def _resume_payload(
    *,
    config: RunConfig,
    run_id: str,
    task_name: str,
    raw_path: Path,
    parquet_path: Path | None,
    manifest_path: Path,
    trace_root: Path,
    expected: int,
) -> tuple[int, JsonObject]:
    rows = JsonlResultStore(raw_path).results(run_id=run_id)
    if not rows:
        raise ManifestError("an immutable manifest exists but the run has no result rows")
    if len(rows) != expected:
        raise ManifestError(
            "immutable manifest terminal count differs from the requested subset "
            f"({len(rows)} != {expected})"
        )
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        recorded.get("run_id") != run_id
        or recorded.get("source", {}).get("config_sha") != config_sha256(config)
        or recorded.get("artifacts", {}).get("raw_output_sha")
        != sha256_bytes(raw_path.read_bytes())
    ):
        raise ManifestError("existing vLLM manifest does not match run/config/raw output")
    failed = sum(row.status.value == "failed" for row in rows)
    return (
        0 if failed == 0 else 1,
        {
            "status": "resumed_complete" if failed == 0 else "resumed_with_failures",
            "run_id": run_id,
            "task": task_name,
            "selected_samples": len(rows),
            "failed_samples": failed,
            "backend": "vllm",
            "method": "full_kv",
            "raw_output": str(raw_path),
            "parquet_output": str(parquet_path) if parquet_path is not None else None,
            "manifest_path": str(manifest_path),
            "trace_directory": str((trace_root / run_id).resolve()),
        },
    )


def run_vllm_evaluation(args: argparse.Namespace, config: RunConfig) -> tuple[int, JsonObject]:
    """Execute vLLM FullKV with raw streaming/cache measurements."""

    options = VLLMRuntimeOptions(
        tensor_parallel_size=int(args.tensor_parallel_size),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=cast("int | None", args.vllm_max_model_len),
        cache_probe_repeats=int(args.cache_probe_repeats),
        local_files_only=bool(args.local_files_only),
        enable_mosaickv=bool(args.enable_mosaickv),
    )
    if options.enable_mosaickv:
        require_native_mosaickv_support(
            enabled=True,
            vllm_version="0.7.2",
            enforce_eager=True,
            attention_backend=config.execution.attention_implementation,
        )
    if not config.method.is_full_cache or config.cache.retention_ratio != 1.0:
        raise ValueError(
            "vLLM Stage A supports only --method full_kv --retention-ratio 1.0; "
            "use --enable-mosaickv only after native support is reported"
        )
    _external_cache()
    task_name = _task_name(config, cast("str | None", args.task))
    registry = default_task_registry()
    task = registry.resolve(task_name)
    run_id = str(args.run_id or uuid.uuid4().hex)
    output_root = Path(str(args.output_dir)).resolve()
    run_root = output_root / run_id
    raw_path = Path(args.raw_output).resolve() if args.raw_output else run_root / "results.jsonl"
    parquet_path = Path(args.parquet_output).resolve() if args.parquet_output else None
    manifest_path = Path(args.manifest).resolve() if args.manifest else run_root / "manifest.json"
    trace_root = (
        Path(args.trace_directory).resolve() if args.trace_directory else output_root / "traces"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    expected = (
        min(int(args.subset_size or len(load_synthetic_samples())), len(load_synthetic_samples()))
        if task.local_scorer is not None
        else int(args.subset_size or 20)
    )
    if manifest_path.exists():
        return _resume_payload(
            config=config,
            run_id=run_id,
            task_name=task_name,
            raw_path=raw_path,
            parquet_path=parquet_path,
            manifest_path=manifest_path,
            trace_root=trace_root,
            expected=expected,
        )

    runtime = VLLMFullKVModel.from_config(config, options, trace_directory=trace_root)
    try:
        if task.local_scorer is not None:
            summary: JsonObject = (
                EvaluationHarness(registry)
                .run(
                    run_id=run_id,
                    task_name=task_name,
                    samples=_pil_synthetic_samples(),
                    model=runtime,
                    raw_output=raw_path,
                    manifest_path=str(manifest_path),
                    seed=config.execution.seed,
                    subset_size=cast("int | None", args.subset_size),
                    parquet_output=parquet_path,
                )
                .to_json_object()
            )
        else:
            summary = run_lmms_development_evaluation(
                run_id=run_id,
                task_names=(task_name,),
                model=runtime,
                raw_output=raw_path,
                manifest_path=str(manifest_path),
                seed=config.execution.seed,
                subset_size=int(args.subset_size or 20),
                parquet_output=parquet_path,
                registry=registry,
                dataset_revision=config.dataset.revision,
            )
    finally:
        runtime.close()

    artifacts = ArtifactProvenance(
        raw_output_sha=sha256_bytes(raw_path.read_bytes()),
        metrics_sha=(
            sha256_bytes(parquet_path.read_bytes())
            if parquet_path is not None
            else "not_applicable"
        ),
        log_sha=_trace_sha(trace_root, run_id, raw_path),
    )
    RunManifestWriter().write(
        manifest_path,
        config,
        MeasurementType.REFERENCE,
        _input_provenance(config, task_name, raw_path),
        artifacts,
        run_id=run_id,
        execution_metadata={
            "engine_execution_mode": runtime.runner.engine_metadata.get(
                "execution_mode", "unavailable"
            ),
            "cuda_graph": runtime.runner.engine_metadata.get("cuda_graph", "unavailable"),
            "attention_backend_configuration": runtime.runner.engine_metadata.get(
                "attention_backend", "unavailable"
            ),
            "vllm_block_size": runtime.runner.engine_metadata.get("block_size", "unavailable"),
            "model_source": runtime.runner.engine_metadata.get("model_source", "unavailable"),
        },
        attention_implementation_override=(
            "vllm_"
            + str(runtime.runner.engine_metadata.get("attention_backend", "unavailable")).lower()
        ),
    )
    rows = JsonlResultStore(raw_path).results(run_id=run_id)
    failed = sum(row.status.value == "failed" for row in rows)
    summary.update(
        {
            "status": "completed" if failed == 0 else "completed_with_failures",
            "backend": "vllm",
            "method": "full_kv",
            "manifest_path": str(manifest_path),
            "trace_directory": str((trace_root / run_id).resolve()),
            "failed_samples": failed,
        }
    )
    return (0 if failed == 0 else 1), summary


def native_capability_payload(vllm_version: str = "0.7.2") -> JsonObject:
    """Expose the Stage B verdict to tests, scripts, and diagnostics."""

    return native_integration_capability(vllm_version).to_json_object()


__all__ = [
    "native_capability_payload",
    "resolve_vllm_config",
    "run_vllm_evaluation",
]
