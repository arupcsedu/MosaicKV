"""CLI construction and execution for the unified Hugging Face runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from mosaickv.adapters.huggingface import audited_model_revision, load_hf_adapter
from mosaickv.backends import HuggingFaceMosaicKVModel
from mosaickv.config import (
    CacheConfig,
    DatasetConfig,
    ExecutionConfig,
    ForecastingConfig,
    GenerationConfig,
    GraphConfig,
    LookMConfig,
    ModelConfig,
    PrefixKVConfig,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    RunConfig,
    SelectionConfig,
    VLCacheConfig,
    config_sha256,
    load_config,
)
from mosaickv.evaluation.harness import EvaluationHarness
from mosaickv.evaluation.lmms_adapter import run_lmms_development_evaluation
from mosaickv.evaluation.messages import MediaItem
from mosaickv.evaluation.storage import JsonlResultStore, load_jsonl
from mosaickv.evaluation.tasks import TaskSample, default_task_registry, load_synthetic_samples
from mosaickv.manifest import (
    ArtifactProvenance,
    InputProvenance,
    ManifestError,
    RunManifestWriter,
    sha256_bytes,
    sha256_text,
)
from mosaickv.types import (
    Backend,
    BudgetUnit,
    ForecastMode,
    JsonObject,
    LookMMergeStrategy,
    MeasurementType,
    MosaicKVMethod,
    Precision,
    PrefixKVProfileMode,
    RepairPolicy,
)


def _external_hf_cache() -> Path:
    configured = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not configured:
        raise ValueError(
            "set HF_HOME or HUGGINGFACE_HUB_CACHE to a cache directory outside the home "
            "directory before loading model weights"
        )
    cache = Path(configured).expanduser().resolve()
    home = Path.home().resolve()
    if cache == home or home in cache.parents:
        raise ValueError(f"Hugging Face cache must be outside the home directory: {cache}")
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _direct_config(args: argparse.Namespace) -> RunConfig:
    model_id = str(args.model)
    revision = str(args.model_revision or audited_model_revision(model_id))
    task = default_task_registry().resolve(str(args.task))
    dataset_revision = args.dataset_revision
    if task.name in {"synthetic_ci", "synthetic_smoke"}:
        dataset_revision = dataset_revision or "schema-v1"
    if not dataset_revision:
        raise ValueError("public benchmark runs require --dataset-revision with an immutable value")
    backend = "huggingface" if str(args.backend) == "hf" else str(args.backend)
    mode = ForecastMode(str(args.forecast))
    baseline_prompt_window = int(args.prompt_window)
    prompt_window = baseline_prompt_window
    draft_steps = int(args.draft_tokens)
    if mode is ForecastMode.PROMPT_WINDOW:
        draft_steps = 0
    elif mode is ForecastMode.DRAFT_ROLLOUT:
        prompt_window = 0
    method = MosaicKVMethod(str(args.method))
    repair_policy = RepairPolicy(str(args.repair_policy))
    if method.is_mosaickv:
        forecasting = ForecastingConfig(
            enabled=True,
            mode=mode,
            prompt_window=prompt_window,
            draft_steps=draft_steps,
            centroid_count=int(args.forecast_centroids),
        )
    elif method is MosaicKVMethod.PROMPT_ATTENTION_TOPK:
        forecasting = ForecastingConfig(
            enabled=False,
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=baseline_prompt_window,
            draft_steps=0,
            centroid_count=int(args.forecast_centroids),
        )
    else:
        forecasting = ForecastingConfig(
            enabled=False,
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=max(1, baseline_prompt_window),
            draft_steps=0,
            centroid_count=int(args.forecast_centroids),
        )
    prototype_enabled = method in {
        MosaicKVMethod.MOSAICKV_PROTO,
        MosaicKVMethod.MOSAICKV_FULL,
    }
    residual_enabled = method is MosaicKVMethod.MOSAICKV_FULL
    repair_enabled = (
        method is MosaicKVMethod.MOSAICKV_FULL and repair_policy is not RepairPolicy.NONE
    )
    return RunConfig(
        model=ModelConfig(model_id, revision, Precision(str(args.precision))),
        dataset=DatasetConfig(task.dataset_id, str(dataset_revision), task.split),
        execution=ExecutionConfig(
            Backend(backend), str(args.attention_backend), int(args.seed), True
        ),
        generation=GenerationConfig(max_new_tokens=int(args.max_new_tokens)),
        cache=CacheConfig(
            budget_value=int(args.cache_budget),
            budget_unit=BudgetUnit(str(args.budget_unit)),
            retention_ratio=float(args.retention_ratio),
            block_size=int(args.block_size),
        ),
        method=method,
        forecasting=forecasting,
        graph=GraphConfig(enabled=method.is_mosaickv),
        selection=SelectionConfig(enabled=method.is_mosaickv),
        prototypes=PrototypeConfig(enabled=prototype_enabled),
        residual=ResidualConfig(enabled=residual_enabled),
        repair=RepairConfig(
            enabled=repair_enabled,
            policy=repair_policy if repair_enabled else RepairPolicy.NONE,
            entropy_threshold=float(args.entropy_threshold),
            prototype_risk_threshold=float(args.prototype_risk_threshold),
            max_blocks_per_step=int(args.repair_blocks) if repair_enabled else 0,
            evaluation_only=repair_enabled and repair_policy is RepairPolicy.ORACLE,
        ),
        lookm=LookMConfig(
            enabled=method.is_lookm_reimplementation,
            recent_ratio=float(args.lookm_recent_ratio),
            important_ratio=float(args.lookm_important_ratio),
            merge_strategy=LookMMergeStrategy(str(args.lookm_merge_strategy)),
            text_prior=True,
        ),
        prefixkv=PrefixKVConfig(
            enabled=method.is_prefixkv_reimplementation,
            profile_mode=PrefixKVProfileMode(str(args.prefixkv_profile_mode)),
            profile_path=(
                str(args.prefixkv_profile)
                if args.prefixkv_profile is not None
                else None
            ),
            start_size=int(args.prefixkv_start_size),
            protect_size=int(args.prefixkv_protect_size),
            eviction_distance=int(args.prefixkv_eviction_distance),
        ),
        vl_cache=VLCacheConfig(
            enabled=method.is_vl_cache_reimplementation,
            sparsity_threshold=float(args.vl_cache_sparsity_threshold),
            min_layer_retention=float(args.vl_cache_min_layer_retention),
            max_layer_retention=float(args.vl_cache_max_layer_retention),
            recent_window_fraction=float(args.vl_cache_recent_window_fraction),
            max_post_vision_queries=(
                int(args.vl_cache_max_post_vision_queries)
                if args.vl_cache_max_post_vision_queries is not None
                else None
            ),
        ),
    )


def resolve_hf_config(args: argparse.Namespace) -> RunConfig | None:
    """Resolve a YAML/TOML config or direct flags when HF execution was requested."""

    if args.config is not None:
        config = load_config(str(args.config))
        return config if config.execution.backend is Backend.HUGGINGFACE else None
    if args.model is None:
        return None
    return _direct_config(args)


def _model_kwargs(config: RunConfig, *, local_files_only: bool) -> dict[str, Any]:
    import torch

    dtype = {
        Precision.FP32: torch.float32,
        Precision.FP16: torch.float16,
        Precision.BF16: torch.bfloat16,
    }.get(config.model.precision)
    if dtype is None:
        raise ValueError("the HF runtime supports model precision fp32, fp16, or bf16")
    kwargs: dict[str, Any] = {
        "attn_implementation": config.execution.attention_implementation,
        "dtype": dtype,
        "local_files_only": local_files_only,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    return kwargs


def _pil_synthetic_samples() -> tuple[TaskSample, ...]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required to pass synthetic images to an HF processor"
        ) from error
    converted: list[TaskSample] = []
    for sample in load_synthetic_samples():
        media: list[MediaItem] = []
        for item in sample.media:
            payload = item.payload
            if isinstance(payload, tuple) and len(payload) == 3:
                payload = Image.new("RGB", (16, 16), tuple(int(value) for value in payload))
            media.append(MediaItem(item.kind, payload))
        converted.append(replace(sample, media=tuple(media)))
    return tuple(converted)


def _trace_sha(trace_root: Path, run_id: str, fallback: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted((trace_root / run_id).glob("*.json"))
    if not paths:
        return sha256_bytes(fallback.read_bytes())
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _input_provenance(config: RunConfig, task_name: str, raw_path: Path) -> InputProvenance:
    identities: list[tuple[str, str]] = []
    for row in load_jsonl(raw_path):
        identities.append((row.sample_id, row.reference or ""))
    canonical = json.dumps(sorted(identities), separators=(",", ":"), ensure_ascii=False)
    selection = (
        f"{config.dataset.id}\0{config.dataset.revision}\0{config.dataset.split}\0"
        f"{config.execution.seed}\0{task_name}\0{canonical}"
    )
    return InputProvenance(
        prompt_set_sha=sha256_text(f"prompts\0{selection}"),
        media_set_sha=sha256_text(f"media\0{selection}"),
        preprocessing_sha=sha256_text(
            f"hf-auto-processor\0{config.model.id}\0{config.model.revision}"
        ),
        tokenization_sha=sha256_text(
            f"hf-auto-tokenizer\0{config.model.id}\0{config.model.revision}"
        ),
    )


def run_hf_evaluation(args: argparse.Namespace, config: RunConfig) -> tuple[int, JsonObject]:
    """Load one pinned model and execute local or lmms-eval samples."""

    if config.execution.attention_implementation != "eager":
        raise ValueError("only --attention-backend eager has passed the correctness gate")
    if config.execution.deterministic_algorithms:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(config.execution.deterministic_algorithms)
    torch.manual_seed(config.execution.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.execution.seed)
    _external_hf_cache()
    task_name = str(args.task or config.dataset.id)
    registry = default_task_registry()
    try:
        task = registry.resolve(task_name)
    except LookupError:
        matches = [
            name for name in registry.names() if registry.resolve(name).dataset_id == task_name
        ]
        if len(matches) != 1:
            raise
        task_name = matches[0]
        task = registry.resolve(task_name)
    run_id = str(args.run_id or uuid.uuid4().hex)
    output_root = Path(str(args.output_dir)).resolve()
    run_root = output_root / run_id
    raw_path = Path(args.raw_output).resolve() if args.raw_output else run_root / "results.jsonl"
    parquet_path = (
        Path(args.parquet_output).resolve()
        if args.parquet_output
        else run_root / "aggregate.parquet"
    )
    manifest_path = Path(args.manifest).resolve() if args.manifest else run_root / "manifest.json"
    trace_root = (
        Path(args.trace_directory).resolve() if args.trace_directory else output_root / "traces"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        rows = JsonlResultStore(raw_path).results(run_id=run_id)
        if not rows:
            raise ManifestError("an immutable manifest exists but the run has no completed rows")
        expected = (
            min(
                int(args.subset_size or len(load_synthetic_samples())),
                len(load_synthetic_samples()),
            )
            if task.local_scorer is not None
            else int(args.subset_size or 20)
        )
        if len(rows) != expected:
            raise ManifestError(
                "an immutable manifest exists but its terminal sample count does not match "
                f"the requested subset ({len(rows)} != {expected})"
            )
        recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            recorded.get("run_id") != run_id
            or recorded.get("source", {}).get("config_sha") != config_sha256(config)
            or recorded.get("artifacts", {}).get("raw_output_sha")
            != sha256_bytes(raw_path.read_bytes())
        ):
            raise ManifestError("existing manifest does not match this run/config/raw artifact")
        failed = sum(row.status.value == "failed" for row in rows)
        return (
            0 if failed == 0 else 1,
            {
                "status": "resumed_complete" if failed == 0 else "resumed_with_failures",
                "run_id": run_id,
                "task": task_name,
                "selected_samples": len(rows),
                "resumed_samples": len(rows),
                "failed_samples": failed,
                "method": config.method.value,
                "model": config.model.id,
                "raw_output": str(raw_path),
                "parquet_output": str(parquet_path),
                "manifest_path": str(manifest_path),
                "trace_directory": str((trace_root / run_id).resolve()),
            },
        )

    adapter = load_hf_adapter(
        config.model.id,
        revision=config.model.revision,
        model_kwargs=_model_kwargs(config, local_files_only=bool(args.local_files_only)),
        processor_kwargs={"local_files_only": bool(args.local_files_only)},
    )
    runtime = HuggingFaceMosaicKVModel(adapter, config, trace_directory=trace_root)
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
        subset_size = int(args.subset_size or 20)
        summary = run_lmms_development_evaluation(
            run_id=run_id,
            task_names=(task_name,),
            model=runtime,
            raw_output=raw_path,
            manifest_path=str(manifest_path),
            seed=config.execution.seed,
            subset_size=subset_size,
            parquet_output=parquet_path,
            registry=registry,
            dataset_revision=config.dataset.revision,
        )
    artifacts = ArtifactProvenance(
        raw_output_sha=sha256_bytes(raw_path.read_bytes()),
        metrics_sha=sha256_bytes(parquet_path.read_bytes()),
        log_sha=_trace_sha(trace_root, run_id, raw_path),
    )
    if not manifest_path.exists():
        measurement = (
            MeasurementType.REFERENCE
            if config.method.is_full_cache
            else (
                MeasurementType.BASELINE_SIMPLE
                if config.method.is_simple_baseline
                else (
                    MeasurementType.BASELINE_REIMPL
                    if config.method.is_published_reimplementation
                    else MeasurementType.METHOD
                )
            )
        )
        RunManifestWriter().write(
            manifest_path,
            config,
            measurement,
            _input_provenance(config, task_name, raw_path),
            artifacts,
            run_id=run_id,
        )
    rows = JsonlResultStore(raw_path).results(run_id=run_id)
    failed = sum(row.status.value == "failed" for row in rows)
    summary.update(
        {
            "status": "completed" if failed == 0 else "completed_with_failures",
            "method": config.method.value,
            "model": config.model.id,
            "manifest_path": str(manifest_path),
            "trace_directory": str((trace_root / run_id).resolve()),
            "failed_samples": failed,
        }
    )
    return (0 if failed == 0 else 1), summary


__all__ = ["resolve_hf_config", "run_hf_evaluation"]
