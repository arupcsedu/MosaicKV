"""CLI construction and execution for the unified Hugging Face runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import uuid
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from mosaickv.adapters.huggingface import (
    HuggingFaceMultimodalAdapter,
    audited_model_revision,
    load_hf_adapter,
)
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
    PublishedKVConfig,
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
from mosaickv.evaluation.metrics.registry import default_metric_registry
from mosaickv.evaluation.paper_native_lanes import (
    cache_operating_point,
    load_paper_native_ledger,
    resolve_protocol,
    run_materialized_paper_native_evaluation,
    validate_protocol_run,
)
from mosaickv.evaluation.published_protocols import (
    PublishedBaselineProtocol,
    load_published_protocols,
)
from mosaickv.evaluation.results import EvaluationResult
from mosaickv.evaluation.storage import JsonlResultStore, load_jsonl
from mosaickv.evaluation.tasks import (
    TaskSample,
    TaskSpec,
    default_task_registry,
    load_synthetic_samples,
)
from mosaickv.manifest import (
    ArtifactProvenance,
    EvaluationProvenance,
    InputProvenance,
    ManifestError,
    RunManifestWriter,
    sha256_bytes,
    sha256_text,
)
from mosaickv.types import (
    AblationVariant,
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
    ResultCategory,
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
            Backend(backend),
            str(args.attention_backend),
            int(args.seed),
            True,
            (
                str(args.decode_attention_backend)
                if args.decode_attention_backend is not None
                else None
            ),
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
        selection=SelectionConfig(
            enabled=method.is_mosaickv,
            stop_on_nonpositive_gain=method is not MosaicKVMethod.MOSAICKV_EXACT,
        ),
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
                str(args.prefixkv_profile) if args.prefixkv_profile is not None else None
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
        published_kv=PublishedKVConfig(
            enabled=method.is_published_eviction_reimplementation,
            observation_window=int(args.published_kv_observation_window),
            attention_pool_kernel=int(args.published_kv_pool_kernel),
            attention_pooling=str(args.published_kv_pooling),
            sink_tokens=int(args.published_kv_sink_tokens),
            recent_fraction=float(args.published_kv_recent_fraction),
            pyramid_beta=int(args.published_kv_pyramid_beta),
            roco_protected_fraction=float(args.published_kv_roco_protected_fraction),
            elastic_start_tokens=int(args.published_kv_elastic_start_tokens),
            elastic_protect_tokens=int(args.published_kv_elastic_protect_tokens),
            elastic_decode_offset=int(args.published_kv_elastic_decode_offset),
        ),
    )


def _paper_native_runtime_profile(
    config: RunConfig,
    protocol: PublishedBaselineProtocol,
) -> RunConfig:
    """Apply the audited block-size-one paper-native execution profile.

    Published reimplementations require token-sized blocks.  Applying the
    ordinary full-graph MosaicKV defaults to that representation creates one
    graph node per layer/head/token (hundreds of thousands for LLaVA) and a
    cardinality budget permits uneven head lengths that are padded back to a
    large dense tensor.  This profile changes neither the protocol block
    granularity nor its shared nominal cache cap:

    * byte budgeting gives every layer/head one physical position limit;
    * stable heads use deterministic prompt-attention selection directly;
    * only the highest-uncertainty 1/256 of routes enter the evidence graph;
    * the graph reranks a bounded prompt shortlist; and
    * MosaicKV deliberately uses at most 69% of the shared compressed cap,
      which preregisters a >30% actual-byte target against a baseline that
      consumes the cap.  Actual bytes, not nominal ratios, remain authoritative.

    Prefix teacher forcing also needs a decode horizon covering the observed
    136-token target.  A 144-token bound leaves a small, explicit margin and
    never truncates a target.
    """

    profiled = config
    if protocol.protocol_id == "prefixkv_hf_reimpl_native_metrics_v1":
        profiled = replace(
            profiled,
            generation=replace(
                profiled.generation,
                max_new_tokens=max(144, profiled.generation.max_new_tokens),
            ),
        )
    if (
        not profiled.method.is_mosaickv
        or profiled.cache.retention_ratio >= 1.0
        or profiled.cache.block_size != 1
    ):
        return profiled
    return replace(
        profiled,
        cache=replace(
            profiled.cache,
            budget_unit=BudgetUnit.BYTES,
            exact_gather_mode="source_views",
            packing_mode="preallocated_layer",
        ),
        forecasting=replace(
            profiled.forecasting,
            mode=ForecastMode.PROMPT_WINDOW,
            prompt_window=max(16, profiled.forecasting.prompt_window),
            draft_steps=0,
            adaptive_head_routing=True,
            shifted_head_prompt_weight=0.0,
        ),
        graph=replace(
            profiled.graph,
            max_neighbors=4,
            attention_host_transfer_mode="deferred_layer_vectorized",
            similarity_shortlist_multiplier=3.0,
            similarity_shortlist_min_nodes=64,
            candidate_descriptor_pooling=True,
            candidate_graph_mode="shortlist_rerank",
            candidate_neighbor_hops=1,
            screening_group_blocks=4,
            uncertain_head_only=True,
            # LLaVA-1.5 has 512 layer/KV-head routes.  Routing 1/256 keeps
            # two high-entropy routes in the evidence graph while the other
            # 510 routes use the exact, vectorized Prompt Top-K transaction.
            # This is intentionally a latency profile, not a budget change:
            # all routes still share the same dense physical capacity plan.
            uncertain_head_fraction=1.0 / 256.0,
            uncertain_head_threshold=0.2,
            candidate_nonmandatory_fraction=0.03,
        ),
        selection=replace(
            profiled.selection,
            algorithm="vectorized_coverage",
            stop_on_nonpositive_gain=False,
            minimum_stratum_fraction=0.1,
            coverage_correction_rounds=1,
            stable_head_prompt_routing=True,
            stable_head_uniform_fill=True,
            routed_head_only_planning=True,
            physical_capacity_fraction=0.69,
        ),
        prototypes=replace(
            profiled.prototypes,
            uncertain_head_only=True,
            uncertain_head_fraction=1.0 / 256.0,
            uncertain_head_threshold=0.2,
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
        # Keep the established ``torch_dtype`` spelling for the audited
        # Transformers 4.55.2 checkpoint loaders.
        "torch_dtype": dtype,
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


def _input_provenance(
    config: RunConfig,
    task_name: str,
    raw_path: Path,
    *,
    input_binding: str | None = None,
) -> InputProvenance:
    identities: list[tuple[str, str]] = []
    for row in load_jsonl(raw_path):
        identities.append((row.sample_id, row.reference or ""))
    canonical = json.dumps(sorted(identities), separators=(",", ":"), ensure_ascii=False)
    selection = (
        f"{config.dataset.id}\0{config.dataset.revision}\0{config.dataset.split}\0"
        f"{config.execution.seed}\0{task_name}\0{canonical}\0{input_binding or 'not_used'}"
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


def _source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or len(commit) != 40:
        raise RuntimeError("cannot resolve the evaluator git commit")
    return commit


def _reference_output_sha256(rows: tuple[EvaluationResult, ...]) -> str:
    """Hash the exact sample/reference population used by an evaluation."""

    payload = [
        {
            "sample_id": row.sample_id,
            "reference": row.reference,
        }
        for row in sorted(rows, key=lambda item: item.sample_id)
    ]
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _paper_evaluation_provenance(
    protocol: PublishedBaselineProtocol | None,
    task: TaskSpec,
    method: str,
    rows: tuple[EvaluationResult, ...],
    summary: JsonObject,
) -> EvaluationProvenance:
    """Build required evaluator, reference, and baseline provenance."""

    registry = default_metric_registry()
    definitions = tuple(registry.resolve(name) for name in task.paper_metrics)
    evaluator_name = "+".join(sorted({definition.evaluator_owner for definition in definitions}))
    evaluator_version = "+".join(
        f"{definition.name}:{definition.version}" for definition in definitions
    )
    method_contract = (
        next(item for item in protocol.methods if item.name == method)
        if protocol is not None
        else None
    )
    reference_sha = summary.get("fullkv_reference_artifact_sha256")
    if task.name == "coco_caption" and summary.get("paper_native_ledger_sha256"):
        reference_sha = summary["paper_native_ledger_sha256"]
    if not isinstance(reference_sha, str) or len(reference_sha) != 64:
        reference_sha = _reference_output_sha256(rows)
    source_commit = _source_commit()
    return EvaluationProvenance(
        evaluator_name=evaluator_name,
        evaluator_version=evaluator_version,
        evaluator_commit=source_commit,
        reference_output_sha256=reference_sha,
        baseline_name=method,
        # Official/reimplementation contracts use the pinned upstream commit.
        # Local FullKV/MosaicKV references are identified by this repository's
        # immutable source commit rather than an ambiguous not-applicable marker.
        baseline_commit=(
            method_contract.repository_commit or source_commit
            if method_contract is not None
            else (
                "597f1ab032704951550f93bcc8a23f1454b80aa4"
                if method == "prefixkv_reimpl"
                else source_commit
            )
        ),
    )


def _load_sample_id_file(
    path_value: str | None,
    expected_sha: str | None,
    *,
    task_name: str,
    config: RunConfig,
) -> tuple[str, ...] | None:
    if path_value is None:
        if expected_sha is not None:
            raise ValueError("--sample-id-file-sha requires --sample-id-file")
        return None
    if expected_sha is None:
        raise ValueError("--sample-id-file requires --sample-id-file-sha")
    path = Path(path_value).resolve()
    payload = path.read_bytes()
    actual_sha = sha256_bytes(payload)
    if actual_sha != expected_sha:
        raise ValueError(
            f"sample ID file SHA mismatch: expected {expected_sha}, observed {actual_sha}"
        )
    value = json.loads(payload)
    if not isinstance(value, dict) or value.get("schema_version") not in {1, 2}:
        raise ValueError("sample ID file must be a schema-version-1-or-2 JSON object")
    expected = {
        "task": task_name,
        "dataset_id": config.dataset.id,
        "dataset_revision": config.dataset.revision,
        "split": config.dataset.split,
    }
    mismatched = [key for key, item in expected.items() if value.get(key) != item]
    if mismatched:
        raise ValueError(f"sample ID file identity mismatch: {', '.join(mismatched)}")
    raw_ids = value.get("sample_ids")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or not all(isinstance(sample_id, str) and sample_id.strip() for sample_id in raw_ids)
    ):
        raise ValueError("sample ID file requires a non-empty sample_ids string array")
    sample_ids = tuple(cast("list[str]", raw_ids))
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("sample ID file contains duplicate sample IDs")
    if value.get("schema_version") == 2 and value.get("sample_count") != len(sample_ids):
        raise ValueError("schema-version-2 sample ID file has inconsistent sample_count")
    return sample_ids


def run_hf_evaluation(
    args: argparse.Namespace,
    config: RunConfig,
    *,
    adapter: HuggingFaceMultimodalAdapter | None = None,
) -> tuple[int, JsonObject]:
    """Load one pinned model and execute local or lmms-eval samples."""

    if config.execution.attention_implementation != "eager" and not config.method.is_full_cache:
        raise ValueError(
            "compressed HF methods require --attention-backend eager; "
            "SDPA and FlashAttention-2 are FullKV-only"
        )
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
    protocol_path = cast("str | None", getattr(args, "published_protocol", None))
    protocol_id = cast("str | None", getattr(args, "published_protocol_id", None))
    if (protocol_path is None) != (protocol_id is None):
        raise ValueError("--published-protocol and --published-protocol-id are required together")
    protocol = None
    if protocol_path is not None and protocol_id is not None:
        protocol = resolve_protocol(load_published_protocols(protocol_path), protocol_id)
        config = _paper_native_runtime_profile(config, protocol)
        environment_lock = Path(__file__).resolve().parents[2] / "env/common/requirements.lock"
        validate_protocol_run(
            protocol,
            task=task,
            config=config,
            environment_lock_sha256=sha256_bytes(environment_lock.read_bytes()),
        )
    ledger_path = cast("str | None", getattr(args, "paper_native_ledger", None))
    ledger_sha = cast("str | None", getattr(args, "paper_native_ledger_sha", None))
    if (ledger_path is None) != (ledger_sha is None):
        raise ValueError(
            "--paper-native-ledger and --paper-native-ledger-sha are required together"
        )
    is_materialized_native = task.scoring_owner in {
        "paper_native_prefixkv",
        "paper_native_milebench",
    }
    unified_milebench = bool(getattr(args, "unified_milebench", False))
    unified_prefixkv = bool(getattr(args, "unified_prefixkv", False))
    unified_coco = bool(getattr(args, "unified_coco", False))
    if unified_coco:
        if (
            task.name != "coco_caption"
            or protocol is not None
            or unified_prefixkv
            or unified_milebench
        ):
            raise ValueError("--unified-coco requires COCO without another protocol/route")
        is_materialized_native = True
    if unified_prefixkv and (
        task.name not in {"llava_description", "mmvet_prefixkv"}
        or protocol is not None
        or unified_milebench
    ):
        raise ValueError("--unified-prefixkv requires a PrefixKV task without a published protocol")
    if (
        unified_prefixkv
        and config.method is MosaicKVMethod.PREFIXKV_REIMPL
        and (
            config.prefixkv.decode_policy != "append_only"
            or not config.prefixkv.rescale_offline_profile_to_byte_budget
            or config.prefixkv.question_score_weight != 0
            or config.prefixkv.exclude_protected_from_calibration
            or config.prefixkv.layer_budget_floor_fraction != 0
        )
    ):
        raise ValueError(
            "unified PrefixKV requires upstream scoring and append-only byte rescaling"
        )
    if unified_milebench and (task.name != "milebench" or protocol is not None):
        raise ValueError("--unified-milebench requires MileBench without a published protocol")
    if (
        is_materialized_native
        and protocol is None
        and not (unified_milebench or unified_prefixkv or unified_coco)
    ):
        raise ValueError(
            f"task {task.name!r} is a paper reproduction and requires a published protocol"
        )
    if is_materialized_native and (ledger_path is None or ledger_sha is None):
        raise ValueError(f"task {task.name!r} requires a hash-bound --paper-native-ledger")
    if not is_materialized_native and ledger_path is not None:
        raise ValueError("--paper-native-ledger is only valid for PrefixKV/MileBench native routes")
    metric_observations_output = cast(
        "str | None", getattr(args, "metric_observations_parquet_output", None)
    )
    metric_aggregates_output = cast(
        "str | None", getattr(args, "metric_aggregates_parquet_output", None)
    )
    metric_output = cast("str | None", getattr(args, "metric_output", None))
    if is_materialized_native and (
        metric_output is None
        or metric_observations_output is None
        or metric_aggregates_output is None
    ):
        raise ValueError(
            "paper-native routes require --metric-output plus separate observation and "
            "aggregate Parquet outputs"
        )
    if protocol is not None and (
        metric_observations_output is None or metric_aggregates_output is None
    ):
        raise ValueError(
            "published-protocol runs require separate metric observation and aggregate Parquet"
        )
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
    full_dataset = bool(getattr(args, "full_dataset", False))
    sample_offset = int(getattr(args, "sample_offset", 0))
    explicit_sample_ids = _load_sample_id_file(
        cast("str | None", getattr(args, "sample_id_file", None)),
        cast("str | None", getattr(args, "sample_id_file_sha", None)),
        task_name=task_name,
        config=config,
    )
    if full_dataset and args.subset_size is not None:
        raise ValueError("--full-dataset and --subset-size are mutually exclusive")
    if explicit_sample_ids is not None and (
        full_dataset or args.subset_size is not None or sample_offset
    ):
        raise ValueError(
            "--sample-id-file cannot be combined with --full-dataset, --subset-size, "
            "or --sample-offset"
        )
    if is_materialized_native and explicit_sample_ids is not None:
        raise ValueError("paper-native ledgers cannot be combined with --sample-id-file")
    if sample_offset < 0:
        raise ValueError("--sample-offset must be >= 0")
    if manifest_path.exists():
        rows = JsonlResultStore(raw_path).results(run_id=run_id)
        if not rows:
            raise ManifestError("an immutable manifest exists but the run has no completed rows")
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

    selected_adapter = adapter or load_hf_adapter(
        config.model.id,
        revision=config.model.revision,
        model_kwargs=_model_kwargs(config, local_files_only=bool(args.local_files_only)),
        processor_kwargs={"local_files_only": bool(args.local_files_only)},
    )
    loaded_model_id = str(getattr(selected_adapter.model.config, "_name_or_path", ""))
    cached_repository = f"models--{config.model.id.replace('/', '--')}"
    loaded_path = Path(loaded_model_id)
    is_pinned_cached_snapshot = (
        loaded_path.name == config.model.revision and cached_repository in loaded_path.parts
    )
    if loaded_model_id and loaded_model_id != config.model.id and not is_pinned_cached_snapshot:
        raise ValueError(
            "reused HF adapter model does not match the resolved run config: "
            f"{loaded_model_id!r} != {config.model.id!r}"
        )
    runtime = HuggingFaceMosaicKVModel(selected_adapter, config, trace_directory=trace_root)
    summary: JsonObject
    if is_materialized_native:
        native_subset_size = None if full_dataset else cast("int | None", args.subset_size)
        ledger = load_paper_native_ledger(
            cast("str", ledger_path),
            expected_sha256=cast("str", ledger_sha),
            task=task,
            dataset_revision=config.dataset.revision,
        )
        if protocol is not None and full_dataset:
            task_contract = next(item for item in protocol.tasks if item.name == task.name)
            if (
                task_contract.sample_count is not None
                and len(ledger.samples) != task_contract.sample_count
            ):
                raise ValueError(
                    f"full-dataset protocol requires {task_contract.sample_count} ledger "
                    f"samples, observed {len(ledger.samples)}"
                )
        generation_sha = sha256_text(
            json.dumps(
                asdict(config.generation),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
        )
        comparison_controls_sha = None
        if unified_prefixkv:
            decode_policy = getattr(args, "comparison_decode_policy", None)
            if decode_policy is not None:  # noqa: SIM102 - keep validation grouped
                if (
                    not isinstance(decode_policy, dict)
                    or set(decode_policy) != {"name", "generated_capacity", "sink_tokens"}
                    or decode_policy["name"] != "common_bounded_history_v1_fullkv_reference_exempt"
                    or (
                        not config.method.is_full_cache
                        and (
                            config.bounded_decode is None
                            or asdict(config.bounded_decode)
                            != {k: decode_policy[k] for k in ("generated_capacity", "sink_tokens")}
                        )
                    )
                ):
                    raise ValueError("shared bounded-history reference policy mismatch")
            processor = selected_adapter.processor
            tokenizer = getattr(processor, "tokenizer", processor)
            comparison_controls_sha = sha256_text(
                json.dumps(
                    {
                        "model": asdict(config.model),
                        "dataset": asdict(config.dataset),
                        "execution": asdict(config.execution),
                        "generation": asdict(config.generation),
                        "ledger_sha256": ledger.sha256,
                        "processor_class": type(processor).__qualname__,
                        "chat_template": getattr(processor, "chat_template", None),
                        "tokenizer_chat_template": getattr(tokenizer, "chat_template", None),
                        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
                        "processor_configuration": (
                            processor.to_dict()
                            if callable(getattr(processor, "to_dict", None))
                            else None
                        ),
                        "image_preprocessor": (
                            processor.image_processor.to_dict()
                            if hasattr(processor, "image_processor")
                            else None
                        ),
                        "block_policy": {
                            "block_size": config.cache.block_size,
                            "physical_budget_block_size": config.cache.physical_budget_block_size,
                            "mandatory": config.cache.physical_budget_mandatory_policy,
                            "decode_growth": decode_policy or "append_only",
                        },
                        "environment_lock_sha256": sha256_bytes(
                            (
                                Path(__file__).resolve().parents[2] / "env/common/requirements.lock"
                            ).read_bytes()
                        ),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    default=str,
                )
            )
        summary = run_materialized_paper_native_evaluation(
            run_id=run_id,
            task=task,
            ledger=ledger,
            model=runtime,
            raw_output=raw_path,
            parquet_output=parquet_path,
            manifest_path=str(manifest_path),
            metric_output=cast("str", metric_output),
            metric_observations_parquet_output=cast("str", metric_observations_output),
            metric_aggregates_parquet_output=cast("str", metric_aggregates_output),
            seed=config.execution.seed,
            subset_size=native_subset_size,
            evaluator_commit=_source_commit(),
            model_revision=config.model.revision,
            tokenizer_revision=config.model.revision,
            generation_config_sha256=generation_sha,
            fullkv_reference_input=cast(
                "str | None", getattr(args, "fullkv_reference_input", None)
            ),
            fullkv_reference_sha256=cast("str | None", getattr(args, "fullkv_reference_sha", None)),
            fullkv_reference_output=cast(
                "str | None", getattr(args, "fullkv_reference_output", None)
            ),
            result_category=(
                ResultCategory.UNIFIED_COMPARISON
                if unified_milebench or unified_prefixkv or unified_coco
                else ResultCategory.PAPER_REPRODUCTION
            ),
            comparison_controls_sha256=comparison_controls_sha,
        )
    elif task.local_scorer is not None:
        summary = (
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
        subset_size = (
            None if full_dataset or explicit_sample_ids is not None else int(args.subset_size or 20)
        )
        summary = run_lmms_development_evaluation(
            run_id=run_id,
            task_names=(task_name,),
            model=runtime,
            raw_output=raw_path,
            manifest_path=str(manifest_path),
            seed=config.execution.seed,
            subset_size=subset_size,
            sample_offset=sample_offset,
            explicit_sample_ids=explicit_sample_ids,
            parquet_output=parquet_path,
            metric_output=metric_output,
            metric_parquet_output=cast("str | None", getattr(args, "metric_parquet_output", None)),
            metric_observations_parquet_output=metric_observations_output,
            metric_aggregates_parquet_output=metric_aggregates_output,
            registry=registry,
            dataset_revision=config.dataset.revision,
            result_category=(
                ResultCategory.PAPER_REPRODUCTION
                if protocol is not None
                else ResultCategory.UNIFIED_COMPARISON
            ),
        )
    if protocol is not None:
        operating_point = cache_operating_point(JsonlResultStore(raw_path).results(run_id=run_id))
        summary["cache_operating_point"] = operating_point.to_json_object()
        summary["published_protocol_id"] = protocol.protocol_id
    result_rows = JsonlResultStore(raw_path).results(run_id=run_id)
    artifacts = ArtifactProvenance(
        raw_output_sha=sha256_bytes(raw_path.read_bytes()),
        metrics_sha=sha256_bytes(parquet_path.read_bytes()),
        log_sha=_trace_sha(trace_root, run_id, raw_path),
    )
    if not manifest_path.exists():
        measurement = (
            MeasurementType.ABLATION
            if config.ablation.variant is not AblationVariant.NONE
            else (
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
        )
        RunManifestWriter().write(
            manifest_path,
            config,
            measurement,
            _input_provenance(
                config,
                task_name,
                raw_path,
                input_binding=ledger_sha,
            ),
            artifacts,
            run_id=run_id,
            result_category=(
                ResultCategory.PAPER_REPRODUCTION
                if protocol is not None
                else ResultCategory.UNIFIED_COMPARISON
            ),
            evaluation_provenance=(
                _paper_evaluation_provenance(
                    protocol,
                    task,
                    config.method.value,
                    result_rows,
                    summary,
                )
                if protocol is not None or unified_prefixkv or unified_coco
                else None
            ),
        )
    rows = result_rows
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
