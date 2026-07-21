"""CLI orchestration for measured FullKV reference workloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from mosaickv.adapters.huggingface import InternVLVideo, load_hf_adapter
from mosaickv.adapters.huggingface.base import _torch, validate_hf_revision
from mosaickv.config import RunConfig, config_sha256, load_config
from mosaickv.evaluation.messages import MediaItem, MediaKind, build_multimodal_messages
from mosaickv.fullkv import FullKV, FullKVBenchmarkConfig, FullKVBenchmarkRunner, FullKVSample
from mosaickv.manifest import (
    ArtifactProvenance,
    InputProvenance,
    RunManifestWriter,
    sha256_bytes,
    sha256_text,
)
from mosaickv.measurements.statistics import aggregate_trials
from mosaickv.measurements.storage import (
    write_aggregate_json,
    write_json_object,
    write_trial_jsonl,
)
from mosaickv.measurements.telemetry import capture_gpu_environment
from mosaickv.measurements.types import FullKVTrialMeasurement
from mosaickv.types import Backend, JsonObject, MeasurementType, Precision

FullKVCommand = Literal["debug", "smoke20", "dataset", "latency"]
_PREPROCESSING_SPEC = (
    "mosaickv FullKV workload preprocessing v1: PIL RGB; video frame list; InternVL tensor"
)


@dataclass(frozen=True, slots=True)
class WorkloadMedia:
    """Validated local media descriptor from a FullKV workload JSONL row."""

    kind: MediaKind
    paths: tuple[Path, ...]
    tensor_path: Path | None = None
    num_patches_list: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class WorkloadRow:
    """One prompt and its ordered, local-only media descriptors."""

    sample_id: str
    prompt: str
    system_prompt: str | None
    media: tuple[WorkloadMedia, ...]


def _required_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _resolve_existing_file(value: object, path: str, root: Path) -> Path:
    raw = _required_text(value, path)
    resolved = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if not resolved.is_file():
        raise ValueError(f"{path} does not identify a local file: {resolved}")
    return resolved


def _parse_media(value: object, path: str, root: Path) -> WorkloadMedia:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path} must be an object")
    record = cast("dict[str, object]", value)
    kind_text = _required_text(record.get("kind"), f"{path}.kind")
    try:
        kind = MediaKind(kind_text)
    except ValueError as error:
        raise ValueError(f"{path}.kind must be 'image' or 'video'") from error
    allowed = (
        {"kind", "path", "tensor_path"}
        if kind is MediaKind.IMAGE
        else {"kind", "frame_paths", "tensor_path", "num_patches_list"}
    )
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown fields: {', '.join(unknown)}")
    tensor_value = record.get("tensor_path")
    if tensor_value is not None:
        tensor_path = _resolve_existing_file(tensor_value, f"{path}.tensor_path", root)
        if kind is MediaKind.IMAGE:
            if "path" in record:
                raise ValueError(f"{path} cannot specify both path and tensor_path")
            return WorkloadMedia(kind, (), tensor_path)
        counts_value = record.get("num_patches_list")
        if not isinstance(counts_value, list) or not counts_value:
            raise ValueError(f"{path}.num_patches_list must be a non-empty integer list")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in counts_value
        ):
            raise ValueError(f"{path}.num_patches_list values must be positive integers")
        if "frame_paths" in record:
            raise ValueError(f"{path} cannot specify both frame_paths and tensor_path")
        return WorkloadMedia(kind, (), tensor_path, tuple(cast("list[int]", counts_value)))
    if kind is MediaKind.IMAGE:
        return WorkloadMedia(
            kind,
            (_resolve_existing_file(record.get("path"), f"{path}.path", root),),
        )
    frames = record.get("frame_paths")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{path}.frame_paths must be a non-empty list")
    return WorkloadMedia(
        kind,
        tuple(
            _resolve_existing_file(frame, f"{path}.frame_paths[{index}]", root)
            for index, frame in enumerate(frames)
        ),
    )


def load_workload(path: str | Path) -> tuple[WorkloadRow, ...]:
    """Load a strict local JSONL workload without fetching datasets or media."""

    source = Path(path).resolve()
    if not source.is_file():
        raise ValueError(f"workload does not exist: {source}")
    rows: list[WorkloadRow] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid workload JSON at line {line_number}: {error}") from error
        if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
            raise ValueError(f"workload line {line_number} must be an object")
        record = cast("dict[str, object]", value)
        unknown = sorted(set(record) - {"sample_id", "prompt", "system_prompt", "media"})
        if unknown:
            raise ValueError(
                f"workload line {line_number} contains unknown fields: {', '.join(unknown)}"
            )
        system_value = record.get("system_prompt")
        system_prompt = (
            None
            if system_value is None
            else _required_text(system_value, f"line {line_number}.system_prompt")
        )
        media_value = record.get("media", [])
        if not isinstance(media_value, list):
            raise ValueError(f"line {line_number}.media must be a list")
        rows.append(
            WorkloadRow(
                sample_id=_required_text(record.get("sample_id"), f"line {line_number}.sample_id"),
                prompt=_required_text(record.get("prompt"), f"line {line_number}.prompt"),
                system_prompt=system_prompt,
                media=tuple(
                    _parse_media(item, f"line {line_number}.media[{index}]", source.parent)
                    for index, item in enumerate(media_value)
                ),
            )
        )
    if not rows:
        raise ValueError("workload contains no samples")
    ids = [row.sample_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("workload sample_id values must be unique")
    return tuple(rows)


def select_workload(
    rows: tuple[WorkloadRow, ...], *, seed: int, count: int | None
) -> tuple[WorkloadRow, ...]:
    """Select a stable seed-keyed subset independent of input ordering."""

    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if count is not None and count < 1:
        raise ValueError("sample count must be positive")
    ordered = sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(f"{seed}\0{row.sample_id}".encode()).digest(),
            row.sample_id,
        ),
    )
    return tuple(ordered if count is None else ordered[:count])


def _validate_fullkv_config(config: RunConfig) -> None:
    if config.execution.backend is not Backend.HUGGINGFACE:
        raise ValueError("FullKV currently requires execution.backend='huggingface'")
    if config.execution.attention_implementation != "eager":
        raise ValueError("FullKV correctness gate currently requires eager attention")
    if config.generation.temperature != 0.0 or config.generation.do_sample:
        raise ValueError("FullKV requires temperature=0 and do_sample=false")
    if config.cache.retention_ratio != 1.0:
        raise ValueError("FullKV requires cache.retention_ratio=1.0")
    validate_hf_revision(config.model.revision)
    normalized_dataset_revision = config.dataset.revision.lower()
    if normalized_dataset_revision in {
        "latest",
        "main",
        "master",
    } or normalized_dataset_revision.startswith("replace-"):
        raise ValueError("FullKV requires an immutable, non-placeholder dataset revision")
    enabled = [
        name
        for name, value in (
            ("forecasting", config.forecasting.enabled),
            ("graph", config.graph.enabled),
            ("selection", config.selection.enabled),
            ("prototypes", config.prototypes.enabled),
            ("residual", config.residual.enabled),
            ("repair", config.repair.enabled),
        )
        if value
    ]
    if enabled:
        raise ValueError(
            "FullKV requires all transformation features disabled: " + ", ".join(enabled)
        )


def _precision_dtype(torch: Any, precision: Precision) -> Any:
    mapping = {
        Precision.FP32: torch.float32,
        Precision.FP16: torch.float16,
        Precision.BF16: torch.bfloat16,
    }
    try:
        return mapping[precision]
    except KeyError as error:
        raise ValueError("FullKV requires model precision fp32, fp16, or bf16") from error


def _load_media(media: WorkloadMedia, torch: Any) -> MediaItem:
    if media.tensor_path is not None:
        tensor = torch.load(media.tensor_path, map_location="cpu", weights_only=True)
        if media.kind is MediaKind.VIDEO:
            return MediaItem(media.kind, InternVLVideo(tensor, media.num_patches_list))
        return MediaItem(media.kind, tensor)
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("image/video workloads require Pillow in the HF environment") from error
    frames = []
    for path in media.paths:
        with Image.open(path) as opened:
            frames.append(opened.convert("RGB"))
    payload: Any = frames[0] if media.kind is MediaKind.IMAGE else tuple(frames)
    return MediaItem(media.kind, payload)


def _input_provenance(rows: tuple[WorkloadRow, ...], tokenization_sha: str) -> InputProvenance:
    prompt_payload = [
        {"sample_id": row.sample_id, "prompt": row.prompt, "system_prompt": row.system_prompt}
        for row in rows
    ]
    media_digest = hashlib.sha256()
    for row in rows:
        encoded_id = row.sample_id.encode("utf-8")
        media_digest.update(len(encoded_id).to_bytes(8, "big"))
        media_digest.update(encoded_id)
        for descriptor in row.media:
            media_digest.update(descriptor.kind.value.encode("ascii"))
            media_digest.update(str(descriptor.num_patches_list).encode("ascii"))
            paths = (
                (descriptor.tensor_path,)
                if descriptor.tensor_path is not None
                else descriptor.paths
            )
            for media_path in paths:
                if media_path is None:  # statically impossible; keeps the hash fail-closed
                    raise RuntimeError("media path unexpectedly missing")
                media_digest.update(sha256_bytes(media_path.read_bytes()).encode("ascii"))
    return InputProvenance(
        prompt_set_sha=sha256_text(
            json.dumps(prompt_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        ),
        media_set_sha=media_digest.hexdigest(),
        preprocessing_sha=sha256_text(_PREPROCESSING_SPEC),
        tokenization_sha=tokenization_sha,
    )


def _failed_preparation_rows(
    *,
    reference: FullKV,
    config: RunConfig,
    run_id: str,
    sample_id: str,
    manifest_path: str,
    repeated_trials: int,
    error: Exception,
    torch: Any,
) -> tuple[FullKVTrialMeasurement, ...]:
    snapshot = capture_gpu_environment(torch)
    return tuple(
        FullKVTrialMeasurement(
            run_id=run_id,
            sample_id=sample_id,
            trial_index=index,
            model_id=reference.model_id,
            model_revision=reference.model_revision,
            dataset_id=config.dataset.id,
            dataset_revision=config.dataset.revision,
            manifest_path=manifest_path,
            status="failed",
            error=f"preparation {type(error).__name__}: {error}",
            answer=None,
            generated_token_ids=(),
            timings=None,
            memory=None,
            active_cache_length=None,
            logical_sequence_length=None,
            synchronization_calls=0,
            phase_event_counts={},
            gpu_before=snapshot,
            gpu_after=snapshot,
        )
        for index in range(repeated_trials)
    )


def _sample_limit(command: FullKVCommand) -> int | None:
    return {"debug": 1, "smoke20": 20, "dataset": None, "latency": 1}[command]


def run_fullkv_command(args: argparse.Namespace) -> int:
    """Load one pinned HF model and execute the requested FullKV workload mode."""

    command = cast("FullKVCommand", args.fullkv_mode)
    config = load_config(cast("str", args.config))
    _validate_fullkv_config(config)
    rows = load_workload(cast("str", args.workload))
    count = _sample_limit(command)
    if count is not None and len(rows) < count:
        raise ValueError(f"{command} requires at least {count} workload samples; found {len(rows)}")
    selected = select_workload(rows, seed=config.execution.seed, count=count)

    output_paths = tuple(
        Path(cast("str", value)).resolve()
        for value in (args.raw_output, args.aggregate_output, args.log_output, args.manifest)
    )
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("raw, aggregate, log, and manifest outputs must be different paths")
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite output(s): " + ", ".join(existing))
    raw_path, aggregate_path, log_path, manifest_path_value = output_paths

    cache_root = Path(cast("str", args.cache_root)).resolve()
    home = Path.home().resolve()
    if cache_root == home or home in cache_root.parents:
        raise ValueError("--cache-root must be outside the home directory")
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["HF_HOME"] = str(cache_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(cache_root / "huggingface" / "hub")

    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("FullKV measured commands require a visible CUDA device")
    device = torch.device(cast("str", args.device))
    if device.type != "cuda":
        raise ValueError("--device must identify one CUDA device")
    torch.cuda.set_device(device)
    torch.manual_seed(config.execution.seed)
    torch.cuda.manual_seed_all(config.execution.seed)
    torch.use_deterministic_algorithms(config.execution.deterministic_algorithms)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model_kwargs: dict[str, Any] = {
        "attn_implementation": "eager",
        "cache_dir": str(cache_root / "huggingface" / "hub"),
        "local_files_only": not bool(args.allow_download),
        "torch_dtype": _precision_dtype(torch, config.model.precision),
        "device_map": {"": str(device)},
    }
    processor_kwargs: dict[str, Any] = {
        "cache_dir": str(cache_root / "huggingface" / "hub"),
        "local_files_only": not bool(args.allow_download),
    }
    adapter = load_hf_adapter(
        config.model.id,
        revision=config.model.revision,
        model_kwargs=model_kwargs,
        processor_kwargs=processor_kwargs,
    )
    reference = FullKV(
        adapter,
        model_id=config.model.id,
        model_revision=config.model.revision,
    )
    benchmark_config = FullKVBenchmarkConfig(
        warmups=int(args.warmups),
        repeated_trials=int(args.trials),
        max_new_tokens=config.generation.max_new_tokens,
        bootstrap_samples=int(args.bootstrap_samples),
        confidence_level=float(args.confidence_level),
        seed=config.execution.seed,
    )
    runner = FullKVBenchmarkRunner(reference, benchmark_config)
    run_id = cast("str", args.run_id)
    if not run_id.strip():
        raise ValueError("--run-id must be non-empty")
    trials: list[FullKVTrialMeasurement] = []
    tokenization_digest = hashlib.sha256()
    for row in selected:
        try:
            media = tuple(_load_media(item, torch) for item in row.media)
            messages = build_multimodal_messages(
                row.prompt,
                media,
                system_prompt=row.system_prompt,
            )
            sample = FullKVSample(row.sample_id, adapter.prepare_inputs(messages))
            runner.update_tokenization_digest(tokenization_digest, sample)
            trials.extend(
                runner.run_sample(
                    sample,
                    run_id=run_id,
                    dataset_id=config.dataset.id,
                    dataset_revision=config.dataset.revision,
                    manifest_path=str(manifest_path_value),
                )
            )
        except Exception as error:
            with suppress(Exception):
                # Preserve the original preparation failure in the raw rows.
                torch.cuda.synchronize(device)
            failure_text = f"{row.sample_id}\0{type(error).__name__}\0{error}".encode()
            tokenization_digest.update(len(failure_text).to_bytes(8, "big"))
            tokenization_digest.update(failure_text)
            trials.extend(
                _failed_preparation_rows(
                    reference=reference,
                    config=config,
                    run_id=run_id,
                    sample_id=row.sample_id,
                    manifest_path=str(manifest_path_value),
                    repeated_trials=benchmark_config.repeated_trials,
                    error=error,
                    torch=torch,
                )
            )

    aggregate = aggregate_trials(
        run_id,
        trials,
        warmups=benchmark_config.warmups,
        repeated_trials=benchmark_config.repeated_trials,
        bootstrap_samples=benchmark_config.bootstrap_samples,
        confidence_level=benchmark_config.confidence_level,
        seed=benchmark_config.seed,
    )
    write_trial_jsonl(trials, raw_path)
    write_aggregate_json(aggregate, aggregate_path)
    raw_sha = sha256_bytes(raw_path.read_bytes())
    aggregate_sha = sha256_bytes(aggregate_path.read_bytes())
    payload: JsonObject = {
        "status": "completed" if aggregate.failed_trials == 0 else "completed_with_failures",
        "command": command,
        "run_id": run_id,
        "config_sha": config_sha256(config),
        "samples": len(selected),
        "completed_trials": aggregate.completed_trials,
        "failed_trials": aggregate.failed_trials,
        "deterministic_token_match": aggregate.deterministic_token_match,
        "raw_output": str(raw_path),
        "aggregate_output": str(aggregate_path),
        "log_output": str(log_path),
        "manifest": str(manifest_path_value),
    }
    write_json_object(payload, log_path)
    log_sha = sha256_bytes(log_path.read_bytes())
    inputs = _input_provenance(selected, tokenization_digest.hexdigest())
    RunManifestWriter().write(
        manifest_path_value,
        config,
        MeasurementType.REFERENCE,
        inputs,
        ArtifactProvenance(raw_sha, aggregate_sha, log_sha),
        run_id=run_id,
    )
    print(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":") if bool(args.json) else None,
            indent=None if bool(args.json) else 2,
        )
    )
    deterministic_failure = aggregate.deterministic_token_match is False
    return 1 if aggregate.failed_trials or deterministic_failure else 0


def add_fullkv_subcommands(subparsers: Any) -> None:
    """Register four explicit FullKV execution modes on the main CLI."""

    modes: tuple[tuple[str, FullKVCommand, str, int, int], ...] = (
        ("fullkv-debug", "debug", "measure one deterministically selected sample", 0, 1),
        ("fullkv-smoke", "smoke20", "measure exactly 20 selected samples", 1, 1),
        ("fullkv-run", "dataset", "measure every workload sample", 1, 1),
        ("fullkv-latency", "latency", "run a repeated one-sample microbenchmark", 3, 10),
    )
    for name, mode, help_text, warmups, trials in modes:
        parser = subparsers.add_parser(name, help=help_text)
        parser.add_argument("--config", required=True)
        parser.add_argument("--workload", required=True)
        parser.add_argument("--run-id", required=True)
        parser.add_argument("--raw-output", required=True)
        parser.add_argument("--aggregate-output", required=True)
        parser.add_argument("--log-output", required=True)
        parser.add_argument("--manifest", required=True)
        parser.add_argument("--cache-root", required=True)
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--allow-download", action="store_true")
        parser.add_argument("--warmups", type=int, default=warmups)
        parser.add_argument("--trials", type=int, default=trials)
        parser.add_argument("--bootstrap-samples", type=int, default=2000)
        parser.add_argument("--confidence-level", type=float, default=0.95)
        parser.add_argument("--json", action="store_true", help="emit compact JSON")
        parser.set_defaults(handler=run_fullkv_command, fullkv_mode=mode)


__all__ = [
    "WorkloadMedia",
    "WorkloadRow",
    "add_fullkv_subcommands",
    "load_workload",
    "run_fullkv_command",
    "select_workload",
]
