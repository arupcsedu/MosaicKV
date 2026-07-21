"""Command-line interface for diagnostics and research-workflow preflights."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from mosaickv import __version__
from mosaickv.adapters import default_registry
from mosaickv.config import ConfigurationError, config_sha256, load_config
from mosaickv.doctor import doctor_report
from mosaickv.evaluation.harness import EvaluationHarness
from mosaickv.evaluation.storage import JsonlResultStore, load_jsonl
from mosaickv.evaluation.synthetic import SyntheticColorModel
from mosaickv.evaluation.tasks import (
    default_task_registry,
    load_synthetic_samples,
    select_samples,
)
from mosaickv.fullkv_cli import add_fullkv_subcommands
from mosaickv.logging import configure_logging, get_logger
from mosaickv.manifest import (
    ArtifactProvenance,
    InputProvenance,
    ManifestError,
    RunManifestWriter,
    sha256_bytes,
    sha256_text,
)
from mosaickv.smoke import run_cpu_smoke
from mosaickv.types import JsonObject, MeasurementType, MosaicKVMethod


def _emit(payload: JsonObject, *, compact: bool) -> None:
    if compact:
        print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(payload, sort_keys=True, indent=2))


def _doctor_command(args: argparse.Namespace) -> int:
    report = doctor_report()
    _emit(report, compact=bool(args.json))
    cuda = cast("JsonObject", report["cuda"])
    if bool(args.require_gpu) and not bool(cuda["available"]):
        return 3
    return 0


def _inspect_model_command(args: argparse.Namespace) -> int:
    registry = default_registry()
    adapter = registry.resolve(str(args.model_id))
    report = adapter.inspect(str(args.model_id), cast("str | None", args.revision))
    _emit(report, compact=bool(args.json))
    return 0


def _smoke_hashes() -> InputProvenance:
    return InputProvenance(
        prompt_set_sha=sha256_text("mosaickv synthetic smoke prompt schema v1"),
        media_set_sha=sha256_text("mosaickv synthetic smoke media schema v1"),
        preprocessing_sha=sha256_text("mosaickv synthetic smoke preprocessing schema v1"),
        tokenization_sha=sha256_text("mosaickv synthetic smoke tokenization schema v1"),
    )


def _smoke_command(args: argparse.Namespace) -> int:
    result = run_cpu_smoke(
        seed=int(args.seed),
        layers=int(args.layers),
        sequence_length=int(args.sequence_length),
        kv_heads=int(args.kv_heads),
        head_dim=int(args.head_dim),
        retention_ratio=float(args.retention_ratio),
    )
    payload = result.to_json_object()
    manifest_path = cast("str | None", args.manifest)
    if manifest_path is not None:
        from mosaickv.config import synthetic_smoke_config

        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        result_sha = sha256_text(serialized)
        writer = RunManifestWriter()
        written = writer.write(
            manifest_path,
            synthetic_smoke_config(seed=int(args.seed)),
            MeasurementType.VALIDATION_SMOKE,
            _smoke_hashes(),
            ArtifactProvenance(result_sha, "not_applicable", result_sha),
        )
        payload["manifest_path"] = str(written.resolve())
    _emit(payload, compact=bool(args.json))
    return 0 if result.exact_equivalence else 1


def _preflight_command(args: argparse.Namespace) -> int:
    config = load_config(cast("str", args.config))
    mode = str(args.command)
    payload: JsonObject = {
        "status": "not_run",
        "mode": mode,
        "config_valid": True,
        "config_sha": config_sha256(config),
        "model_id": config.model.id,
        "model_revision": config.model.revision,
        "backend": config.execution.backend.value,
        "reason": (
            f"{mode} execution is intentionally disabled until its milestone implementation "
            "and correctness gates exist"
        ),
    }
    _emit(payload, compact=bool(args.json))
    return 0


def _synthetic_evaluation_inputs() -> InputProvenance:
    return InputProvenance(
        prompt_set_sha=sha256_text("mosaickv synthetic evaluation prompts v1"),
        media_set_sha=sha256_text("mosaickv synthetic RGB pixels v1"),
        preprocessing_sha=sha256_text("identity RGB tuple preprocessing v1"),
        tokenization_sha=sha256_text("synthetic whitespace token count v1"),
    )


def _evaluate_command(args: argparse.Namespace) -> int:
    registry = default_task_registry()
    if bool(args.list_tasks):
        _emit(
            {
                "tasks": [
                    {
                        "name": name,
                        "scoring_owner": registry.resolve(name).scoring_owner,
                        "development_lmms_task": registry.resolve(name).development_lmms_task,
                        "requires_video": registry.resolve(name).requires_video,
                    }
                    for name in registry.names()
                ]
            },
            compact=bool(args.json),
        )
        return 0
    from mosaickv.hf_cli import resolve_hf_config, run_hf_evaluation
    from mosaickv.sglang_cli import resolve_sglang_config, run_sglang_evaluation
    from mosaickv.vllm_cli import resolve_vllm_config, run_vllm_evaluation

    sglang_config = resolve_sglang_config(args)
    if sglang_config is not None:
        status, payload = run_sglang_evaluation(args, sglang_config)
        _emit(payload, compact=bool(args.json))
        return status

    vllm_config = resolve_vllm_config(args)
    if vllm_config is not None:
        status, payload = run_vllm_evaluation(args, vllm_config)
        _emit(payload, compact=bool(args.json))
        return status

    hf_config = resolve_hf_config(args)
    if hf_config is not None:
        status, payload = run_hf_evaluation(args, hf_config)
        _emit(payload, compact=bool(args.json))
        return status
    task_name = cast("str | None", args.task)
    config_path = cast("str | None", args.config)
    if task_name is None:
        if config_path is None:
            raise ValueError("evaluate requires --task, --list-tasks, or --config")
        return _preflight_command(args)
    if config_path is not None:
        raise ValueError("--task and --config cannot be used together")
    if task_name not in {"synthetic_ci", "synthetic_smoke"}:
        task = registry.resolve(task_name)
        raise ValueError(
            f"task {task.name!r} requires a local model object and the lmms-eval Python API; "
            "the CLI currently runs only synthetic_ci"
        )
    run_id = cast("str | None", args.run_id)
    raw_output = cast("str | None", args.raw_output)
    manifest = cast("str | None", args.manifest)
    if run_id is None or raw_output is None or manifest is None:
        raise ValueError(
            "synthetic_ci/synthetic_smoke requires --run-id, --raw-output, and --manifest"
        )
    manifest_path = str(Path(manifest).resolve())
    samples = load_synthetic_samples()
    selected = select_samples(
        samples,
        seed=int(args.seed),
        subset_size=cast("int | None", args.subset_size),
    )
    destination = Path(manifest_path)
    foreign_run_ids = {row.run_id for row in load_jsonl(raw_output) if row.run_id != run_id}
    if foreign_run_ids:
        raise ValueError(
            "the CLI requires one run per raw file; found other run IDs: "
            + ", ".join(sorted(foreign_run_ids))
        )
    if destination.exists():
        prior = JsonlResultStore(raw_output).completed_sample_ids(run_id)
        selected_ids = {sample.sample_id for sample in selected}
        if not selected_ids <= prior:
            raise ManifestError(
                "an immutable manifest already exists but the requested run has pending samples"
            )
    summary = EvaluationHarness(registry).run(
        run_id=run_id,
        task_name=task_name,
        samples=samples,
        model=SyntheticColorModel(),
        raw_output=raw_output,
        manifest_path=manifest_path,
        seed=int(args.seed),
        subset_size=cast("int | None", args.subset_size),
        parquet_output=cast("str | None", args.parquet_output),
    )
    output_path = Path(raw_output)
    parquet_path = summary.parquet_output
    artifacts = ArtifactProvenance(
        raw_output_sha=sha256_bytes(output_path.read_bytes()),
        metrics_sha=(
            sha256_bytes(Path(parquet_path).read_bytes())
            if parquet_path is not None
            else "not_applicable"
        ),
        log_sha=sha256_bytes(output_path.read_bytes()),
    )
    if not destination.exists():
        from mosaickv.config import synthetic_evaluation_config

        RunManifestWriter().write(
            destination,
            synthetic_evaluation_config(seed=int(args.seed)),
            MeasurementType.VALIDATION_SMOKE,
            _synthetic_evaluation_inputs(),
            artifacts,
            run_id=run_id,
        )
    else:
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if (
            existing.get("run_id") != run_id
            or existing.get("artifacts", {}).get("raw_output_sha") != artifacts.raw_output_sha
        ):
            raise ManifestError("existing manifest does not match the resumed raw artifact")
    payload = summary.to_json_object()
    payload["manifest_path"] = manifest_path
    _emit(payload, compact=bool(args.json))
    return 0 if summary.failed_samples == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI parser for tests and console entry points."""

    parser = argparse.ArgumentParser(prog="mosaickv")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="report Python, CUDA, and backend availability"
    )
    doctor_parser.add_argument("--json", action="store_true", help="emit compact JSON")
    doctor_parser.add_argument(
        "--require-gpu", action="store_true", help="return nonzero when no GPU is visible"
    )
    doctor_parser.set_defaults(handler=_doctor_command)

    inspect_parser = subparsers.add_parser(
        "inspect-model", help="inspect audited model capabilities without loading weights"
    )
    inspect_parser.add_argument("model_id")
    inspect_parser.add_argument("--revision")
    inspect_parser.add_argument("--json", action="store_true", help="emit compact JSON")
    inspect_parser.set_defaults(handler=_inspect_model_command)

    smoke_parser = subparsers.add_parser("smoke", help="run a CPU-only synthetic tensor smoke test")
    smoke_parser.add_argument("--seed", type=int, default=0)
    smoke_parser.add_argument("--layers", type=int, default=2)
    smoke_parser.add_argument("--sequence-length", type=int, default=32)
    smoke_parser.add_argument("--kv-heads", type=int, default=2)
    smoke_parser.add_argument("--head-dim", type=int, default=8)
    smoke_parser.add_argument("--retention-ratio", type=float, default=1.0)
    smoke_parser.add_argument("--manifest", help="write an immutable validation manifest")
    smoke_parser.add_argument("--json", action="store_true", help="emit compact JSON")
    smoke_parser.set_defaults(handler=_smoke_command)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="run synthetic CI evaluation or validate an evaluation config"
    )
    evaluate_parser.add_argument("--config")
    evaluate_parser.add_argument("--task")
    evaluate_parser.add_argument("--model")
    evaluate_parser.add_argument("--model-revision")
    evaluate_parser.add_argument("--dataset-revision")
    evaluate_parser.add_argument(
        "--backend", default="hf", choices=["hf", "huggingface", "vllm", "sglang"]
    )
    evaluate_parser.add_argument("--attention-backend", default="eager")
    evaluate_parser.add_argument(
        "--method",
        default="full_kv",
        choices=[method.value for method in MosaicKVMethod],
    )
    evaluate_parser.add_argument("--retention-ratio", type=float, default=1.0)
    evaluate_parser.add_argument("--cache-budget", type=int, default=2_147_483_647)
    evaluate_parser.add_argument(
        "--budget-unit", default="blocks", choices=["blocks", "retained_slots", "bytes"]
    )
    evaluate_parser.add_argument("--block-size", type=int, default=16)
    evaluate_parser.add_argument(
        "--forecast",
        default="hybrid",
        choices=["prompt_window", "draft_rollout", "hybrid"],
    )
    evaluate_parser.add_argument("--prompt-window", type=int, default=16)
    evaluate_parser.add_argument("--draft-tokens", type=int, default=4)
    evaluate_parser.add_argument("--forecast-centroids", type=int, default=4)
    evaluate_parser.add_argument("--lookm-recent-ratio", type=float, default=0.1)
    evaluate_parser.add_argument("--lookm-important-ratio", type=float, default=0.1)
    evaluate_parser.add_argument(
        "--lookm-merge-strategy",
        default="pivotal",
        choices=["averaged", "pivotal", "weighted"],
    )
    evaluate_parser.add_argument(
        "--prefixkv-profile-mode",
        default="offline_profile",
        choices=["offline_profile", "fixed_global"],
    )
    evaluate_parser.add_argument("--prefixkv-profile")
    evaluate_parser.add_argument("--prefixkv-start-size", type=int, default=1)
    evaluate_parser.add_argument("--prefixkv-protect-size", type=int, default=1)
    evaluate_parser.add_argument("--prefixkv-eviction-distance", type=int, default=-25)
    evaluate_parser.add_argument("--vl-cache-sparsity-threshold", type=float, default=0.01)
    evaluate_parser.add_argument("--vl-cache-min-layer-retention", type=float, default=0.01)
    evaluate_parser.add_argument("--vl-cache-max-layer-retention", type=float, default=1.0)
    evaluate_parser.add_argument("--vl-cache-recent-window-fraction", type=float, default=0.1)
    evaluate_parser.add_argument("--vl-cache-max-post-vision-queries", type=int)
    evaluate_parser.add_argument(
        "--repair-policy",
        default="entropy_or_prototype_risk",
        choices=[
            "none",
            "entropy",
            "prototype_risk",
            "entropy_or_prototype_risk",
            "oracle",
        ],
    )
    evaluate_parser.add_argument("--entropy-threshold", type=float, default=0.5)
    evaluate_parser.add_argument("--prototype-risk-threshold", type=float, default=0.25)
    evaluate_parser.add_argument("--repair-blocks", type=int, default=2)
    evaluate_parser.add_argument("--max-new-tokens", type=int, default=16)
    evaluate_parser.add_argument("--precision", default="bf16", choices=["fp32", "fp16", "bf16"])
    evaluate_parser.add_argument("--output-dir", default="runs")
    evaluate_parser.add_argument("--trace-directory")
    evaluate_parser.add_argument("--local-files-only", action="store_true")
    evaluate_parser.add_argument(
        "--enable-mosaickv",
        action="store_true",
        help="request a fail-closed experimental native serving-backend integration",
    )
    evaluate_parser.add_argument("--tensor-parallel-size", type=int, default=1)
    evaluate_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    evaluate_parser.add_argument("--vllm-max-model-len", type=int)
    evaluate_parser.add_argument("--sglang-context-length", type=int, default=4096)
    evaluate_parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.8)
    evaluate_parser.add_argument("--sglang-port", type=int, default=0)
    evaluate_parser.add_argument("--sglang-startup-timeout", type=float, default=1800.0)
    evaluate_parser.add_argument(
        "--cache-probe-repeats",
        type=int,
        default=2,
        help="repeat each identical request to measure prefix/multimodal cache behavior",
    )
    evaluate_parser.add_argument("--list-tasks", action="store_true")
    evaluate_parser.add_argument("--run-id")
    evaluate_parser.add_argument("--raw-output")
    evaluate_parser.add_argument("--parquet-output")
    evaluate_parser.add_argument("--manifest")
    evaluate_parser.add_argument("--seed", type=int, default=0)
    evaluate_parser.add_argument("--subset-size", type=int)
    evaluate_parser.add_argument("--json", action="store_true", help="emit compact JSON")
    evaluate_parser.set_defaults(handler=_evaluate_command)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="validate a benchmark configuration without starting a measured run"
    )
    benchmark_parser.add_argument("--config", required=True)
    benchmark_parser.add_argument("--json", action="store_true", help="emit compact JSON")
    benchmark_parser.set_defaults(handler=_preflight_command)
    add_fullkv_subcommands(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the MosaicKV CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    logger = get_logger("mosaickv.cli")
    handler = cast("Callable[[argparse.Namespace], int]", args.handler)
    try:
        logger.info("command started", extra={"event": "command_started", "command": args.command})
        return handler(args)
    except (
        ConfigurationError,
        ManifestError,
        LookupError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        logger.error(
            "command failed",
            extra={"event": "command_failed", "command": args.command, "error": str(error)},
        )
        print(
            json.dumps(
                {"status": "error", "command": args.command, "error": str(error)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
