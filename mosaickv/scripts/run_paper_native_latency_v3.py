#!/usr/bin/env python3
"""Run the registered persistent PrefixKV-versus-MosaicKV latency trial."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import torch

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.cli import build_parser
from mosaickv.evaluation.paper_native_latency import (
    load_paper_native_latency_protocol,
)
from mosaickv.hf_cli import _model_kwargs, resolve_hf_config, run_hf_evaluation
from mosaickv.measurements.telemetry import capture_gpu_environment

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
DATASET_REVISION = "407c2d2f8fc13611d13f72a5e18b8c33bf9c44e6e05278bf0b4da41deeacbe6a"
PROTOCOL_ID = "prefixkv_hf_reimpl_native_metrics_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_once(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _git_sha(project_root: Path) -> str:
    observed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty.strip():
        raise RuntimeError("paper latency trials require a clean worktree")
    return observed


def _validated_output_root(path: Path) -> Path:
    """Resolve an output path while accepting the cluster's scratch alias.

    On this cluster ``/scratch`` is a symlink to the mounted
    ``/sfs/weka/scratch`` filesystem.  Comparing a resolved candidate with
    the unresolved spelling of ``/scratch/djy8hg`` rejects a valid artifact
    directory before any model work begins.  Resolve both boundaries, then
    retain the strict ancestry check.
    """

    output_root = path.resolve()
    scratch_root = Path("/scratch/djy8hg").resolve()
    try:
        output_root.relative_to(scratch_root)
    except ValueError as error:
        raise ValueError("paper latency output must be under /scratch/djy8hg") from error
    return output_root


def _warmup_ledger(source: Path, destination: Path) -> str:
    payload = json.loads(source.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("PrefixKV ledger contains no samples")
    payload["samples"] = samples[:1]
    _write_once(destination, cast("dict[str, Any]", payload))
    return _sha256(destination)


def _base_cli(method: str) -> list[str]:
    retention_ratio = "1.0" if method == "full_kv" else "0.2"
    return [
        "evaluate",
        "--task",
        "llava_description",
        "--model",
        MODEL_ID,
        "--model-revision",
        MODEL_REVISION,
        "--dataset-revision",
        DATASET_REVISION,
        "--backend",
        "hf",
        "--attention-backend",
        "eager",
        "--precision",
        "bf16",
        "--method",
        method,
        "--retention-ratio",
        retention_ratio,
        "--budget-unit",
        "blocks",
        "--block-size",
        "1",
        "--forecast",
        "prompt_window",
        "--prompt-window",
        "16",
        "--draft-tokens",
        "0",
        "--repair-policy",
        "none",
        "--max-new-tokens",
        "144",
        "--seed",
        "0",
        "--prefixkv-profile-mode",
        "fixed_global",
        "--local-files-only",
    ]


def _paired_method_order(
    methods: tuple[str, str], repetition: int
) -> tuple[str, str]:
    """Alternate a two-method trial order without changing either method."""

    if len(methods) != 2 or len(set(methods)) != 2:
        raise ValueError("latency verification requires exactly two distinct methods")
    if repetition < 0:
        raise ValueError("repetition must be nonnegative")
    return methods if repetition % 2 == 0 else tuple(reversed(methods))


def _run_once(
    *,
    method: str,
    adapter: Any,
    run_id: str,
    output_root: Path,
    ledger: Path,
    ledger_sha: str,
    published_protocol: Path,
    fullkv_reference: Path,
    fullkv_reference_sha: str | None,
    write_fullkv_reference: bool = False,
) -> dict[str, Any]:
    run_root = output_root / "runs" / run_id
    trace_root = output_root / "traces"
    argv = [
        *_base_cli(method),
        "--output-dir",
        str(run_root),
        "--trace-directory",
        str(trace_root),
        "--run-id",
        run_id,
        "--raw-output",
        str(run_root / "raw.jsonl"),
        "--parquet-output",
        str(run_root / "results.parquet"),
        "--metric-output",
        str(run_root / "metrics.jsonl"),
        "--metric-observations-parquet-output",
        str(run_root / "metric-observations.parquet"),
        "--metric-aggregates-parquet-output",
        str(run_root / "metric-aggregates.parquet"),
        "--manifest",
        str(run_root / "manifest.json"),
        "--published-protocol",
        str(published_protocol),
        "--published-protocol-id",
        PROTOCOL_ID,
        "--paper-native-ledger",
        str(ledger),
        "--paper-native-ledger-sha",
        ledger_sha,
    ]
    if method == "full_kv":
        # The paper-native lane requires every FullKV invocation to emit a
        # reference artifact.  The initial reference is shared by compressed
        # runs; repeated FullKV timing trials use a private write-once output
        # so they cannot overwrite that immutable comparison reference.
        reference_output = (
            fullkv_reference
            if write_fullkv_reference
            else run_root / "fullkv-reference-output.json"
        )
        argv.extend(("--fullkv-reference-output", str(reference_output)))
    else:
        if fullkv_reference_sha is None:
            raise ValueError("compressed timing trials require a FullKV reference SHA")
        argv.extend(
            (
                "--fullkv-reference-input",
                str(fullkv_reference),
                "--fullkv-reference-sha",
                fullkv_reference_sha,
            )
        )
    args = build_parser().parse_args(argv)
    config = resolve_hf_config(args)
    if config is None:
        raise RuntimeError("paper latency worker did not resolve an HF configuration")
    if torch.cuda.is_available():
        torch.cuda.synchronize(adapter.device)
        torch.cuda.reset_peak_memory_stats(adapter.device)
    status, summary = run_hf_evaluation(args, config, adapter=adapter)
    if torch.cuda.is_available():
        torch.cuda.synchronize(adapter.device)
    failed_samples = summary.get("failed_samples")
    if status or not isinstance(failed_samples, int) or failed_samples != 0:
        raise RuntimeError(f"latency trial failed: {run_id}: {summary}")
    result_path = Path(cast("str", summary["parquet_output"]))
    frame = pd.read_parquet(result_path)
    if len(frame) < 1 or set(frame["status"].astype(str)) != {"completed"}:
        raise RuntimeError(f"latency trial is incomplete: {run_id}")
    return {
        "run_id": run_id,
        "method": method,
        "result_path": str(result_path),
        "manifest_path": str(summary["manifest_path"]),
        "metric_path": str(run_root / "metrics.jsonl"),
        "trace_root": str(trace_root / run_id),
        "sample_ids": sorted(str(value) for value in frame["sample_id"]),
        "mean_end_to_end_seconds": float(frame["end_to_end_time"].mean()),
        "mean_active_kv_bytes": float(frame["active_kv_bytes"].mean()),
    }


def _settle(adapter: Any) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(adapter.device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--published-protocol", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--ledger-sha", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument(
        "--verification-fullkv-mosaickv-only",
        action="store_true",
        help=(
            "run a non-paper FullKV-versus-MosaicKV latency verification; "
            "this deliberately excludes PrefixKV when its variable layer capacities "
            "are not representable by the common HF cache transport"
        ),
    )
    parser.add_argument("--verification-warmups", type=int, default=2)
    parser.add_argument("--verification-repetitions", type=int, default=3)
    args = parser.parse_args()

    if args.verification_warmups < 1 or args.verification_repetitions < 1:
        raise ValueError("verification warmups and repetitions must be positive")

    project_root = Path(__file__).resolve().parents[2]
    observed_sha = _git_sha(project_root)
    if observed_sha != args.source_git_sha or len(observed_sha) != 40:
        raise RuntimeError("latency worker source SHA differs from the authorized commit")
    protocol_path = args.protocol.resolve(strict=True)
    protocol = load_paper_native_latency_protocol(protocol_path)
    published_protocol = args.published_protocol.resolve(strict=True)
    ledger = args.ledger.resolve(strict=True)
    if _sha256(ledger) != args.ledger_sha:
        raise RuntimeError("PrefixKV ledger hash mismatch")
    output_root = _validated_output_root(args.output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"latency output root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    warmup_ledger = output_root / "inputs/warmup-ledger.json"
    warmup_sha = _warmup_ledger(ledger, warmup_ledger)

    config_args = build_parser().parse_args(_base_cli("mosaickv_exact"))
    model_config = resolve_hf_config(config_args)
    if model_config is None:
        raise RuntimeError("could not resolve model configuration")
    adapter = load_hf_adapter(
        model_config.model.id,
        revision=model_config.model.revision,
        model_kwargs=_model_kwargs(model_config, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    fullkv_reference = output_root / "inputs/fullkv-reference.json"
    reference_run = _run_once(
        method="full_kv",
        adapter=adapter,
        run_id="latency-fullkv-reference",
        output_root=output_root,
        ledger=ledger,
        ledger_sha=args.ledger_sha,
        published_protocol=published_protocol,
        fullkv_reference=fullkv_reference,
        fullkv_reference_sha=None,
        write_fullkv_reference=True,
    )
    fullkv_reference_sha = _sha256(fullkv_reference)
    _settle(adapter)
    verification_only = bool(args.verification_fullkv_mosaickv_only)
    methods = (
        ("full_kv", "mosaickv_exact") if verification_only else protocol.timing.methods
    )
    warmup_count = (
        args.verification_warmups
        if verification_only
        else protocol.timing.warmups_per_method
    )
    repetition_count = (
        args.verification_repetitions
        if verification_only
        else protocol.timing.timed_repetitions_per_method
    )
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "experiment_id": protocol.experiment_id,
        "classification": (
            "preflight_fullkv_mosaickv_latency_verification"
            if verification_only
            else protocol.classification
        ),
        "paper_measurement_eligible": not verification_only,
        "verification_only": verification_only,
        "methods": list(methods),
        "source_git_sha": observed_sha,
        "source_git_dirty": False,
        "protocol_path": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "published_protocol_path": str(published_protocol),
        "published_protocol_sha256": _sha256(published_protocol),
        "ledger_path": str(ledger),
        "ledger_sha256": args.ledger_sha,
        "fullkv_reference_path": str(fullkv_reference),
        "fullkv_reference_sha256": fullkv_reference_sha,
        "fullkv_reference_run": reference_run,
        "warmups_per_method": warmup_count,
        "timed_repetitions_per_method": repetition_count,
        "trial_order": protocol.timing.trial_order,
        "persistent_model_process": True,
        "before_measurement": None,
        "warmups": [],
        "trials": [],
        "status": "started",
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "not_used"),
            "node": os.environ.get("SLURMD_NODENAME", "not_used"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        },
    }
    telemetry_path = output_root / "telemetry.json"
    try:
        for repetition in range(warmup_count):
            ordered_methods = _paired_method_order(methods, repetition)
            for order, method in enumerate(ordered_methods):
                record = _run_once(
                    method=method,
                    adapter=adapter,
                    run_id=f"latency-warmup-{repetition:02d}-{order}-{method}",
                    output_root=output_root,
                    ledger=warmup_ledger,
                    ledger_sha=warmup_sha,
                    published_protocol=published_protocol,
                    fullkv_reference=fullkv_reference,
                    fullkv_reference_sha=fullkv_reference_sha,
                )
                record.update({"phase": "warmup", "repetition": repetition, "order": order})
                cast("list[dict[str, Any]]", telemetry["warmups"]).append(record)
                _settle(adapter)
        telemetry["before_measurement"] = asdict(capture_gpu_environment(torch))
        for repetition in range(repetition_count):
            ordered_methods = _paired_method_order(methods, repetition)
            for order, method in enumerate(ordered_methods):
                if _git_sha(project_root) != observed_sha:
                    raise RuntimeError("repository changed during persistent latency trial")
                record = _run_once(
                    method=method,
                    adapter=adapter,
                    run_id=f"latency-trial-{repetition:02d}-{order}-{method}",
                    output_root=output_root,
                    ledger=ledger,
                    ledger_sha=args.ledger_sha,
                    published_protocol=published_protocol,
                    fullkv_reference=fullkv_reference,
                    fullkv_reference_sha=fullkv_reference_sha,
                )
                record.update({"phase": "measured", "repetition": repetition, "order": order})
                cast("list[dict[str, Any]]", telemetry["trials"]).append(record)
                _settle(adapter)
        telemetry["status"] = "completed"
    except Exception as error:
        telemetry["status"] = "failed"
        telemetry["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        telemetry["after_measurement"] = asdict(capture_gpu_environment(torch))
        _write_once(telemetry_path, telemetry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
