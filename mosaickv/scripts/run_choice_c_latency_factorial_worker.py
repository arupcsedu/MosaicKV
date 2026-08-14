#!/usr/bin/env python3
"""Run one isolated cell of the Choice-C online-latency factorial experiment."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]
import torch
import yaml
from run_choice_c_gate import (
    _load_ledger,
    _mosaic_config,
    _sha,
    _write_config,
)

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import FidelityDiagnosticsConfig, config_sha256
from mosaickv.hf_cli import _model_kwargs, run_hf_evaluation
from mosaickv.measurements.telemetry import capture_gpu_environment


def factor_settings(index: int) -> tuple[str, str, str, str]:
    """Return stable factor ID plus graph, exact, and prototype gather modes."""

    if index < 0 or index > 7:
        raise ValueError("factor index must be in [0, 7]")
    graph_optimized = bool(index & 4)
    exact_optimized = bool(index & 2)
    prototype_optimized = bool(index & 1)
    factor_id = (
        f"g{int(graph_optimized)}e{int(exact_optimized)}p{int(prototype_optimized)}"
    )
    return (
        factor_id,
        "deferred_layer" if graph_optimized else "per_group_reference",
        "batched_layer" if exact_optimized else "per_head_reference",
        "batched_layer" if prototype_optimized else "per_head_reference",
    )


def _run_online(
    config: Any,
    adapter: Any,
    *,
    run_id: str,
    output_root: Path,
    ledger_path: Path,
    ledger_sha: str,
    expected_sample_ids: tuple[str, ...],
    task: str = "docvqa",
) -> pd.DataFrame:
    if config.diagnostics.enabled:
        raise RuntimeError("online-only timing requires evaluation diagnostics to be disabled")
    print(
        f"choice-c online start run_id={run_id} method={config.method.value} "
        f"samples={len(expected_sample_ids)}",
        flush=True,
    )
    config_path = output_root / "configs" / f"{run_id}-{config_sha256(config)}.yaml"
    _write_config(config, config_path)
    args = argparse.Namespace(
        task=task,
        run_id=run_id,
        output_dir=str(output_root),
        raw_output=None,
        parquet_output=None,
        manifest=None,
        trace_directory=str(output_root / "traces"),
        subset_size=None,
        full_dataset=False,
        sample_offset=0,
        sample_id_file=str(ledger_path),
        sample_id_file_sha=ledger_sha,
        local_files_only=True,
    )
    status, summary = run_hf_evaluation(args, config, adapter=adapter)
    failed_samples = summary.get("failed_samples", 0)
    if not isinstance(failed_samples, int):
        raise RuntimeError(f"run returned invalid failure count: {run_id}")
    if status or failed_samples:
        raise RuntimeError(f"run failed or contains failed samples: {run_id}: {summary}")
    frame = pd.read_parquet(cast("str", summary["parquet_output"]))
    if (
        len(frame) != len(expected_sample_ids)
        or set(frame["sample_id"]) != set(expected_sample_ids)
        or set(frame["status"]) != {"completed"}
    ):
        raise RuntimeError(f"run is incomplete: {run_id}")
    print(f"choice-c online complete run_id={run_id} rows={len(frame)}", flush=True)
    return frame


def _write_telemetry(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-index", required=True, type=int)
    parser.add_argument("--round", required=True, type=int)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.round not in {0, 1}:
        raise ValueError("round must be 0 or 1")
    if args.rank < 0 or args.rank > 7:
        raise ValueError("rank must be in [0, 7]")

    factor_id, graph_mode, exact_mode, prototype_mode = factor_settings(args.factor_index)
    spec_path = args.spec.resolve()
    profile_path = args.profile.resolve()
    ledger_path = args.ledger.resolve()
    output_root = args.output_root.resolve()
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text()))
    profile = json.loads(profile_path.read_text())
    parameters = profile.get("parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("frozen fidelity profile has no parameter mapping")
    ledger, ledger_sha = _load_ledger(ledger_path)
    sample_ids = tuple(cast("list[str]", ledger["sample_ids"]))
    config = _mosaic_config(
        parameters,
        ledger,
        ledger_sha,
        retention_ratio=float(spec["retention_ratio"]),
        output_length=int(spec["output_length"]),
        seed=int(spec["seed"]),
    )
    config = replace(
        config,
        diagnostics=FidelityDiagnosticsConfig(),
        cache=replace(config.cache, exact_gather_mode=exact_mode),
        graph=replace(config.graph, attention_host_transfer_mode=graph_mode),
        prototypes=replace(config.prototypes, source_gather_mode=prototype_mode),
    )
    run_id = (
        f"choice-c-online-{factor_id}-round{args.round}-rank{args.rank}-"
        f"{_sha(spec_path)[:10]}"
    )
    telemetry_path = output_root / "telemetry" / f"{run_id}.json"
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_system_latency_factorial",
        "run_id": run_id,
        "factor_index": args.factor_index,
        "factor_id": factor_id,
        "round": args.round,
        "rank": args.rank,
        "graph_host_transfer_mode": graph_mode,
        "exact_gather_mode": exact_mode,
        "prototype_source_gather_mode": prototype_mode,
        "fixed_clock_status": os.environ.get("MOSAICKV_FIXED_CLOCK_STATUS", "not_attempted"),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "not_used"),
            "step_id": os.environ.get("SLURM_STEP_ID", "not_used"),
            "node": os.environ.get("SLURMD_NODENAME", "not_used"),
            "local_id": os.environ.get("SLURM_LOCALID", "not_used"),
            "cpu_bind": os.environ.get("SLURM_CPU_BIND", "not_set"),
            "mem_bind": os.environ.get("SLURM_MEM_BIND", "not_set"),
            "step_gpus": os.environ.get("SLURM_STEP_GPUS", "not_set"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        "before_model_load": asdict(capture_gpu_environment(torch)),
        "status": "started",
    }
    try:
        adapter = load_hf_adapter(
            config.model.id,
            revision=config.model.revision,
            model_kwargs=_model_kwargs(config, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        warmup_ledger = dict(ledger)
        warmup_ledger["sample_ids"] = [sample_ids[0]]
        warmup_ledger["sample_count"] = 1
        warmup_path = output_root / "warmup_ledgers" / f"{factor_id}-round{args.round}.json"
        warmup_payload = json.dumps(warmup_ledger, indent=2, sort_keys=True) + "\n"
        warmup_path.parent.mkdir(parents=True, exist_ok=True)
        if warmup_path.exists() and warmup_path.read_text() != warmup_payload:
            raise RuntimeError(f"refusing to change existing warmup ledger: {warmup_path}")
        if not warmup_path.exists():
            warmup_path.write_text(warmup_payload)
        warmup_sha = _sha(warmup_path)
        _run_online(
            config,
            adapter,
            run_id=f"choice-c-online-warmup-{factor_id}-round{args.round}-rank{args.rank}",
            output_root=output_root,
            ledger_path=warmup_path,
            ledger_sha=warmup_sha,
            expected_sample_ids=(sample_ids[0],),
        )
        telemetry["before_measurement"] = asdict(capture_gpu_environment(torch))
        _run_online(
            config,
            adapter,
            run_id=run_id,
            output_root=output_root,
            ledger_path=ledger_path,
            ledger_sha=ledger_sha,
            expected_sample_ids=sample_ids,
        )
        telemetry["status"] = "completed"
    except Exception as error:
        telemetry["status"] = "failed"
        telemetry["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        telemetry["after_measurement"] = asdict(capture_gpu_environment(torch))
        _write_telemetry(telemetry_path, telemetry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
