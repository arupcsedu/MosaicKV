#!/usr/bin/env python3
"""Run one reference or batched-compute Choice-C latency cell."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from run_choice_c_gate import _load_ledger, _mosaic_config, _sha
from run_choice_c_latency_factorial_worker import _run_online, _write_telemetry

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import FidelityDiagnosticsConfig
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-mode", choices=("reference", "optimized"), required=True)
    parser.add_argument("--round", choices=(0, 1), required=True, type=int)
    parser.add_argument("--rank", choices=(0, 1), required=True, type=int)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

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
    optimized = args.compute_mode == "optimized"
    config = replace(
        config,
        diagnostics=FidelityDiagnosticsConfig(capture_selection_evidence=True),
        cache=replace(config.cache, exact_gather_mode="batched_layer"),
        graph=replace(config.graph, attention_host_transfer_mode="deferred_layer"),
        prototypes=replace(config.prototypes, source_gather_mode="per_head_reference"),
        forecasting=replace(
            config.forecasting,
            centroid_execution_mode=("batched_layer" if optimized else "reference_per_head"),
        ),
        utility=replace(
            config.utility,
            execution_mode="vectorized_edges" if optimized else "reference_sparse",
        ),
    )
    run_id = (
        f"choice-c-compute-{args.compute_mode}-round{args.round}-rank{args.rank}-"
        f"{_sha(spec_path)[:10]}"
    )
    telemetry_path = output_root / "telemetry" / f"{run_id}.json"
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_system_latency_compute_ab",
        "run_id": run_id,
        "compute_mode": args.compute_mode,
        "round": args.round,
        "rank": args.rank,
        "graph_host_transfer_mode": config.graph.attention_host_transfer_mode,
        "exact_gather_mode": config.cache.exact_gather_mode,
        "prototype_source_gather_mode": config.prototypes.source_gather_mode,
        "centroid_execution_mode": config.forecasting.centroid_execution_mode,
        "utility_execution_mode": config.utility.execution_mode,
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
        warmup_path = output_root / "warmup_ledgers" / f"{args.compute_mode}-round{args.round}.json"
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
            run_id=(
                f"choice-c-compute-warmup-{args.compute_mode}-round{args.round}-rank{args.rank}"
            ),
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
