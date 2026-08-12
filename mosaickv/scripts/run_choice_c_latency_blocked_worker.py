#!/usr/bin/env python3
"""Run alternating reference/optimized latency cells on one physical GPU."""

from __future__ import annotations

import argparse
import gc
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
from mosaickv.types import JsonValue


def mode_order(rank: int, cycle: int) -> tuple[str, str]:
    """Counterbalance A/B order across GPUs and consecutive cycles."""

    if rank not in {0, 1}:
        raise ValueError("rank must be 0 or 1")
    if cycle < 0:
        raise ValueError("cycle must be nonnegative")
    if (rank + cycle) % 2 == 0:
        return "reference", "optimized"
    return "optimized", "reference"


def _mode_config(config: Any, mode: str) -> Any:
    if mode not in {"reference", "optimized"}:
        raise ValueError(f"unsupported compute mode: {mode}")
    optimized = mode == "optimized"
    return replace(
        config,
        forecasting=replace(
            config.forecasting,
            centroid_execution_mode=("batched_layer" if optimized else "reference_per_head"),
        ),
        utility=replace(
            config.utility,
            execution_mode="vectorized_edges" if optimized else "reference_sparse",
        ),
        graph=replace(
            config.graph,
            attention_host_transfer_mode=(
                "deferred_layer_vectorized" if optimized else "deferred_layer"
            ),
            edge_accumulation_mode="preselected" if optimized else "dictionary",
        ),
        cache=replace(
            config.cache,
            # v33 established that the layer-batched indexed scatter was
            # 37.88% slower on A100. Keep packing identical in both arms so
            # this gate isolates the beneficial compute optimizations.
            packing_mode="per_head_reference",
        ),
        residual=replace(
            config.residual,
            index_mode="compact" if optimized else "materialized",
        ),
    )


def _settle(adapter: Any) -> None:
    """Release Python cycles and finish device work outside measured regions."""

    gc.collect()
    if torch.cuda.is_available() and getattr(adapter.device, "type", "cpu") == "cuda":
        torch.cuda.synchronize(adapter.device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", choices=(0, 1), required=True, type=int)
    parser.add_argument("--cycles", default=4, type=int)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.cycles < 2:
        raise ValueError("cycles must be at least 2 for within-GPU counterbalancing")

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
    base_config = _mosaic_config(
        parameters,
        ledger,
        ledger_sha,
        retention_ratio=float(spec["retention_ratio"]),
        output_length=int(spec["output_length"]),
        seed=int(spec["seed"]),
    )
    base_config = replace(
        base_config,
        diagnostics=FidelityDiagnosticsConfig(capture_selection_evidence=True),
        cache=replace(base_config.cache, exact_gather_mode="batched_layer"),
        graph=replace(base_config.graph, attention_host_transfer_mode="deferred_layer"),
        prototypes=replace(base_config.prototypes, source_gather_mode="per_head_reference"),
    )
    worker_id = f"choice-c-blocked-worker-rank{args.rank}-{_sha(spec_path)[:10]}"
    telemetry_path = output_root / "telemetry" / f"{worker_id}.json"
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_same_gpu_blocked_latency_ab",
        "worker_id": worker_id,
        "rank": args.rank,
        "cycles": args.cycles,
        "mode_orders": [list(mode_order(args.rank, cycle)) for cycle in range(args.cycles)],
        "model_load_count": 1,
        "between_cell_policy": "gc_collect_and_cuda_synchronize_no_empty_cache",
        "reference_pipeline": {
            "centroids": "reference_per_head",
            "utility": "reference_sparse",
            "graph_descriptors": "deferred_layer",
            "graph_edges": "dictionary",
            "mixed_tier_packing": "per_head_reference",
            "residual_index": "materialized",
        },
        "optimized_pipeline": {
            "centroids": "batched_layer",
            "utility": "vectorized_edges",
            "graph_descriptors": "deferred_layer_vectorized",
            "graph_edges": "preselected",
            "mixed_tier_packing": "per_head_reference_after_v33_rejection",
            "residual_index": "compact",
        },
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
        "measured_runs": [],
        "status": "started",
    }
    try:
        adapter = load_hf_adapter(
            base_config.model.id,
            revision=base_config.model.revision,
            model_kwargs=_model_kwargs(base_config, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        warmup_ledger = dict(ledger)
        warmup_ledger["sample_ids"] = [sample_ids[0]]
        warmup_ledger["sample_count"] = 1
        warmup_path = output_root / "warmup_ledgers" / f"rank{args.rank}.json"
        warmup_payload = json.dumps(warmup_ledger, indent=2, sort_keys=True) + "\n"
        warmup_path.parent.mkdir(parents=True, exist_ok=True)
        with warmup_path.open("x", encoding="utf-8") as stream:
            stream.write(warmup_payload)
        warmup_sha = _sha(warmup_path)
        for order_index, mode in enumerate(mode_order(args.rank, 0)):
            _run_online(
                _mode_config(base_config, mode),
                adapter,
                run_id=f"choice-c-blocked-warmup-rank{args.rank}-order{order_index}-{mode}",
                output_root=output_root,
                ledger_path=warmup_path,
                ledger_sha=warmup_sha,
                expected_sample_ids=(sample_ids[0],),
            )
            _settle(adapter)
        telemetry["before_measurement"] = asdict(capture_gpu_environment(torch))
        for cycle in range(args.cycles):
            for order_index, mode in enumerate(mode_order(args.rank, cycle)):
                run_id = (
                    f"choice-c-blocked-rank{args.rank}-cycle{cycle}-order{order_index}-"
                    f"{mode}-{_sha(spec_path)[:10]}"
                )
                _run_online(
                    _mode_config(base_config, mode),
                    adapter,
                    run_id=run_id,
                    output_root=output_root,
                    ledger_path=ledger_path,
                    ledger_sha=ledger_sha,
                    expected_sample_ids=sample_ids,
                )
                cast("list[JsonValue]", telemetry["measured_runs"]).append(
                    {
                        "run_id": run_id,
                        "cycle": cycle,
                        "order_index": order_index,
                        "compute_mode": mode,
                    }
                )
                _settle(adapter)
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
