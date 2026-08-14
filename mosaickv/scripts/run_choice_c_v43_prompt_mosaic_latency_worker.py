#!/usr/bin/env python3
"""Counterbalanced same-GPU Prompt Top-K versus MosaicKV latency gate."""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
import yaml
from run_choice_c_gate import _load_ledger, _sha
from run_choice_c_latency_factorial_worker import _run_online, _write_telemetry
from run_choice_c_v36_recommended_worker import mode_config as v36_mode_config
from run_choice_c_v38_bottleneck_worker import mode_config as v38_mode_config

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment


def mode_order(rank: int, cycle: int) -> tuple[str, str]:
    """Counterbalance order across two ranks and consecutive cycles."""

    return (
        ("prompt_attention_topk", "mosaickv_light_graph")
        if (rank + cycle) % 2 == 0
        else ("mosaickv_light_graph", "prompt_attention_topk")
    )


def _settle(adapter: Any) -> None:
    gc.collect()
    if torch.cuda.is_available() and getattr(adapter.device, "type", "cpu") == "cuda":
        torch.cuda.synchronize(adapter.device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", choices=(0, 1), required=True, type=int)
    parser.add_argument("--cycles", default=3, type=int)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.cycles < 2:
        raise ValueError("same-GPU latency gate requires at least two cycles")

    spec_path = args.spec.resolve()
    profile_path = args.profile.resolve()
    ledger_path = args.ledger.resolve()
    output_root = args.output_root.resolve()
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text()))
    profile = cast("dict[str, Any]", yaml.safe_load(profile_path.read_text()))
    parameters = profile.get("parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("frozen fidelity profile has no parameter mapping")
    ledger, ledger_sha = _load_ledger(ledger_path)
    if ledger.get("classification") != "retrospective_non_claim_system_validation":
        raise RuntimeError("v43 latency gate requires the fixed retrospective system ledger")
    sample_ids = tuple(cast("list[str]", ledger["sample_ids"]))
    if len(sample_ids) != 2:
        raise RuntimeError("v43 latency gate requires exactly two fixed samples")

    output_length = int(spec["output_length"])
    seed = int(spec["seed"])
    prompt_config = v36_mode_config(
        "prompt_attention_topk",
        parameters,
        ledger,
        ledger_sha,
        output_length=output_length,
        seed=seed,
    )
    mosaic_config = v38_mode_config(
        "light_graph",
        parameters,
        ledger,
        ledger_sha,
        output_length=output_length,
        seed=seed,
    )
    configs = {
        "prompt_attention_topk": prompt_config,
        "mosaickv_light_graph": mosaic_config,
    }
    worker_id = f"v43-rank{args.rank}-{_sha(spec_path)[:10]}"
    telemetry_path = output_root / "telemetry" / f"{worker_id}.json"
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_same_gpu_counterbalanced_latency_gate",
        "worker_id": worker_id,
        "rank": args.rank,
        "cycles": args.cycles,
        "sample_ids": list(sample_ids),
        "mode_orders": [list(mode_order(args.rank, cycle)) for cycle in range(args.cycles)],
        "comparison": {
            "prompt_method": "prompt_attention_topk",
            "prompt_retention_ratio": prompt_config.cache.retention_ratio,
            "mosaic_method": "mosaickv_full",
            "mosaic_retention_ratio": mosaic_config.cache.retention_ratio,
            "latency_target": "mosaickv_at_least_5_percent_faster",
            "memory_target": "mosaickv_at_least_30_percent_smaller_active_kv",
        },
        "before_model_load": asdict(capture_gpu_environment(torch)),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "not_used"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "not_used"),
            "node": os.environ.get("SLURMD_NODENAME", "not_used"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        },
        "measured_runs": [],
        "status": "started",
    }
    try:
        adapter = load_hf_adapter(
            prompt_config.model.id,
            revision=prompt_config.model.revision,
            model_kwargs=_model_kwargs(prompt_config, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        warmup_ledger = dict(ledger)
        warmup_ledger["sample_ids"] = [sample_ids[0]]
        warmup_ledger["sample_count"] = 1
        warmup_path = output_root / "warmup_ledgers" / f"rank{args.rank}.json"
        warmup_path.parent.mkdir(parents=True, exist_ok=True)
        warmup_path.write_text(json.dumps(warmup_ledger, indent=2, sort_keys=True) + "\n")
        warmup_sha = _sha(warmup_path)
        for order_index, mode in enumerate(mode_order(args.rank, 0)):
            _run_online(
                configs[mode],
                adapter,
                run_id=f"v43-warmup-rank{args.rank}-order{order_index}-{mode}",
                output_root=output_root,
                ledger_path=warmup_path,
                ledger_sha=warmup_sha,
                expected_sample_ids=(sample_ids[0],),
            )
            _settle(adapter)
        telemetry["before_measurement"] = asdict(capture_gpu_environment(torch))
        measured_runs = cast("list[dict[str, Any]]", telemetry["measured_runs"])
        for cycle in range(args.cycles):
            for order_index, mode in enumerate(mode_order(args.rank, cycle)):
                run_id = (
                    f"v43-rank{args.rank}-cycle{cycle}-order{order_index}-{mode}-"
                    f"{_sha(spec_path)[:10]}"
                )
                _run_online(
                    configs[mode],
                    adapter,
                    run_id=run_id,
                    output_root=output_root,
                    ledger_path=ledger_path,
                    ledger_sha=ledger_sha,
                    expected_sample_ids=sample_ids,
                )
                measured_runs.append(
                    {
                        "run_id": run_id,
                        "cycle": cycle,
                        "order_index": order_index,
                        "mode": mode,
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
