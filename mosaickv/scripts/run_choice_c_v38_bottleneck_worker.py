#!/usr/bin/env python3
"""Measure cumulative MosaicKV compression-bottleneck fixes on two A100 workers."""

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
from run_choice_c_gate import _load_ledger, _sha
from run_choice_c_latency_factorial_worker import _run_online, _write_telemetry
from run_choice_c_v36_recommended_worker import mode_config as v36_mode_config

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment

RANK_MODES = {
    0: ("implementation_only", "light_graph"),
    1: ("larger_groups", "block32"),
}


def mode_config(
    mode: str,
    parameters: dict[str, Any],
    ledger: dict[str, Any],
    ledger_sha: str,
    *,
    output_length: int,
    seed: int,
) -> Any:
    """Resolve one cumulative, auditable latency candidate."""

    config = v36_mode_config(
        "mosaickv_recommended",
        parameters,
        ledger,
        ledger_sha,
        output_length=output_length,
        seed=seed,
    )
    config = replace(
        config,
        cache=replace(
            config.cache,
            exact_gather_mode="source_views",
            packing_mode="preallocated_layer",
        ),
    )
    if mode == "implementation_only":
        return config
    config = replace(
        config,
        graph=replace(
            config.graph,
            max_neighbors=2,
            similarity_shortlist_multiplier=1.25,
        ),
    )
    if mode == "light_graph":
        return config
    if mode == "larger_groups":
        return replace(
            config,
            prototypes=replace(
                config.prototypes,
                group_size=31,
                max_position_span=511,
            ),
        )
    if mode == "block32":
        return replace(
            config,
            cache=replace(config.cache, block_size=32),
            prototypes=replace(
                config.prototypes,
                group_size=7,
                max_position_span=255,
            ),
        )
    raise ValueError(f"unknown v38 bottleneck mode: {mode}")


def _settle(adapter: Any) -> None:
    gc.collect()
    if torch.cuda.is_available() and getattr(adapter.device, "type", "cpu") == "cuda":
        torch.cuda.synchronize(adapter.device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", choices=tuple(RANK_MODES), required=True, type=int)
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
    profile = cast("dict[str, Any]", yaml.safe_load(profile_path.read_text()))
    parameters = profile.get("parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("frozen fidelity profile has no parameter mapping")
    ledger, ledger_sha = _load_ledger(ledger_path)
    if ledger.get("classification") != "retrospective_non_claim_system_validation":
        raise RuntimeError("v38 requires the retrospective implementation-validation ledger")
    sample_ids = tuple(cast("list[str]", ledger["sample_ids"]))
    if len(sample_ids) != 2:
        raise RuntimeError("v38 validation requires exactly two fixed samples")
    modes = RANK_MODES[args.rank]
    configs = {
        mode: mode_config(
            mode,
            parameters,
            ledger,
            ledger_sha,
            output_length=int(spec["output_length"]),
            seed=int(spec["seed"]),
        )
        for mode in modes
    }
    worker_id = f"v38-rank{args.rank}-{_sha(spec_path)[:10]}"
    telemetry_path = output_root / "telemetry" / f"{worker_id}.json"
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_retrospective_non_claim_validation",
        "worker_id": worker_id,
        "rank": args.rank,
        "mode_order": list(modes),
        "sample_ids": list(sample_ids),
        "before_model_load": asdict(capture_gpu_environment(torch)),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "not_used"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "not_used"),
            "node": os.environ.get("SLURMD_NODENAME", "not_used"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        },
        "status": "started",
    }
    try:
        first_config = configs[modes[0]]
        adapter = load_hf_adapter(
            first_config.model.id,
            revision=first_config.model.revision,
            model_kwargs=_model_kwargs(first_config, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        warmup_ledger = dict(ledger)
        warmup_ledger["sample_ids"] = [sample_ids[0]]
        warmup_ledger["sample_count"] = 1
        warmup_path = output_root / "warmup_ledgers" / f"rank{args.rank}.json"
        warmup_path.parent.mkdir(parents=True, exist_ok=True)
        warmup_payload = json.dumps(warmup_ledger, indent=2, sort_keys=True) + "\n"
        if warmup_path.exists() and warmup_path.read_text() != warmup_payload:
            raise RuntimeError("refusing to mutate an existing v38 warmup ledger")
        if not warmup_path.exists():
            warmup_path.write_text(warmup_payload)
        warmup_sha = _sha(warmup_path)
        for mode in modes:
            _run_online(
                configs[mode],
                adapter,
                run_id=f"v38-warmup-{mode}-rank{args.rank}",
                output_root=output_root,
                ledger_path=warmup_path,
                ledger_sha=warmup_sha,
                expected_sample_ids=(sample_ids[0],),
            )
            _settle(adapter)
        telemetry["before_measurement"] = asdict(capture_gpu_environment(torch))
        for mode in modes:
            _run_online(
                configs[mode],
                adapter,
                run_id=f"v38-measured-{mode}-rank{args.rank}-{_sha(spec_path)[:10]}",
                output_root=output_root,
                ledger_path=ledger_path,
                ledger_sha=ledger_sha,
                expected_sample_ids=sample_ids,
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
