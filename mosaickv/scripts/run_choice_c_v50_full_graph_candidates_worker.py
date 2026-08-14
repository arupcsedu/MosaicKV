#!/usr/bin/env python3
"""Screen full-graph latency/quality candidates on declared calibration samples."""

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
from run_choice_c_v38_bottleneck_worker import mode_config as v38_mode_config
from run_choice_c_v46_prompt_mosaic_latency_worker import _source_snapshot

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment

RANK_MODES = {
    0: ("prompt_attention_topk", "current_full_graph"),
    1: ("block32_full_graph", "prompt_heavy_full_graph", "block32_prompt_heavy_full_graph"),
    2: ("prompt_attention_topk", "group31_full_graph"),
}


def candidate_config(
    mode: str,
    parameters: dict[str, Any],
    ledger: dict[str, Any],
    ledger_sha: str,
    *,
    output_length: int,
    seed: int,
) -> Any:
    if mode == "prompt_attention_topk":
        return v36_mode_config(
            mode,
            parameters,
            ledger,
            ledger_sha,
            output_length=output_length,
            seed=seed,
        )
    config = v38_mode_config(
        "implementation_only",
        parameters,
        ledger,
        ledger_sha,
        output_length=output_length,
        seed=seed,
    )
    if mode == "current_full_graph":
        return config
    if mode == "group31_full_graph":
        return replace(
            config,
            prototypes=replace(
                config.prototypes,
                group_size=31,
                max_position_span=511,
            ),
        )
    if mode in {"block32_full_graph", "block32_prompt_heavy_full_graph"}:
        config = replace(
            config,
            cache=replace(config.cache, block_size=32),
            prototypes=replace(
                config.prototypes,
                group_size=7,
                max_position_span=255,
            ),
        )
    if mode in {"prompt_heavy_full_graph", "block32_prompt_heavy_full_graph"}:
        config = replace(
            config,
            selection=replace(
                config.selection,
                prompt_anchor_fraction=0.60,
                evidence_preservation_fraction=0.15,
            ),
        )
    if mode not in {
        "block32_full_graph",
        "prompt_heavy_full_graph",
        "block32_prompt_heavy_full_graph",
        "group31_full_graph",
    }:
        raise ValueError(f"unknown v50 candidate mode: {mode}")
    return config


def _settle(adapter: Any) -> None:
    gc.collect()
    if torch.cuda.is_available() and getattr(adapter.device, "type", "cpu") == "cuda":
        torch.cuda.synchronize(adapter.device)


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError(f"refusing to mutate immutable derived ledger: {path}")
    if not path.exists():
        path.write_text(encoded)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", choices=tuple(RANK_MODES), required=True, type=int)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--sample-count", default=4, type=int)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.sample_count < 2:
        raise ValueError("candidate screening requires at least two calibration samples")

    project_root = Path(__file__).resolve().parents[2]
    source_snapshot = _source_snapshot(project_root)
    spec_path = args.spec.resolve()
    profile_path = args.profile.resolve()
    ledger_path = args.ledger.resolve()
    output_root = args.output_root.resolve()
    spec = cast("dict[str, Any]", yaml.safe_load(spec_path.read_text()))
    profile = cast("dict[str, Any]", yaml.safe_load(profile_path.read_text()))
    parameters = profile.get("parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("frozen fidelity profile has no parameter mapping")
    parent, parent_sha = _load_ledger(ledger_path)
    parent_ids = tuple(cast("list[str]", parent["sample_ids"]))
    if len(parent_ids) < args.sample_count:
        raise RuntimeError("declared calibration ledger is smaller than the requested screen")
    sample_ids = parent_ids[: args.sample_count]
    derived = dict(parent)
    derived.update(
        {
            "classification": "existing_declared_calibration_subset_non_claim_screen",
            "canonical_eligible": False,
            "parent_ledger": str(ledger_path),
            "parent_ledger_sha256": parent_sha,
            "selection_rule": "first_N_from_existing_declared_calibration_ledger",
            "sample_count": len(sample_ids),
            "sample_ids": list(sample_ids),
        }
    )
    derived_path = output_root / "calibration_subset_ledger.json"
    _write_immutable(derived_path, derived)
    derived_sha = _sha(derived_path)
    modes = RANK_MODES[args.rank]
    configs = {
        mode: candidate_config(
            mode,
            parameters,
            derived,
            derived_sha,
            output_length=int(spec["output_length"]),
            seed=int(spec["seed"]),
        )
        for mode in modes
    }
    worker_id = f"v50-rank{args.rank}-{_sha(spec_path)[:10]}"
    telemetry_path = output_root / "telemetry" / f"{worker_id}.json"
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_calibration_candidate_screen",
        "worker_id": worker_id,
        "rank": args.rank,
        "modes": list(modes),
        "sample_ids": list(sample_ids),
        "source_snapshot": source_snapshot,
        "light_graph_used": False,
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
        first = configs[modes[0]]
        adapter = load_hf_adapter(
            first.model.id,
            revision=first.model.revision,
            model_kwargs=_model_kwargs(first, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        warmup = dict(derived)
        warmup["sample_count"] = 1
        warmup["sample_ids"] = [sample_ids[0]]
        warmup_path = output_root / "warmup_ledgers" / f"rank{args.rank}.json"
        _write_immutable(warmup_path, warmup)
        warmup_sha = _sha(warmup_path)
        for mode in modes:
            _run_online(
                configs[mode],
                adapter,
                run_id=f"v50-warmup-rank{args.rank}-{mode}",
                output_root=output_root,
                ledger_path=warmup_path,
                ledger_sha=warmup_sha,
                expected_sample_ids=(sample_ids[0],),
            )
            _settle(adapter)
        telemetry["before_measurement"] = asdict(capture_gpu_environment(torch))
        for mode in modes:
            if _source_snapshot(project_root) != source_snapshot:
                raise RuntimeError("repository source changed during the frozen candidate screen")
            _run_online(
                configs[mode],
                adapter,
                run_id=f"v50-calibration-rank{args.rank}-{mode}-{_sha(spec_path)[:10]}",
                output_root=output_root,
                ledger_path=derived_path,
                ledger_sha=derived_sha,
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
