#!/usr/bin/env python3
"""Run a frozen fresh-held-out quality gate for the MosaicKV light graph."""

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

MODES = ("implementation_only", "light_graph", "prompt_attention_topk")


def _settle(adapter: Any) -> None:
    gc.collect()
    if torch.cuda.is_available() and getattr(adapter.device, "type", "cpu") == "cuda":
        torch.cuda.synchronize(adapter.device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    if ledger.get("classification") not in {
        "fresh_disjoint_heldout_quality_validation",
        "exploratory_dirty_tree_fresh_disjoint_heldout_quality_validation",
    }:
        raise RuntimeError("v44 requires a declared fresh disjoint held-out ledger")
    sample_ids = tuple(cast("list[str]", ledger["sample_ids"]))
    if len(sample_ids) < 8:
        raise RuntimeError("v44 quality gate requires at least eight fresh held-out samples")
    output_length = int(spec["output_length"])
    seed = int(spec["seed"])
    configs = {
        "implementation_only": v38_mode_config(
            "implementation_only",
            parameters,
            ledger,
            ledger_sha,
            output_length=output_length,
            seed=seed,
        ),
        "light_graph": v38_mode_config(
            "light_graph",
            parameters,
            ledger,
            ledger_sha,
            output_length=output_length,
            seed=seed,
        ),
        "prompt_attention_topk": v36_mode_config(
            "prompt_attention_topk",
            parameters,
            ledger,
            ledger_sha,
            output_length=output_length,
            seed=seed,
        ),
    }
    worker_id = f"v44-quality-{_sha(spec_path)[:10]}"
    telemetry_path = output_root / "telemetry" / f"{worker_id}.json"
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": ledger["classification"],
        "canonical_eligible": bool(ledger.get("canonical_eligible", False)),
        "worker_id": worker_id,
        "mode_order": list(MODES),
        "sample_ids": list(sample_ids),
        "preregistered_acceptance": {
            "light_graph_vs_implementation": "exact_per_sample_answer_and_score_identity",
            "mosaic_vs_prompt_score_delta_minimum": -0.01,
            "mosaic_vs_prompt_active_kv_reduction_minimum": 0.30,
        },
        "before_model_load": asdict(capture_gpu_environment(torch)),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "not_used"),
            "node": os.environ.get("SLURMD_NODENAME", "not_used"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        },
        "status": "started",
    }
    try:
        adapter = load_hf_adapter(
            configs["implementation_only"].model.id,
            revision=configs["implementation_only"].model.revision,
            model_kwargs=_model_kwargs(configs["implementation_only"], local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        warmup_ledger = dict(ledger)
        warmup_ledger["sample_ids"] = [sample_ids[0]]
        warmup_ledger["sample_count"] = 1
        warmup_path = output_root / "warmup_ledger.json"
        warmup_path.parent.mkdir(parents=True, exist_ok=True)
        warmup_path.write_text(json.dumps(warmup_ledger, indent=2, sort_keys=True) + "\n")
        warmup_sha = _sha(warmup_path)
        for mode in MODES:
            _run_online(
                configs[mode],
                adapter,
                run_id=f"v44-warmup-{mode}",
                output_root=output_root,
                ledger_path=warmup_path,
                ledger_sha=warmup_sha,
                expected_sample_ids=(sample_ids[0],),
            )
            _settle(adapter)
        telemetry["before_measurement"] = asdict(capture_gpu_environment(torch))
        for mode in MODES:
            _run_online(
                configs[mode],
                adapter,
                run_id=f"v44-heldout-{mode}-{_sha(spec_path)[:10]}",
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
