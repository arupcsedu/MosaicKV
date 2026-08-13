#!/usr/bin/env python3
"""Run the recommended MosaicKV runtime sequence on a tiny V100 subset."""

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
from run_choice_c_gate import _baseline_config, _load_ledger, _mosaic_config, _sha
from run_choice_c_incremental_worker import _optimized_common
from run_choice_c_latency_factorial_worker import _run_online, _write_telemetry

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import FidelityDiagnosticsConfig
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment
from mosaickv.types import ForecastMode, Precision

RANK_MODES = {
    0: ("full_kv", "prompt_attention_topk", "mosaickv_recommended"),
    1: ("lookm_reimpl", "vl_cache_reimpl", "mosaickv_recommended"),
}


def _common(config: Any) -> Any:
    optimized = _optimized_common(config)
    return replace(
        optimized,
        model=replace(optimized.model, precision=Precision.FP16),
        diagnostics=FidelityDiagnosticsConfig(),
        cache=replace(optimized.cache, packing_mode="preallocated_layer"),
    )


def mode_config(
    mode: str,
    parameters: dict[str, Any],
    ledger: dict[str, Any],
    ledger_sha: str,
    *,
    output_length: int,
    seed: int,
) -> Any:
    """Resolve one immutable comparison cell."""

    if mode in {"full_kv", "prompt_attention_topk", "lookm_reimpl", "vl_cache_reimpl"}:
        return _common(
            _baseline_config(
                mode,
                ledger,
                ledger_sha,
                retention_ratio=0.2,
                output_length=output_length,
                seed=seed,
            )
        )
    if mode != "mosaickv_recommended":
        raise ValueError(f"unknown v36 mode: {mode}")
    config = _common(
        _mosaic_config(
            parameters,
            ledger,
            ledger_sha,
            retention_ratio=0.13,
            output_length=output_length,
            seed=seed,
        )
    )
    return replace(
        config,
        forecasting=replace(
            config.forecasting,
            mode=ForecastMode.PROMPT_WINDOW,
            draft_steps=0,
            conditional_draft_enabled=False,
        ),
        graph=replace(
            config.graph,
            similarity_shortlist_multiplier=2.0,
            similarity_shortlist_min_nodes=64,
            candidate_descriptor_pooling=True,
        ),
        prototypes=replace(
            config.prototypes,
            budget_planning_mode="one_pass_anchor_cover",
        ),
        residual=replace(
            config.residual,
            max_bytes=1024 * 1024,
            max_source_blocks=128,
            selection_policy="risk_weighted_reconstruction_error",
        ),
    )


def _settle(adapter: Any) -> None:
    gc.collect()
    if torch.cuda.is_available() and getattr(adapter.device, "type", "cpu") == "cuda":
        torch.cuda.synchronize(adapter.device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    profile = cast("dict[str, Any]", json.loads(profile_path.read_text()))
    parameters = profile.get("parameters")
    if not isinstance(parameters, dict):
        raise RuntimeError("frozen fidelity profile has no parameter mapping")
    ledger, ledger_sha = _load_ledger(ledger_path)
    if ledger.get("classification") != "retrospective_non_claim_system_validation":
        raise RuntimeError("v36 requires its explicitly retrospective validation ledger")
    sample_ids = tuple(cast("list[str]", ledger["sample_ids"]))
    if len(sample_ids) != 2:
        raise RuntimeError("v36 validation requires exactly two fixed samples")
    output_length = int(spec["output_length"])
    seed = int(spec["seed"])
    modes = RANK_MODES[args.rank]
    configs = {
        mode: mode_config(
            mode,
            parameters,
            ledger,
            ledger_sha,
            output_length=output_length,
            seed=seed,
        )
        for mode in modes
    }
    worker_id = f"v36-rank{args.rank}-{_sha(spec_path)[:10]}"
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
            raise RuntimeError("refusing to mutate an existing v36 warmup ledger")
        if not warmup_path.exists():
            warmup_path.write_text(warmup_payload)
        warmup_sha = _sha(warmup_path)
        for mode in modes:
            _run_online(
                configs[mode],
                adapter,
                run_id=f"v36-warmup-{mode}-rank{args.rank}",
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
                run_id=f"v36-measured-{mode}-rank{args.rank}-{_sha(spec_path)[:10]}",
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
