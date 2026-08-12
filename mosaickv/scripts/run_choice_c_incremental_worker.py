#!/usr/bin/env python3
"""Run small, incremental MosaicKV latency ideas on one isolated GPU."""

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
from run_choice_c_latency_factorial_worker import _run_online, _write_telemetry

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import FidelityDiagnosticsConfig
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment

MODES = (
    "prompt_topk_rr20",
    "retry_reference_rr20",
    "one_pass_rr20",
    "shortlist_rr20",
    "conditional_draft_rr20",
    "bounded_residual_rr20",
    "combined_rr14",
)


def _optimized_common(config: Any) -> Any:
    """Apply only optimizations already admitted by the v34 parity gate."""

    return replace(
        config,
        diagnostics=FidelityDiagnosticsConfig(),
        forecasting=replace(config.forecasting, centroid_execution_mode="batched_layer"),
        utility=replace(config.utility, execution_mode="vectorized_edges"),
        graph=replace(
            config.graph,
            attention_host_transfer_mode="deferred_layer_vectorized",
            edge_accumulation_mode="preselected",
        ),
        cache=replace(
            config.cache,
            exact_gather_mode="batched_layer",
            packing_mode="per_head_reference",
        ),
        prototypes=replace(config.prototypes, source_gather_mode="per_head_reference"),
        residual=replace(config.residual, index_mode="compact"),
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
    """Build one cumulative idea cell with explicit predecessor semantics."""

    if mode not in MODES:
        raise ValueError(f"unknown incremental mode: {mode}")
    if mode == "prompt_topk_rr20":
        return _optimized_common(
            _baseline_config(
                "prompt_attention_topk",
                ledger,
                ledger_sha,
                retention_ratio=0.2,
                output_length=output_length,
                seed=seed,
            )
        )
    ratio = 0.14 if mode == "combined_rr14" else 0.2
    config = _optimized_common(
        _mosaic_config(
            parameters,
            ledger,
            ledger_sha,
            retention_ratio=ratio,
            output_length=output_length,
            seed=seed,
        )
    )
    if mode == "retry_reference_rr20":
        return replace(
            config,
            prototypes=replace(
                config.prototypes,
                budget_planning_mode="bounded_search_reference",
            ),
        )
    config = replace(
        config,
        prototypes=replace(
            config.prototypes,
            budget_planning_mode="one_pass_anchor_cover",
        ),
    )
    if mode == "one_pass_rr20":
        return config
    config = replace(
        config,
        graph=replace(
            config.graph,
            similarity_shortlist_multiplier=2.0,
            similarity_shortlist_min_nodes=64,
        ),
    )
    if mode == "shortlist_rr20":
        return config
    config = replace(
        config,
        forecasting=replace(
            config.forecasting,
            conditional_draft_enabled=True,
            conditional_draft_uncertainty_threshold=0.25,
        ),
    )
    if mode == "conditional_draft_rr20":
        return config
    return replace(
        config,
        cache=replace(
            config.cache,
            total_storage_budget_bytes=80 * 1024 * 1024,
        ),
        residual=replace(
            config.residual,
            max_bytes=8 * 1024 * 1024,
            max_source_blocks=256,
            selection_policy="risk_weighted_reconstruction_error",
        ),
    )


def _settle(adapter: Any) -> None:
    gc.collect()
    if torch.cuda.is_available() and getattr(adapter.device, "type", "cpu") == "cuda":
        torch.cuda.synchronize(adapter.device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rank",
        choices=(0, 1),
        default=int(os.environ.get("SLURM_PROCID", "0")),
        type=int,
    )
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
    sample_ids = tuple(cast("list[str]", ledger["sample_ids"]))
    if len(sample_ids) != 2:
        raise RuntimeError("incremental smoke ledger must contain exactly two samples")
    output_length = int(spec["output_length"])
    seed = int(spec["seed"])
    configs = {
        mode: mode_config(
            mode,
            parameters,
            ledger,
            ledger_sha,
            output_length=output_length,
            seed=seed,
        )
        for mode in MODES
    }
    first_config = configs[MODES[0]]
    worker_id = f"incremental-rank{args.rank}-{_sha(spec_path)[:10]}"
    telemetry_path = output_root / "telemetry" / f"{worker_id}.json"
    order = MODES if args.rank == 0 else tuple(reversed(MODES))
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_incremental_small_sample_test",
        "worker_id": worker_id,
        "rank": args.rank,
        "mode_order": list(order),
        "sample_ids": list(sample_ids),
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
        with warmup_path.open("x", encoding="utf-8") as stream:
            json.dump(warmup_ledger, stream, indent=2, sort_keys=True)
            stream.write("\n")
        warmup_sha = _sha(warmup_path)
        for mode in order:
            _run_online(
                configs[mode],
                adapter,
                run_id=f"incremental-warmup-{mode}-rank{args.rank}",
                output_root=output_root,
                ledger_path=warmup_path,
                ledger_sha=warmup_sha,
                expected_sample_ids=(sample_ids[0],),
            )
            _settle(adapter)
        telemetry["before_measurement"] = asdict(capture_gpu_environment(torch))
        for mode in order:
            _run_online(
                configs[mode],
                adapter,
                run_id=f"incremental-measured-{mode}-rank{args.rank}-{_sha(spec_path)[:10]}",
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
