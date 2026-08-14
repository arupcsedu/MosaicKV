#!/usr/bin/env python3
"""Run a paired Prompt Top-K versus MosaicKV SDPA latency control."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch
from run_choice_c_gate import _load_ledger, _sha
from run_choice_c_latency_factorial_worker import _run_online, _write_telemetry
from run_choice_c_v46_prompt_mosaic_latency_worker import _source_snapshot
from run_choice_c_v63_anchor_routing_worker import _settle, _write_immutable
from run_choice_c_v65_routed_quality_worker import VARIANTS, variant_config

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import load_config
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment
from mosaickv.types import MosaicKVMethod, RepairPolicy

METHOD_ORDER = ("prompt_attention_topk", "mosaickv_full")


def _matched_configs(base: Any, *, routed_head_only: bool = False) -> dict[str, Any]:
    variant = next(item for item in VARIANTS if item["id"] == "routed_direct_sdpa")
    mosaic = variant_config(base, variant)
    if routed_head_only:
        mosaic = replace(
            mosaic,
            selection=replace(
                mosaic.selection,
                routed_head_only_planning=True,
                physical_capacity_fraction=0.70,
            ),
        )
    if mosaic.execution.attention_implementation != "eager":
        raise RuntimeError("v67 requires eager prefill")
    if mosaic.execution.decode_attention_implementation != "sdpa":
        raise RuntimeError("v67 requires SDPA decode")
    prompt = replace(
        mosaic,
        method=MosaicKVMethod.PROMPT_ATTENTION_TOPK,
        forecasting=replace(mosaic.forecasting, enabled=False),
        graph=replace(mosaic.graph, enabled=False),
        selection=replace(
            mosaic.selection,
            enabled=False,
            stable_head_prompt_routing=False,
            stable_head_uniform_fill=False,
            routed_head_only_planning=False,
            physical_capacity_fraction=1.0,
        ),
        prototypes=replace(mosaic.prototypes, enabled=False),
        residual=replace(mosaic.residual, enabled=False),
        repair=replace(
            mosaic.repair,
            enabled=False,
            policy=RepairPolicy.NONE,
            counterfactual_draft_steps=0,
            counterfactual_route_heads=False,
        ),
    )
    return {
        MosaicKVMethod.PROMPT_ATTENTION_TOPK.value: prompt,
        MosaicKVMethod.MOSAICKV_FULL.value: mosaic,
    }


def _run_record(
    config: Any,
    adapter: Any,
    *,
    run_id: str,
    output_root: Path,
    ledger_path: Path,
    ledger_sha: str,
    sample_ids: tuple[str, ...],
    phase: str,
    repetition: int,
    order: int,
) -> dict[str, Any]:
    frame = _run_online(
        config,
        adapter,
        run_id=run_id,
        output_root=output_root,
        ledger_path=ledger_path,
        ledger_sha=ledger_sha,
        expected_sample_ids=sample_ids,
    )
    rows = []
    for row in frame.itertuples():
        rows.append(
            {
                "sample_id": str(row.sample_id),
                "answer": str(row.answer),
                "task_score": float(row.task_score),
                "active_kv_bytes": int(row.active_kv_bytes),
                "residual_kv_bytes": int(row.residual_kv_bytes),
                "compression_time": float(row.compression_time),
                "decode_time": float(row.decode_time),
                "end_to_end_time": float(row.end_to_end_time),
                "generated_tokens": int(row.generated_tokens),
                "repair_count": int(row.repair_count),
                "manifest_path": str(row.manifest_path),
            }
        )
    return {
        "phase": phase,
        "repetition": repetition,
        "order": order,
        "method": config.method.value,
        "run_id": run_id,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--warmups", default=2, type=int)
    parser.add_argument("--repetitions", default=5, type=int)
    parser.add_argument("--routed-head-only", action="store_true")
    args = parser.parse_args()
    if args.warmups < 1 or args.repetitions < 2:
        raise ValueError("v67 requires at least one warmup and two measured repetitions")

    project_root = Path(__file__).resolve().parents[2]
    source_snapshot = _source_snapshot(project_root)
    base_path = args.base_config.resolve()
    ledger_path = args.ledger.resolve()
    output_root = args.output_root.resolve()
    configs = _matched_configs(
        load_config(base_path),
        routed_head_only=args.routed_head_only,
    )
    ledger, ledger_sha = _load_ledger(ledger_path)
    sample_ids = tuple(cast("list[str]", ledger["sample_ids"]))
    if len(sample_ids) != 2:
        raise RuntimeError("v67 requires exactly two fixed retrospective samples")
    warmup = dict(ledger)
    warmup.update({"sample_count": 1, "sample_ids": [sample_ids[0]]})
    warmup_path = output_root / "warmup_ledgers" / "paired_sdpa.json"
    _write_immutable(warmup_path, warmup)
    warmup_sha = _sha(warmup_path)

    planner_label = "routed-head-only" if args.routed_head_only else "full-node"
    worker_id = f"v67-matched-sdpa-{planner_label}-{_sha(base_path)[:10]}"
    telemetry_path = output_root / "telemetry" / f"{worker_id}.json"
    if telemetry_path.exists():
        prior = json.loads(telemetry_path.read_text())
        if prior.get("status") == "completed":
            print(f"v67 already completed: {worker_id}", flush=True)
            return 0
        raise RuntimeError(f"v67 has non-complete immutable telemetry: {telemetry_path}")
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_paired_backend_control",
        "worker_id": worker_id,
        "base_config": str(base_path),
        "base_config_sha256": _sha(base_path),
        "ledger": str(ledger_path),
        "ledger_sha256": ledger_sha,
        "sample_ids": list(sample_ids),
        "warmups_per_method": args.warmups,
        "measured_repetitions_per_method": args.repetitions,
        "trial_order": "alternating_prompt_first_then_mosaic_first",
        "planner_scope": planner_label,
        "source_snapshot": source_snapshot,
        "before_model_load": asdict(capture_gpu_environment(torch)),
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID", "not_used"),
            "node": os.environ.get("SLURMD_NODENAME", "not_used"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
            "fixed_clock_status": os.environ.get(
                "MOSAICKV_FIXED_CLOCK_STATUS", "not_permitted_or_not_requested"
            ),
        },
        "trials": [],
        "status": "started",
    }
    try:
        model_config = configs[MosaicKVMethod.MOSAICKV_FULL.value]
        adapter = load_hf_adapter(
            model_config.model.id,
            revision=model_config.model.revision,
            model_kwargs=_model_kwargs(model_config, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        for warmup_index in range(args.warmups):
            order = METHOD_ORDER if warmup_index % 2 == 0 else tuple(reversed(METHOD_ORDER))
            for order_index, method in enumerate(order):
                _run_record(
                    configs[method],
                    adapter,
                    run_id=f"v67-warmup-{warmup_index:02d}-{method}",
                    output_root=output_root,
                    ledger_path=warmup_path,
                    ledger_sha=warmup_sha,
                    sample_ids=(sample_ids[0],),
                    phase="warmup",
                    repetition=warmup_index,
                    order=order_index,
                )
                _settle(adapter)
        telemetry["before_measurement"] = asdict(capture_gpu_environment(torch))
        for repetition in range(args.repetitions):
            order = METHOD_ORDER if repetition % 2 == 0 else tuple(reversed(METHOD_ORDER))
            for order_index, method in enumerate(order):
                if _source_snapshot(project_root) != source_snapshot:
                    raise RuntimeError("repository source changed during v67 verification")
                record = _run_record(
                    configs[method],
                    adapter,
                    run_id=f"v67-trial-{repetition:02d}-{method}",
                    output_root=output_root,
                    ledger_path=ledger_path,
                    ledger_sha=ledger_sha,
                    sample_ids=sample_ids,
                    phase="measured",
                    repetition=repetition,
                    order=order_index,
                )
                cast("list[dict[str, Any]]", telemetry["trials"]).append(record)
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
