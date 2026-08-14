#!/usr/bin/env python3
"""Validate stable-head prompt routing and decoupled repair on two samples."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch
from run_choice_c_gate import _load_ledger, _sha
from run_choice_c_latency_factorial_worker import _run_online, _write_telemetry
from run_choice_c_v46_prompt_mosaic_latency_worker import _source_snapshot
from run_choice_c_v63_anchor_routing_worker import _settle, _write_immutable

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import load_config
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment
from mosaickv.types import RepairCandidateRanking, RepairPolicy

VARIANTS = (
    {"id": "routed_anchor75", "prompt_anchor_fraction": 0.75, "repair": False},
    {"id": "routed_anchor15", "prompt_anchor_fraction": 0.15, "repair": False},
    {"id": "routed_anchor75_cluster_repair", "prompt_anchor_fraction": 0.75, "repair": True},
    {
        "id": "routed_direct",
        "prompt_anchor_fraction": 0.0,
        "evidence_preservation_fraction": 0.0,
        "repair": False,
    },
    {
        "id": "routed_light_eager",
        "prompt_anchor_fraction": 0.0,
        "evidence_preservation_fraction": 0.0,
        "repair": False,
        "light_planner": True,
        "decode_attention": None,
    },
    {
        "id": "routed_light_sdpa",
        "prompt_anchor_fraction": 0.0,
        "evidence_preservation_fraction": 0.0,
        "repair": False,
        "light_planner": True,
        "decode_attention": "sdpa",
    },
    {
        "id": "routed_uniform_eager",
        "prompt_anchor_fraction": 0.0,
        "evidence_preservation_fraction": 0.0,
        "repair": False,
        "light_planner": True,
        "uniform_fill": True,
        "decode_attention": None,
    },
    {
        "id": "routed_uniform_sdpa",
        "prompt_anchor_fraction": 0.0,
        "evidence_preservation_fraction": 0.0,
        "repair": False,
        "light_planner": True,
        "uniform_fill": True,
        "decode_attention": "sdpa",
    },
    {
        "id": "routed_direct_sdpa",
        "prompt_anchor_fraction": 0.0,
        "evidence_preservation_fraction": 0.0,
        "repair": False,
        "decode_attention": "sdpa",
    },
)


def variant_config(base: Any, variant: dict[str, object]) -> Any:
    config = replace(
        base,
        execution=replace(
            base.execution,
            decode_attention_implementation=cast(
                "str | None", variant.get("decode_attention")
            ),
        ),
        selection=replace(
            base.selection,
            prompt_anchor_fraction=float(variant["prompt_anchor_fraction"]),
            evidence_preservation_fraction=float(
                variant.get(
                    "evidence_preservation_fraction",
                    base.selection.evidence_preservation_fraction,
                )
            ),
            stable_head_prompt_routing=True,
            stable_head_uniform_fill=bool(variant.get("uniform_fill", False)),
        ),
    )
    if bool(variant.get("light_planner", False)):
        config = replace(
            config,
            graph=replace(
                config.graph,
                max_neighbors=1,
                similarity_shortlist_multiplier=1.0,
                similarity_shortlist_min_nodes=16,
                candidate_nonmandatory_fraction=0.02,
                screening_group_blocks=8,
                uncertain_head_fraction=0.03125,
            ),
            selection=replace(
                config.selection,
                minimum_stratum_fraction=0.0,
                coverage_correction_rounds=0,
            ),
            prototypes=replace(
                config.prototypes,
                uncertain_head_fraction=0.03125,
                maximum_prototypes_per_head=2,
            ),
        )
    if not bool(variant["repair"]):
        return config
    return replace(
        config,
        repair=replace(
            config.repair,
            enabled=True,
            policy=RepairPolicy.DECISION_AWARE,
            prototype_risk_threshold=1.0e6,
            decision_margin_threshold=0.0,
            decision_high_risk_threshold=None,
            draft_kl_threshold=0.01,
            counterfactual_draft_steps=4,
            draft_require_token_mismatch=True,
            counterfactual_route_heads=True,
            max_blocks_per_step=31,
            candidate_ranking=RepairCandidateRanking.DECISION_INFLUENCE,
        ),
        residual=replace(
            config.residual,
            enabled=True,
            max_source_blocks=None,
            max_bytes=None,
            selection_policy="prototype_coverage_then_risk",
            gpu_rescue_max_blocks=0,
            gpu_rescue_max_bytes=None,
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", choices=range(len(VARIANTS)), required=True, type=int)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    source_snapshot = _source_snapshot(project_root)
    variant = VARIANTS[args.index]
    variant_id = cast("str", variant["id"])
    base_path = args.base_config.resolve()
    ledger_path = args.ledger.resolve()
    output_root = args.output_root.resolve()
    config = variant_config(load_config(base_path), variant)
    ledger, ledger_sha = _load_ledger(ledger_path)
    sample_ids = tuple(cast("list[str]", ledger["sample_ids"]))
    if len(sample_ids) != 2:
        raise RuntimeError("v65 requires exactly two fixed retrospective samples")

    worker_id = f"v65-{variant_id}-{_sha(base_path)[:10]}"
    telemetry_path = output_root / "telemetry" / f"{worker_id}.json"
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_retrospective_routed_quality_gate",
        "worker_id": worker_id,
        "variant": variant,
        "base_config": str(base_path),
        "base_config_sha256": _sha(base_path),
        "ledger": str(ledger_path),
        "ledger_sha256": ledger_sha,
        "sample_ids": list(sample_ids),
        "source_snapshot": source_snapshot,
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
        adapter = load_hf_adapter(
            config.model.id,
            revision=config.model.revision,
            model_kwargs=_model_kwargs(config, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        warmup = dict(ledger)
        warmup.update({"sample_count": 1, "sample_ids": [sample_ids[0]]})
        warmup_path = output_root / "warmup_ledgers" / f"{variant_id}.json"
        _write_immutable(warmup_path, warmup)
        _run_online(
            config,
            adapter,
            run_id=f"v65-warmup-{variant_id}",
            output_root=output_root,
            ledger_path=warmup_path,
            ledger_sha=_sha(warmup_path),
            expected_sample_ids=(sample_ids[0],),
        )
        _settle(adapter)
        telemetry["before_measurement"] = asdict(capture_gpu_environment(torch))
        if _source_snapshot(project_root) != source_snapshot:
            raise RuntimeError("repository source changed during v65 verification")
        frame = _run_online(
            config,
            adapter,
            run_id=f"v65-validation-{variant_id}",
            output_root=output_root,
            ledger_path=ledger_path,
            ledger_sha=ledger_sha,
            expected_sample_ids=sample_ids,
        )
        telemetry["answers"] = {
            str(row.sample_id): str(row.answer) for row in frame.itertuples()
        }
        telemetry["repair_counts"] = {
            str(row.sample_id): int(row.repair_count) for row in frame.itertuples()
        }
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
