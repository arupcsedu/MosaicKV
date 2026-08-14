#!/usr/bin/env python3
"""Screen prompt anchoring and sparse prototype routing on fixed DocVQA samples."""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch
from run_choice_c_gate import _load_ledger, _sha
from run_choice_c_latency_factorial_worker import _run_online, _write_telemetry
from run_choice_c_v46_prompt_mosaic_latency_worker import _source_snapshot

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import load_config
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment

VARIANTS: tuple[dict[str, float | int | str], ...] = (
    {
        "id": "control",
        "prompt_anchor_fraction": 0.15,
        "evidence_preservation_fraction": 0.15,
        "routed_head_fraction": 0.25,
        "candidate_nonmandatory_fraction": 0.10,
        "maximum_prototypes_per_head": 2,
    },
    {
        "id": "balanced_anchor",
        "prompt_anchor_fraction": 0.50,
        "evidence_preservation_fraction": 0.10,
        "routed_head_fraction": 0.125,
        "candidate_nonmandatory_fraction": 0.05,
        "maximum_prototypes_per_head": 2,
    },
    {
        "id": "anchor_heavy",
        "prompt_anchor_fraction": 0.75,
        "evidence_preservation_fraction": 0.05,
        "routed_head_fraction": 0.0625,
        "candidate_nonmandatory_fraction": 0.03,
        "maximum_prototypes_per_head": 2,
    },
    {
        "id": "exact_heavy",
        "prompt_anchor_fraction": 0.85,
        "evidence_preservation_fraction": 0.05,
        "routed_head_fraction": 0.03125,
        "candidate_nonmandatory_fraction": 0.02,
        "maximum_prototypes_per_head": 1,
    },
)


def variant_config(base: Any, variant: dict[str, float | int | str]) -> Any:
    """Apply one declared system variant without changing its byte budget."""

    routed_fraction = float(variant["routed_head_fraction"])
    return replace(
        base,
        selection=replace(
            base.selection,
            prompt_anchor_fraction=float(variant["prompt_anchor_fraction"]),
            evidence_preservation_fraction=float(
                variant["evidence_preservation_fraction"]
            ),
        ),
        graph=replace(
            base.graph,
            candidate_nonmandatory_fraction=float(
                variant["candidate_nonmandatory_fraction"]
            ),
            uncertain_head_fraction=routed_fraction,
        ),
        prototypes=replace(
            base.prototypes,
            uncertain_head_fraction=routed_fraction,
            maximum_prototypes_per_head=int(
                variant["maximum_prototypes_per_head"]
            ),
        ),
    )


def _settle(adapter: Any) -> None:
    gc.collect()
    if torch.cuda.is_available() and getattr(adapter.device, "type", "cpu") == "cuda":
        torch.cuda.synchronize(adapter.device)


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError(f"refusing to mutate immutable v63 input: {path}")
    if not path.exists():
        path.write_text(encoded)


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
    base = load_config(base_path)
    config = variant_config(base, variant)
    ledger, ledger_sha = _load_ledger(ledger_path)
    sample_ids = tuple(cast("list[str]", ledger["sample_ids"]))
    if len(sample_ids) != 2:
        raise RuntimeError("v63 requires exactly two fixed retrospective samples")

    worker_id = f"v63-{variant_id}-{_sha(base_path)[:10]}"
    telemetry_path = output_root / "telemetry" / f"{worker_id}.json"
    if telemetry_path.exists():
        prior = json.loads(telemetry_path.read_text())
        if prior.get("status") == "completed":
            print(f"v63 already completed: {worker_id}", flush=True)
            return 0
        raise RuntimeError(f"v63 has non-complete immutable telemetry: {telemetry_path}")
    telemetry: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_dirty_tree_retrospective_candidate_screen",
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
            run_id=f"v63-warmup-{variant_id}",
            output_root=output_root,
            ledger_path=warmup_path,
            ledger_sha=_sha(warmup_path),
            expected_sample_ids=(sample_ids[0],),
        )
        _settle(adapter)
        telemetry["before_measurement"] = asdict(capture_gpu_environment(torch))
        if _source_snapshot(project_root) != source_snapshot:
            raise RuntimeError("repository source changed during v63 verification")
        frame = _run_online(
            config,
            adapter,
            run_id=f"v63-validation-{variant_id}",
            output_root=output_root,
            ledger_path=ledger_path,
            ledger_sha=ledger_sha,
            expected_sample_ids=sample_ids,
        )
        telemetry["answers"] = {
            str(row.sample_id): str(row.answer) for row in frame.itertuples()
        }
        telemetry["active_kv_bytes"] = {
            str(row.sample_id): int(row.active_kv_bytes) for row in frame.itertuples()
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
