#!/usr/bin/env python3
"""Run both HF selectors at five identical physical KV byte caps."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import torch
from scripts.run_paper_candidate_v1_preflight_worker import (
    _control_configs,
    _record,
    _settle,
)

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import (
    RepairConfig,
    ResidualConfig,
    config_sha256,
    load_config,
)
from mosaickv.evaluation.paper_candidate import PAPER_CANDIDATE_TASK_ROUTES
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment
from mosaickv.types import MosaicKVMethod, RepairPolicy

TASKS = ("mmstar", "textvqa", "docvqa")
METHODS = ("prompt_attention_topk", "mosaickv_exact")
BUDGETS_MIB = (16, 22, 30, 40, 60)
CLASSIFICATION = "canonical_byte_matched_selector_curve"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_snapshot(repository: Path) -> tuple[str, bool]:
    sha = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return sha, bool(status.strip())


def curve_configs(
    template: Any,
    *,
    task: str,
    seed: int,
    output_length: int,
    budget_bytes: int,
    retention_ratio_upper_bound: float,
) -> dict[str, Any]:
    """Resolve method configs with one identical physical-cache contract."""

    controls = _control_configs(template, task=task, seed=seed, output_length=output_length)
    shared_cache = replace(
        template.cache,
        budget_value=budget_bytes,
        budget_unit=template.cache.budget_unit,
        retention_ratio=retention_ratio_upper_bound,
        block_size=4,
        physical_budget_block_size=4,
        physical_budget_mandatory_policy="last_token",
        exact_gather_mode="source_views",
        packing_mode="preallocated_layer",
        total_storage_budget_bytes=None,
    )
    prompt = replace(controls["prompt_attention_topk"], cache=shared_cache)
    mosaic = replace(
        template,
        dataset=PAPER_CANDIDATE_TASK_ROUTES[task],
        execution=replace(template.execution, seed=seed),
        generation=replace(template.generation, max_new_tokens=output_length),
        cache=shared_cache,
        method=MosaicKVMethod.MOSAICKV_EXACT,
        prototypes=replace(template.prototypes, enabled=False),
        residual=ResidualConfig(enabled=False),
        repair=RepairConfig(enabled=False, policy=RepairPolicy.NONE, max_blocks_per_step=0),
    )
    configs = {"prompt_attention_topk": prompt, "mosaickv_exact": mosaic}
    if prompt.cache != mosaic.cache:
        raise RuntimeError("paired methods do not share the exact cache contract")
    if prompt.model != mosaic.model or prompt.dataset != mosaic.dataset:
        raise RuntimeError("paired methods do not share model and dataset inputs")
    if prompt.execution != mosaic.execution or prompt.generation != mosaic.generation:
        raise RuntimeError("paired methods do not share execution and generation controls")
    return configs


def _load_job(path: Path, expected_sha: str) -> dict[str, Any]:
    if _sha(path) != expected_sha:
        raise RuntimeError("immutable job specification hash changed")
    job = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
    if job.get("classification") != CLASSIFICATION:
        raise RuntimeError("worker received the wrong experiment classification")
    if tuple(job.get("methods", ())) != METHODS:
        raise RuntimeError("worker method order changed")
    if tuple(int(value) for value in job.get("budgets_mib", ())) != BUDGETS_MIB:
        raise RuntimeError("worker byte-budget sweep changed")
    return job


def _write_completion(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--job-spec-sha256", required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    job_path = args.job_spec.resolve()
    job = _load_job(job_path, str(args.job_spec_sha256))
    source_sha, dirty = _source_snapshot(repository)
    if dirty or source_sha != job["source_git_sha"]:
        raise RuntimeError("byte-curve worker requires its clean immutable source SHA")
    protocol_path = Path(str(job["protocol_path"])).resolve()
    if _sha(protocol_path) != job["protocol_sha256"]:
        raise RuntimeError("byte-curve protocol hash changed")
    candidate_path = Path(str(job["candidate_config_path"])).resolve()
    if _sha(candidate_path) != job["candidate_config_sha256"]:
        raise RuntimeError("candidate configuration hash changed")
    ledger_path = Path(str(job["sample_ledger_path"])).resolve()
    if _sha(ledger_path) != job["sample_ledger_sha256"]:
        raise RuntimeError("sample ledger hash changed")
    ledger = cast("dict[str, Any]", json.loads(ledger_path.read_text(encoding="utf-8")))
    task = str(job["task"])
    if task not in TASKS or ledger.get("task") != task:
        raise RuntimeError("sample ledger task changed")
    sample_ids = tuple(str(value) for value in cast("list[str]", ledger["sample_ids"]))
    if len(sample_ids) != int(job["sample_count"]) or len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("sample ledger identities are incomplete or duplicated")

    output_root = Path(str(job["output_root"])).resolve()
    shard_root = output_root / task / f"{int(job['shard_index']):02d}"
    completion = shard_root / "job_complete.json"
    if completion.exists():
        previous = json.loads(completion.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "completed"
            and previous.get("job_spec_sha256") == args.job_spec_sha256
        ):
            return 0
        raise RuntimeError("existing completion record has a different identity")
    shard_root.mkdir(parents=True, exist_ok=True)

    template = load_config(candidate_path)
    adapter = load_hf_adapter(
        template.model.id,
        revision=template.model.revision,
        model_kwargs=_model_kwargs(template, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    records: list[dict[str, Any]] = []
    byte_checks: list[dict[str, Any]] = []
    before = asdict(capture_gpu_environment(torch))
    namespace = f"hf-byte-curve-{source_sha[:10]}-{task}-{int(job['shard_index']):02d}"
    try:
        for budget_index, budget_mib in enumerate(BUDGETS_MIB):
            budget_bytes = budget_mib * 2**20
            configs = curve_configs(
                template,
                task=task,
                seed=int(job["selection_seed"]),
                output_length=int(job["output_length"]),
                budget_bytes=budget_bytes,
                retention_ratio_upper_bound=float(job["retention_ratio_upper_bound"]),
            )
            # Counterbalance method order across budgets while retaining an
            # immutable, deterministic order in the completion record.
            order = METHODS if budget_index % 2 == 0 else tuple(reversed(METHODS))
            paired_rows: dict[str, dict[str, dict[str, Any]]] = {}
            for method in order:
                run_id = f"{namespace}-b{budget_mib:03d}-{method}"
                record = _record(
                    configs[method],
                    adapter,
                    task=task,
                    run_id=run_id,
                    output_root=shard_root,
                    ledger_path=ledger_path,
                    ledger_sha=str(job["sample_ledger_sha256"]),
                    sample_ids=sample_ids,
                    phase=CLASSIFICATION,
                    repetition=0,
                    order=len(records),
                )
                record.update(
                    {
                        "budget_mib": budget_mib,
                        "budget_bytes": budget_bytes,
                        "config_sha256": config_sha256(configs[method]),
                    }
                )
                records.append(record)
                paired_rows[method] = {
                    str(row["sample_id"]): row
                    for row in cast("list[dict[str, Any]]", record["rows"])
                }
                _settle(adapter)
            prompt_rows = paired_rows["prompt_attention_topk"]
            mosaic_rows = paired_rows["mosaickv_exact"]
            if set(prompt_rows) != set(sample_ids) or set(mosaic_rows) != set(sample_ids):
                raise RuntimeError("paired method sample identities changed")
            for sample_id in sample_ids:
                prompt_bytes = int(prompt_rows[sample_id]["active_kv_bytes"])
                mosaic_bytes = int(mosaic_rows[sample_id]["active_kv_bytes"])
                check = {
                    "sample_id": sample_id,
                    "budget_mib": budget_mib,
                    "configured_budget_bytes": budget_bytes,
                    "prompt_active_kv_bytes": prompt_bytes,
                    "mosaickv_active_kv_bytes": mosaic_bytes,
                    "active_byte_difference": mosaic_bytes - prompt_bytes,
                    "exact_byte_match": mosaic_bytes == prompt_bytes,
                    "within_hard_cap": max(prompt_bytes, mosaic_bytes) <= budget_bytes,
                }
                byte_checks.append(check)
                if not check["exact_byte_match"] or not check["within_hard_cap"]:
                    raise RuntimeError(f"realized active bytes are not exactly matched: {check}")
    finally:
        after = asdict(capture_gpu_environment(torch))
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    _write_completion(
        completion,
        {
            "schema_version": 1,
            "status": "completed",
            "classification": CLASSIFICATION,
            "source_git_sha": source_sha,
            "source_git_dirty": False,
            "job_spec": str(job_path),
            "job_spec_sha256": str(args.job_spec_sha256),
            "protocol_sha256": str(job["protocol_sha256"]),
            "candidate_config_sha256": str(job["candidate_config_sha256"]),
            "sample_ledger_sha256": str(job["sample_ledger_sha256"]),
            "task": task,
            "shard_index": int(job["shard_index"]),
            "sample_count": len(sample_ids),
            "methods": list(METHODS),
            "budgets_mib": list(BUDGETS_MIB),
            "records": records,
            "byte_match_checks": byte_checks,
            "before_model_load": before,
            "after_measurement": after,
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID", "not_used"),
                "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "not_used"),
                "node": os.environ.get("SLURMD_NODENAME", "not_used"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
