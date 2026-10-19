#!/usr/bin/env python3
"""Run one two-sample PrefixKV native-metric shard with one model load."""

from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path

from run_byte_capped_published_baseline_pilot import _require_clean_source
from run_prefixkv_decode_preflight import digest, write_once

from mosaickv.config import config_sha256, load_config
from mosaickv.evaluation.worker_support import record_hf_run, settle


def _inputs(root: Path, plan_sha: str, group_index: int):
    source = Path(__file__).resolve().parents[2]
    plan_path = root / "plan.json"
    if digest(plan_path) != plan_sha:
        raise ValueError("PrefixKV byte-validation plan changed")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _require_clean_source(source, plan["source_sha"])
    parity_gate = Path(plan["parity_gate"])
    parity_comparison = Path(plan["parity_comparison"])
    gate = json.loads(parity_gate.read_text())
    comparison = json.loads(parity_comparison.read_text())
    if (
        not plan.get("byte_validation_authorized")
        or plan.get("full_campaign_authorized")
        or digest(parity_gate) != plan.get("parity_gate_sha256")
        or digest(parity_comparison) != plan.get("parity_comparison_sha256")
        or gate.get("passed") is not True
        or gate.get("checked_samples") != 4
        or gate.get("failures")
        or comparison.get("status") != "comparable"
    ):
        raise ValueError("PrefixKV byte validation lacks a passed immutable fidelity gate")
    if (plan["max_global_a100s"], plan["gpus_per_worker"], plan["cpus_per_worker"]) != (
        4,
        1,
        12,
    ):
        raise ValueError("unexpected PrefixKV worker resources")
    for path, sha in plan["bound_files"].items():
        if digest(Path(path)) != sha:
            raise ValueError(f"bound PrefixKV input changed: {path}")
    groups = sorted({(str(job["task"]), int(job["shard"])) for job in plan["jobs"]})
    if not 0 <= group_index < len(groups):
        raise ValueError("invalid PrefixKV shard-group index")
    task, shard = groups[group_index]
    jobs = [job for job in plan["jobs"] if (job["task"], job["shard"]) == (task, shard)]
    if len(jobs) != 16 or sum(job["method"] == "full_kv" for job in jobs) != 1:
        raise ValueError("PrefixKV native shard must contain one FullKV and 15 compressed arms")
    return source, plan, task, shard, jobs


def run(args: argparse.Namespace) -> int:
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    _source, plan, task, shard, jobs = _inputs(args.root, args.plan_sha, args.group_index)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("PrefixKV native validation requires one allocated GPU")
    if "A100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("PrefixKV native validation requires an A100")
    completion = args.root / "workers" / task / f"{shard:02d}.json"
    if completion.exists():
        raise FileExistsError(completion)
    full_job = next(job for job in jobs if job["method"] == "full_kv")
    ordered = [
        full_job,
        *sorted(
            (job for job in jobs if job is not full_job),
            key=lambda job: (int(job["cap_mib"]), str(job["method"])),
        ),
    ]
    first = load_config(full_job["config"])
    before = asdict(capture_gpu_environment(torch))
    records = []
    adapter = None
    try:
        torch.manual_seed(first.execution.seed)
        torch.use_deterministic_algorithms(True)
        adapter = load_hf_adapter(
            first.model.id,
            revision=first.model.revision,
            model_kwargs=_model_kwargs(first, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        reference = Path(full_job["reference"])
        for order, job in enumerate(ordered):
            config = load_config(job["config"])
            if config_sha256(config) != job["config_sha"]:
                raise ValueError("PrefixKV resolved configuration changed")
            is_full = job["method"] == "full_kv"
            record = record_hf_run(
                config,
                adapter,
                task=task,
                run_id=f"prefix-native-{task}-{shard:02d}-{job['method']}-{job['cap_mib']}",
                output_root=Path(job["output"]),
                ledger_path=Path(job["ledger"]),
                ledger_sha=job["ledger_sha"],
                sample_ids=tuple(job["sample_ids"]),
                phase="native_metric_byte_validation",
                repetition=0,
                order=order,
                export_metrics=True,
                materialized_prefixkv=True,
                fullkv_reference_output=reference if is_full else None,
                fullkv_reference_input=None if is_full else reference,
                fullkv_reference_sha256=None if is_full else digest(reference),
            )
            record.update(cap_mib=job["cap_mib"], config_sha=job["config_sha"])
            records.append(record)
            settle(adapter)
        payload = {
            "schema_version": 1,
            "status": "passed",
            "plan_sha256": args.plan_sha,
            "source_sha": plan["source_sha"],
            "task": task,
            "shard": shard,
            "sample_ids": full_job["sample_ids"],
            "reference": str(reference),
            "reference_sha256": digest(reference),
            "records": records,
            "gpu_before": before,
            "gpu_after": asdict(capture_gpu_environment(torch)),
            "slurm": {
                "job_id": os.environ.get("SLURM_JOB_ID", "not_used"),
                "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "not_used"),
                "node": os.environ.get("SLURMD_NODENAME", "not_used"),
            },
        }
    finally:
        if adapter is not None:
            del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    write_once(completion, payload)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--group-index", type=int, required=True)
    raise SystemExit(run(parser.parse_args()))
