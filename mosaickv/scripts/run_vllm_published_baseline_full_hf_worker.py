#!/usr/bin/env python3
"""Run three HF baseline reimplementations on one immutable sample shard."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import torch
from scripts.run_paper_candidate_v1_preflight_worker import _control_configs, _record, _settle

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import LookMConfig, PrefixKVConfig, VLCacheConfig, load_config
from mosaickv.hf_cli import _model_kwargs
from mosaickv.types import LookMMergeStrategy, MosaicKVMethod, PrefixKVProfileMode

METHODS = ("lookm_reimpl", "prefixkv_reimpl", "vl_cache_reimpl")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_snapshot(repository: Path) -> dict[str, Any]:
    commit = subprocess.run(
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
    return {"commit": commit, "git_dirty": bool(status.strip())}


def _configs(template: Any, *, task: str, seed: int, output_length: int) -> dict[str, Any]:
    prompt = _control_configs(
        template,
        task=task,
        seed=seed,
        output_length=output_length,
    )["prompt_attention_topk"]
    token_cache = replace(
        prompt.cache,
        block_size=1,
        physical_budget_block_size=1,
        physical_budget_mandatory_policy="method",
        retention_ratio=0.13,
        budget_value=2**63 - 1,
        total_storage_budget_bytes=None,
    )
    return {
        "lookm_reimpl": replace(
            prompt,
            method=MosaicKVMethod.LOOKM_REIMPL,
            cache=token_cache,
            lookm=LookMConfig(
                enabled=True,
                recent_ratio=0.065,
                important_ratio=0.065,
                merge_strategy=LookMMergeStrategy.PIVOTAL,
                text_prior=True,
            ),
        ),
        "prefixkv_reimpl": replace(
            prompt,
            method=MosaicKVMethod.PREFIXKV_REIMPL,
            cache=token_cache,
            prefixkv=PrefixKVConfig(
                enabled=True,
                profile_mode=PrefixKVProfileMode.OFFLINE_PROFILE,
                profile_path=(
                    "/scratch/djy8hg/workdir/MosaicKV/third_party/PrefixKV/"
                    "confs/prefixkv_llava-v1.5-7b_0.8.json"
                ),
                start_size=1,
                protect_size=1,
                eviction_distance=-25,
            ),
        ),
        "vl_cache_reimpl": replace(
            prompt,
            method=MosaicKVMethod.VL_CACHE_REIMPL,
            cache=token_cache,
            vl_cache=VLCacheConfig(
                enabled=True,
                sparsity_threshold=0.01,
                min_layer_retention=0.01,
                max_layer_retention=1.0,
                recent_window_fraction=0.1,
            ),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    package_root = repository / "mosaickv"
    job_path = args.job_spec.resolve()
    job = cast("dict[str, Any]", json.loads(job_path.read_text()))
    expected_sha = str(job["source_git_sha"])
    if _source_snapshot(repository) != {"commit": expected_sha, "git_dirty": False}:
        raise RuntimeError("full baseline worker requires its clean immutable source SHA")
    if tuple(cast("list[str]", job["methods"])) != METHODS:
        raise RuntimeError("baseline method order changed")
    ledger_path = Path(str(job["sample_ledger_path"])).resolve()
    if _sha(ledger_path) != str(job["sample_ledger_sha256"]):
        raise RuntimeError("sample ledger SHA-256 changed")
    ledger = cast("dict[str, Any]", json.loads(ledger_path.read_text()))
    sample_ids = tuple(str(value) for value in cast("list[str]", ledger["sample_ids"]))
    if len(sample_ids) != int(job["sample_count"]):
        raise RuntimeError("sample shard count changed")

    task = str(job["task"])
    shard = int(job["shard_index"])
    namespace = f"baseline-full-{expected_sha[:10]}-{task}-{shard:03d}"
    output_root = args.output_root.resolve()
    complete = output_root / "hf_complete.json"
    if complete.exists():
        recorded = json.loads(complete.read_text())
        if recorded.get("status") == "completed" and recorded.get("job_spec_sha256") == _sha(
            job_path
        ):
            return 0
        raise RuntimeError("existing HF completion marker is inconsistent")

    template = load_config(package_root / "configs" / "paper_candidate_llava_v1.yaml")
    configs = _configs(
        template,
        task=task,
        seed=int(job["seed"]),
        output_length=int(job["output_length"]),
    )
    adapter = load_hf_adapter(
        template.model.id,
        revision=template.model.revision,
        model_kwargs=_model_kwargs(configs[METHODS[0]], local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    records: list[dict[str, Any]] = []
    try:
        for order, method in enumerate(METHODS):
            records.append(
                _record(
                    configs[method],
                    adapter,
                    task=task,
                    run_id=f"{namespace}-{method}",
                    output_root=output_root,
                    ledger_path=ledger_path,
                    ledger_sha=str(job["sample_ledger_sha256"]),
                    sample_ids=sample_ids,
                    phase="canonical_published_baseline_extension_to_existing_9000",
                    repetition=0,
                    order=order,
                )
            )
            _settle(adapter)
    finally:
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    payload = {
        "schema_version": 1,
        "status": "completed",
        "namespace": namespace,
        "source_git_sha": expected_sha,
        "job_spec": str(job_path),
        "job_spec_sha256": _sha(job_path),
        "sample_ledger_sha256": str(job["sample_ledger_sha256"]),
        "methods": list(METHODS),
        "sample_count": len(sample_ids),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_used"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "not_used"),
        "runs": records,
    }
    complete.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
