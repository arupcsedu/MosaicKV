#!/usr/bin/env python3
"""Run one fail-closed physical-byte-matched published-baseline pilot shard.

This narrow runner intentionally uses the existing immutable TextVQA/DocVQA
sample ledgers.  It is a diagnostic for physical byte matching, not a claim
of reproduction on either baseline's native paper benchmark.
"""

from __future__ import annotations

import argparse
import gc

# This worker is invoked directly by Slurm and can also be imported by tests.
# Load only its sibling's baseline configuration helpers by file path; shared
# HF execution helpers live in the installed package above.
import importlib.util
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch

from mosaickv.adapters.huggingface import load_hf_adapter
from mosaickv.config import load_config
from mosaickv.evaluation.worker_support import record_hf_run, settle
from mosaickv.hf_cli import _model_kwargs
from mosaickv.measurements.telemetry import capture_gpu_environment

_BASELINE_WORKER_PATH = Path(__file__).with_name("run_baseline_fidelity_review_worker.py")
_BASELINE_WORKER_SPEC = importlib.util.spec_from_file_location(
    "mosaickv_baseline_fidelity_worker", _BASELINE_WORKER_PATH
)
if _BASELINE_WORKER_SPEC is None or _BASELINE_WORKER_SPEC.loader is None:  # pragma: no cover
    raise ImportError(f"cannot load baseline worker helpers from {_BASELINE_WORKER_PATH}")
_BASELINE_WORKER = importlib.util.module_from_spec(_BASELINE_WORKER_SPEC)
_BASELINE_WORKER_SPEC.loader.exec_module(_BASELINE_WORKER)
FAMILY_METHOD = _BASELINE_WORKER.FAMILY_METHOD
_load_json = _BASELINE_WORKER._load_json
_sha = _BASELINE_WORKER._sha
_snapshot = _BASELINE_WORKER._snapshot
byte_capped_configs = _BASELINE_WORKER.byte_capped_configs

CLASSIFICATION = "byte_capped_published_baseline_pilot"
METHOD_ORDER = ("full_kv", "prompt_attention_topk", "baseline", "mosaickv_exact")


def _write_once(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"immutable pilot output already differs: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def _require_clean_source(repository: Path, expected_sha: str) -> str:
    observed, dirty = _snapshot(repository)
    if dirty or observed != expected_sha:
        raise RuntimeError("byte-capped pilot requires the declared clean immutable source")
    return observed


def _profile_path(repository: Path, family: str, profile_forget_ratio: str | None) -> str | None:
    if family != "prefixkv":
        return None
    if profile_forget_ratio is None:
        raise ValueError("PrefixKV pilot requires --profile-forget-ratio")
    path = (
        repository
        / "third_party/PrefixKV/confs"
        / (f"prefixkv_llava-v1.5-7b_{profile_forget_ratio}.json")
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(FAMILY_METHOD), required=True)
    parser.add_argument("--task", choices=("textvqa", "docvqa"), required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--ledger-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--byte-cap", type=int, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--seed", type=int, default=20271111)
    parser.add_argument("--output-length", type=int, default=64)
    parser.add_argument("--profile-forget-ratio")
    parser.add_argument("--profile-reference-retention-ratio", type=float)
    parser.add_argument(
        "--prefixkv-transport",
        default="common_guarded",
        choices=("common_guarded", "ragged_llama_eager_v1"),
    )
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument(
        "--mandatory-policy", choices=("last_token", "first_and_last"), default="last_token"
    )
    args = parser.parse_args()

    repository = Path(__file__).resolve().parents[2]
    source_sha = _require_clean_source(repository, args.expected_git_sha)
    ledger_path = args.ledger.resolve()
    if _sha(ledger_path) != args.ledger_sha256:
        raise RuntimeError("immutable sample ledger hash changed")
    ledger = _load_json(ledger_path)
    sample_ids = tuple(str(item) for item in cast("list[object]", ledger["sample_ids"]))
    if not sample_ids or len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError("sample ledger identities are absent or duplicate")
    if ledger.get("task") != args.task:
        raise RuntimeError("ledger task differs from requested task")
    output_root = args.output_root.resolve()
    completion = output_root / "job_complete.json"
    if completion.exists():
        prior = _load_json(completion)
        if prior.get("status") == "completed" and prior.get("source_git_sha") == source_sha:
            return 0
        raise RuntimeError("pilot completion exists with a different identity")

    template = load_config(repository / "mosaickv/configs/paper_candidate_llava_v1.yaml")
    profile = _profile_path(repository, args.family, args.profile_forget_ratio)
    if profile is not None:
        # ``byte_capped_configs`` resolves this exact pinned path itself.  The
        # local guard above exists solely to reject a missing official profile.
        del profile
    configs = byte_capped_configs(
        template,
        repository=repository,
        family=args.family,
        task=args.task,
        model_id=args.model_id,
        model_revision=args.model_revision,
        seed=args.seed,
        output_length=args.output_length,
        byte_cap=args.byte_cap,
        profile_forget_ratio=args.profile_forget_ratio,
        profile_reference_retention_ratio=args.profile_reference_retention_ratio,
        prefixkv_transport=args.prefixkv_transport,
        mandatory_policy=args.mandatory_policy,
    )
    baseline_name = FAMILY_METHOD[args.family]
    expected_methods = (
        "full_kv",
        "prompt_attention_topk",
        baseline_name,
        "mosaickv_exact",
    )
    if tuple(configs) != expected_methods:
        raise RuntimeError("byte-capped method order changed")

    adapter = load_hf_adapter(
        args.model_id,
        revision=args.model_revision,
        model_kwargs=_model_kwargs(template, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    records: list[dict[str, Any]] = []
    before = asdict(capture_gpu_environment(torch))
    namespace = f"bytecap-{source_sha[:10]}-{args.family}-{args.task}-{args.byte_cap}"
    try:
        for order, method in enumerate(expected_methods):
            record = record_hf_run(
                configs[method],
                adapter,
                task=args.task,
                run_id=f"{namespace}-{method}",
                output_root=output_root,
                ledger_path=ledger_path,
                ledger_sha=args.ledger_sha256,
                sample_ids=sample_ids,
                phase=CLASSIFICATION,
                repetition=0,
                order=order,
            )
            record.update(
                {
                    "configured_byte_cap": args.byte_cap,
                    "profile_forget_ratio": args.profile_forget_ratio,
                    "profile_reference_retention_ratio": args.profile_reference_retention_ratio,
                }
            )
            records.append(record)
            settle(adapter)
    finally:
        after = asdict(capture_gpu_environment(torch))
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    completed = {
        record["method"]: {str(row["sample_id"]): row for row in record["rows"]}
        for record in records
    }
    compressed = ("prompt_attention_topk", baseline_name, "mosaickv_exact")
    checks: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        realized = {
            method: int(completed[method][sample_id]["active_kv_bytes"]) for method in compressed
        }
        check = {
            "sample_id": sample_id,
            "configured_byte_cap": args.byte_cap,
            "realized_active_kv_bytes": realized,
            "exact_byte_match": len(set(realized.values())) == 1,
            "within_hard_cap": max(realized.values()) <= args.byte_cap,
        }
        checks.append(check)
    if not all(check["exact_byte_match"] and check["within_hard_cap"] for check in checks):
        raise RuntimeError(f"physical-byte matching failed: {checks}")
    _write_once(
        completion,
        {
            "schema_version": 1,
            "status": "completed",
            "classification": CLASSIFICATION,
            "source_git_sha": source_sha,
            "source_git_dirty": False,
            "family": args.family,
            "task": args.task,
            "model_id": args.model_id,
            "model_revision": args.model_revision,
            "sample_ledger": str(ledger_path),
            "sample_ledger_sha256": args.ledger_sha256,
            "sample_count": len(sample_ids),
            "configured_byte_cap": args.byte_cap,
            "methods": list(expected_methods),
            "records": records,
            "byte_match_checks": checks,
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
