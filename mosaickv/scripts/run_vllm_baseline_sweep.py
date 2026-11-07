#!/usr/bin/env python3
"""Matched-byte, explicitly exploratory sweep of published vLLM projections.

HF selectors and vLLM engines run in separate processes. No package mutation,
no fallback methods, no claim of online native selection or paper reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from prepare_vllm_byte_capped_mirror import (
    controls,
    development_source_binding,
    prepare,
    sha,
    write_once,
)
from run_byte_capped_published_baseline_pilot import byte_capped_configs

from mosaickv.config import LookMConfig, VLCacheConfig, canonical_config_json, load_config
from mosaickv.types import MosaicKVMethod

METHODS = (
    "full_kv",
    "prompt_attention_topk",
    "lookm_reimpl",
    "prefixkv_reimpl",
    "vl_cache_reimpl",
    "mosaickv_exact",
)
TASKS = ("mmstar", "textvqa", "docvqa")
CAPS_MIB = (40, 80, 120)
MODEL = "llava-hf/llava-1.5-7b-hf"
REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"


def configurations(source: Path, task: str, cap: int) -> dict:
    if task not in TASKS or cap not in {n * 2**20 for n in CAPS_MIB}:
        raise ValueError("unregistered task or byte cap")
    base = byte_capped_configs(
        load_config(source / "mosaickv/configs/paper_candidate_llava_v1.yaml"),
        repository=source,
        family="prefixkv",
        task=task,
        model_id=MODEL,
        model_revision=REVISION,
        seed=20271001,
        output_length=64,
        byte_cap=cap,
        profile_forget_ratio="0.8",
        profile_reference_retention_ratio=0.2,
        prefixkv_transport="ragged_llama_eager_v1",
        mandatory_policy="first_and_last",
    )
    prompt = base["prompt_attention_topk"]
    base["prefixkv_reimpl"] = replace(
        base["prefixkv_reimpl"],
        prefixkv=replace(base["prefixkv_reimpl"].prefixkv, decode_policy="effective_profile_ratio"),
    )
    base["lookm_reimpl"] = replace(
        prompt,
        method=MosaicKVMethod.LOOKM_REIMPL,
        # Physical capacity, not these nominal placeholders, controls selection.
        lookm=LookMConfig(enabled=True, recent_ratio=0.495, important_ratio=0.495),
    )
    base["vl_cache_reimpl"] = replace(
        prompt,
        method=MosaicKVMethod.VL_CACHE_REIMPL,
        vl_cache=VLCacheConfig(
            enabled=True,
            sparsity_threshold=0.01,
            min_layer_retention=0.01,
            max_layer_retention=1.0,
            recent_window_fraction=0.1,
        ),
    )
    shared = controls(json.loads(canonical_config_json(prompt)))
    for name in METHODS[1:]:
        if controls(json.loads(canonical_config_json(base[name]))) != shared:
            raise ValueError(f"mismatched source controls: {name}")
    return {name: base[name] for name in METHODS}


def check(root: Path, expected: str, index: int | None = None) -> tuple[dict, dict | None]:
    if sha(root / "plan.json") != expected:
        raise ValueError("sweep plan changed")
    plan = json.loads((root / "plan.json").read_text())
    if plan["canonical_eligible"] or plan["maximum_a100s"] != 4:
        raise ValueError("unexpected sweep classification or resources")
    source = Path(plan["source_root"])
    if Path(__file__).resolve().parents[2] != source.resolve():
        raise ValueError("worker imported from outside frozen snapshot")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if commit != plan["source_git_sha"]:
        raise ValueError("source commit changed")
    for name, expected_hash in plan["bound_files"].items():
        if sha(root / name) != expected_hash:
            raise ValueError(f"bound file changed: {name}")
    development_source_binding(root / "source_attestation.json", source, "preflight")
    job = None if index is None else plan["jobs"][index]
    if job is not None:
        ledger = json.loads((root / job["ledger"]).read_text())
        if len(ledger["sample_ids"]) != 8 or len(set(ledger["sample_ids"])) != 8:
            raise ValueError("sweep requires eight paired identities per task")
        if ledger["task"] != job["task"]:
            raise ValueError("ledger task changed")
    return plan, job


def collect_hf(root: Path, expected: str, index: int) -> None:
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.evaluation.worker_support import record_hf_run, settle
    from mosaickv.hf_cli import _model_kwargs

    plan, job = check(root, expected, index)
    source = Path(plan["source_root"])
    configs = configurations(source, job["task"], job["cap_bytes"])
    output = root / "outputs" / str(index) / "hf_sources"
    completion = output / "completion.json"
    if completion.exists():
        if json.loads(completion.read_text())["status"] != "completed":
            raise ValueError("failed source must be retried in a new immutable output root")
        return
    ledger_path = root / job["ledger"]
    ledger = json.loads(ledger_path.read_text())
    records, failures = [], []
    torch.manual_seed(20271001)
    torch.use_deterministic_algorithms(True)
    adapter = load_hf_adapter(
        MODEL,
        revision=REVISION,
        model_kwargs=_model_kwargs(configs["full_kv"], local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    for order, (method, config) in enumerate(configs.items()):
        try:
            records.append(
                record_hf_run(
                    config,
                    adapter,
                    task=job["task"],
                    run_id=f"vllm-source-{index}-{method}",
                    output_root=output,
                    ledger_path=ledger_path,
                    ledger_sha=sha(ledger_path),
                    sample_ids=tuple(ledger["sample_ids"]),
                    phase="development_selector_source",
                    repetition=0,
                    order=order,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "method": method,
                    "error": f"{type(exc).__name__}: {exc}",
                    "planned_sample_ids": ledger["sample_ids"],
                }
            )
        settle(adapter)
        print(
            json.dumps({"phase": "hf_source", "method": method, "failures": len(failures)}),
            flush=True,
        )
        write_once(output / f"progress-{order}.json", {"records": records, "failures": failures})
    write_once(
        completion,
        {
            "status": "failed" if failures else "completed",
            "classification": "byte_capped_published_baseline_pilot",
            "source_git_sha": plan["source_git_sha"],
            "source_git_dirty": True,
            "task": job["task"],
            "configured_byte_cap": job["cap_bytes"],
            "sample_ledger": str(ledger_path),
            "sample_ledger_sha256": sha(ledger_path),
            "records": records,
            "failures": failures,
        },
    )
    if failures:
        raise RuntimeError("source failures preserved; no partial comparison authorized")


def run(root: Path, expected: str, index: int) -> int:
    plan, job = check(root, expected, index)
    source = Path(plan["source_root"])
    output = root / "outputs" / str(index)
    terminal = output / "terminal.json"
    if terminal.exists():
        raise ValueError("terminal record exists; use a new retry root")
    result = {
        "status": "failed",
        "index": index,
        "task": job["task"],
        "cap_bytes": job["cap_bytes"],
        "plan_sha256": expected,
        "classification": "development_only",
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
    }
    try:
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--root",
                str(root),
                "--plan-sha",
                expected,
                "--index",
                str(index),
                "--hf-source",
            ],
            check=True,
        )
        check(root, expected, index)
        mirror = output / "vllm"
        prepare(
            output / "hf_sources/completion.json",
            mirror,
            cap=job["cap_bytes"],
            sample_limit=None,
            stage="preflight",
            source_root=source,
            attestation=root / "source_attestation.json",
            allow_development_hf=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(source / "mosaickv/scripts/run_vllm_byte_capped_mirror.py"),
                "--job",
                str(mirror / "job.json"),
                "--job-sha256",
                sha(mirror / "job.json"),
            ],
            check=True,
        )
        validation = json.loads((mirror / "validation.json").read_text())
        if not validation["telemetry_checks_passed"]:
            raise ValueError("native cache telemetry gate failed")
        result.update(
            status="passed",
            validation=str(mirror / "validation.json"),
            validation_sha256=sha(mirror / "validation.json"),
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    write_once(terminal, result)
    return 0 if result["status"] == "passed" else 1


def analyze(root: Path, expected: str) -> int:
    import numpy as np
    import pandas as pd

    plan, _ = check(root, expected)
    failures, observations, summaries = [], [], []
    for index, job in enumerate(plan["jobs"]):
        output = root / "outputs" / str(index)
        terminal = output / "terminal.json"
        state = json.loads(terminal.read_text()) if terminal.exists() else {"status": "missing"}
        if state["status"] != "passed":
            failures.append({"index": index, **job, **state})
            continue
        validation_path = Path(state["validation"])
        if sha(validation_path) != state["validation_sha256"]:
            raise ValueError("validation changed")
        spec = json.loads((output / "vllm/job.json").read_text())
        arms = {}
        for arm in spec["arms"]:
            name = arm["name"]
            files = list((output / "vllm/outputs" / name).glob("*/results.jsonl"))
            if len(files) != 1:
                raise ValueError("missing/ambiguous predictions")
            rows = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
            arms[name] = {row["sample_id"]: row for row in rows}
            observations.extend(
                dict(
                    row,
                    sweep_task=job["task"],
                    shared_cap_bytes=job["cap_bytes"],
                    sweep_arm=name,
                    classification="development_only",
                )
                for row in rows
            )
        ids = spec["sample_ids"]
        for name, rows in arms.items():
            if name == "native_identity":
                continue
            delta = np.array(
                [
                    arms["mosaickv_exact"][sid]["task_score"] - rows[sid]["task_score"]
                    for sid in ids
                ],
                dtype=float,
            )
            seed = int(hashlib.sha256(f"{job['task']}/{name}".encode()).hexdigest()[:8], 16)
            boot = (
                np.random.default_rng(seed).choice(delta, (10000, len(ids)), replace=True).mean(1)
            )
            summaries.append(
                {
                    "task": job["task"],
                    "cap_mib": job["cap_bytes"] / 2**20,
                    "method": name if name == "full_kv" else name + "__vllm_requestwide_projection",
                    "sample_count": len(ids),
                    "failed_samples": 0,
                    "task_score": sum(row["task_score"] for row in rows.values()) / len(ids),
                    "active_kv_bytes": sum(row["active_kv_bytes"] for row in rows.values())
                    / len(ids),
                    "mosaic_score_delta": float(delta.mean()),
                    "paired_ci_low": float(np.quantile(boot, 0.025)),
                    "paired_ci_high": float(np.quantile(boot, 0.975)),
                    "decode_seconds": None
                    if name == "full_kv"
                    else sum(row["decode_time"] for row in rows.values()) / len(ids),
                    "reconstructed_total_seconds": None
                    if name == "full_kv"
                    else sum(row["end_to_end_time"] for row in rows.values()) / len(ids),
                    "online_native_latency_claim": False,
                    "canonical_eligible": False,
                }
            )
    analysis = root / "analysis"
    analysis.mkdir(exist_ok=False)
    pd.DataFrame(observations).to_parquet(analysis / "observations.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(analysis / "summary.parquet", index=False)
    write_once(
        analysis / "report.json",
        {
            "status": "failed" if failures else "completed",
            "failures": failures,
            "summaries": summaries,
            "plan_sha256": expected,
            "canonical_eligible": False,
            "paper_reproduction_claim": False,
            "failed_shards_excluded_from_comparisons_but_reported": True,
        },
    )
    return bool(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--hf-source", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.analyze:
        raise SystemExit(analyze(args.root, args.plan_sha))
    if args.index is None:
        parser.error("worker requires --index")
    if args.hf_source:
        collect_hf(args.root, args.plan_sha, args.index)
    else:
        raise SystemExit(run(args.root, args.plan_sha, args.index))
