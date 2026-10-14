#!/usr/bin/env python3
"""Small frozen development cohort: native schedule and shared-byte ablation.

No profile fitting; no full campaign submission. FullKV is quality-only.
All direct comparisons share 40 MiB prefill storage and first/last protection.
The native 50% arm is a separate diagnostic, never a byte-matched competitor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from run_byte_capped_published_baseline_pilot import _require_clean_source, byte_capped_configs

from mosaickv.config import PrefixKVConfig, load_config
from mosaickv.types import BudgetUnit, PrefixKVProfileMode

MODEL = "llava-hf/llava-1.5-7b-hf"
REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
CAP = 40 * 2**20


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def configs(root, task):
    base = byte_capped_configs(
        load_config(root / "mosaickv/configs/paper_candidate_llava_v1.yaml"),
        repository=root,
        family="prefixkv",
        task=task,
        model_id=MODEL,
        model_revision=REVISION,
        seed=20271001,
        output_length=64,
        byte_cap=CAP,
        profile_forget_ratio="0.8",
        profile_reference_retention_ratio=0.2,
        prefixkv_transport="ragged_llama_eager_v1",
        mandatory_policy="first_and_last",
    )
    prefix = base["prefixkv_reimpl"]
    corrected = replace(
        prefix, prefixkv=replace(prefix.prefixkv, decode_policy="effective_profile_ratio")
    )
    native = replace(
        prefix,
        cache=replace(
            prefix.cache,
            budget_unit=BudgetUnit.BLOCKS,
            budget_value=2**63 - 1,
            retention_ratio=0.5,
            physical_budget_block_size=None,
            physical_budget_mandatory_policy="method",
        ),
        prefixkv=PrefixKVConfig(
            enabled=True,
            profile_path=str(root / "third_party/PrefixKV/confs/prefixkv_llava-v1.5-7b_0.5.json"),
            hf_cache_transport="ragged_llama_eager_v1",
            decode_boundary_policy="protected",
            allocation_policy="upstream_round",
            decode_policy="profile_ratio",
        ),
    )
    return {
        "full_kv": base["full_kv"],
        "native_r050_prefixkv_reimpl": native,
        "prompt_attention_topk": base["prompt_attention_topk"],
        "byte_rescaled_prefixkv_reimpl": corrected,
        "prefixkv_selection_only_reimpl": replace(
            corrected, prefixkv=replace(corrected.prefixkv, decode_policy="append_only")
        ),
        "mosaickv_exact": base["mosaickv_exact"],
    }


def derived_ids(parent_ids):
    """Fixed ID-only continuation, excludes calibration [2:12] and prior [12:24]."""
    if len(parent_ids) < 32 or len(set(parent_ids)) != len(parent_ids):
        raise ValueError("need at least 32 distinct parent identities")
    return tuple(parent_ids[24:32])


def allocation_configs(root, task, profile_path, profile):
    """Fixed, label-blind allocation ablations; does not change the old retry.

    Freeze one exact-40-MiB profile on disjoint calibration IDs first. Callers
    must use new evaluation IDs and record all arms, including failures. This
    helper neither materializes samples nor authorizes/submits a GPU campaign.
    """
    if (
        profile.calibration_byte_cap != CAP
        or profile.calibration_bytes_per_layer_position != 16384
        or profile.model_id != MODEL
        or profile.model_revision != REVISION
        or profile.start_size != 1
        or profile.protect_size != 1
        or profile.search_policy != "upstream_protected_cdf"
    ):
        raise ValueError("allocation preflight requires the protected exact-40-MiB LLaVA profile")
    from mosaickv.baselines.prefixkv import PrefixKVOfflineProfile

    if (
        PrefixKVOfflineProfile.from_json_object(json.loads(Path(profile_path).read_text()))
        != profile
    ):
        raise ValueError("frozen allocation profile differs from supplied profile")
    base = configs(root, task)
    upstream = base["byte_rescaled_prefixkv_reimpl"]
    calibrated = replace(
        upstream,
        prefixkv=replace(
            upstream.prefixkv,
            profile_path=str(profile_path),
            profile_reference_retention_ratio=profile.target_retention_ratio,
        ),
    )
    uniform = replace(
        upstream,
        prefixkv=replace(
            upstream.prefixkv,
            profile_mode=PrefixKVProfileMode.FIXED_GLOBAL,
            profile_path=None,
            rescale_offline_profile_to_byte_budget=False,
            profile_reference_retention_ratio=None,
        ),
    )
    arms = {
        "full_kv": base["full_kv"],
        "prompt_attention_topk": base["prompt_attention_topk"],
        "prefixkv_upstream_shape_reimpl": upstream,
        "prefixkv_byte_calibrated_reimpl": calibrated,
        "prefixkv_capacity_floor_reimpl": replace(
            calibrated,
            prefixkv=replace(calibrated.prefixkv, layer_budget_floor_fraction=0.5),
        ),
        "prefixkv_uniform_layer_reimpl": uniform,
        "mosaickv_exact": base["mosaickv_exact"],
    }
    for name, config in arms.items():
        if name != "full_kv":
            for field in ("cache", "execution", "generation", "model", "dataset"):
                if getattr(config, field) != getattr(upstream, field):
                    raise ValueError(f"allocation preflight changed shared {field}: {name}")
    return arms


def check_plan(root, expected_hash):
    plan_path = root / "plan.json"
    if digest(plan_path) != expected_hash:
        raise RuntimeError("immutable plan hash mismatch")
    plan = json.loads(plan_path.read_text())
    source = Path(__file__).resolve().parents[2]
    _require_clean_source(source, plan["source_sha"])
    if plan["full_campaign_authorized"] or plan["max_a100s_global"] != 4:
        raise RuntimeError("unexpected scope or GPU limit")
    for file, expected in plan["bound_files"].items():
        if digest(root / file) != expected:
            raise RuntimeError(f"bound file changed: {file}")
    return plan, source


def run(args):
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.evaluation.worker_support import record_hf_run, settle
    from mosaickv.hf_cli import _model_kwargs

    plan, source = check_plan(args.root, args.plan_sha)
    job = plan["jobs"][args.index]
    task = job["task"]
    parity = json.loads((args.root / f"upstream-parity-{task}.json").read_text())
    if parity["status"] != "passed" or set(parity["points"]) != {
        "native_profile_r050",
        "byte_rescaled_40mib",
    }:
        raise RuntimeError("upstream algorithm parity must pass before evaluating quality")
    ledger_path = args.root / job["evaluation"]
    ledger = json.loads(ledger_path.read_text())
    ids = ledger["sample_ids"]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise RuntimeError("requires eight distinct preregistered IDs")
    output = args.root / "outputs" / task
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "status": "running",
        "task": task,
        "records": [],
        "failures": [],
        "source_sha": plan["source_sha"],
        "plan_sha256": args.plan_sha,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "full_campaign_authorized": False,
        "classification": "retrospective_disjoint_development_only",
        "parity_artifact_sha256": digest(args.root / f"upstream-parity-{task}.json"),
    }
    try:
        torch.manual_seed(20271001)
        torch.use_deterministic_algorithms(True)
        arms = configs(source, task)
        adapter = load_hf_adapter(
            MODEL,
            revision=REVISION,
            model_kwargs=_model_kwargs(arms["full_kv"], local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        for order, (name, config) in enumerate(arms.items()):
            try:
                result = record_hf_run(
                    config,
                    adapter,
                    task=task,
                    run_id=f"prefix-decode-{task}-{name}",
                    output_root=output,
                    ledger_path=ledger_path,
                    ledger_sha=digest(ledger_path),
                    sample_ids=tuple(ids),
                    phase=summary["classification"],
                    repetition=0,
                    order=order,
                )
                result["variant"] = name
                summary["records"].append(result)
            except Exception as exc:
                summary["failures"].append(
                    {"variant": name, "error": f"{type(exc).__name__}: {exc}"}
                )
            settle(adapter)
            print(
                json.dumps(
                    {"phase": "quality", "variant": name, "failure_count": len(summary["failures"])}
                ),
                flush=True,
            )
        summary["status"] = "completed" if not summary["failures"] else "failed"
    except Exception as exc:
        summary.update(status="failed", fatal_error=f"{type(exc).__name__}: {exc}")
    finally:
        write_once(output / "completion.json", summary)
    return int(summary["status"] != "completed")


def analyze(args):
    import numpy as np

    plan, _ = check_plan(args.root, args.plan_sha)
    report = {
        "classification": "development_only",
        "full_campaign_authorized": False,
        "fullkv_latency_comparator": False,
        "timing_claim": "informational_0_warmups_1_trial",
        "source_sha": plan["source_sha"],
        "plan_sha256": args.plan_sha,
        "tasks": {},
    }
    for job in plan["jobs"]:
        task = job["task"]
        path = args.root / "outputs" / task / "completion.json"
        if not path.exists():
            report["tasks"][task] = {"gate": False, "error": "missing completion artifact"}
            continue
        completion = json.loads(path.read_text())
        ids = json.loads((args.root / job["evaluation"]).read_text())["sample_ids"]
        arms = {
            r["variant"]: {row["sample_id"]: row for row in r["rows"]}
            for r in completion["records"]
        }
        outcomes = {}
        for name, rows in arms.items():
            if set(rows) != set(ids) or any(r["status"] != "completed" for r in rows.values()):
                raise RuntimeError(
                    "missing or failed samples must not be dropped from denominators"
                )
            byte_matched = name not in {"full_kv", "native_r050_prefixkv_reimpl"}
            values = np.array([rows[i]["task_score"] for i in ids])
            out = {
                "n": len(ids),
                "task_score": float(values.mean()),
                "prefill_bytes": sorted({rows[i]["active_kv_bytes"] for i in ids}),
                "byte_matching_applicable": byte_matched,
                "byte_gate": all(rows[i]["active_kv_bytes"] == CAP for i in ids)
                if byte_matched
                else None,
            }
            if name != "full_kv":
                out["mean_end_to_end_seconds"] = float(
                    np.mean([rows[i]["end_to_end_time"] for i in ids])
                )
            if byte_matched and "prompt_attention_topk" in arms:
                control = arms["prompt_attention_topk"]
                difference = values - np.array([control[i]["task_score"] for i in ids])
                rng = np.random.default_rng(20271001)
                replicates = difference[rng.integers(0, len(ids), size=(10000, len(ids)))].mean(1)
                out.update(
                    paired_score_delta=float(difference.mean()),
                    paired_score_delta_ci95=np.quantile(replicates, [0.025, 0.975]).tolist(),
                    empirical_quality_gate=bool(difference.mean() >= -1e-12),
                    answer_agreement=float(
                        np.mean([rows[i]["answer"] == control[i]["answer"] for i in ids])
                    ),
                )
            # Decode growth differs: report the actual final storage independently.
            traces = [
                json.loads(p.read_text())
                for p in (
                    args.root / "outputs" / task / "traces" / f"prefix-decode-{task}-{name}"
                ).rglob("*.json")
            ]
            completed = [
                t for t in traces if t.get("status") == "completed" and "final_active_kv_bytes" in t
            ]
            out["trace_count"] = len(completed)
            if len(completed) != len(ids):
                raise RuntimeError(f"incomplete traces for {task}/{name}")
            out["final_active_bytes"] = sorted({t["final_active_kv_bytes"] for t in completed})
            out["effective_methods"] = sorted({t["effective_method"] for t in completed})
            out["eviction_count"] = sum(
                e.get("removed_physical_position") is not None
                for t in completed
                for e in t.get("prefixkv_decode_events", [])
            )
            if name == "prefixkv_selection_only_reimpl" and (
                out["eviction_count"] != 0
                or out["effective_methods"] != ["prefixkv_selection_only_reimpl"]
            ):
                raise RuntimeError("selection-only ablation mislabeled or unexpectedly evicted")
            outcomes[name] = out
        report["tasks"][task] = {
            "status": completion["status"],
            "failures": completion["failures"],
            "arms": outcomes,
            "planned_denominator": len(ids),
        }
    write_once(args.root / "analysis" / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--index", type=int, choices=(0, 1))
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if not args.analyze and args.index is None:
        parser.error("--index is required for a GPU worker")
    raise SystemExit(analyze(args) if args.analyze else run(args))
