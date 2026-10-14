#!/usr/bin/env python3
"""Explicitly exploratory pretrained A/B/C validation from a hash-bound dirty snapshot.

Never substitutes for a clean canonical run. No submission or full-campaign promotion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from run_prefixkv_allocation_preflight import collect
from run_prefixkv_decode_preflight import CAP, MODEL, REVISION, configs, digest, write_once

from mosaickv.config import canonical_config, config_sha256
from mosaickv.evaluation.prefixkv_evidence_ablation import (
    EVIDENCE_ABLATIONS,
    evidence_ablation_configs,
)
from mosaickv.manifest import _worktree_fingerprint

CLASSIFICATION = "exploratory_nonheldout_prefixkv_evidence_ablations_v1"
FIRST = "first_and_last"
QUESTION = "question_and_boundaries"
ORDER = (
    (FIRST, "full_kv"),
    (FIRST, "prompt_attention_topk"),
    (FIRST, "mosaickv_exact"),
    (FIRST, "control"),
    (FIRST, "a_sink"),
    (QUESTION, "full_kv"),
    (QUESTION, "prompt_attention_topk"),
    (QUESTION, "mosaickv_exact"),
    (QUESTION, "b_protect"),
    (QUESTION, "ab_sink_protect"),
    (FIRST, "c_rank"),
    (FIRST, "ac_sink_rank"),
    (QUESTION, "bc_protect_rank"),
    (QUESTION, "abc_combined"),
)


def select_ids(parent_ids, prior_ids):
    """ID-only fixed continuation; not a globally untouched/held-out claim."""
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("duplicate parent identities")
    available = [s for s in parent_ids if s not in set(prior_ids)]
    if len(available) < 26:
        raise ValueError("need ten calibration and sixteen evaluation identities")
    return tuple(available[:10]), tuple(available[10:26])


def check_plan(root, expected_hash):
    if digest(root / "plan.json") != expected_hash:
        raise RuntimeError("plan changed")
    plan = json.loads((root / "plan.json").read_text())
    source = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if (
        str(source.resolve()) != plan["source_root"]
        or sha != plan["source_sha"]
        or _worktree_fingerprint(source) != plan["source_patch_sha256"]
    ):
        raise RuntimeError("isolated source snapshot changed")
    if (
        plan["classification"] != CLASSIFICATION
        or not plan["source_dirty"]
        or plan["canonical_eligible"]
        or plan["full_campaign_authorized"]
        or not plan["gpu_preflight_authorized"]
        or plan["arms"] != [list(p) for p in ORDER]
    ):
        raise RuntimeError("scope mismatch; only exploratory preflight is authorized")
    for path, expected in plan["bound_files"].items():
        if digest(root / path) != expected:
            raise RuntimeError(f"bound input changed: {path}")
    return plan, source


def run(args):
    import torch

    from mosaickv.adapters.huggingface import (
        compare_cache_reinjection,
        compare_with_generate,
        load_hf_adapter,
    )
    from mosaickv.baselines.prefixkv import generate_prefixkv_byte_profile
    from mosaickv.evaluation.worker_support import record_hf_run, settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan, source = check_plan(args.root, args.plan_sha)
    job = plan["jobs"][args.index]
    task = job["task"]
    cal = json.loads((args.root / job["calibration"]).read_text())
    ev = json.loads((args.root / job["evaluation"]).read_text())
    if len(cal["sample_ids"]) != 10 or len(ev["sample_ids"]) != 16:
        raise RuntimeError("wrong fixed calibration/evaluation size")
    if set(cal["sample_ids"]) & set(ev["sample_ids"]):
        raise RuntimeError("calibration/evaluation overlap")
    if any(cal[k] != ev[k] for k in ("task", "dataset_id", "dataset_revision", "split")):
        raise RuntimeError("ledger provenance mismatch")
    output = args.root / "outputs" / task
    output.mkdir(parents=True, exist_ok=False)
    summary = dict(
        status="running",
        records=[],
        failures=[],
        task=task,
        classification=CLASSIFICATION,
        canonical_eligible=False,
        source_sha=plan["source_sha"],
        source_patch_sha256=plan["source_patch_sha256"],
        source_dirty=True,
        plan_sha256=args.plan_sha,
        slurm_job_id=os.getenv("SLURM_JOB_ID"),
        planned_denominator=16,
    )
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("requires exactly one allocated GPU")
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        base = configs(source, task)
        controls = {n: base[n] for n in ("full_kv", "prompt_attention_topk", "mosaickv_exact")}
        adapter = load_hf_adapter(
            MODEL,
            revision=REVISION,
            model_kwargs=_model_kwargs(controls["full_kv"], local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        summary["gpu_environment"] = asdict(capture_gpu_environment(torch))
        observations, captures, prepared, prefill = collect(
            adapter, cal, output, capture_question_evidence=True
        )
        profile_paths = {}
        for arm in EVIDENCE_ABLATIONS:
            profile = generate_prefixkv_byte_profile(
                observations,
                byte_cap=CAP,
                bytes_per_layer_position=16384,
                model_id=MODEL,
                model_revision=REVISION,
                dataset_id=cal["dataset_id"],
                dataset_revision=cal["dataset_revision"],
                calibration_split=cal["split"],
                evaluation_sample_ids=tuple(ev["sample_ids"]),
                seed=plan["seed"],
                exclude_protected_from_calibration=arm.sink_neutral,
                prompt_reservation_policy=arm.profile_policy,
                question_score_weight=arm.question_weight,
            )
            profile_paths[arm.name] = profile.write(output / "profiles" / (arm.name + ".json"))
        groups = evidence_ablation_configs(controls, profile_paths)
        write_once(
            output / "frozen_ablations.json",
            {
                "profiles": {
                    n: {"path": str(p), "sha256": digest(p)} for n, p in profile_paths.items()
                },
                "configs": {
                    f"{policy}/{name}": {
                        "config": canonical_config(cfg),
                        "config_sha256": config_sha256(cfg),
                    }
                    for policy, arms in groups.items()
                    for name, cfg in arms.items()
                },
                "calibration_captures": captures,
                "frozen_before_evaluation": True,
                "offline_calibration_seconds": time.perf_counter() - started,
            },
        )
        parity = {"status": "running", "atol": 1e-4, "tokens": 16}
        try:
            for name, function in (
                ("ordinary_generate", compare_with_generate),
                ("retention_one_reinjection", compare_cache_reinjection),
            ):
                result = function(adapter, prepared, max_new_tokens=16)
                parity[name] = asdict(result)
                if result.token_agreement != 1.0 or result.maximum_logit_difference > 1e-4:
                    raise RuntimeError(f"{name} parity failed")
            parity["status"] = "passed"
        finally:
            write_once(output / "parity.json", parity)
        del prefill, prepared
        settle(adapter)
        for order, (policy, name) in enumerate(ORDER):
            check_plan(args.root, args.plan_sha)
            config = groups[policy][name]
            run_id = f"prefix-evidence-{task}-{policy}-{name}"
            try:
                record = record_hf_run(
                    config,
                    adapter,
                    task=task,
                    run_id=run_id,
                    output_root=output,
                    ledger_path=args.root / job["evaluation"],
                    ledger_sha=digest(args.root / job["evaluation"]),
                    sample_ids=tuple(ev["sample_ids"]),
                    phase=CLASSIFICATION,
                    repetition=0,
                    order=order,
                )
                record.update(variant=name, mandatory_policy=policy)
                for row in record["rows"]:
                    if name != "full_kv" and row["active_kv_bytes"] != CAP:
                        raise RuntimeError("initial byte match failed")
                    manifest = json.loads(Path(row["manifest_path"]).read_text())
                    if (
                        manifest["source"]["canonical_eligible"]
                        or manifest["source"]["patch_sha"] != plan["source_patch_sha256"]
                    ):
                        raise RuntimeError("incorrect exploratory source provenance")
                summary["records"].append(record)
            except Exception as exc:
                summary["failures"].append(
                    {
                        "variant": name,
                        "mandatory_policy": policy,
                        "run_id": run_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "raw_output": str(output / run_id / "raw.jsonl"),
                    }
                )
            settle(adapter)
            write_once(output / "progress" / f"{order:02d}.json", summary)
            print(
                json.dumps(
                    {
                        "phase": "quality",
                        "variant": name,
                        "policy": policy,
                        "failures": len(summary["failures"]),
                    }
                ),
                flush=True,
            )
        summary["status"] = "completed" if not summary["failures"] else "failed"
    except Exception as exc:
        summary.update(status="failed", fatal_error=f"{type(exc).__name__}: {exc}")
    finally:
        summary["wall_seconds"] = time.perf_counter() - started
        write_once(output / "completion.json", summary)
    return int(summary["status"] != "completed")


def summarize(completion, ids):
    import numpy as np

    records = {(r["mandatory_policy"], r["variant"]): r for r in completion["records"]}
    result = {
        "status": completion["status"],
        "failures": completion.get("failures", []),
        "fatal_error": completion.get("fatal_error"),
        "groups": {},
        "ablation_deltas": {},
    }
    pairs = {}
    for policy, name in ORDER:
        rows = records.get((policy, name), {}).get("rows", [])
        successful = [r for r in rows if r["status"] == "completed"]
        complete = len(successful) == len(ids) and {r["sample_id"] for r in successful} == set(ids)
        out = {
            "attempted": len(ids),
            "completed": len(successful),
            "failed_or_unavailable": len(ids) - len(successful),
            "complete": complete,
        }
        result["groups"].setdefault(policy, {})[name] = out
        if not complete:
            out["quality_gate"] = False
            continue
        if any(not math.isfinite(float(r["task_score"])) for r in rows):
            raise ValueError("non-finite task score")
        pairs[(policy, name)] = {r["sample_id"]: r for r in rows}
        out.update(
            task_score=float(np.mean([r["task_score"] for r in rows])),
            active_bytes=sorted({r["active_kv_bytes"] for r in rows}),
        )
        if name != "full_kv":
            out.update(
                byte_gate=all(r["active_kv_bytes"] == CAP for r in rows),
                mean_end_to_end_seconds=float(np.mean([r["end_to_end_time"] for r in rows])),
                mean_compression_seconds=float(np.mean([r["compression_time"] for r in rows])),
            )

    def paired(left, right):
        delta = np.array(
            [pairs[left][s]["task_score"] - pairs[right][s]["task_score"] for s in ids]
        )
        rng = np.random.default_rng(20271001)
        boot = delta[rng.integers(0, len(ids), size=(10000, len(ids)))].mean(1)
        ci = np.quantile(boot, [0.025, 0.975]).tolist()
        return {
            "paired_score_delta": float(delta.mean()),
            "ci95": ci,
            "positive_point_estimate": bool(delta.mean() > 0),
            "ci_excludes_zero_positive": bool(ci[0] > 0),
            "answer_agreement": float(
                np.mean([pairs[left][s]["answer"] == pairs[right][s]["answer"] for s in ids])
            ),
        }

    for policy, name in pairs:
        if name == "full_kv" or (policy, "prompt_attention_topk") not in pairs:
            continue
        out = result["groups"][policy][name]
        out["vs_same_policy_topk"] = paired((policy, name), (policy, "prompt_attention_topk"))
        informative = (policy, "full_kv") in pairs and any(
            r["task_score"] > 0 for r in pairs[(policy, "full_kv")].values()
        )
        out["quality_informative"] = informative
        out["quality_gate"] = bool(
            out["byte_gate"]
            and informative
            and out["vs_same_policy_topk"]["paired_score_delta"] >= -1e-12
        )
    comparisons = (
        ("A_vs_control", (FIRST, "a_sink"), (FIRST, "control")),
        ("C_vs_control", (FIRST, "c_rank"), (FIRST, "control")),
        ("AC_vs_A", (FIRST, "ac_sink_rank"), (FIRST, "a_sink")),
        ("AB_vs_B", (QUESTION, "ab_sink_protect"), (QUESTION, "b_protect")),
        ("BC_vs_B", (QUESTION, "bc_protect_rank"), (QUESTION, "b_protect")),
        ("ABC_vs_AB", (QUESTION, "abc_combined"), (QUESTION, "ab_sink_protect")),
        ("B_vs_control_policy_change", (QUESTION, "b_protect"), (FIRST, "control")),
        ("ABC_vs_control_policy_change", (QUESTION, "abc_combined"), (FIRST, "control")),
    )
    for label, left, right in comparisons:
        result["ablation_deltas"][label] = (
            {**paired(left, right), "same_mandatory_policy": left[0] == right[0]}
            if left in pairs and right in pairs
            else {"status": "incomplete"}
        )
    return result


def analyze(args):
    plan, _ = check_plan(args.root, args.plan_sha)
    report = dict(
        classification=CLASSIFICATION,
        canonical_eligible=False,
        full_campaign_authorized=False,
        source_sha=plan["source_sha"],
        source_patch_sha256=plan["source_patch_sha256"],
        tasks={},
        inputs={},
        latency="informational_only; FullKV excluded; initial budget only",
    )
    for job in plan["jobs"]:
        path = args.root / "outputs" / job["task"] / "completion.json"
        if not path.exists():
            progress = sorted((path.parent / "progress").glob("*.json"))
            if not progress:
                report["tasks"][job["task"]] = {"status": "missing", "quality_gate": False}
                continue
            path = progress[-1]
        completed = json.loads(path.read_text())
        if completed["plan_sha256"] != args.plan_sha:
            raise RuntimeError("worker plan mismatch")
        ids = json.loads((args.root / job["evaluation"]).read_text())["sample_ids"]
        report["tasks"][job["task"]] = summarize(completed, ids)
        report["inputs"][str(path)] = digest(path)
    write_once(args.root / "analysis/report.json", report)
    lines = [
        "# PrefixKV evidence ablations — development-only GPU check",
        "",
        "Not held-out or paper-eligible. Quality gain is not assumed. "
        "All compressed comparisons use 40 MiB initial physical KV; decode growth differs.",
        "",
    ]
    for task, result in report["tasks"].items():
        lines += [f"## {task}", "", f"Worker status: {result['status']}", ""]
        for policy, arms in result.get("groups", {}).items():
            lines += [
                f"### Mandatory policy: {policy}",
                "",
                "| Arm | Completed/planned | Score | Delta vs Top-K (95% CI) "
                "| Active MiB | Quality gate |",
                "|---|---|---|---|---|---|",
            ]
            for name, a in arms.items():
                d = a.get("vs_same_policy_topk")
                delta = (
                    f"{d['paired_score_delta']:.4f} [{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]"
                    if d
                    else "not_applicable"
                )
                score = f"{a['task_score']:.4f}" if a.get("complete") else "failed/unavailable"
                memory = ", ".join(f"{v / 2**20:.3f}" for v in a.get("active_bytes", []))
                lines += [
                    f"| {name} | {a['completed']}/{a['attempted']} | {score} | {delta} | "
                    f"{memory} | {a.get('quality_gate', 'reference')} |"
                ]
            lines += [""]
        lines += ["A zero-score FullKV cohort is non-informative and cannot pass quality.", ""]
    with (args.root / "analysis/report.md").open("x") as stream:
        stream.write("\n".join(lines))
    print(json.dumps({"status": "analyzed", "report": str(args.root / "analysis/report.json")}))
    return 0 if all(t["status"] == "completed" for t in report["tasks"].values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--index", type=int, choices=(0, 1))
    mode.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    raise SystemExit(analyze(args) if args.analyze else run(args))
