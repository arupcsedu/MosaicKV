#!/usr/bin/env python3
"""Same-input graph/arena controls -> paired PrefixKV gate -> all baselines."""

import argparse
import os
import subprocess
from pathlib import Path

import evicastkv_prompt_graph_latency as previous

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, driver = previous.SOURCE, previous.driver
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_graph_arena_latency.sbatch"
OWNED = Path("/scratch/djy8hg/runs/evicastkv/prompt-graph-owned-consumed-20261104-v1")
POLICY = dict(history_arena=True, captured_code_replay=True, row_major_codes=False)
ORIGINAL_BUILDER = previous.make_records
ORIGINAL_EXECUTE = previous.direct.execute


def make_records(parent, methods):
    result = ORIGINAL_BUILDER(parent, methods)
    for method, arms in result.items():
        arms["graph-original"] = arms["graph"]
        arms["graph"] = dict(arms["graph"], prompt_graph_policy=dict(POLICY))
        if method == "prefixkv_reimpl":
            arms["whole-graph-original"] = arms["whole-graph"]
            arms["whole-graph"] = dict(arms["whole-graph"], prompt_graph_policy=dict(POLICY))
    return result


def required(final):
    methods = previous.METHODS if final else ("mosaickv_exact", "prefixkv_reimpl")
    return {(m, a) for m in methods for a in ("direct", "graph-original", "graph")} | {
        ("prefixkv_reimpl", a) for a in ("whole-6291456", "whole-graph-original", "whole-graph")
    }


def execute(adapter, refs, messages, profile, plan, method, record, **kwargs):
    measured = ORIGINAL_EXECUTE(adapter, refs, messages, profile, plan, method, record, **kwargs)
    if record.get("prompt_graph_policy") is not None:
        info = measured["prompt_graph_reader"]
        if (
            record["prompt_graph_policy"] != POLICY
            or info["optimized_policy"] != POLICY
            or info["history_borrows"] < 1
            or info["history_borrow_misses"] != 0
            or info["history_borrow_maximum_bytes"] < 1
            or info["row4_workspace_peak_bytes"] != 0
            or info["row4_replays"] != 0
            or not info["retired"]
        ):
            raise RuntimeError("complete charged graph/arena policy engagement required")
        if method in plan.get("code_delta_methods", []) and (
            info["captured_code_replays"] < 1 or info["captured_code_kernel_dispatches"] < 1
        ):
            raise RuntimeError("ordered code replay must genuinely execute in captured kernels")
    return measured


def summarize(rows, seed, *, final=False):
    import numpy as np

    expected = {(m, a, t) for m, a in required(final) for t in range(-5, 20)}
    keys = [(r["method"], r["arm"], r["trial"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("all original AND optimized readers, five warmups/twenty pairs required")
    arrays, summaries = {}, []
    for m, a in sorted(required(final)):
        selected = sorted(
            (r for r in rows if (r["method"], r["arm"]) == (m, a)), key=lambda r: r["trial"]
        )
        for r in selected:
            measured = r["measurement"]
            if (
                r["warmup"] is not (r["trial"] < 0)
                or measured["logits_bitwise"] is not True
                or measured["no_FullKV_fallback"] is not True
            ):
                raise ValueError(
                    "bitwise parent logits, exact warmup identity, no fallback required"
                )
            value = measured["total_wall_seconds"]
            if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
                raise ValueError("finite positive nonboolean full-request latency required")
        values = np.array([r["measurement"]["total_wall_seconds"] for r in selected[5:]])
        arrays[m, a] = values
        summaries.append(
            dict(
                method=m,
                arm=a,
                mean=float(values.mean()),
                p50=float(np.median(values)),
                p95=float(np.quantile(values, 0.95)),
            )
        )
    draws = np.random.default_rng(seed).integers(0, 20, (10000, 20))
    candidate, comparisons = arrays["mosaickv_exact", "graph"], []
    tail = 0.05 / (2 * 64)
    for (m, a), values in arrays.items():
        if (m, a) == ("mosaickv_exact", "graph"):
            continue
        ratios = candidate[draws].mean(1) / values[draws].mean(1)
        ci = np.quantile(ratios, [tail, 1 - tail]).tolist()
        comparisons.append(
            dict(
                method=m,
                arm=a,
                mean_ratio=float(candidate.mean() / values.mean()),
                adjusted_ratio_CI=ci,
                equal_or_faster=bool(ci[1] <= 1),
                four_percent_faster=bool(ci[1] <= 0.96),
            )
        )
    baselines = [r for r in comparisons if r["method"] != "mosaickv_exact"]
    return dict(
        summaries=summaries,
        comparisons=comparisons,
        four_percent_latency_gate=all(r["four_percent_faster"] for r in baselines),
        equal_or_faster_gate=all(r["equal_or_faster"] for r in baselines),
        acceptance_ratio_upper_bound=0.96,
        bootstrap_replicates=10000,
        family_maximum_engineering_contrasts=64,
        intervals_over_repetitions_not_independent_examples=True,
    )


def submit(root, action, *, dependency=None):
    plan = driver.checked(root, sha(root / "plan.json"))
    gpu = action in ("study", "timing", "all_baselines")
    args = ["sbatch", "--parsable", "--account=bii_dsc_community", "--nodes=1", "--ntasks=1"]
    if gpu:
        driver.cpu_proof(root, plan)
        args += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--array=0-1%2",
            "--time="
            + {"study": "00:35:00", "timing": "01:15:00", "all_baselines": "01:55:00"}[action],
        ]
    else:
        args += ["--partition=bii,standard", "--cpus-per-task=2", "--mem=16G", "--time=00:25:00"]
    if dependency:
        args.append(f"--dependency=afterany:{dependency}")
    (root / "slurm").mkdir(exist_ok=True)
    args += [
        f"--job-name=mk-garena-{action}",
        f"--output={root}/slurm/{action}-%A_%a.out",
        f"--error={root}/slurm/{action}-%A_%a.err",
        str(SCRIPT),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        action,
    ]
    job = subprocess.check_output(args, text=True).strip()
    if gpu:
        from control_a100_schedule import tick

        policy = dict(
            timezone="America/New_York",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[job],
        )
        write_once(root / "scheduler/policy.json", policy)
        write_once(
            root / "scheduler/registration.json",
            tick(policy, {}, root / "scheduler/state.json", False),
        )
    return job


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new root and clean immutable source required")
    old = driver.read(OWNED / "plan.json")
    bound = dict(old["bound_files"])
    for path in (
        OWNED / "plan.json",
        OWNED / "analysis/decision.json",
        SOURCE / "mosaickv/docs/graph_arena_latency_20261104.md",
    ):
        bound[str(path)] = sha(path)
    for index in (0, 1):
        path = OWNED / f"workers/{index}/study.json"
        bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_captured_code_and_verification_arena",
            prompt_graph_root=str(root),
            optimized_graph_policy=POLICY,
            automatic_quality_submission=False,
            paper_campaign_authorized=False,
            registered_request_ratio_upper=0.96,
            acceptance_ratio_upper_bound=0.96,
        ),
    )
    previous.stage(root, "cpu")


def install():
    previous.make_records, previous.summarize, previous.submit = make_records, summarize, submit
    previous.direct.execute = execute


def study(root, expected, index):
    from mosaickv.evaluation import captured_code_gpu_control, prompt_graph_reader

    ordinary = prompt_graph_reader.gpu_controls

    def controls():
        return dict(
            ordinary_prompt_graph=ordinary(), captured_codes=captured_code_gpu_control.run()
        )

    prompt_graph_reader.gpu_controls = controls
    try:
        previous.study(root, expected, index)
    finally:
        prompt_graph_reader.gpu_controls = ordinary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "cpu",
            "cpu_analysis",
            "study",
            "study_analysis",
            "timing",
            "timing_analysis",
            "all_baselines",
            "all_baselines_analysis",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    install()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "cpu":
        driver.pair.common.previous.cpu(args.root, args.plan_sha)
    elif args.action.endswith("_analysis"):
        previous.analyze(args.root, args.plan_sha, args.action)
    elif args.action == "study":
        study(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        previous.timing(
            args.root,
            args.plan_sha,
            int(os.environ["SLURM_ARRAY_TASK_ID"]),
            all_methods=args.action == "all_baselines",
        )


if __name__ == "__main__":
    main()
