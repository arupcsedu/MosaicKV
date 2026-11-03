#!/usr/bin/env python3
"""Finite event-driven latency engineering; no new quality cohort or paper run."""

import argparse
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import evicastkv_paired_request_readers as pair

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = pair.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/prompt-source-subphases-consumed-20261102-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_source_hotpath_followup.sbatch"
DOC = SOURCE / "mosaickv/docs/source_hotpath_followup_20261102.md"


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("clean immutable source and new output root required")
    decision = pair.common.previous.read(PARENT / "analysis/decision.json")
    if (
        not decision["complete"]
        or decision["failures"]
        or decision["next_target"] != "projection_dispatch"
        or any(
            decision["mean_exclusive_host_seconds"][str(i)]["projection_dispatch"] < 0.5
            or decision["mean_exclusive_host_seconds"][str(i)]["indexing_views"] < 0.5
            for i in (0, 1)
        )
    ):
        raise ValueError("complete measured source dispatch/view opportunity required")
    old = pair.common.previous.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    for p in (PARENT / "plan.json", PARENT / "analysis/decision.json", DOC):
        bound[str(p)] = sha(p)
    for i in (0, 1):
        folder = PARENT / f"workers/{i}"
        result = pair.common.previous.read(folder / "result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("both operation-profile workers must be complete")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, value in result["outputs"].items():
            if sha(folder / name) != value:
                raise ValueError("changed measured phase evidence")
            bound[str(folder / name)] = value
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            paired_reader_stage="source_hotpath",
            reader_modes=["anchor_hotpath", "candidate_hotpath"],
            anchor_hotpath="original",
            candidate_hotpath="prepared",
            hotpath_attempt=0,
            maximum_hotpath_attempts=2,
            maximum_worker_minutes=60,
            maximum_simultaneous_workers=2,
            chain_root=str(root),
            study_root=str(root),
            selected_reader="carry_literals",
            base_dispatch=False,
            reuse_code_descriptors=False,
            source_hotpath="original",
            paired_common_streaming_timing=False,
            stage="consumed_prepared_source_launch_pair",
            fresh_materialization_authorized=False,
            paper_campaign_authorized=False,
            independent_quality_N=0,
        ),
    )
    pair.common.previous.cpu(root, sha(root / "plan.json"))
    precheck(root)


def cpu_proof(root, plan):
    proof = pair.common.previous.read(root / "cpu.json")
    if (
        proof["source_sha"] != plan["source_sha"]
        or proof["returncode"]
        or proof["failed_or_skipped"]
        or proof["test_count"] < 3613
        or proof["xml_sha256"] != sha(root / "cpu.xml")
    ):
        raise ValueError("complete same-clean-source CPU suite required")


def precheck(root):
    expected = sha(root / "plan.json")
    plan = pair.common.previous.common.inputs(root, expected)
    cpu_proof(root, plan)
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%b"], text=True
    )
    if any("gpu" in row.lower() for row in queue.splitlines()):
        raise ValueError("preceding owned GPU work must finish before the bounded follow-up")
    (root / "slurm").mkdir(exist_ok=True)
    args = [str(SOURCE), str(root), expected]
    gpu = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii-gpu",
            "--account=bii_dsc_community",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=12",
            "--mem=16G",
            "--time=00:10:00",
            "--job-name=mk-source-hot-precheck",
            f"--output={root}/slurm/precheck-%j.out",
            f"--error={root}/slurm/precheck-%j.err",
            str(SCRIPT),
            *args,
            "precheck",
        ],
        text=True,
    ).strip()
    begin = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:10:00",
            f"--dependency=afterany:{gpu}",
            "--job-name=mk-source-hot-begin",
            f"--output={root}/slurm/begin-%j.out",
            f"--error={root}/slurm/begin-%j.err",
            str(SCRIPT),
            *args,
            "begin",
        ],
        text=True,
    ).strip()
    write_once(root / "precheck_submission.json", dict(gpu_job=gpu, cpu_begin=begin))


def submit(root, *, final=False):
    from control_a100_schedule import tick

    expected = sha(root / "plan.json")
    plan = pair.common.previous.common.inputs(root, expected)
    cpu_proof(root, plan)
    if plan["maximum_worker_minutes"] != 60 or len(plan["jobs"]) != 2:
        raise ValueError("two bounded consumed-input workers only")
    (root / "slurm").mkdir(exist_ok=True)
    args = [str(SOURCE), str(root), expected]
    gpu = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii-gpu",
            "--account=bii_dsc_community",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--time=01:00:00",
            "--array=0-1%2",
            "--job-name=mk-source-hot-final" if final else "--job-name=mk-source-hot-pair",
            f"--output={root}/slurm/gpu-%A_%a.out",
            f"--error={root}/slurm/gpu-%A_%a.err",
            str(SCRIPT),
            *args,
            "final_worker" if final else "worker",
        ],
        text=True,
    ).strip()
    analysis = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:10:00",
            f"--dependency=afterany:{gpu}",
            "--job-name=mk-source-hot-analysis",
            f"--output={root}/slurm/analysis-%j.out",
            f"--error={root}/slurm/analysis-%j.err",
            str(SCRIPT),
            *args,
            "final_analysis" if final else "analyze",
        ],
        text=True,
    ).strip()
    policy = dict(
        timezone="America/New_York",
        maximum_worker_minutes=60,
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
        arrays=[gpu],
    )
    write_once(root / "scheduler/policy.json", policy)
    write_once(
        root / "scheduler/registration.json", tick(policy, {}, root / "scheduler/state.json", False)
    )
    write_once(root / "submission.json", dict(gpu_array=gpu, cpu_analysis=analysis))


def improvement(summaries, failures):
    return (
        not failures
        and len(summaries) == 2
        and all(
            x["within_method"]["mosaickv_exact"]["familywise_ratio_bounds"][1] <= 0.99
            for x in summaries
        )
    )


def begin(root, expected):
    with patch.object(pair, "submit", submit):
        pair.begin(root, expected)
    blocked = root / "precheck/blocked.json"
    if blocked.exists():
        plan = pair.common.previous.common.inputs(root, expected)
        reason = pair.common.previous.read(blocked)["error"]
        decision = dict(
            complete=False,
            failures=[reason],
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            summaries=[],
            independent_quality_N=0,
            paper_superiority_established=False,
            fresh_or_large_campaign_submitted=False,
        )
        write_once(root / "analysis/decision.json", decision)
        closeout(root, decision)


def analyze(root, expected):
    plan = pair.common.previous.common.inputs(root, expected)
    summaries, profiles, failures = pair.read_producers(root, plan, expected)
    passed = improvement(summaries, failures)
    selected = plan["candidate_hotpath"] if passed else plan["anchor_hotpath"]
    decision = dict(
        complete=not failures,
        failures=failures,
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        summaries=summaries,
        profiles=profiles,
        hotpath_end_to_end_improvement_gate=passed,
        selected_hotpath=selected,
        independent_quality_N=0,
        paper_superiority_established=False,
        fresh_or_large_campaign_submitted=False,
    )
    write_once(root / "analysis/decision.json", decision)
    if failures:
        closeout(root, decision)
        return
    if plan["hotpath_attempt"] == 0:
        # A separate measured near-dominant opportunity, not a budget change.
        target = root / "restore"
        bound = dict(plan["bound_files"])
        for path in (root / "plan.json", root / "analysis/decision.json"):
            bound[str(path)] = sha(path)
        write_once(
            target / "plan.json",
            dict(
                plan,
                bound_files=bound,
                hotpath_attempt=1,
                anchor_hotpath=selected,
                candidate_hotpath="prepared_restore" if selected == "prepared" else "restore",
                study_root=str(target),
                stage="consumed_fused_sparse_restore_pair",
            ),
        )
        pair.reuse_cpu(root, target)
        precheck(target)
    elif plan["hotpath_attempt"] == 1:
        if selected == "original":
            closeout(root, decision)
        else:
            final_timing(root, plan, selected)
    else:
        raise ValueError("at most two registered source hotpath attempts")


def final_timing(root, plan, selected):
    target = Path(plan["chain_root"]) / "timing"
    bound = dict(plan["bound_files"])
    companions = {}
    boundaries = {}
    for path in (root / "plan.json", root / "analysis/decision.json"):
        bound[str(path)] = sha(path)
    for i in (0, 1):
        for name in ("audited_storage_controls.json", "all_method_controls.json", "result.json"):
            path = root / f"workers/{i}/{name}"
            bound[str(path)] = sha(path)
        path = root / f"workers/{i}/audited_storage_controls.json"
        companions[str(i)] = dict(path=str(path), sha256=sha(path))
        controls = pair.common.previous.read(root / f"workers/{i}/all_method_controls.json")["rows"]
        boundaries[str(i)] = {x["method"]: x["storage_boundaries"] for x in controls}
    write_once(
        target / "plan.json",
        dict(
            plan,
            bound_files=bound,
            paired_common_streaming_timing=True,
            reconstruction_readers="carry_literals",
            reuse_kernel_dispatch=False,
            reuse_code_descriptors=False,
            source_hotpath=selected,
            audited_storage_companions=companions,
            expected_storage_boundaries=boundaries,
            stage="consumed_all_five_hotpath_request_verification",
        ),
    )
    pair.reuse_cpu(root, target)
    submit(target, final=True)


def closeout(root, decision):
    plan = pair.common.previous.read(root / "plan.json")
    terminal = Path(plan["chain_root"])
    same = decision.get("all_method_similar_time_gate", False)
    no_slower = bool(
        same
        and all(
            p["familywise_eight_comparison_ratio_bounds"][1] <= 1.0
            for s in decision.get("summaries", [])
            for p in s.get("pairs", {}).values()
        )
    )
    write_once(
        terminal / "engineering_closeout.json",
        dict(
            complete=decision["complete"],
            failures=decision["failures"],
            source_sha=plan["source_sha"],
            final_root=str(root),
            final_decision_sha256=sha(root / "analysis/decision.json"),
            selected_hotpath=decision.get(
                "selected_hotpath", plan.get("source_hotpath", "original")
            ),
            all_method_similar_time_gate=same,
            all_method_no_slower_gate=no_slower,
            independent_quality_N=0,
            paper_superiority_established=False,
            fresh_or_large_campaign_submitted=False,
            whole_request_GPU_peak_certified=False,
        ),
    )
    lines = [
        "# Bounded source-hotpath engineering closeout",
        "",
        "Two consumed inputs only. No independent quality or byte-matched algorithm claim.",
        f"Complete: {decision['complete']}; failures: {decision['failures']}.",
        f"Similar-time gate: {same}; strict no-slower gate: {no_slower}.",
        "",
    ]
    for i, s in enumerate(decision.get("summaries", [])):
        lines.extend(
            [
                f"## Consumed input {i}",
                "",
                "| Method | Mean full-request seconds | p50 | p95 |",
                "|---|---:|---:|---:|",
            ]
        )
        distributions = s.get("distributions", [])
        if isinstance(distributions, dict):
            distributions = [dict(x, method=method) for method, x in distributions.items()]
        for x in distributions:
            lines.append(
                f"| {x['method']} {x.get('reader', '')} | {x['mean']:.6f} "
                f"| {x['p50']:.6f} | {x['p95']:.6f} |"
            )
    with (terminal / "engineering_report.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=[
            "prepare",
            "precheck",
            "begin",
            "worker",
            "analyze",
            "final_worker",
            "final_analysis",
        ],
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source-sha")
    p.add_argument("--plan-sha")
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.root, a.source_sha)
    elif a.action == "precheck":
        pair.precheck(a.root, a.plan_sha)
    elif a.action == "begin":
        begin(a.root, a.plan_sha)
    elif a.action == "worker":
        pair.worker(a.root, a.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif a.action == "analyze":
        analyze(a.root, a.plan_sha)
    elif a.action == "final_worker":
        pair.common.worker(a.root, a.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        pair.common.analyze(a.root, a.plan_sha)
        closeout(a.root, pair.common.previous.read(a.root / "analysis/decision.json"))
