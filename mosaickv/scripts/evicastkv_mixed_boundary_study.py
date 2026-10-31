#!/usr/bin/env python3
"""Common mixed-boundary codec for all selectors; consumed development only."""

import argparse
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import evicastkv_request_validation as previous
import evicastkv_text_row_study as study

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
RUNS = Path("/scratch/djy8hg/runs/evicastkv")
PARENT = RUNS / "draft-task-study-consumed-20261031-v1"
DIAGNOSTIC = RUNS / "isolated-draft-evidence-consumed-20261031-v1"
SETTINGS = {
    "balanced_refit2": ("head3k2v_refit", 0),
    "mixed_refit2_boundary8": ("head3k2v_refit_boundary8", 0),
}


def construct(*args, policy, **kwargs):
    from mosaickv.evaluation.head_local_precision import bundle

    if args[-1] != "query3" or policy not in (*SETTINGS, "union3"):
        raise ValueError("registered forecast/allocation policy required")
    codec, steps = SETTINGS.get(policy, ("union3", 0))
    allocation = (
        "profile"
        if policy == "union3" or args[5] not in ("mosaickv_exact", "prompt_attention_topk")
        else "balanced"
    )
    formats, proof = bundle(
        *args[:-1], policies=(codec,), allocation_mode=allocation, future_steps=steps, **kwargs
    )
    return formats[codec], dict(
        proof,
        precision=policy,
        allocation_mode=allocation,
        future_steps=steps,
        generated_history="lossless_consolidated_source_append_only",
        explicit_common_mixed_codec_variant_not_graph_exact_paper_method=True,
    )


def worker(root, expected, index):
    with patch.object(study, "construct", construct):
        study.worker(root, expected, index)


def prepare(root):
    if root.exists():
        raise ValueError("new immutable output root required")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "5df1fb2", "HEAD", "--", "mosaickv/src"],
        cwd=SOURCE,
        text=True,
    ).splitlines()
    if changed != ["mosaickv/src/mosaickv/evaluation/head_local_precision.py"]:
        raise ValueError("only the explicit common codec composition may change")
    old = previous.read(PARENT / "plan.json")
    gate = previous.read(PARENT / "analysis/GATE.json")
    if not gate["complete"] or gate["failures"]:
        raise ValueError("complete consumed baseline evidence required")
    chosen = old["selected_inputs"]
    if len(chosen) != 16 or len(set(chosen)) != 16:
        raise ValueError("only sixteen already consumed inputs permitted")
    bound = dict(old["bound_files"])
    for path in (
        PARENT / "plan.json",
        PARENT / "analysis/GATE.json",
        PARENT / "analysis/rows.json",
    ):
        bound[str(path)] = sha(path)
    for job in old["jobs"]:
        folder = PARENT / f"workers/{job['index']}"
        result = previous.read(folder / "result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("reused producer incomplete")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("reused producer changed")
            bound[str(folder / name)] = digest
    rows = previous.read(PARENT / "analysis/rows.json")["rows"]
    if len(rows) != 656:
        raise ValueError("all 41 previous measured points required, including stronger baselines")
    write_once(root / "reused_rows.json", dict(rows=rows))
    bound[str(root / "reused_rows.json")] = sha(root / "reused_rows.json")
    protocol = SOURCE / "mosaickv/docs/mixed_boundary_study_20261031.md"
    bound[str(protocol)] = sha(protocol)
    mixed = "mixed_refit2_boundary8"
    arms = [(m, mixed) for m in ("lookm_reimpl", "vl_cache_reimpl", "prefixkv_reimpl")]
    arms += [(m, p) for m in ("prompt_attention_topk", "mosaickv_exact") for p in SETTINGS]
    work = [(("mosaickv_exact", mixed), 0)] + [
        (a, s) for a in arms for s in range(0, 16, 4) if (a, s) != (("mosaickv_exact", mixed), 0)
    ]
    jobs = [
        dict(index=i, inputs=chosen[start : start + 4], method_arms=[list(arm)])
        for i, (arm, start) in enumerate(work)
    ]
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            jobs=jobs,
            arms=[],
            precision_policies=tuple(SETTINGS),
            reused_rows=str(root / "reused_rows.json"),
            warmups=1,
            timed_repetitions=1,
            new_expected_rows=112,
            total_expected_rows=768,
            stage="consumed_common_mixed_boundary_codec_study",
            bootstrap_comparisons=94,
            no_fresh_or_paper_promotion=True,
            paper_campaign_authorized=False,
            independent_quality_N=0,
            no_untouched_confirmation_claim=True,
            constructor_frozen_before_new_codec_results=True,
            task_quality_not_a_32_token_identity_claim=True,
            explicit_per_policy_overrides={
                p: dict(
                    codec=codec,
                    future_steps=steps,
                    allocation_mode="balanced_for_mosaickv_and_topk_native_otherwise",
                    prompt_to_draft_weight=0.5 if steps else 0.0,
                )
                for p, (codec, steps) in SETTINGS.items()
            },
        ),
    )
    print(sha(root / "plan.json"))


def diagnostic_valid(root):
    decision = previous.read(root / "analysis/decision.json")
    if not decision["complete"] or decision["failures"]:
        return False
    rows = decision["rows"]
    if len(rows) != 6 or not all(r["original_source_unchanged"] for r in rows):
        return False
    for index in range(2):
        folder = root / f"workers/{index}"
        result = previous.read(folder / "result.json")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("diagnostic artifact changed")
        control = previous.read(folder / "clone_repack_control.json")
        if control["maximum_logit_difference"] > 0.001 or control["token_agreement"] != 1:
            return False
    # Lossy compression's registered errors remain measured/negative where
    # applicable. This permits only a consumed-image rate/distortion study, not
    # a claim of token identity or a fresh cohort after failed numerical guards.
    return all(
        r["fullkv_draft_control"] is None
        or (
            r["fullkv_draft_control"]["maximum_logit_difference"] <= 0.001
            and r["fullkv_draft_control"]["token_agreement"] == 1
        )
        for r in rows
    )


def submit(root, expected, *, remaining=False):
    plan = previous.common.inputs(root, expected)
    cpu = previous.read(root / "cpu.json")
    if (
        cpu["source_sha"] != plan["source_sha"]
        or cpu["returncode"]
        or cpu["failed_or_skipped"]
        or cpu["test_count"] < 2892
        or cpu["xml_sha256"] != sha(root / "cpu.xml")
    ):
        raise ValueError("complete CPU verification required")
    if not diagnostic_valid(DIAGNOSTIC):
        write_once(
            root / "blocked.json",
            dict(
                reason="FullKV draft/source/clone control failed", paper_campaign_authorized=False
            ),
        )
        return
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%b"], text=True
    )
    if any("gpu" in line.lower() for line in queue.splitlines()):
        raise ValueError("admit no extra GPU work until previous arrays finish")
    script = str(SOURCE / "mosaickv/slurm/evicastkv_mixed_boundary_study.sbatch")
    args = [str(SOURCE), str(root), expected]
    (root / "slurm").mkdir(exist_ok=True)
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
            "--time=00:30:00",
            "--array=1-27%6" if remaining else "--array=0",
            "--job-name=mk-mixed-codec",
            f"--output={root}/slurm/gpu-%A_%a.out",
            f"--error={root}/slurm/gpu-%A_%a.err",
            script,
            *args,
            "worker",
        ],
        text=True,
    ).strip()
    cpu_job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:20:00",
            "--job-name=mk-draft-score",
            f"--dependency=afterany:{gpu}",
            f"--output={root}/slurm/analysis-%j.out",
            f"--error={root}/slurm/analysis-%j.err",
            script,
            *args,
            "analyze" if remaining else "promote",
        ],
        text=True,
    ).strip()
    from control_a100_schedule import tick

    policy = dict(
        timezone="America/New_York",
        maximum_worker_minutes=30,
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
        arrays=[gpu],
    )
    folder = root / "scheduler" / ("remaining" if remaining else "canary")
    write_once(folder / "policy.json", policy)
    write_once(folder / "registration.json", tick(policy, {}, folder / "state.json", False))
    write_once(
        root / ("submission.json" if remaining else "canary_submission.json"),
        dict(gpu_array=gpu, cpu_analysis=cpu_job),
    )
    print(json.dumps(dict(gpu_array=gpu, cpu_analysis=cpu_job)))


def promote(root, expected):
    previous.common.inputs(root, expected)
    passed = False
    try:
        folder = root / "workers/0"
        result = previous.read(folder / "result.json")
        for name, digest in result["outputs"].items():
            if sha(folder / name) != digest:
                raise ValueError("canary artifact changed")
        rows = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
        scored = [r for r in rows if not r["warmup"]]
        passed = (
            result["status"] == "completed"
            and not result["failures"]
            and len(scored) == 4
            and len({r["sample_id"] for r in scored}) == 4
            and all(r["status"] == "completed" and not r["used_fullkv_fallback"] for r in scored)
            and previous.read(root / "union_controls/0.json")["bitwise_logit_hash_match"]
        )
    except (OSError, ValueError, KeyError):
        pass
    write_once(root / "canary_decision.json", dict(passed=passed, execution_controls_only=True))
    if passed:
        submit(root, expected, remaining=True)
    else:
        study.analyze(root, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "bootstrap", "cpu", "submit", "worker", "promote", "analyze")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "bootstrap":
        prepare(args.root)
        digest = sha(args.root / "plan.json")
        previous.cpu(args.root, digest)
        submit(args.root, digest)
    elif args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif args.mode == "analyze":
        study.analyze(args.root, args.plan_sha)
    else:
        globals()[args.mode](args.root, args.plan_sha)
