#!/usr/bin/env python3
"""Cheap consumed-input screen before conditional formal latency validation."""

import argparse
import json
import os
import random
import subprocess
import traceback
from pathlib import Path

import evicastkv_lt_latency as common

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, driver = common.SOURCE, common.driver
PARENT = Path("/scratch/djy8hg/runs/evicastkv/lt-reader-consumed-20261104-v1")
PROFILE = Path("/scratch/djy8hg/runs/evicastkv/lt-model-profile-consumed-20261104-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_lt_direct_constructor.sbatch"
CANDIDATE = ("mosaickv_exact", "graph-lt-direct")
ORIGINAL_RECORDS, ORIGINAL_REQUIRED = common.make_records, common.required
ORIGINAL_EXECUTE, ORIGINAL_SUBMIT = common.execute, common.submit
SCREEN_ARMS = (
    ("mosaickv_exact", "graph"),
    ("mosaickv_exact", "graph-lt"),
    CANDIDATE,
    ("prefixkv_reimpl", "graph"),
    ("prefixkv_reimpl", "graph-original"),
)


def make_records(parent, methods):
    result = ORIGINAL_RECORDS(parent, methods)
    if "mosaickv_exact" in result:
        result["mosaickv_exact"]["graph-lt-direct"] = dict(
            result["mosaickv_exact"]["graph-lt"], lt_fit_policy="direct"
        )
    return result


def required(final):
    previous = ORIGINAL_REQUIRED(final)
    if final:
        previous = {x for x in previous if x[0] != "mosaickv_exact"}
    return previous | {CANDIDATE}


def execute(adapter, refs, messages, profile, plan, method, record, **kwargs):
    value = ORIGINAL_EXECUTE(adapter, refs, messages, profile, plan, method, record, **kwargs)
    if record.get("lt_fit_policy") == "direct":
        proof = value["lt_source_conversion"]
        if (
            proof["constructor_policy"] != "direct"
            or not proof["full_parent_KV_word_validation"]
            or proof["exhaustive_stride_audit"] is not kwargs.get("audit", False)
        ):
            raise RuntimeError("actual direct word validation and audited stride controls required")
    return value


def submit(root, action, *, dependency=None):
    if action not in ("screen", "screen_analysis"):
        return ORIGINAL_SUBMIT(root, action, dependency=dependency)
    plan = driver.checked(root, sha(root / "plan.json"))
    gpu = action == "screen"
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
            "--time=00:20:00",
        ]
    else:
        args += ["--partition=bii", "--cpus-per-task=2", "--mem=16G", "--time=00:25:00"]
    if dependency:
        args.append("--dependency=afterany:" + dependency)
    (root / "slurm").mkdir(exist_ok=True)
    args += [
        f"--job-name=mk-ltfix-{action}",
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


def install():
    common.make_records, common.required = make_records, required
    common.execute, common.submit = execute, submit
    common.CANDIDATE, common.SCRIPT = CANDIDATE, SCRIPT


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("clean immutable source and new root required")
    negative = driver.read(PARENT / "paired-timing/analysis/decision.json")
    attribution = driver.read(PROFILE / "analysis/decision.json")
    if (
        not negative["complete"]
        or negative["four_percent_latency_gate"]
        or not attribution["complete"]
    ):
        raise ValueError(
            "preserved complete negative and complete actual-model attribution required"
        )
    old = driver.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    paths = [
        PARENT / "plan.json",
        PARENT / "analysis/decision.json",
        PARENT / "paired-timing/analysis/decision.json",
        PROFILE / "analysis/decision.json",
        SOURCE / "mosaickv/docs/lt_direct_constructor_20261105.md",
    ]
    paths += [PARENT / f"workers/{i}/study.json" for i in (0, 1)]
    bound.update({str(p): sha(p) for p in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="direct_exact_constructor_screen",
            prompt_graph_root=str(root / "formal"),
            candidate_constructor="direct",
            screen_warmups=5,
            screen_repetitions=3,
            screen_all_paired_ratio_threshold=0.94,
            screen_is_NOT_acceptance=True,
            automatic_quality_submission=False,
            paper_campaign_authorized=False,
            independent_quality_N=0,
        ),
    )
    common.stage(root, "cpu")


def screen_summary(rows):
    import math
    import statistics

    keys = [(r["method"], r["arm"], r["trial"]) for r in rows]
    expected = {(m, a, t) for m, a in SCREEN_ARMS for t in range(-5, 3)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("complete unique five-warmup/three-repeat screen required")
    values = {}
    for r in rows:
        value = r["measurement"]["total_wall_seconds"]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value <= 0
            or r["warmup"] is not (r["trial"] < 0)
            or r["measurement"]["logits_bitwise"] is not True
            or r["measurement"]["no_FullKV_fallback"] is not True
        ):
            raise ValueError("finite full requests and exact parent trajectory required")
        if not r["warmup"]:
            values[r["method"], r["arm"], r["trial"]] = value
    ratios = [
        values[CANDIDATE[0], CANDIDATE[1], t] / values["prefixkv_reimpl", a, t]
        for a in ("graph", "graph-original")
        for t in range(3)
    ]
    return dict(
        screen_pass=max(ratios) <= 0.94,
        all_paired_ratios=ratios,
        preliminary_only_NOT_latency_acceptance=True,
        independent_quality_N=0,
        summaries=[
            dict(method=m, arm=a, mean=statistics.mean(values[m, a, t] for t in range(3)))
            for m, a in SCREEN_ARMS
        ],
    )


def screen(root, expected, index):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_direct_constructor_screen",
                methods=["mosaickv_exact", "prefixkv_reimpl"],
                actual_timing_protocol=dict(
                    output_predictions=128, warmups=5, repetitions=3, not_acceptance=True
                ),
            ),
        )
        records = driver.read(PARENT / f"workers/{index}/study.json")["records"]
        records["mosaickv_exact"]["graph-lt-direct"] = dict(
            records["mosaickv_exact"]["graph-lt"], lt_fit_policy="direct"
        )
        candidate = records[CANDIDATE[0]][CANDIDATE[1]]

        def verify(packets):
            actual = common.previous.direct.packet_signature(packets)["digest"]
            if actual != candidate["packet_sha256"]:
                raise RuntimeError("direct constructor changed original parent payload")

        audited = execute(
            adapter,
            refs,
            messages,
            profile,
            plan,
            CANDIDATE[0],
            candidate,
            audit=True,
            final_validator=verify,
        )
        if any(audited[k] != candidate["audited"][k] for k in driver.STORAGE[:2]):
            raise RuntimeError("constructor changed persistent backing")
        write_once(out / "audited_candidate.json", audited)
        records[CANDIDATE[0]][CANDIDATE[1]] = dict(candidate, audited=audited)
        rng = random.Random(plan["seed"])
        rows = []
        for trial in range(-5, 3):
            arms = list(SCREEN_ARMS)
            rng.shuffle(arms)
            for method, arm in arms:
                record = records[method][arm]
                measured = execute(adapter, refs, messages, profile, plan, method, record)
                if any(measured[k] != record["audited"][k] for k in driver.STORAGE[:2]):
                    raise RuntimeError(
                        "screen persistent bytes differ from bound audited companion"
                    )
                row = dict(
                    method=method,
                    arm=arm,
                    trial=trial,
                    warmup=trial < 0,
                    measurement=measured,
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                )
                with (out / "trials.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
        write_once(out / "screen.json", screen_summary(rows))
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def analyze(root, expected, action):
    plan = driver.checked(root, expected)
    failures, rows = [], []
    if action == "cpu_analysis":
        try:
            driver.cpu_proof(root, plan)
        except Exception as error:
            failures.append(str(error))
    else:
        rows, failures = driver.load_producers(root, plan, "screen.json")
    passed = not failures and len(rows) == 2 and all(r["screen_pass"] for r in rows)
    write_once(
        root / ("cpu-analysis" if action == "cpu_analysis" else "analysis") / "decision.json",
        dict(
            complete=not failures,
            failures=failures,
            rows=rows,
            screen_pass=passed,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            independent_quality_N=0,
            latency_or_paper_superiority_established=False,
            automatic_quality_submission=False,
        ),
    )
    if failures:
        return
    if action == "cpu_analysis":
        common.stage(root, "screen")
    elif passed:
        target = common.previous.direct.common.previous.child(root, "formal")
        common.stage(target, "study")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "cpu",
            "cpu_analysis",
            "screen",
            "screen_analysis",
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
    elif args.action in ("cpu_analysis", "screen_analysis"):
        analyze(args.root, args.plan_sha, args.action)
    elif args.action == "screen":
        screen(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif args.action.endswith("_analysis"):
        common.analyze(args.root, args.plan_sha, args.action)
    elif args.action == "study":
        common.study(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        common.timing(
            args.root,
            args.plan_sha,
            int(os.environ["SLURM_ARRAY_TASK_ID"]),
            final=args.action == "all_baselines",
        )


if __name__ == "__main__":
    main()
