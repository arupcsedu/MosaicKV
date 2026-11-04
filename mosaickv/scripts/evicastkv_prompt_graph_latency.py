#!/usr/bin/env python3
"""Finite CPU -> controls -> paired timing -> conditional all-baseline chain.

Same two consumed inputs only. Every method receives the same prompt-graph
option; original fast controls remain. Four-percent superiority is an adjusted
upper paired request-latency ratio <=0.96, not a best-trial or reader-only gain.
"""

import argparse
import json
import os
import random
import subprocess
import traceback
from pathlib import Path

import evicastkv_direct_text_latency as direct

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, driver = direct.SOURCE, direct.driver
PARENT = Path("/scratch/djy8hg/runs/evicastkv/predicate-span-consumed-20261104-v1")
DOC = SOURCE / "mosaickv/docs/prompt_graph_latency_20261104.md"
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_prompt_graph_latency.sbatch"
METHODS = tuple(direct.common.previous.ALL_METHODS)


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
            + {"study": "00:45:00", "timing": "01:00:00", "all_baselines": "01:55:00"}[action],
        ]
    else:
        args += ["--partition=bii,standard", "--cpus-per-task=2", "--mem=16G", "--time=00:15:00"]
    if dependency:
        args.append(f"--dependency=afterany:{dependency}")
    (root / "slurm").mkdir(exist_ok=True)
    args += [
        f"--job-name=mk-pgraph-{action}",
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


def stage(root, action):
    worker = submit(root, action)
    analysis = submit(root, action + "_analysis", dependency=worker)
    write_once(root / f"{action}_submission.json", dict(worker=worker, analysis=analysis))


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    parent = driver.read(PARENT / "paired-timing/all-baselines/analysis/decision.json")
    if not parent["complete"] or parent["all_input_latency_gate"]:
        raise ValueError("preserved complete all-baseline strict negative required")
    old = driver.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    for p in (
        PARENT / "plan.json",
        PARENT / "analysis/decision.json",
        PARENT / "paired-timing/all-baselines/analysis/decision.json",
        DOC,
    ):
        bound[str(p)] = sha(p)
    for i in (0, 1):
        p = PARENT / f"workers/{i}/study.json"
        bound[str(p)] = sha(p)
    failed = Path("/scratch/djy8hg/runs/evicastkv/prompt-graph-consumed-20261104-v1")
    attribution = Path("/scratch/djy8hg/runs/evicastkv/prompt-graph-allocation-20261104-v1")
    for p in (
        failed / "analysis/decision.json",
        failed / "plan.json",
        attribution / "diagnostic.py",
        attribution / "result.json",
    ):
        bound[str(p)] = sha(p)
    allocation = driver.read(attribution / "result.json")
    if any(
        r["captured"] - r["before"] != 1024 or r["after_del"] != r["before"]
        for r in allocation["records"]
    ):
        raise ValueError("positive separately measured generator-lifetime attribution required")
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            prompt_graph_root=str(root),
            stage="consumed_private_prompt_only_cuda_graph_reader",
            registered_request_ratio_upper=0.96,
            warmups=5,
            repetitions=20,
            bootstrap_replicates=10000,
            family_maximum_engineering_contrasts=64,
            no_fresh_identities=True,
            independent_quality_N=0,
            paper_campaign_authorized=False,
            automatic_quality_submission=False,
            main_six_mechanisms_disabled_precision_variant=True,
            parent_packet_words_unchanged=True,
            graph_private_generator_scalar_bytes_charged=1024,
            graph_unowned_tensor_bytes_allowed=0,
            whole_request_GPU_peak_certified=False,
        ),
    )
    stage(root, "cpu")


def make_records(parent, methods):
    result = {}
    for method in methods:
        old = parent["records"][method]
        result[method] = {
            "direct": dict(old["direct"], prompt_graph_reader=False),
            "graph": dict(old["direct"], prompt_graph_reader=True),
        }
        if method == "prefixkv_reimpl":
            result[method].update(
                {
                    "whole-6291456": dict(old["whole-6291456"], prompt_graph_reader=False),
                    "whole-graph": dict(old["whole-6291456"], prompt_graph_reader=True),
                }
            )
    return result


def study(root, expected, index):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        driver.checked(root, expected)
        from mosaickv.evaluation.prompt_graph_reader import gpu_controls

        write_once(out / "primitive.json", gpu_controls())
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_prompt_graph_parent_controls",
                methods=list(METHODS),
                actual_timing_protocol=dict(output_predictions=128, not_latency_evidence=True),
            ),
        )
        parent = driver.read(PARENT / f"workers/{index}/study.json")
        records = make_records(parent, METHODS)
        for method, arms in records.items():
            signatures = {}
            for arm, record in arms.items():
                graph = record["prompt_graph_reader"]
                family = "whole" if arm.startswith("whole") else "direct"

                def verify(
                    packets, family=family, graph=graph, signatures=signatures, method=method
                ):
                    actual = direct.packet_signature(packets)
                    if graph and actual != signatures[family]:
                        raise RuntimeError("graph policy changed frozen parent packet words")
                    if not graph:
                        signatures[family] = actual
                        if (
                            family == "direct"
                            and actual["digest"] != parent["records"][method]["packet_sha256"]
                        ):
                            raise RuntimeError("original representation differs from frozen parent")

                measured = direct.execute(
                    adapter,
                    refs,
                    messages,
                    profile,
                    plan,
                    method,
                    record,
                    audit=True,
                    final_validator=verify,
                )
                anchor = parent["records"][method][
                    "whole-6291456" if family == "whole" else "direct"
                ]["audited"]
                if any(measured[k] != anchor[k] for k in driver.STORAGE[:2]):
                    raise RuntimeError("prompt graphs changed initial or final persistent bytes")
                if graph:
                    info = measured["prompt_graph_reader"]
                    if (
                        info["captures"] != 64
                        or info["replays"] != 8128
                        or info["graph_private_tensor_bytes"] != 1024
                        or info["graph_private_live_bytes"] != 0
                        or info["unowned_tensor_bytes"] != 0
                        or not info["retired"]
                        or info["active_workspace_bytes"]
                    ):
                        raise RuntimeError("complete isolated, fully charged graph replay required")
                record["audited"] = measured
                record["packet_sha256"] = signatures[family]["digest"]
                write_once(out / f"control-{method}-{arm}.json", record)
        write_once(out / "study.json", dict(records=records, independent_quality_N=0))
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def timing(root, expected, index, *, all_methods=False):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        methods = METHODS if all_methods else ("mosaickv_exact", "prefixkv_reimpl")
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_prompt_graph_paired_full_requests",
                methods=list(methods),
                actual_timing_protocol=dict(
                    output_predictions=128,
                    warmups=5,
                    repetitions=20,
                    randomized_order=True,
                    capture_and_retirement_included=True,
                ),
            ),
        )
        original = driver.read(Path(plan["prompt_graph_root"]) / f"workers/{index}/study.json")
        arms = [(m, a, r) for m in methods for a, r in original["records"][m].items()]
        rng = random.Random(plan["seed"])
        for trial in range(-5, 20):
            placement = list(arms)
            rng.shuffle(placement)
            for method, arm, record in placement:
                measured = direct.execute(adapter, refs, messages, profile, plan, method, record)
                if any(measured[k] != record["audited"][k] for k in driver.STORAGE[:2]):
                    raise RuntimeError("timed bytes differ from audited graph companion")
                if record["prompt_graph_reader"] and (
                    measured["prompt_graph_reader"]["replays"] != 8128
                    or not measured["prompt_graph_reader"]["retired"]
                ):
                    raise RuntimeError("timed prompt graph branch not completely exercised")
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
        write_once(out / "timing.json", dict(complete=True, independent_quality_N=0))
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def summarize(rows, seed, *, final=False):
    import numpy as np

    methods = METHODS if final else ("mosaickv_exact", "prefixkv_reimpl")
    required = {(m, a) for m in methods for a in ("direct", "graph")} | {
        ("prefixkv_reimpl", "whole-6291456"),
        ("prefixkv_reimpl", "whole-graph"),
    }
    keys = [(x["method"], x["arm"], x["trial"]) for x in rows]
    if len(keys) != len(set(keys)) or set(keys) != {
        (m, a, t) for m, a in required for t in range(-5, 20)
    }:
        raise ValueError("complete unique five-warmup/twenty-repeat registered graph arms required")
    for x in rows:
        if (
            x["warmup"] is not (x["trial"] < 0)
            or x["measurement"]["logits_bitwise"] is not True
            or x["measurement"]["no_FullKV_fallback"] is not True
        ):
            raise ValueError("warmup identity, parent-logit parity and no fallback required")
    arrays, summaries = {}, []
    for m, a in sorted(required):
        selected = sorted(
            (x for x in rows if x["method"] == m and x["arm"] == a), key=lambda x: x["trial"]
        )
        raw = [x["measurement"]["total_wall_seconds"] for x in selected]
        if any(type(v) not in (int, float) or not np.isfinite(v) or v <= 0 for v in raw):
            raise ValueError("finite positive nonboolean timings including warmups required")
        v = np.asarray(raw[5:])
        arrays[m, a] = v
        summaries.append(
            dict(
                method=m,
                arm=a,
                mean=float(v.mean()),
                p50=float(np.median(v)),
                p95=float(np.quantile(v, 0.95)),
            )
        )
    draws = np.random.default_rng(seed).integers(0, 20, (10000, 20))
    candidate = arrays["mosaickv_exact", "graph"]
    comparisons = []
    tail = 0.05 / (2 * 64)
    for (m, a), v in arrays.items():
        if (m, a) == ("mosaickv_exact", "graph"):
            continue
        ratios = candidate[draws].mean(1) / v[draws].mean(1)
        ci = np.quantile(ratios, [tail, 1 - tail]).tolist()
        comparisons.append(
            dict(
                method=m,
                arm=a,
                mean_ratio=float(candidate.mean() / v.mean()),
                adjusted_ratio_CI=ci,
                equal_or_faster=bool(ci[1] <= 1),
                four_percent_faster=bool(ci[1] <= 0.96),
            )
        )
    baselines = [x for x in comparisons if x["method"] != "mosaickv_exact"]
    return dict(
        summaries=summaries,
        comparisons=comparisons,
        four_percent_latency_gate=all(x["four_percent_faster"] for x in baselines),
        equal_or_faster_gate=all(x["equal_or_faster"] for x in baselines),
        acceptance_ratio_upper_bound=0.96,
        bootstrap_replicates=10000,
        family_maximum_engineering_contrasts=64,
        intervals_over_repetitions_not_independent_examples=True,
    )


def analyze(root, expected, action):
    plan = driver.checked(root, expected)
    failures, records, timings, memory = [], [], [], []
    if action == "cpu_analysis":
        try:
            driver.cpu_proof(root, plan)
        except Exception as error:
            failures.append(str(error))
    else:
        records, failures = driver.load_producers(
            root, plan, "study.json" if action == "study_analysis" else "timing.json"
        )
        if not failures and action == "study_analysis":
            for i, item in enumerate(records):
                candidate = item["records"]["mosaickv_exact"]["graph"]["audited"]
                baselines = item["records"]["prefixkv_reimpl"]
                passed_memory = all(
                    candidate[k] < r["audited"][k]
                    for r in baselines.values()
                    for k in driver.STORAGE
                )
                memory.append(
                    dict(
                        input_index=i,
                        smaller_than_all_PrefixKV_storage_controls=passed_memory,
                        candidate={k: candidate[k] for k in driver.STORAGE},
                    )
                )
                if not passed_memory:
                    failures.append(f"input {i}: fixed workspace erased PrefixKV storage advantage")
        if not failures and action != "study_analysis":
            for i in (0, 1):
                try:
                    rows = [
                        json.loads(x)
                        for x in (root / f"workers/{i}/trials.jsonl").read_text().splitlines()
                    ]
                    if any(
                        x["source_sha"] != plan["source_sha"] or x["plan_sha256"] != expected
                        for x in rows
                    ):
                        raise ValueError("raw trial provenance mismatch")
                    timings.append(
                        summarize(rows, plan["seed"], final=action == "all_baselines_analysis")
                    )
                except Exception as error:
                    failures.append(f"input {i}: {type(error).__name__}: {error}")
    passed = (
        not failures and len(timings) == 2 and all(x["four_percent_latency_gate"] for x in timings)
    )
    decision = dict(
        complete=not failures,
        failures=failures,
        timings=timings,
        four_percent_latency_gate=passed,
        acceptance_ratio_upper_bound=0.96,
        execution_source_sha=plan["source_sha"],
        plan_sha256=expected,
        independent_quality_N=0,
        no_reader_promoted=True,
        paper_superiority_established=False,
        whole_request_GPU_peak_certified=False,
        not_byte_matched_algorithm_evidence=True,
        storage_controls=memory,
    )
    target = root / ("cpu-analysis" if action == "cpu_analysis" else "analysis")
    write_once(target / "decision.json", decision)
    if failures:
        return
    if action == "cpu_analysis":
        stage(root, "study")
    elif action == "study_analysis":
        stage(direct.common.previous.child(root, "paired-timing"), "timing")
    elif action == "timing_analysis" and passed:
        stage(direct.common.previous.child(root, "all-baselines"), "all_baselines")


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
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "cpu":
        driver.pair.common.previous.cpu(args.root, args.plan_sha)
    elif args.action.endswith("_analysis"):
        analyze(args.root, args.plan_sha, args.action)
    elif args.action == "study":
        study(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        timing(
            args.root,
            args.plan_sha,
            int(os.environ["SLURM_ARRAY_TASK_ID"]),
            all_methods=args.action == "all_baselines",
        )


if __name__ == "__main__":
    main()
