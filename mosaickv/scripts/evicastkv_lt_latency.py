#!/usr/bin/env python3
"""Finite consumed-input Lt controls, full-request timing and baseline follow-up."""

import argparse
import json
import os
import random
import subprocess
import traceback
from pathlib import Path

import evicastkv_graph_arena_latency as arena

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

previous, driver, SOURCE = arena.previous, arena.driver, arena.SOURCE
OPERATOR = Path("/scratch/djy8hg/runs/evicastkv/lt128-operator-20261104-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_lt_latency.sbatch"
CANDIDATE = ("mosaickv_exact", "graph-lt")


def make_records(parent, methods):
    records = arena.make_records(parent, methods)
    for method, arms in records.items():
        arms["graph-lt"] = dict(arms["graph"], lt_source_reader=True)
        if method == "prefixkv_reimpl":
            arms["whole-graph-lt"] = dict(arms["whole-graph"], lt_source_reader=True)
    return records


def required(final):
    methods = previous.METHODS if final else ("mosaickv_exact", "prefixkv_reimpl")
    result = {(m, a) for m in methods for a in ("direct", "graph-original", "graph", "graph-lt")}
    result |= {
        ("prefixkv_reimpl", a)
        for a in ("whole-6291456", "whole-graph-original", "whole-graph", "whole-graph-lt")
    }
    if final:
        result = {x for x in result if x[0] != "mosaickv_exact" or x == CANDIDATE}
    return result


def validate(measured, plan, method, record):
    arena.validate_engagement(measured, plan, method, record)
    if measured["logits_bitwise"] is not True or measured["no_FullKV_fallback"] is not True:
        raise RuntimeError("all 128 parent logits bitwise and no fallback required")
    if record["prompt_graph_reader"]:
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
            raise RuntimeError("fully owned and retired prompt graph required")
    if record.get("lt_source_reader", False):
        conversion, info = measured["lt_source_conversion"], measured["prompt_graph_reader"]
        count = conversion["eligible_layers"]
        if (
            count not in (0, 32)
            or not conversion["parent_KV_bitwise"]
            or not conversion["no_labels_or_answers_used"]
            or conversion["added_patch_bytes"] > 2**20
        ):
            raise RuntimeError("complete bounded native-source conversion required")
        if method == "mosaickv_exact" and count != 32:
            raise RuntimeError("EviCastKV must genuinely convert every source layer")
        if count:
            if (
                info["lt_workspace_peak_bytes"] != 2**21
                or info["lt_replays"] != 8128
                or info["lt_matmul_dispatches"] != 40640
                or info["lt_engine"]["algorithm_workspace_bytes"] != 0
            ):
                raise RuntimeError("genuine bounded native GEMM graph replay required")
        elif info["lt_workspace_peak_bytes"] or info["lt_replays"]:
            raise RuntimeError("ineligible baseline must remain a truthful unchanged no-op")


def execute(adapter, refs, messages, profile, plan, method, record, **kwargs):
    measured = previous.direct.execute(
        adapter, refs, messages, profile, plan, method, record, **kwargs
    )
    validate(measured, plan, method, record)
    return measured


def summarize(rows, seed, *, final=False):
    import numpy as np

    expected = {(m, a, t) for m, a in required(final) for t in range(-5, 20)}
    keys = [(r["method"], r["arm"], r["trial"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError(
            "all original and optimized readers and 25 complete paired trials required"
        )
    arrays, summaries = {}, []
    for m, a in sorted(required(final)):
        selected = sorted(
            (r for r in rows if (r["method"], r["arm"]) == (m, a)), key=lambda r: r["trial"]
        )
        raw = []
        for r in selected:
            value = r["measurement"]["total_wall_seconds"]
            if (
                r["warmup"] is not (r["trial"] < 0)
                or r["measurement"]["logits_bitwise"] is not True
                or r["measurement"]["no_FullKV_fallback"] is not True
                or type(value) not in (int, float)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError("positive finite synchronized parent-exact full requests required")
            raw.append(value)
        values = np.asarray(raw[5:])
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
    candidate, comparisons = arrays[CANDIDATE], []
    tail = 0.05 / (2 * 64)
    for (m, a), values in arrays.items():
        if (m, a) == CANDIDATE:
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
        args += ["--partition=bii", "--cpus-per-task=2", "--mem=16G", "--time=00:25:00"]
    if dependency:
        args.append(f"--dependency=afterany:{dependency}")
    (root / "slurm").mkdir(exist_ok=True)
    args += [
        f"--job-name=mk-lt-{action}",
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
    decision = driver.read(OPERATOR / "analysis/decision.json")
    if not decision["complete"] or not decision["operator_gate"]:
        raise ValueError("positive two-geometry native operator gate required")
    old = driver.read(previous.PARENT / "plan.json")
    bound = dict(old["bound_files"])
    paths = [
        OPERATOR / "plan.json",
        OPERATOR / "analysis/decision.json",
        SOURCE / "mosaickv/docs/lt_latency_20261104.md",
    ]
    for index in (0, 1):
        paths.extend(
            [
                previous.PARENT / f"workers/{index}/study.json",
                OPERATOR / f"workers/{index}/operator.json",
                OPERATOR / f"workers/{index}/result.json",
            ]
        )
    bound.update({str(p): sha(p) for p in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            prompt_graph_root=str(root),
            stage="bounded_native_source_full_requests",
            registered_request_ratio_upper=0.96,
            acceptance_ratio_upper_bound=0.96,
            warmups=5,
            repetitions=20,
            bootstrap_replicates=10000,
            family_maximum_engineering_contrasts=64,
            independent_quality_N=0,
            no_fresh_identities=True,
            paper_campaign_authorized=False,
            automatic_quality_submission=False,
            main_six_mechanisms_disabled_precision_variant=True,
            whole_request_GPU_peak_certified=False,
            graph_unowned_tensor_bytes_allowed=0,
            controls_reused=False,
            operator_fitting_NOT_request_latency=True,
            bounded_native_workspace_bytes=2**21,
            numerical_patch_request_cap=2**20,
            optimized_graph_policy=arena.POLICY,
        ),
    )
    stage(root, "cpu")


def study(root, expected, index):
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
                variant="consumed_bounded_native_reader_controls",
                methods=list(previous.METHODS),
                actual_timing_protocol=dict(
                    output_predictions=128,
                    fitting_included_in_request=True,
                    old_controls_reused=False,
                    not_latency_evidence=True,
                ),
            ),
        )
        parent = driver.read(previous.PARENT / f"workers/{index}/study.json")
        records = make_records(parent, previous.METHODS)
        for method, arms in records.items():
            signatures = {"direct": parent["records"][method]["packet_sha256"]}
            for arm, record in arms.items():
                family = "whole-6291456" if arm.startswith("whole") else "direct"

                def verify(packets, family=family, arm=arm, signatures=signatures):
                    actual = previous.direct.packet_signature(packets)["digest"]
                    if family not in signatures:
                        if arm != "whole-6291456":
                            raise RuntimeError(
                                "original whole reader must establish its digest first"
                            )
                        signatures[family] = actual
                    elif actual != signatures[family]:
                        raise RuntimeError("original parent headers/codes/positions changed")

                measured = execute(
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
                digest = signatures[family]
                anchor = parent["records"][method][family]["audited"]
                added = (
                    measured["lt_source_conversion"]["added_patch_bytes"]
                    if record.get("lt_source_reader", False)
                    else 0
                )
                if any(measured[k] != anchor[k] + added for k in driver.STORAGE[:2]):
                    raise RuntimeError(
                        "native persistent bytes must equal parent plus exact repairs"
                    )
                records[method][arm] = dict(record, audited=measured, packet_sha256=digest)
                write_once(out / f"control-{method}-{arm}.json", records[method][arm])
        write_once(out / "study.json", dict(records=records, independent_quality_N=0))
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def timing(root, expected, index, *, final=False):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        methods = previous.METHODS if final else ("mosaickv_exact", "prefixkv_reimpl")
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_native_full_request_paired_timing",
                methods=list(methods),
                actual_timing_protocol=dict(
                    output_predictions=128,
                    warmups=5,
                    repetitions=20,
                    randomized_order=True,
                    fitting_capture_and_retirement_included=True,
                ),
            ),
        )
        original = driver.read(Path(plan["prompt_graph_root"]) / f"workers/{index}/study.json")
        arms = [(m, a, original["records"][m][a]) for m, a in sorted(required(final))]
        rng = random.Random(plan["seed"])
        for trial in range(-5, 20):
            placement = list(arms)
            rng.shuffle(placement)
            for method, arm, record in placement:
                measured = execute(adapter, refs, messages, profile, plan, method, record)
                if any(measured[k] != record["audited"][k] for k in driver.STORAGE[:2]):
                    raise RuntimeError("timed persistent bytes differ from audited companion")
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


def analyze(root, expected, action):
    plan = driver.checked(root, expected)
    failures, timings, memory = [], [], []
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
            for index, item in enumerate(records):
                candidate = item["records"][CANDIDATE[0]][CANDIDATE[1]]["audited"]
                baselines = [
                    r["audited"]
                    for m, arms in item["records"].items()
                    if m != CANDIDATE[0]
                    for r in arms.values()
                ]
                passed = all(candidate[k] < b[k] for b in baselines for k in driver.STORAGE)
                memory.append(
                    dict(
                        input_index=index,
                        smaller_than_ALL_baseline_controls=passed,
                        candidate={k: candidate[k] for k in driver.STORAGE},
                    )
                )
                if not passed:
                    failures.append(
                        f"input {index}: native workspace/repairs erase storage advantage"
                    )
        if not failures and action != "study_analysis":
            for index in (0, 1):
                try:
                    rows = [
                        json.loads(x)
                        for x in (root / f"workers/{index}/trials.jsonl").read_text().splitlines()
                    ]
                    if any(
                        r["source_sha"] != plan["source_sha"] or r["plan_sha256"] != expected
                        for r in rows
                    ):
                        raise ValueError("timing provenance mismatch")
                    timings.append(
                        summarize(rows, plan["seed"], final=action == "all_baselines_analysis")
                    )
                except Exception as error:
                    failures.append(f"input {index}: {type(error).__name__}: {error}")
    passed = (
        not failures and len(timings) == 2 and all(t["four_percent_latency_gate"] for t in timings)
    )
    write_once(
        root / ("cpu-analysis" if action == "cpu_analysis" else "analysis") / "decision.json",
        dict(
            complete=not failures,
            failures=failures,
            timings=timings,
            storage_controls=memory,
            four_percent_latency_gate=passed,
            acceptance_ratio_upper_bound=0.96,
            execution_source_sha=plan["source_sha"],
            plan_sha256=expected,
            independent_quality_N=0,
            no_reader_promoted=True,
            paper_superiority_established=False,
            whole_request_GPU_peak_certified=False,
            not_byte_matched_algorithm_evidence=True,
        ),
    )
    if failures:
        return
    if action == "cpu_analysis":
        stage(root, "study")
    elif action == "study_analysis":
        stage(previous.direct.common.previous.child(root, "paired-timing"), "timing")
    elif action == "timing_analysis" and passed:
        stage(previous.direct.common.previous.child(root, "all-baselines"), "all_baselines")


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
            final=args.action == "all_baselines",
        )


if __name__ == "__main__":
    main()
