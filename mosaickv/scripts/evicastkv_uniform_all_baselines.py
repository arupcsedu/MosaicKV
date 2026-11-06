#!/usr/bin/env python3
"""Finite consumed-input canary -> repeated all-baseline engineering controls.

No task-quality promotion, new identities, runtime changes or paper claims.
Every comparator is paired with its own recomputed candidate in the SAME job.
"""

import argparse
import json
import math
import os
import random
import subprocess
import traceback
from pathlib import Path
from unittest.mock import patch

import evicastkv_direct_text_latency as direct
import evicastkv_uniform_core_paired as paired
import evicastkv_uniform_core_screen as screen
import numpy as np

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, driver = screen.SOURCE, screen.driver
PARENT = Path("/scratch/djy8hg/runs/evicastkv/uniform-core-paired-consumed-20261106-v1")
METHODS = ("prompt_attention_topk", "vl_cache_reimpl", "lookm_reimpl")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_uniform_all_baselines.sbatch"
DOC = SOURCE / "mosaickv/docs/uniform_all_baselines.md"
CANDIDATE = ("mosaickv_exact", "joint-async-uniform-core")


def configurations(records, method, plan):
    if method not in METHODS:
        raise ValueError("registered baseline required")
    with (
        patch.object(screen.base, "ARMS", screen.ARMS),
        patch.object(screen.base.base, "ARMS", screen.ARMS),
        patch.object(screen.base, "overrides", screen.overrides),
    ):
        source = screen.base.configurations(records)[CANDIDATE]
    # The original source worker initialized several historical controllers
    # from this immutable plan before constructing options. A standalone
    # launcher must not inherit their import-time keys/False defaults.
    policies = plan["frozen_runtime_policies"]
    source = dict(
        source,
        joint_incoming_projection="kv",
        post_attention_segment=policies["segment"],
        exact_rotary_dispatch=policies["rotary"],
        reuse_source_input_copy=policies["copy"],
        frozen_joint_plan=policies["frozen"],
        overlap_native_fit=policies["overlap"],
        input_norm_segment=plan["common_original_norm_replay"],
        reuse_norm_input=plan["selected_common_input_reuse"],
    )
    from mosaickv.evaluation.joint_incoming_projection import validate_source_rows

    validate_source_rows(
        source["source_tile_rows"],
        source["joint_incoming_projection"],
        source["frozen_joint_plan"],
        source["native_history_reader"],
    )
    result = {CANDIDATE: source}
    for arm in ("graph", "graph-original"):
        result[method, arm] = dict(
            records[method][arm],
            graph_accounting="event_sum",
            reuse_patch_dispatch=True,
            reuse_graph_replay_plan=True,
            post_attention_segment=True,
            exact_rotary_dispatch=True,
            reuse_source_input_copy=True,
            frozen_joint_plan=True,
            overlap_native_fit=True,
            native_history_reader=True,
            native_history_graph=True,
            input_norm_segment=True,
            reuse_norm_input=True,
            verified_constructor="handoff",
            uniform_attention_core=True,
            fused_transport_predicate=False,
        )
    return result


def submit(root, action, *, dependency=None, gpu=False):
    args = ["sbatch", "--parsable", "--account=bii_dsc_community", "--nodes=1", "--ntasks=1"]
    if gpu:
        driver.cpu_proof(root, driver.checked(root, sha(root / "plan.json")))
        args += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--array=0-1%2",
            "--time=00:45:00",
        ]
    else:
        args += [
            "--partition=bii",
            "--nodelist=udc-aj36-2c0",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:25:00",
        ]
    if dependency:
        args.append("--dependency=afterany:" + str(dependency))
    (root / "slurm").mkdir(exist_ok=True)
    args += [
        f"--job-name=mk-allcore-{action}",
        f"--output={root}/slurm/{action}-%A_%a.out",
        f"--error={root}/slurm/{action}-%A_%a.err",
        str(SCRIPT),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        action,
    ]
    return subprocess.check_output(args, text=True).strip()


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new root and clean immutable source required")
    old = driver.read(PARENT / "plan.json")
    for index in (0, 1):
        records = driver.read(screen.parent.PARENT / f"workers/{index}/study.json")["records"]
        for method in METHODS:
            configurations(records, method, old)  # Real CPU-only launch-boundary check.
    for filename in paired.RUNTIME:
        rel = "mosaickv/src/mosaickv/evaluation/" + filename
        original = subprocess.check_output(
            ["git", "show", f"{old['source_sha']}:{rel}"], cwd=SOURCE, text=True
        )
        if paired.semantic(original) != paired.semantic((SOURCE / rel).read_text()):
            raise ValueError("all-baseline engineering study must freeze runtime semantics")
    bound = dict(old["bound_files"])
    failed = Path(
        "/scratch/djy8hg/runs/evicastkv/uniform-core-all-baselines-consumed-20261106-v1/canary/analysis/decision.json"
    )
    if failed.exists():
        bound[str(failed)] = sha(failed)
    for path in (DOC, PARENT / "plan.json", PARENT / "profile_submission.json"):
        bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_uniform_core_all_baseline_engineering_controls",
            baseline_methods=METHODS,
            warmups=5,
            repetitions=20,
            root_of_all_baseline_chain=str(root),
            required_prefix_confirmation=str(PARENT / "analysis/decision.json"),
            required_prefix_source_sha=old["source_sha"],
            required_prefix_plan_sha256=sha(PARENT / "plan.json"),
            same_consumed_inputs_only=True,
            independent_quality_N=0,
            baseline_labels="local_reimplementations_or_Prompt_TopK_NOT_official",
            no_fresh_or_paper_campaign_authorized=True,
        ),
    )
    cpu = submit(root, "cpu")
    prior_analysis = driver.read(PARENT / "profile_submission.json")["analysis"]
    release = submit(root, "release", dependency=f"{cpu}:{prior_analysis}")
    write_once(
        root / "submission.json", dict(cpu=cpu, release=release, required_analysis=prior_analysis)
    )


def launch(root, phase):
    plan = driver.checked(root, sha(root / "plan.json"))
    jobs = []
    for method in METHODS:
        target = root / phase / method
        bound = dict(plan["bound_files"])
        bound[str(root / "plan.json")] = sha(root / "plan.json")
        bound[str(root / "release.json")] = sha(root / "release.json")
        if phase == "timing":
            path = root / "canary/analysis/decision.json"
            bound[str(path)] = sha(path)
        write_once(
            target / "plan.json",
            dict(
                plan,
                bound_files=bound,
                baseline_method=method,
                phase=phase,
                uniform_core_root=str(target),
                warmups=0 if phase == "canary" else 5,
                repetitions=1 if phase == "canary" else 20,
            ),
        )
        driver.copy_cpu(root, target)
        job = submit(target, "worker", gpu=True)
        jobs.append(job)
        write_once(root / phase / f"submission-{method}.json", dict(array=job))
    from control_a100_schedule import tick

    policy = dict(
        timezone="America/New_York",
        maximum_worker_minutes=120,
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
        arrays=jobs,
    )
    folder = root / phase / "scheduler"
    write_once(folder / "policy.json", policy)
    write_once(folder / "registration.json", tick(policy, {}, folder / "state.json", False))
    action = "canary_analysis" if phase == "canary" else "analysis"
    analysis = submit(root, action, dependency=":".join(jobs))
    write_once(root / phase / "submission.json", dict(arrays=jobs, analysis=analysis))


def release(root):
    plan = driver.checked(root, sha(root / "plan.json"))
    driver.cpu_proof(root, plan)
    path = Path(plan["required_prefix_confirmation"])
    decision = driver.read(path)
    passed = (
        decision["source_sha"] == plan["required_prefix_source_sha"]
        and decision["plan_sha256"] == plan["required_prefix_plan_sha256"]
        and decision["complete"]
        and not decision["failures"]
        and decision["formal_latency_acceptance_against_both_Prefix_controls"]
    )
    write_once(
        root / "release.json",
        dict(
            released=passed,
            parent_decision=str(path),
            parent_decision_sha256=sha(path),
            independent_quality_N=0,
        ),
    )
    if passed:
        launch(root, "canary")


def check_core(value, enabled, *, audit=False):
    info = value["uniform_attention_core"]
    if info["requested"] is not enabled or not info["graphs_retired"] or info["aborted"]:
        raise RuntimeError("complete declared request-private core retirement required")
    if info["retained_tensor_owners"] or info["extra_KV_buffer_bytes"]:
        raise RuntimeError("no retained core owners or extra KV buffer allowed")
    if info["eligible"]:
        if info["calls"] != 4064 or info["captures"] != 254 or info["replays"] != 8128:
            raise RuntimeError("incomplete uniform graph execution")
        if not 0 < info["workspace_bytes"] <= 384 * 1024:
            raise RuntimeError("charged fixed workspace bound violated")
        if info["capture_owner_checks"] != (254 if audit else 0):
            raise RuntimeError("audited companion must inspect every capture owner")
    elif info["calls"] or info["captures"] or info["workspace_bytes"]:
        raise RuntimeError("adaptive cache must remain unchanged original eager")


def result_summary(rows, audited, config, seed, warmups, repetitions):
    keys = [(r["method"], r["arm"], r["trial"]) for r in rows]
    expected = {(m, a, t) for m, a in config for t in range(-warmups, repetitions)}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("complete unique paired engineering rows required")
    arrays = {}
    for key in config:
        selected = sorted(
            (r for r in rows if (r["method"], r["arm"]) == key and not r["warmup"]),
            key=lambda r: r["trial"],
        )
        values = [r["measurement"]["total_wall_seconds"] for r in selected]
        if any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError("finite positive synchronized request times required")
        arrays[key] = np.asarray(values)
    candidate = arrays[CANDIDATE]
    draws = np.random.default_rng(seed).integers(0, repetitions, (10000, repetitions))
    comparisons = []
    tail = 0.05 / (2 * 64)
    for key, values in arrays.items():
        if key == CANDIDATE:
            continue
        interval = np.quantile(
            candidate[draws].mean(1) / values[draws].mean(1), [tail, 1 - tail]
        ).tolist()
        comparisons.append(
            dict(
                method=key[0],
                arm=key[1],
                mean_seconds=float(values.mean()),
                ratio=float(candidate.mean() / values.mean()),
                adjusted_ratio_CI=interval,
                four_percent_faster=repetitions == 20 and bool(interval[1] <= 0.96),
            )
        )
    smaller = all(
        audited[CANDIDATE][k] < audited[key][k]
        for key in config
        if key != CANDIDATE
        for k in driver.STORAGE
    )
    return dict(
        complete=True,
        candidate_mean_seconds=float(candidate.mean()),
        comparisons=comparisons,
        smaller_at_all_three_decode_owned_boundaries=smaller,
        four_percent_engineering_gate=repetitions == 20
        and smaller
        and all(r["four_percent_faster"] for r in comparisons),
        physical_storage={
            f"{m}:{a}": {k: v[k] for k in driver.STORAGE} for (m, a), v in audited.items()
        },
        all_parent_words_and_packets_identical=True,
        no_FullKV_fallback=True,
        independent_quality_N=0,
        intervals_over_repetitions_NOT_independent_examples=True,
        whole_request_GPU_peak_certified=False,
        no_paper_quality_superiority_claim=True,
    )


def worker(root, expected, index):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        from mosaickv.evaluation.reservation_window import window

        admission = window(
            subprocess.check_output(
                ["scontrol", "show", "reservation", "bi_fox_dgx", "-o"], text=True
            ),
            seconds=45 * 60,
        )
        if admission["begin"] is not None:
            raise RuntimeError("reservation cannot fit the complete 45-minute shard")
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_all_baseline_uniform_core_engineering",
                methods=["mosaickv_exact", driver.read(root / "plan.json")["baseline_method"]],
                actual_timing_protocol=dict(
                    warmups=driver.read(root / "plan.json")["warmups"],
                    repetitions=driver.read(root / "plan.json")["repetitions"],
                    same_GPU_paired_order=True,
                    independent_quality_N=0,
                ),
            ),
        )
        screen.pre_request_controls(plan)
        config = configurations(
            driver.read(screen.parent.PARENT / f"workers/{index}/study.json")["records"],
            plan["baseline_method"],
            plan,
        )
        audited = {}

        def execute(key, *, audit=False):
            record = config[key]

            def verify(packets):
                if direct.packet_signature(packets)["digest"] != record["packet_sha256"]:
                    raise RuntimeError("frozen original preconversion packet words changed")

            value = direct.execute(
                adapter,
                refs,
                messages,
                profile,
                plan,
                key[0],
                record,
                audit=audit,
                final_validator=verify,
            )
            check_core(value, True, audit=audit)
            if not value["logits_bitwise"] or not value["no_FullKV_fallback"]:
                raise RuntimeError("full 128 parent vocabulary vectors and no fallback required")
            return value

        for key in config:
            audited[key] = execute(key, audit=True)
            write_once(out / f"audited-{key[0]}-{key[1]}.json", audited[key])
        rows, rng = [], random.Random(plan["seed"])
        for trial in range(-plan["warmups"], plan["repetitions"]):
            order = list(config)
            rng.shuffle(order)
            for place, key in enumerate(order):
                value = execute(key)
                if any(value[k] != audited[key][k] for k in driver.STORAGE[:2]):
                    raise RuntimeError("timed initial/final storage differs from audited companion")
                row = dict(
                    method=key[0],
                    arm=key[1],
                    trial=trial,
                    warmup=trial < 0,
                    order_index=place,
                    measurement=value,
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                )
                rows.append(row)
                with (out / "trials.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_once(
            out / "profile.json",
            result_summary(
                rows, audited, config, plan["seed"] + index, plan["warmups"], plan["repetitions"]
            ),
        )
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def analyze(root, phase):
    driver.checked(root, sha(root / "plan.json"))
    rows, failures = [], []
    for method in METHODS:
        target = root / phase / method
        plan = driver.checked(target, sha(target / "plan.json"))
        found, errors = driver.load_producers(target, plan, "profile.json")
        rows.extend(dict(baseline=method, **r) for r in found)
        failures.extend(f"{method}: {error}" for error in errors)
    complete = not failures and len(rows) == 6
    passed = complete and all(r["smaller_at_all_three_decode_owned_boundaries"] for r in rows)
    write_once(
        root / phase / "analysis/decision.json",
        dict(
            complete=complete,
            failures=failures,
            rows=rows,
            canary_passed=phase == "canary" and passed,
            four_percent_engineering_gate=phase == "timing"
            and passed
            and all(r["four_percent_engineering_gate"] for r in rows),
            source_sha=clean_source(SOURCE),
            independent_quality_N=0,
            whole_request_GPU_peak_certified=False,
            no_fresh_or_paper_campaign_authorized=True,
        ),
    )
    if phase == "canary" and passed:
        launch(root, "timing")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "cpu", "release", "worker", "canary_analysis", "analysis")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "cpu":
        driver.pair.common.previous.cpu(args.root, args.plan_sha)
    elif args.action == "release":
        release(args.root)
    elif args.action == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        analyze(args.root, "canary" if args.action == "canary_analysis" else "timing")


if __name__ == "__main__":
    main()
