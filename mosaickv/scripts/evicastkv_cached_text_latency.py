#!/usr/bin/env python3
"""Finite exact-text-row study -> paired latency -> conditional all-method DAG.

Two consumed inputs only. No new benchmark evidence, threshold relaxation,
fresh identities, opaque byte estimates or automatic paper promotion.
"""

import argparse
import json
import os
import random
import traceback
from pathlib import Path
from unittest.mock import patch

import evicastkv_hybrid_latency as previous

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

driver = previous.driver
SOURCE = driver.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/hybrid-latency-consumed-20261103-v2")
GUARD = Path("/scratch/djy8hg/runs/evicastkv/fused-model-guard-consumed-20261103-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_cached_text_latency.sbatch"


def submit(root, action, **kwargs):
    with patch.object(previous, "SCRIPT", SCRIPT):
        return previous.submit(root, action, **kwargs)


def submit_stage(root, action, analyzer):
    gpu = submit(root, action, gpu=True)
    cpu = submit(root, analyzer, dependency=gpu)
    write_once(root / "submission.json", dict(gpu_array=gpu, analysis=cpu))


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable output root and clean source required")
    parent = driver.read(PARENT / "hybrid-study/paired-timing/analysis/decision.json")
    guard = driver.read(GUARD / "analysis/decision.json")
    if not parent["complete"] or parent["all_input_latency_gate"] or not guard["complete"]:
        raise ValueError("completed preserved negative controls required")
    plan = driver.read(PARENT / "plan.json")
    bound = dict(plan["bound_files"])
    paths = [
        PARENT / "plan.json",
        PARENT / "hybrid-study/paired-timing/analysis/decision.json",
        GUARD / "analysis/decision.json",
        SOURCE / "mosaickv/docs/cached_text_latency_20261103.md",
    ]
    paths.extend(PARENT / f"hybrid-study/workers/{i}/study.json" for i in (0, 1))
    for path in paths:
        bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            plan,
            source_sha=revision,
            bound_files=bound,
            cached_text_root=str(root),
            stage="cached_exact_text_rows",
            maximum_simultaneous_workers=2,
            maximum_worker_minutes=120,
            cached_text_allowances=[0, 6 * 2**20],
            family_maximum_engineering_contrasts=48,
            automatic_bounded_followup=True,
            automatic_quality_submission=False,
            fresh_materialization_authorized=False,
            paper_campaign_authorized=False,
            independent_quality_N=0,
            exact_parent_words_required=True,
            all_baselines_receive_same_reader_choice=True,
            whole_request_GPU_peak_certified=False,
        ),
    )
    write_once(root / "submission.json", dict(cpu=submit(root, "cpu")))


def cpu(root, expected):
    driver.pair.common.previous.cpu(root, expected)
    driver.cpu_proof(root, driver.checked(root, expected))
    gpu = submit(root, "study", gpu=True)
    analysis = submit(root, "study_analysis", dependency=gpu)
    write_once(root / "gpu_submission.json", dict(gpu_array=gpu, analysis=analysis))


def run(adapter, refs, messages, profile, plan, method, record, *, audit=False):
    return previous.execute(
        adapter,
        refs,
        messages,
        profile,
        plan,
        method,
        reuse=True,
        audit=audit,
        policy=record.get("whole_policy"),
        text_policy=record.get("text_policy"),
    )


def study(root, expected, index):
    from mosaickv.evaluation.cached_text_reader import profile as cost_profile
    from mosaickv.evaluation.hybrid_reader_study import plans

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
                variant="consumed_exact_text_row_operator_and_storage_study",
                methods=list(previous.ALL_METHODS),
                actual_timing_protocol=dict(
                    operator_warmups=5,
                    operator_repetitions=20,
                    output_predictions=128,
                    not_request_latency_evidence=True,
                ),
            ),
        )
        attentions = tuple(layer.self_attn for layer in adapter._language_layers())
        old = driver.read(PARENT / f"hybrid-study/workers/{index}/study.json")
        records = {}
        for method in previous.ALL_METHODS:
            costs = {}

            def collect(packets, costs=costs):
                costs.update(cost_profile(packets, attentions, seed=plan["seed"]))

            base = previous.execute(
                adapter,
                refs,
                messages,
                profile,
                plan,
                method,
                reuse=True,
                audit=True,
                collector=collect,
            )
            write_once(out / f"cost-{method}.json", costs)
            proposals = plans(costs)
            arms = dict(reuse=dict(status="passed", audited=base))
            if method in previous.METHODS:
                prior = old["records"][method]["arms"].get("hybrid-6291456")
                if prior is None or prior["status"] != "passed":
                    raise ValueError("previous strongest whole-layer control required")
                arms["whole-6291456"] = dict(whole_policy=prior["policy"])
            for item in proposals:
                policy = item["policy"]
                if any(m != "source" for row in policy["modes"] for m in row):
                    arms[f"text-{policy['maximum_added_bytes']}"] = dict(text_policy=policy)
            for name, record in arms.items():
                if name == "reuse":
                    continue
                try:
                    measured = run(
                        adapter, refs, messages, profile, plan, method, record, audit=True
                    )
                    allowance = record.get("text_policy", record.get("whole_policy"))[
                        "maximum_added_bytes"
                    ]
                    if (
                        any(measured[k] > base[k] + allowance for k in driver.STORAGE)
                        or not measured["peak_measurement_admissible"]
                    ):
                        raise RuntimeError("actual audited cache storage exceeds allowance")
                    record.update(status="passed", audited=measured)
                except Exception as error:
                    record.update(
                        status="rejected",
                        error=f"{type(error).__name__}: {error}",
                        traceback=traceback.format_exc(),
                    )
            records[method] = dict(arms=arms, proposals=proposals)
            write_once(out / f"controls-{method}.json", records[method])
        write_once(out / "study.json", dict(records=records, independent_quality_N=0))
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def study_analysis(root, expected):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "study.json")
    meaningful = []
    for index, row in enumerate(rows):
        accepted = previous.validated_arms(row["records"]["mosaickv_exact"]["arms"])
        meaningful.append({name for name in accepted if name.startswith("text-")})
        for method in previous.ALL_METHODS:
            if "reuse" not in previous.validated_arms(row["records"][method]["arms"]):
                failures.append(
                    dict(input_index=index, method=method, error="parent controls failed")
                )
    common = set.intersection(*meaningful) if len(meaningful) == 2 else set()
    previous.closeout(
        root,
        not failures,
        failures,
        studies=rows,
        meaningful_common_text_arms=sorted(common),
        no_reader_promoted=True,
    )
    if not failures and common:
        submit_stage(previous.child(root, "paired-timing"), "timing", "timing_analysis")


def contrasts(rows, seed):
    import numpy as np

    keys = [(r["method"], r["arm"], r["trial"]) for r in rows]
    arms = sorted({(m, a) for m, a, _ in keys})
    if not {
        ("mosaickv_exact", "reuse"),
        ("prefixkv_reimpl", "reuse"),
        ("prefixkv_reimpl", "whole-6291456"),
    }.issubset(arms):
        raise ValueError("own reuse and strongest registered PrefixKV controls required")
    if len(keys) != len(set(keys)) or set(keys) != {
        (m, a, t) for m, a in arms for t in range(-5, 20)
    }:
        raise ValueError("complete unique 5-warmup/20-trial arms required")
    arrays, summaries = {}, []
    for m, a in arms:
        ordered = sorted(
            (r for r in rows if (r["method"], r["arm"]) == (m, a) and r["trial"] >= 0),
            key=lambda r: r["trial"],
        )
        values = np.asarray([r["measurement"]["total_wall_seconds"] for r in ordered])
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError("finite positive synchronized request times required")
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
    tail = 0.05 / (2 * 48)
    comparisons = []
    for (m, a), values in arrays.items():
        if m != "mosaickv_exact" or not a.startswith("text-"):
            continue

        def interval(reference, values=values):
            return np.quantile(
                values[draws].mean(1) / reference[draws].mean(1), [tail, 1 - tail]
            ).tolist()

        prefixes = [
            dict(
                reference_arm=b,
                mean_ratio=float(values.mean() / v.mean()),
                adjusted_ratio_CI=interval(v),
            )
            for (n, b), v in arrays.items()
            if n == "prefixkv_reimpl"
        ]
        own = interval(arrays[m, "reuse"])
        comparisons.append(
            dict(
                arm=a,
                own_adjusted_ratio_CI=own,
                prefixes=prefixes,
                own_speedup_gate=own[1] <= 0.99,
                strongest_prefix_similar_gate=all(
                    p["adjusted_ratio_CI"][1] <= 1.05 for p in prefixes
                ),
                strongest_prefix_no_slower_gate=all(
                    p["adjusted_ratio_CI"][1] <= 1 for p in prefixes
                ),
            )
        )
    return dict(
        summaries=summaries,
        comparisons=comparisons,
        family_maximum_engineering_contrasts=48,
        intervals_over_repetitions_not_independent_examples=True,
    )


def timing(root, expected, index, *, all_methods=False):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        methods = previous.ALL_METHODS if all_methods else previous.METHODS
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_exact_text_row_paired_full_request_timing",
                methods=list(methods),
                actual_timing_protocol=dict(
                    warmups=5,
                    repetitions=20,
                    output_predictions=128,
                    conversion_included=True,
                    randomized_method_order=True,
                ),
            ),
        )
        source = Path(plan["cached_text_root"])
        study_row = driver.read(source / f"workers/{index}/study.json")
        arms = [
            (m, name, record)
            for m in methods
            for name, record in previous.validated_arms(study_row["records"][m]["arms"]).items()
        ]
        rng, raw = random.Random(plan["seed"]), []
        for trial in range(-5, 20):
            placement = list(arms)
            rng.shuffle(placement)
            for method, name, record in placement:
                measured = run(adapter, refs, messages, profile, plan, method, record)
                if (
                    any(measured[k] != record["audited"][k] for k in driver.STORAGE[:2])
                    or measured["decode_owned_peak_bytes"] is not None
                    or measured["peak_measurement_admissible"]
                ):
                    raise RuntimeError("timed cache differs from audited storage companion")
                row = dict(
                    method=method,
                    arm=name,
                    trial=trial,
                    warmup=trial < 0,
                    measurement=measured,
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                )
                raw.append(row)
                with (out / "trials.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_once(out / "timing.json", contrasts(raw, plan["seed"]))
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def timing_analysis(root, expected, *, final=False):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "timing.json")
    groups = [
        {
            c["arm"]
            for c in row["comparisons"]
            if c["own_speedup_gate"] and c["strongest_prefix_similar_gate"]
        }
        for row in rows
    ]
    common = set.intersection(*groups) if len(groups) == 2 else set()
    previous.closeout(
        root,
        not failures,
        failures,
        timings=rows,
        all_input_latency_gate=bool(common) and not failures,
        eligible_engineering_arms=sorted(common),
        no_reader_promoted=True,
        no_paper_quality_inference=True,
        previous_fusion_and_prefetch_negatives_preserved=True,
    )
    if not failures and common and not final:
        submit_stage(previous.child(root, "all-baselines"), "all_baselines", "all_analysis")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "cpu",
            "study",
            "study_analysis",
            "timing",
            "timing_analysis",
            "all_baselines",
            "all_analysis",
        ),
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "all_baselines":
        timing(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]), all_methods=True)
    elif args.action == "all_analysis":
        timing_analysis(args.root, args.plan_sha, final=True)
    elif args.action in ("study", "timing"):
        globals()[args.action](args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.action](args.root, args.plan_sha)


if __name__ == "__main__":
    main()
