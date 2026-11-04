#!/usr/bin/env python3
"""Separate strict acceptance from legacy <=5%-slower diagnostic progression.

No model code is changed. Execution and analysis SHAs remain distinct. Finite
afterany follow-ups use the original clean runtime worktree, never this one.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import evicastkv_direct_text_latency as direct

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_strict_latency.sbatch"
DOC = SOURCE / "mosaickv/docs/strict_latency_acceptance_20261103.md"
METHODS = set(direct.common.previous.ALL_METHODS)
FAMILY = 64


def summarize(rows, *, kind, seed, final=False):
    """Paired repeat bootstrap; not independent-example quality inference."""
    import numpy as np

    candidate, anchor = ("direct", "cached") if kind == "direct" else ("text-6291456", "reuse")
    required = {
        ("mosaickv_exact", candidate),
        ("mosaickv_exact", anchor),
        ("prefixkv_reimpl", "whole-6291456"),
        ("prefixkv_reimpl", candidate),
        ("prefixkv_reimpl", anchor),
    }
    keys = [(r["method"], r["arm"], r["trial"]) for r in rows]
    arms = {(m, a) for m, a, _ in keys}
    if not required <= arms or {m for m, _ in arms} - METHODS:
        raise ValueError("registered anchor, candidate, and strongest Prefix controls required")
    if final and {m for m, _ in arms} != METHODS:
        raise ValueError("all five methods required for final all-baseline acceptance")
    if len(keys) != len(set(keys)) or set(keys) != {
        (m, a, t) for m, a in arms for t in range(-5, 20)
    }:
        raise ValueError("complete unique five warmups and twenty timed repetitions required")
    if any(r["warmup"] is not (r["trial"] < 0) for r in rows):
        raise ValueError("warmup flags must match registered trial indices")
    arrays, summaries = {}, []
    for method, arm in sorted(arms):
        selected = sorted(
            (r for r in rows if (r["method"], r["arm"]) == (method, arm)),
            key=lambda r: r["trial"],
        )
        values = np.asarray([r["measurement"]["total_wall_seconds"] for r in selected])
        if not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError("all timings, including warmups, must be finite and positive")
        for row in selected:
            measured = row["measurement"]
            if (
                measured.get("logits_bitwise") is not True
                or measured.get("no_FullKV_fallback") is not True
            ):
                raise ValueError("parent vocabulary-word parity and no fallback required")
        values = values[5:]
        arrays[method, arm] = values
        summaries.append(
            dict(
                method=method,
                arm=arm,
                mean=float(values.mean()),
                p50=float(np.median(values)),
                p95=float(np.quantile(values, 0.95)),
            )
        )
    draws = np.random.default_rng(seed).integers(0, 20, (10000, 20))
    v = arrays["mosaickv_exact", candidate]
    tail = 0.05 / (2 * FAMILY)

    def contrast(method, arm):
        other = arrays[method, arm]
        ratios = v[draws].mean(1) / other[draws].mean(1)
        differences = (v[draws] - other[draws]).mean(1)
        ci = np.quantile(ratios, [tail, 1 - tail]).tolist()
        return dict(
            method=method,
            arm=arm,
            mean_ratio=float(v.mean() / other.mean()),
            mean_difference_seconds=float((v - other).mean()),
            adjusted_ratio_CI=ci,
            adjusted_difference_CI=np.quantile(differences, [tail, 1 - tail]).tolist(),
            equal_or_faster=bool(ci[1] <= 1.00),
            five_percent_faster=bool(ci[1] <= 0.95),
            within_five_percent_tradeoff=bool(ci[1] <= 1.05),
        )

    comparisons = [contrast(m, a) for m, a in sorted(arms) if m != "mosaickv_exact"]
    if len(comparisons) * 2 * 2 > FAMILY:
        raise ValueError("registered two-stage/two-input multiplicity family exceeded")
    return dict(
        summaries=summaries,
        own=contrast("mosaickv_exact", anchor),
        baselines=comparisons,
        strict_latency_gate=all(r["equal_or_faster"] for r in comparisons),
        five_percent_faster_gate=all(r["five_percent_faster"] for r in comparisons),
        within_five_percent_tradeoff=all(r["within_five_percent_tradeoff"] for r in comparisons),
        acceptance_ratio_upper_bound=1.00,
        tradeoff_only_ratio_upper_bound=1.05,
        bootstrap_replicates=10000,
        family_maximum_engineering_contrasts=FAMILY,
        intervals_over_repetitions_not_independent_examples=True,
    )


def validate(root, expected, runtime, analyzer):
    if clean_source(SOURCE) != analyzer or sha(root / "plan.json") != expected:
        raise ValueError("clean analyzer SHA and immutable execution plan required")
    plan = direct.driver.read(root / "plan.json")
    if clean_source(runtime) != plan["source_sha"]:
        raise ValueError("runtime must retain its original clean execution SHA")
    for path, checksum in plan["bound_files"].items():
        if sha(Path(path)) != checksum:
            raise ValueError(f"changed execution dependency: {path}")
    direct.driver.cpu_proof(root, plan)
    return plan


def submit_analysis(root, runtime, dependency, action, kind):
    revision = clean_source(SOURCE)
    args = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        "--partition=bii,standard",
        "--cpus-per-task=2",
        "--mem=16G",
        "--time=00:15:00",
        f"--dependency=afterany:{dependency}",
        f"--job-name=mk-strict-{action}",
        f"--output={root}/slurm/strict-{action}-%j.out",
        f"--error={root}/slurm/strict-{action}-%j.err",
        str(SCRIPT),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        revision,
        str(runtime),
        action,
        kind,
    ]
    return subprocess.check_output(args, text=True).strip()


def launch_child(root, runtime, name, action):
    """Only original frozen runtime builds plans or launches model workers."""
    code = (
        "import json; from pathlib import Path; import evicastkv_direct_text_latency as d; "
        f"r=d.common.previous.child(Path({str(root)!r}),{name!r}); "
        f"j=d.submit(r,{action!r},gpu=True); print(json.dumps(dict(root=str(r),gpu_array=j)))"
    )
    env = dict(os.environ, PYTHONPATH=f"{runtime}/mosaickv/src:{runtime}/mosaickv/scripts")
    output = subprocess.check_output(
        ["/scratch/djy8hg/env/mosaickv/bin/python", "-c", code], cwd=runtime, env=env, text=True
    )
    submission = json.loads(output.strip().splitlines()[-1])
    child = Path(submission["root"])
    submission["analysis"] = submit_analysis(
        child,
        runtime,
        submission["gpu_array"],
        "all_analysis" if action == "all_baselines" else "timing",
        "direct",
    )
    submission["strict_analyzer_sha"] = clean_source(SOURCE)
    write_once(child / "submission.json", submission)


def analyze(root, expected, runtime, analyzer, action, kind, output=None):
    plan = validate(root, expected, runtime, analyzer)
    filename = "study.json" if action == "study" else "timing.json"
    records, failures = direct.driver.load_producers(root, plan, filename)
    summaries = []
    if not failures and action != "study":
        for index in (0, 1):
            try:
                path = root / f"workers/{index}/trials.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if any(
                    r["source_sha"] != plan["source_sha"] or r["plan_sha256"] != expected
                    for r in rows
                ):
                    raise ValueError("raw trial provenance differs from execution plan")
                summaries.append(
                    summarize(rows, kind=kind, seed=plan["seed"], final=action == "all_analysis")
                )
            except Exception as error:
                failures.append(dict(input_index=index, error=f"{type(error).__name__}: {error}"))
    strict = (
        not failures and len(summaries) == 2 and all(r["strict_latency_gate"] for r in summaries)
    )
    result = dict(
        complete=not failures,
        failures=failures,
        timings=summaries,
        all_input_latency_gate=bool(strict),
        acceptance_ratio_upper_bound=1.00,
        five_percent_slower_is_tradeoff_only=True,
        execution_source_sha=plan["source_sha"],
        analyzer_source_sha=analyzer,
        plan_sha256=expected,
        amendment_sha256=sha(DOC),
        producer_result_hashes={
            str(root / f"workers/{i}/result.json"): sha(root / f"workers/{i}/result.json")
            for i in (0, 1)
            if (root / f"workers/{i}/result.json").exists()
        },
        independent_quality_N=0,
        paper_superiority_established=False,
        whole_request_GPU_peak_certified=False,
        no_reader_promoted=True,
        not_byte_matched_algorithm_evidence=True,
        retrospective_reclassification=output is not None,
    )
    if action == "study":
        result.update(studies=records, packet_and_storage_identity=not failures)
    target = output or root / "analysis"
    write_once(target / "decision.json", result)
    if output is None and not failures:
        if action == "study":
            launch_child(root, runtime, "paired-timing", "timing")
        elif action == "timing" and strict:
            launch_child(root, runtime, "all-baselines", "all_baselines")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("study", "timing", "all_analysis"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--analyzer-sha", required=True)
    parser.add_argument("--kind", choices=("direct", "cached"), default="direct")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analyze(
        args.root,
        args.plan_sha,
        args.runtime_source,
        args.analyzer_sha,
        args.action,
        args.kind,
        args.output,
    )


if __name__ == "__main__":
    main()
