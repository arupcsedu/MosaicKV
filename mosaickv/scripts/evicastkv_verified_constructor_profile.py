#!/usr/bin/env python3
"""Attribute the failed constructor screen; no runtime or numerical changes."""

import json
import pstats
import subprocess
import traceback
from pathlib import Path

import evicastkv_verified_constructor as base

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.optimized_runtime_profile import collect

SOURCE, driver, parent = base.SOURCE, base.driver, base.parent
PREVIOUS = Path("/scratch/djy8hg/runs/evicastkv/verified-constructor-screen-consumed-20261106-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_verified_constructor_profile.sbatch"
DOC = SOURCE / "mosaickv/docs/verified_constructor_profile.md"
ARMS = (*base.ARMS[:3], base.ARMS[3])
APPROVED = {
    "mosaickv/scripts/evicastkv_verified_constructor_profile.py",
    "mosaickv/slurm/evicastkv_verified_constructor_profile.sbatch",
    "mosaickv/docs/verified_constructor_profile.md",
    "mosaickv/tests/unit/test_verified_constructor_profile.py",
}


def attribution(stats, limit=30):
    rows = [
        dict(
            file=file,
            line=line,
            function=function,
            calls=value[1],
            exclusive_host_seconds=value[2],
            cumulative_host_seconds=value[3],
        )
        for (file, line, function), value in stats.items()
    ]
    return dict(
        top_exclusive=sorted(
            rows, key=lambda r: (-r["exclusive_host_seconds"], r["file"], r["line"])
        )[:limit],
        constructor=[r for r in rows if r["file"].endswith("/verified_constructor.py")],
        torch_equal=[r for r in rows if "equal of" in r["function"]],
        instrumented_NOT_latency_acceptance=True,
        background_thread_CPU_not_attributed=True,
    )


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    prior = driver.read(PREVIOUS / "analysis/decision.json")
    if not prior["complete"] or prior["failures"]:
        raise ValueError("complete preceding screen required, including negative findings")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", prior["source_sha"], revision],
        cwd=SOURCE,
        text=True,
    ).splitlines()
    if not set(changed) <= APPROVED:
        raise ValueError("profiling forbids changes to the validated runtime")
    old = driver.read(PREVIOUS / "plan.json")
    bound = dict(old["bound_files"])
    for path in (DOC, PREVIOUS / "plan.json", PREVIOUS / "analysis/decision.json"):
        bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            arms=ARMS,
            resolved_arm_overrides=[
                row for row in old["resolved_arm_overrides"] if (row["method"], row["arm"]) in ARMS
            ],
            stage="consumed_verified_constructor_host_attribution",
            correctness_only=False,
            warmups=0,
            repetitions=0,
            profile_requests_per_arm=1,
            main_thread_only=True,
            source_runtime_unchanged=True,
            instrumentation_NOT_latency_acceptance=True,
            independent_quality_N=0,
            no_automatic_fresh_or_quality_submission=True,
        ),
    )
    cpu = parent.submit(root, "cpu")
    release = parent.submit(root, "release", cpu)
    write_once(root / "cpu_submission.json", dict(cpu=cpu, release=release))


def worker(root, expected, index):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        base.control.actual(at_worker_start=True)
        release = driver.read(root / "release.json")
        if (
            not release["profile_released"]
            or sha(Path(release["parent_decision"])) != release["parent_sha256"]
        ):
            raise ValueError("unchanged successful numerical predecessor required")
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_verified_constructor_host_attribution",
                methods=["mosaickv_exact", "prefixkv_reimpl"],
                actual_timing_protocol=dict(
                    output_predictions=128,
                    profile_requests_per_arm=1,
                    construction="prepare_inputs_to_first_decode_entry",
                    decode_updates=list(range(1, 127)),
                    initial_capture_excluded=True,
                    instrumented_NOT_latency_acceptance=True,
                    main_thread_only=True,
                    background_thread_CPU_not_attributed=True,
                ),
            ),
        )
        records = base.configurations(
            driver.read(parent.PARENT / f"workers/{index}/study.json")["records"]
        )
        rows = []
        for method, arm in ARMS:
            folder = out / f"profile-{method}-{arm}"
            folder.mkdir()

            def call(method=method, arm=arm):
                return base.execute(
                    adapter, refs, messages, profile, plan, method, arm, records[method, arm]
                )

            value = collect(adapter, call, folder)
            write_once(folder / "profile.json", value)
            phases = {}
            for phase, item in value["phases"].items():
                path = Path(item["profile"])
                if sha(path) != item["profile_sha256"]:
                    raise ValueError("profile changed before attribution")
                phases[phase] = attribution(pstats.Stats(str(path)).stats)
            rows.append(
                dict(
                    method=method,
                    arm=arm,
                    profile_sha256=sha(folder / "profile.json"),
                    phases=phases,
                )
            )
        write_once(
            out / "profile.json",
            dict(
                records=rows, independent_quality_N=0, instrumentation_NOT_latency_acceptance=True
            ),
        )
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def analyze(root, expected):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "profile.json")
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not failures and len(rows) == 2,
            failures=failures,
            rows=rows,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            instrumentation_NOT_latency_acceptance=True,
            independent_quality_N=0,
            whole_request_GPU_peak_certified=False,
            no_automatic_fresh_or_quality_submission=True,
        ),
    )
    print(json.dumps(dict(complete=not failures and len(rows) == 2, failures=failures)))


def main():
    base.prepare, base.SCRIPT, base.DOC, base.ARMS = prepare, SCRIPT, DOC, ARMS
    base.base.ORIGINAL_WORKER, base.base.analyze = worker, analyze
    base.main()


if __name__ == "__main__":
    main()
