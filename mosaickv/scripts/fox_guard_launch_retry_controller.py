#!/usr/bin/env python3
"""Finite launch-only retry after PrefixKV's quality lease ends; no polling."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from evicastkv_review_closure_v2 import cpu_suite
from fox_installed_guard_followup_v2 import PREFIX_ROOT, prepare, submit

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
CONTROL = Path("/scratch/djy8hg/runs/evicastkv/fox-guard-launch-retry1-controller")
ROOT = Path("/scratch/djy8hg/runs/evicastkv/fox-installed-guard-20261029-v2-retry1")


def schedule(mode, dependency=None):
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii",
        "--job-name=mk-fox-launch-fix",
        "--cpus-per-task=2",
        "--mem=12G",
        "--time=00:10:00",
        f"--output={CONTROL}/slurm/{mode}-%j.out",
        f"--error={CONTROL}/slurm/{mode}-%j.err",
        str(SOURCE / "mosaickv/slurm/fox_guard_launch_retry_controller.sbatch"),
        str(SOURCE),
        mode,
    ]
    if dependency:
        command.insert(1, f"--dependency=afterany:{dependency}")
    job = subprocess.check_output(command, text=True, timeout=30).strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("no Slurm job ID")
    write_once(CONTROL / f"submissions/{mode}-{job}.json", dict(job_id=job, command=command))
    return job


def assert_gpu_lease():
    from control_a100_schedule import gpu_count, read_queue

    timing = PREFIX_ROOT / "decisions/timing.json"
    allowed = {json.loads(timing.read_text())["gpu_array"]} if timing.exists() else set()
    gpus = [r for r in read_queue() if gpu_count(r["gres"]) > 0]
    if any(r["array"] not in allowed for r in gpus) or sum(gpu_count(r["gres"]) for r in gpus) > 2:
        raise ValueError("PrefixKV quality or unknown GPU work remains; retry cannot use four GPUs")


def run(mode):
    revision = clean_source(SOURCE)
    write_once(CONTROL / f"decisions/{mode}-started.json", dict(source_sha=revision, mode=mode))
    if mode == "cpu":
        cpu_suite(CONTROL / "verification")
        return
    cpu = json.loads((CONTROL / "verification/cpu-tests.json").read_text())
    if cpu["status"] != "passed" or cpu["source_sha"] != revision:
        raise ValueError("corrective clean-source CPU suite did not pass")
    if mode == "follow-fox":
        decision = ROOT / "decisions/after-canary.json"
        if decision.exists():
            final = schedule("report", json.loads(decision.read_text())["cpu_analysis"])
            write_once(CONTROL / "decisions/report-queued.json", dict(job_id=final))
        else:
            report()
        return
    if mode == "report":
        report()
        return
    if mode in {"after-references", "after-canary"}:
        path = PREFIX_ROOT / f"decisions/{mode}.json"
        if path.exists():
            state = json.loads(path.read_text())
            next_mode = "after-canary" if mode == "after-references" else "start"
            job = schedule(next_mode, state["cpu_callback"])
            write_once(
                CONTROL / f"decisions/{mode}.json",
                dict(next_mode=next_mode, job_id=job, parent_decision_sha256=sha(path)),
            )
            return
        blocked = PREFIX_ROOT / f"decisions/{mode}-blocked.json"
        if not blocked.exists():
            raise ValueError("missing PrefixKV terminal/next decision; do not guess dependency")
    assert_gpu_lease()
    # This assertion does not require PrefixKV quality to be positive. It only
    # proves the six-worker quality phase has ended and at most two timing tasks
    # remain. PrefixKV's negative result cannot change this already frozen policy.
    prepare(ROOT, cpu_gate=CONTROL / "verification/cpu-tests.json", after_prefix_calibration=True)
    job = submit(ROOT, "evidence_canary", "worker")
    callback = submit(ROOT, "evidence_canary", "analyze", job)
    write_once(
        ROOT / "decisions/start.json",
        dict(
            gpu_array=job,
            cpu_analysis=callback,
            max_fox_workers=4,
            max_prefix_timing_workers=2,
            no_prefix_source_or_running_jobs_changed=True,
        ),
    )
    write_once(
        CONTROL / "decisions/retry-submitted.json",
        dict(gpu_array=job, cpu_analysis=callback, root=str(ROOT), source_sha=revision),
    )
    dependencies = [callback]
    timing = PREFIX_ROOT / "decisions/timing.json"
    if timing.exists():
        dependencies.append(json.loads(timing.read_text())["cpu_callback"])
    final = schedule("follow-fox", ":".join(dependencies))
    write_once(
        CONTROL / "decisions/closing-analysis.json", dict(job_id=final, dependencies=dependencies)
    )


def report():
    """All outcomes, including failed gates and slower timings, remain visible."""
    import pandas as pd

    def optional(path):
        return (
            json.loads(path.read_text())
            if path.exists()
            else dict(status="missing", path=str(path))
        )

    prefix = optional(PREFIX_ROOT / "analysis/GATE.json")
    timing = optional(PREFIX_ROOT / "timing/analysis/report.json")
    fox = optional(ROOT / "analysis/installed_guard_development/GATE.json")
    canary = optional(ROOT / "analysis/evidence_canary/GATE.json")
    text = [
        "# October 29 frozen quality follow-up",
        "",
        "Retrospective development and technical timing; not independent confirmation.",
        "Original raw attempts and source SHAs remain unchanged.",
        "",
        "## PrefixKV native-metric calibration",
        "",
        json.dumps(prefix, indent=2),
    ]
    metrics = PREFIX_ROOT / "analysis/native_metrics.parquet"
    out = CONTROL / "report"
    out.mkdir(parents=True, exist_ok=False)
    if metrics.exists():
        frame = pd.read_parquet(metrics)
        text += ["", frame.to_string(index=False)]
        frame.to_latex(
            out / "prefix_native_metrics.tex",
            index=False,
            escape=True,
            na_rep="--",
            float_format="%.5f",
        )
    text += [
        "",
        "## Repeated timing",
        "",
        "One preregistered input per task, 20 repeats; not population-latency evidence.",
        "FullKV is excluded from compressed latency comparisons.",
        json.dumps(timing, indent=2),
        "",
        "## Fox installed-plan guard",
        "",
        json.dumps(canary, indent=2),
        json.dumps(fox, indent=2),
    ]
    stats_path = ROOT / "analysis/installed_guard_development/quality_statistics.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        text += ["", json.dumps(stats, indent=2)]
    text += [
        "",
        "## Claim restrictions",
        "",
        "PPL, ROUGE-L and CIDEr are never averaged together.",
        "Cross-budget points are not byte-matched; measured storage governs memory claims.",
        "Any slower timing is a measured trade-off, not suppressed.",
        "Independent confirmation, broad superiority, prototypes and repair remain unclaimed.",
    ]
    (out / "report.md").write_text("\n".join(text) + "\n")
    bindings = {
        str(p): sha(p)
        for p in [
            PREFIX_ROOT / "plan.json",
            PREFIX_ROOT / "analysis/GATE.json",
            PREFIX_ROOT / "timing/analysis/report.json",
            ROOT / "protocol_lock.json",
            stats_path,
        ]
        if p.exists()
    }
    write_once(
        out / "manifest.json",
        dict(
            analysis_source_sha=clean_source(SOURCE),
            bindings=bindings,
            source_producer_prefix=prefix.get("source_sha"),
            source_producer_fox=fox.get("source_sha"),
            independent_confirmation_executed=False,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "mode", choices=["cpu", "after-references", "after-canary", "start", "follow-fox", "report"]
    )
    a = p.parse_args()
    try:
        run(a.mode)
    except Exception as exc:
        write_once(
            CONTROL / f"failures/{a.mode}.json",
            dict(error=str(exc), status="blocked", source_sha=clean_source(SOURCE)),
        )
        raise
