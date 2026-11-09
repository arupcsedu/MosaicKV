#!/usr/bin/env python3
"""One preregistered operator diagnostic and a separate CPU-only metadata audit."""

import argparse
import json
import os
import subprocess
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import prepared_address_study as parent

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

base, SOURCE = parent.base, parent.SOURCE
CONFIG = SOURCE / "mosaickv/configs/experiments/dispatch_affinity_diagnostic_20261109.json"
PARENT = Path("/scratch/djy8hg/runs/evicastkv/prepared-address-dispatch-20261108-v2")
WRAPPER = SOURCE / "mosaickv/slurm/dispatch_affinity_diagnostic.sbatch"
base.EXTRA_TESTS = (
    *base.EXTRA_TESTS,
    "test_dispatch_affinity_audit.py",
    "test_history_bookkeeping_audit.py",
)


def protocol():
    p = base.common.read(CONFIG)
    fixed = dict(
        warmups=5,
        timed_repetitions=20,
        launches_per_burst=50,
        bootstrap_replicates=10000,
        gate_family_comparisons=4,
        strict_adjusted_upper_wall_ratio=1.0,
        workers=2,
    )
    if any(p.get(k) != v for k, v in fixed.items()):
        raise ValueError("original fixed timing gate and sample counts required")
    for key in (
        "model_execution_authorized",
        "candidate_combination_authorized",
        "candidate_promotion_authorized",
        "fresh_cohort_authorized",
        "outlier_removal_authorized",
        "repeat_until_pass_authorized",
    ):
        if p.get(key) is not False:
            raise ValueError("operator/profile diagnostic only; no promotion or selective repeats")
    return p


def submit(root, action, after=None, stage=None):
    gpu = action == "operator"
    args = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        "--job-name=mk-affinity-" + action,
    ]
    if gpu:
        args += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--time=00:10:00",
            "--array=0-1%2",
            "--hold",
        ]
    else:
        mem, minutes = (
            ("16G", 15)
            if action == "cpu"
            else (("12G", 10) if action == "cpu_profile" else ("4G", 5))
        )
        args += [
            "--partition=bii",
            "--cpus-per-task=2",
            "--mem=" + mem,
            f"--time=00:{minutes:02}:00",
        ]
    if after:
        args += ["--dependency=afterany:" + after]
    args += [
        f"--output={root}/logs/{action}-%A_%a.out",
        f"--error={root}/logs/{action}-%A_%a.err",
        str(WRAPPER),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        action,
        stage or "none",
    ]
    return subprocess.check_output(args, text=True).strip()


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable diagnostic root required")
    spec = protocol()
    p = base.common.read(PARENT / "plan.json")
    paths = [CONFIG, PARENT / "plan.json", PARENT / "operator-decision.json"]
    paths += [PARENT / f"operator/{i}/result.json" for i in (0, 1)]
    p["bound_files"].update({str(path): sha(path) for path in paths})
    history_path = SOURCE / "mosaickv/src/mosaickv/evaluation/history_value_first.py"
    frozen = subprocess.check_output(
        [
            "git",
            "show",
            spec["history_value_first_source_sha"] + ":" + str(history_path.relative_to(SOURCE)),
        ],
        cwd=SOURCE,
    )
    import hashlib

    if sha(history_path) != hashlib.sha256(frozen).hexdigest():
        raise ValueError("CPU audit must profile the unchanged frozen scheduling implementation")
    p.update(
        source_sha=clean_source(SOURCE),
        preregistered_at_utc=datetime.now(timezone.utc).isoformat(),
        exploratory_remeasurement_NOT_confirmatory_quality_evidence=True,
        protocol=spec,
        study=spec["study"],
        measurement_type=spec["measurement_type"],
        frozen_history_value_first_sha256=sha(history_path),
        old_negative_result_preserved_and_NOT_pooled=True,
        model_execution_authorized=False,
        fresh_or_paper_campaign_authorized=False,
        independent_quality_validation_authorized=False,
    )
    root.mkdir()
    (root / "logs").mkdir()
    write_once(root / "plan.json", p)
    cpu = submit(root, "cpu")
    follow = submit(root, "after_cpu", cpu)
    profile_job = submit(root, "cpu_profile", cpu)
    write_once(
        root / "submission.json",
        dict(
            cpu=cpu,
            follow=follow,
            cpu_profile=profile_job,
            cpu_profile_analysis=submit(root, "analyze_cpu_profile", profile_job),
            model_and_candidate_promotion_disabled=True,
        ),
    )
    print(json.dumps(base.common.read(root / "submission.json")), flush=True)


def validate_trials(rows):
    expected = Counter((rep, flag) for rep in range(-5, 20) for flag in (False, True))
    actual = Counter((r["repetition"], r["raw_addresses"]) for r in rows)
    if actual != expected or any(r["warmup"] != (r["repetition"] < 0) for r in rows):
        raise ValueError("retain every unique preregistered warmup and timed observation")


def operator(root, p, folder, i):
    import numpy as np
    import torch
    from compact_hybrid_study import manifest

    from mosaickv.evaluation.dispatch_affinity_audit import enqueue, pinned_thread
    from mosaickv.evaluation.exact_storage import encode
    from mosaickv.evaluation.fused_history_unpack import launch
    from mosaickv.evaluation.fused_history_unpack import use as fused
    from mosaickv.evaluation.history_source_precision import pack_source
    from mosaickv.evaluation.prepared_address_dispatch import gpu_controls
    from mosaickv.evaluation.prepared_address_dispatch import use as addresses

    rows = []
    with pinned_thread() as affinity:
        manifest(p, sha(root / "plan.json"), folder, i, gpu=True)
        controls = gpu_controls(backend="cpp")
        recent = tuple(
            SimpleNamespace(
                shape=(1, 32, 16, 128),
                archive=SimpleNamespace(
                    source=encode(torch.randn(1, 1, 16, 4096, device="cuda").bfloat16())
                ),
            )
            for _ in range(4)
        )
        older = SimpleNamespace(
            shape=(1, 32, 48, 128),
            body=pack_source(torch.randn(1, 1, 48, 4096, device="cuda").bfloat16(), group=32),
        )

        @dataclass
        class Packet:
            shape: tuple
            recent: tuple
            older: object

        packet = Packet((1, 32, 112, 128), recent, older)
        target = torch.empty(128, 4096, device="cuda", dtype=torch.bfloat16)
        rng = np.random.default_rng(p["seed"] + i)
        with (folder / "trials.jsonl").open("x") as output, fused():
            launch(packet, target)
            expected = target[:112].clone()
            for rep in range(-5, 20):
                for flag in rng.permutation([False, True]):
                    with addresses(bool(flag), backend="cpp") as report:
                        torch.cuda.synchronize()
                        a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        a.record()
                        row = enqueue(lambda: launch(packet, target), 50)
                        b.record()
                        torch.cuda.synchronize()
                        row.update(CUDA_stream_seconds_per_launch=a.elapsed_time(b) / 50000)
                    row.update(
                        repetition=rep,
                        warmup=rep < 0,
                        raw_addresses=bool(flag),
                        dispatch=report,
                        affinity=affinity,
                        CUDA_interval_includes_host_gaps_and_clock_instrumentation=True,
                        words_bitwise=bool(
                            torch.equal(expected.view(torch.int16), target[:112].view(torch.int16))
                        ),
                    )
                    output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    output.flush()
                    rows.append(row)
                    if not row["words_bitwise"]:
                        raise RuntimeError("same binary changed an original source word")
        validate_trials(rows)
        stats = {}
        for metric in ("host_enqueue_seconds_per_launch", "CUDA_stream_seconds_per_launch"):
            values = [
                [r[metric] for r in rows if not r["warmup"] and r["raw_addresses"] == flag]
                for flag in (False, True)
            ]
            stats[metric] = parent.statistics.paired(
                values, seed=p["seed"] + i, replicates=10000, family_comparisons=4
            )
        diagnostics = []
        for flag in (False, True):
            local = [r for r in rows if not r["warmup"] and r["raw_addresses"] == flag]
            diagnostics.append(
                dict(
                    raw_addresses=flag,
                    thread_CPU_mean=float(
                        np.mean([r["thread_CPU_seconds_per_launch"] for r in local])
                    ),
                    largest_wall_minus_thread=max(
                        r["wall_minus_thread_seconds_per_launch"] for r in local
                    ),
                    voluntary_context_switches=sum(r["voluntary_context_switches"] for r in local),
                    involuntary_context_switches=sum(
                        r["involuntary_context_switches"] for r in local
                    ),
                    observations=local,
                )
            )
        passed = stats["host_enqueue_seconds_per_launch"]["equal_or_faster"]
        write_once(
            folder / "result.json",
            dict(
                status="completed",
                passed=passed,
                correctness_passed=True,
                paired=stats,
                diagnostics=diagnostics,
                trials_path=str(folder / "trials.jsonl"),
                trials_sha256=sha(folder / "trials.jsonl"),
                affinity=affinity,
                controls=controls,
                all_observations_retained=True,
                thread_time_did_NOT_replace_wall_gate=True,
                model_execution_or_promotion_authorized=False,
            ),
        )


def analyze(root, p):
    results, errors = [], []
    for i in (0, 1):
        path = root / f"operator/{i}/result.json"
        r = base.common.read(path) if path.exists() else dict(status="missing")
        if r.get("status") != "completed":
            errors.append(dict(input=i, error=r))
        else:
            trials = Path(r["trials_path"])
            if sha(trials) != r["trials_sha256"]:
                raise ValueError("immutable raw diagnostic trials changed")
            validate_trials([json.loads(x) for x in trials.read_text().splitlines()])
        results.append(dict(input=i, **r))
    folder = root / "analysis"
    folder.mkdir()
    complete = not errors
    passed = complete and all(r.get("passed") is True for r in results)
    write_once(
        folder / "decision.json",
        dict(
            complete=complete,
            strict_operator_gate_passed=passed,
            results=results,
            errors=errors,
            historical_negative_result_NOT_overwritten_or_pooled=True,
            fresh_or_model_execution_authorized=False,
            candidate_combined_or_promoted=False,
            end_to_end_or_quality_superiority_established=False,
        ),
    )
    lines = [
        "# CPU-affined C++ dispatch diagnostic",
        "",
        "Operator measurements only; all observations retained, no promotion.",
        "",
        "| Input | Wall ratio | Adjusted CI | Strict gate |",
        "|---|---|---|---|",
    ]
    for r in results:
        if r.get("status") == "completed":
            s = r["paired"]["host_enqueue_seconds_per_launch"]
            lines.append(
                f"| {r['input']} | {s['mean_latency_ratio']:.6f} | "
                f"{s['ratio_adjusted_CI']} | {r['passed']} |"
            )
    lines += [
        "",
        "Thread-time and context-switch observations diagnose off-CPU intervals; "
        "they do not prove their cause or replace the wall-time gate.",
    ]
    with (folder / "report.md").open("x") as out:
        out.write("\n".join(lines) + "\n")


def cpu_profile(root, p):
    from mosaickv.evaluation.history_bookkeeping_audit import profile

    folder = root / "cpu-profile"
    try:
        if base.common.read(root / "cpu.json")["status"] != "completed":
            raise ValueError("canonical CPU regression suite did not pass")
        report = profile(
            folder,
            transactions=p["protocol"]["cpu_profile_transactions_per_mode"],
            warmups=p["protocol"]["cpu_profile_warmups"],
        )
        import importlib.metadata

        import torch

        fixture_path = SOURCE / "mosaickv/src/mosaickv/evaluation/history_bookkeeping_audit.py"
        write_once(
            folder / "manifest.json",
            dict(
                git_sha=p["source_sha"],
                config_sha=sha(root / "plan.json"),
                source_dirty=False,
                model=dict(id="not_used", revision="not_used"),
                dataset=dict(id="synthetic_metadata_fixture", revision=sha(fixture_path)),
                CUDA="not_used",
                driver="not_used",
                torch=str(torch.__version__),
                transformers=importlib.metadata.version("transformers"),
                vllm="not_used",
                sglang="not_used",
                gpu="not_used",
                gpu_count=0,
                backend="CPU_metadata_mock",
                attention_implementation="not_used",
                seed=p["seed"],
                measurement_type=report["measurement_type"],
                frozen_schedule_sha256=p["frozen_history_value_first_sha256"],
                environment=dict(
                    name="common",
                    cache_root="/scratch/djy8hg/cache/mosaickv",
                    lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
                ),
            ),
        )
        report["manifest_path"] = str(folder / "manifest.json")
        report["manifest_sha256"] = sha(folder / "manifest.json")
        write_once(folder / "result.json", dict(status="completed", **report))
    except Exception:
        folder.mkdir(exist_ok=True)
        write_once(folder / "result.json", dict(status="failed", error=traceback.format_exc()))
        raise


def analyze_profile(root, p):
    path = root / "cpu-profile/result.json"
    r = base.common.read(path) if path.exists() else dict(status="missing")
    out = root / "cpu-profile-analysis"
    out.mkdir()
    write_once(
        out / "decision.json",
        dict(
            complete=r["status"] == "completed",
            producer=r,
            source_sha=p["source_sha"],
            frozen_schedule_sha256=p["frozen_history_value_first_sha256"],
            GPU_execution_or_combination_or_promotion_authorized=False,
        ),
    )
    lines = [
        "# CPU-only frozen History-V-first bookkeeping audit",
        "",
        "GPU/projector work is mocked. These are not model/GPU latency measurements; "
        "CUDA stream-query overhead is also absent. No implementation is changed.",
        "",
    ]
    if r["status"] == "completed":
        for arm in r["results"]:
            lines += [
                f"## Enabled: {arm['enabled']}",
                "",
                "| Function | Calls | Self seconds | Cumulative seconds |",
                "|---|---|---|---|",
            ]
            for f in arm["functions"][:20]:
                lines.append(
                    f"| {f['file']}:{f['line']} {f['function']} | {f['calls']} | "
                    f"{f['self_seconds']:.6f} | {f['cumulative_seconds']:.6f} |"
                )
            lines += [""]
    else:
        lines.append("Profile incomplete: " + r.get("error", "missing producer artifact"))
    with (out / "report.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "cpu",
            "after_cpu",
            "operator",
            "after_operator",
            "controller",
            "cpu_profile",
            "analyze_cpu_profile",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    parser.add_argument("--stage")
    args = parser.parse_args()
    base.common.submit = submit
    if args.action == "prepare":
        prepare(args.root)
        return
    p = base.common.checked(args.root, args.plan_sha)
    if args.action == "cpu":
        base.cpu(args.root, p)
    elif args.action == "after_cpu":
        if (args.root / "cpu.json").exists() and base.common.read(args.root / "cpu.json")[
            "status"
        ] == "completed":
            base.common.launch(args.root, "operator")
        else:
            write_once(args.root / "CPU-negative.json", dict(no_GPU_submission=True))
    elif args.action == "operator":
        import torch

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(p["seed"])
        i = int(os.environ["SLURM_ARRAY_TASK_ID"])
        folder = args.root / f"operator/{i}"
        folder.mkdir(parents=True)
        try:
            operator(args.root, p, folder, i)
        except Exception:
            write_once(folder / "result.json", dict(status="failed", error=traceback.format_exc()))
            raise
    elif args.action == "after_operator":
        analyze(args.root, p)
    elif args.action == "cpu_profile":
        cpu_profile(args.root, p)
    elif args.action == "analyze_cpu_profile":
        analyze_profile(args.root, p)
    else:
        # One admission decision is sufficient for this fixed, two-task diagnostic;
        # released pending tasks count toward the global six-GPU safety limit.
        subprocess.run(
            [
                base.PYTHON,
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(args.root / "admission-operator.json"),
                "--state",
                str(args.root / "admission-operator-state.json"),
                "--apply",
            ],
            check=True,
            cwd=SOURCE,
        )


if __name__ == "__main__":
    main()
