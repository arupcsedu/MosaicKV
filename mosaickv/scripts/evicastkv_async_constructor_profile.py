#!/usr/bin/env python3
"""Attribute the remaining consumed-input construction gap, including fit thread."""

import cProfile
import pstats
import subprocess
import threading
import traceback
from pathlib import Path
from unittest.mock import patch

import evicastkv_async_text_native as base
from evicastkv_verified_constructor_profile import attribution

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.optimized_runtime_profile import collect
from mosaickv.evaluation.prefill_native_overlap import Pipeline

SOURCE, parent, driver = base.SOURCE, base.parent, base.driver
PREVIOUS = Path("/scratch/djy8hg/runs/evicastkv/async-text-native-consumed-20261106-launch-retry1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_async_constructor_profile.sbatch"
DOC = SOURCE / "mosaickv/docs/async_constructor_profile.md"
ARMS = base.ARMS[:3]
APPROVED = {
    "mosaickv/scripts/evicastkv_async_constructor_profile.py",
    "mosaickv/slurm/evicastkv_async_constructor_profile.sbatch",
    "mosaickv/docs/async_constructor_profile.md",
    "mosaickv/tests/unit/test_async_constructor_profile.py",
}


class FitProfile:
    """CPU-only observations owned by the existing private fit thread."""

    def __init__(self):
        self.profiler, self.thread, self.calls = None, None, 0

    def wrap(self, original):
        def fitted(instance, *args, **kwargs):
            identity = threading.get_ident()
            if self.thread is None:
                self.thread, self.profiler = identity, cProfile.Profile()
            elif identity != self.thread:
                raise RuntimeError("one original private fit thread required")
            self.calls += 1
            self.profiler.enable()
            try:
                return original(instance, *args, **kwargs)
            finally:
                self.profiler.disable()

        return fitted

    def finish(self, folder, *, expected):
        if self.calls != expected:
            raise RuntimeError("complete original background fit count required")
        if expected == 0:
            return dict(calls=0, threads=0, background_fit_ineligible=True)
        path = folder / "background-fit.prof"
        self.profiler.dump_stats(str(path))
        return dict(
            attribution(pstats.Stats(str(path)).stats, limit=60),
            calls=self.calls,
            threads=1,
            profile=str(path),
            profile_sha256=sha(path),
            background_thread_CPU_not_attributed=False,
            CPU_service_NOT_additive_to_request_wall=True,
            CUDA_kernel_execution_NOT_measured=True,
        )


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable output and clean source required")
    decision = driver.read(PREVIOUS / "analysis/decision.json")
    if not decision["complete"] or decision["failures"]:
        raise ValueError("complete preceding numerical and timing screen required")
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", decision["source_sha"], revision], cwd=SOURCE, text=True
    ).splitlines()
    if not set(changed) <= APPROVED:
        raise ValueError("attribution forbids validated runtime changes")
    old = driver.read(PREVIOUS / "plan.json")
    bound = dict(old["bound_files"])
    for path in (DOC, PREVIOUS / "plan.json", PREVIOUS / "analysis/decision.json"):
        bound[str(path)] = sha(path)
    for index in (0, 1):
        for method, arm in ARMS:
            path = PREVIOUS / f"workers/{index}/audited-{method}-{arm}.json"
            bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            arms=ARMS,
            stage="consumed_async_constructor_main_and_background_attribution",
            resolved_arm_overrides=[
                row for row in old["resolved_arm_overrides"] if (row["method"], row["arm"]) in ARMS
            ],
            correctness_only=False,
            warmups=0,
            repetitions=0,
            untimed_warmups_per_arm=1,
            profile_requests_per_arm=1,
            steady_profile_must_not_compile=True,
            background_fit_calls_per_eligible_arm=32,
            source_runtime_unchanged=True,
            CPU_service_times_must_not_be_added_to_wall_time=True,
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
        base.base.control.actual(at_worker_start=True)
        release = driver.read(root / "release.json")
        if (
            not release["profile_released"]
            or sha(Path(release["parent_decision"])) != release["parent_sha256"]
        ):
            raise ValueError("unchanged successful predecessor required")
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_async_constructor_main_and_background_attribution",
                methods=["mosaickv_exact", "prefixkv_reimpl"],
                actual_timing_protocol=dict(
                    profile_requests_per_arm=1,
                    untimed_warmups_per_arm=1,
                    output_predictions=128,
                    instrumentation_NOT_latency_acceptance=True,
                    CPU_service_times_must_not_be_added_to_wall_time=True,
                ),
            ),
        )
        records = base.base.configurations(
            driver.read(parent.PARENT / f"workers/{index}/study.json")["records"]
        )
        rows = []
        for method, arm in ARMS:
            folder = out / f"profile-{method}-{arm}"
            folder.mkdir()
            observed = None

            def call(method=method, arm=arm):
                nonlocal observed
                observed = base.execute(
                    adapter, refs, messages, profile, plan, method, arm, records[method, arm]
                )
                return observed

            call()  # Same warm count; never profile cold compilation.
            fits = FitProfile()
            with patch.object(Pipeline, "fit", fits.wrap(Pipeline.fit)):
                captured = collect(adapter, call, folder)
            write_once(folder / "profile.json", captured)
            phases = {}
            for phase, item in captured["phases"].items():
                path = Path(item["profile"])
                if sha(path) != item["profile_sha256"]:
                    raise ValueError("profile changed before attribution")
                phases[phase] = attribution(pstats.Stats(str(path)).stats, limit=60)
            background = fits.finish(folder, expected=32 if method == "mosaickv_exact" else 0)
            if any(p["jit_compilation"] for p in phases.values()) or background.get(
                "jit_compilation"
            ):
                raise RuntimeError("profile still contains cold compilation")
            anchor = driver.read(PREVIOUS / f"workers/{index}/audited-{method}-{arm}.json")
            if any(observed[key] != anchor[key] for key in driver.STORAGE[:2]):
                raise RuntimeError("instrumentation changed persistent physical cache bytes")
            rows.append(
                dict(
                    method=method,
                    arm=arm,
                    phases=phases,
                    background_fit=background,
                    initial_final_bytes={k: observed[k] for k in driver.STORAGE[:2]},
                    whole_request_GPU_peak_certified=False,
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
            CPU_service_times_must_not_be_added_to_wall_time=True,
            independent_quality_N=0,
            whole_request_GPU_peak_certified=False,
            no_automatic_fresh_or_quality_submission=True,
        ),
    )


def main():
    base.prepare, base.SCRIPT, base.DOC, base.ARMS = prepare, SCRIPT, DOC, ARMS
    base.base.base.ORIGINAL_WORKER, base.base.base.analyze = worker, analyze
    base.main()


if __name__ == "__main__":
    main()
