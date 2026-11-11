#!/usr/bin/env python3
"""Bounded provider recovery, provenance proofs and matched-source engineering."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import time
from pathlib import Path

import evicastkv_review_v8_diagnostics as shared

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
V12 = Path("/scratch/djy8hg/workdir/fox-review-v12/run-20261111-1")
V13 = Path("/scratch/djy8hg/workdir/fox-review-v13/run-20261111-1")
BROAD_RULES = Path(
    "/scratch/djy8hg/workdir/fox-source-rules-draft-v2-20261111-1/broad_rules_draft.json"
)
MODULES = {
    "fetch": ("evicastkv_historical_provider_fetch", "GATE_historical_provider_fetch.json"),
    "providers": (
        "evicastkv_historical_provider_superset",
        "GATE_historical_provider_superset.json",
    ),
    "derived": ("evicastkv_derived_artifact_proofs", "GATE_derived_artifact_proofs.json"),
    "timing": ("evicastkv_lossless_streaming_timing", "GATE_lossless_streaming_timing.json"),
    "broad": ("evicastkv_source_trace_adjudication", "GATE_source_trace_adjudication.json"),
}


def read(path):
    return json.loads(Path(path).read_text())


def module(lane):
    return importlib.import_module(MODULES[lane][0])


def load(root):
    plan = read(root / "registration.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["execute"] is not True or plan["paper_campaign_authorized"] is not False:
        raise ValueError("Only finite engineering/metadata scope is executable")
    if any(sha(Path(p)) != value for p, value in plan["bindings"].items()):
        raise ValueError("Immutable follow-up evidence changed")
    return plan


def submit(
    root,
    label,
    mode,
    *,
    dependencies=(),
    gpu=False,
    array=False,
    tests=False,
    dependency_resolution=None,
):
    if (root / f"jobs/{label}-request.json").exists():
        raise FileExistsError("Do not retry an attempted scheduler request blindly")
    retained = list(dict.fromkeys(shared.identity(d) for d in dependencies))
    numeric = mode == "after-timing"
    cpus = 12 if gpu or tests or numeric else 1 if array or mode == "fetch-worker" else 4
    minutes = 60 if gpu else 20 if tests else 5 if array else 15
    cmd = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name=fox14-{label}",
        f"--cpus-per-task={cpus}",
        "--mem=64G" if gpu else "--mem=32G" if mode == "broad-worker" else "--mem=16G",
        f"--time={minutes // 60:02d}:{minutes % 60:02d}:00",
        f"--output={root}/slurm/{label}-%A_%a.out",
        f"--error={root}/slurm/{label}-%A_%a.err",
        "--partition=bii-gpu" if gpu else "--partition=bii,standard",
    ]
    if gpu:
        cmd += ["--reservation=bi_fox_dgx", "--gres=gpu:a100:1"]
    if array:
        cmd += ["--array=0-1%2" if gpu else "--array=0-7%8"]
    if retained:
        cmd += ["--dependency=afterany:" + ":".join(retained)]
    launcher = "evicastkv_review_closure_v2" if tests else "evicastkv_review_v14"
    cmd += [
        str(SOURCE / f"mosaickv/slurm/{launcher}.sbatch"),
        str(SOURCE),
        str(root / "full-cpu") if tests else str(root),
        "tests" if tests else mode,
    ]
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(
        root / f"jobs/{label}-request.json",
        dict(command=cmd, dependency_resolution=dependency_resolution, automatic_retry=False),
    )
    try:
        job = shared.identity(subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE))
    except Exception as error:
        write_once(
            root / f"jobs/{label}-submission-error.json",
            dict(error=f"{type(error).__name__}: {error}", automatic_retry=False),
        )
        raise shared.SubmissionError("Failed or unacknowledged V14 submission") from error
    write_once(
        root / f"jobs/{label}-submission.json",
        dict(job=job, command=cmd, acknowledged_at_unix_seconds=time.time()),
    )
    return job


def register(root, *, execute=False):
    root = Path(root).resolve()
    allowed = Path("/scratch/djy8hg/workdir").resolve()
    if root.exists() or root == allowed or not root.is_relative_to(allowed):
        raise ValueError("New write-once scratch output root required")
    revision = clean_source(SOURCE)
    _, parent_bindings = module("timing").require_parent(V13 / "streaming")
    paths = [
        Path(__file__),
        SOURCE / "mosaickv/env/common/requirements.lock",
        SOURCE / "mosaickv/scripts/evicastkv_completion_v14.py",
    ]
    paths += [SOURCE / f"mosaickv/scripts/{name}.py" for name, _ in MODULES.values()]
    paths += [
        BROAD_RULES,
        V13 / "ownership/worker/result.json",
        V13 / "ownership/worker/trace_pre_attribution_receipt.json",
    ]
    selection = V12 / "frontier-selection.json"
    selected = read(selection)
    parent = Path(selected["path"]).resolve()
    if (
        not 1 <= selected["index"] <= 4
        or parent
        != (
            V12 / ("frontier" if selected["index"] == 1 else f"frontier-{selected['index']}")
        ).resolve()
        or sha(parent / "GATE_freshness_empty_parquet_frontier.json") != selected["gate_sha256"]
    ):
        raise ValueError("Exact final state-bearing V12 frontier required")
    paths.append(selection)
    bindings = dict(parent_bindings)
    bindings.update({str(p): sha(p) for p in paths})
    write_once(
        root / "registration.json",
        dict(
            source_sha=revision,
            source_root=str(SOURCE),
            execute=execute,
            bindings=bindings,
            derived_parent=str(parent),
            maximum_a100s=6,
            maximum_new_gpu_workers=2,
            six_a100s_explicitly_authorized=True,
            consumed_timing_inputs_only=True,
            no_benchmark_label_access=True,
            all_model_guards_unchanged=True,
            paper_campaign_authorized=False,
            confirmation_authorized=False,
        ),
    )
    if not execute:
        return dict(status="dry_run", jobs=0)
    cpu = submit(root, "cpu", "tests", tests=True)
    return dict(cpu=cpu, prepare=submit(root, "prepare", "prepare", dependencies=[cpu]))


def current():
    return [shared.identity(os.environ["SLURM_JOB_ID"])] if os.environ.get("SLURM_JOB_ID") else []


def prepare(root):
    plan = load(root)
    try:
        write_once(root / "GATE_cpu_suite.json", shared.cpu_gate(root, plan["source_sha"]))
    except Exception as error:
        shared.failure(root, "cpu", error)
        return finish(root)
    static = []
    for lane in ("derived", "timing", "broad"):
        try:
            kwargs = dict(cpu_gate_root=root)
            if lane == "derived":
                kwargs.update(
                    parent=plan["derived_parent"],
                    parent_gate_filename="GATE_freshness_empty_parquet_frontier.json",
                )
            if lane != "broad":
                module(lane).prepare(root / lane, **kwargs)
            dependencies = []
            if lane == "timing":
                snapshot = shared.gpu_snapshot()
                dependencies = sorted(
                    {r["job"].split("_")[0] for r in snapshot["active"] + snapshot["pending"]}
                )
                write_once(
                    root / "GPU-admission.json",
                    dict(
                        snapshot=snapshot,
                        maximum_a100s=6,
                        maximum_new_gpu_workers=2,
                        wait_for_existing=dependencies,
                    ),
                )
            job = submit(
                root,
                lane,
                f"{lane}-worker",
                dependencies=dependencies,
                gpu=lane == "timing",
                array=lane != "broad",
            )
            static.append(job)
            static.append(submit(root, f"after-{lane}", f"after-{lane}", dependencies=[job]))
        except Exception as error:
            shared.failure(root, f"prepare-{lane}", error)
    # Persist these IDs before the fetch can finish or reuse cached bodies.
    write_once(root / "static-dependencies.json", dict(jobs=static))
    try:
        rescue = importlib.import_module("evicastkv_completion_v14")
        write_once(
            root / "V13-report-rescue-registration.json",
            rescue.register(
                root / "V13-report-rescue",
                parent_campaign=V13,
                source=Path("/scratch/djy8hg/workdir/fox-review-v13-code"),
                execute=True,
                cpu_gate_root=root,
            ),
        )
    except Exception as error:
        shared.failure(root, "V13-report-rescue-registration", error)
    try:
        module("fetch").prepare(root / "fetch", cpu_gate_root=root)
        job = submit(root, "fetch", "fetch-worker")
        return dict(
            fetch=job,
            analysis=submit(root, "after-fetch", "after-fetch", dependencies=[job, *current()]),
        )
    except Exception as error:
        shared.failure(root, "prepare-fetch", error)
        return submit_final(root)


def submit_final(root):
    # Include every acknowledged producer, even if analyzer submission failed.
    # Do not query newly acknowledged IDs/current callback before the database
    # necessarily knows them. Only aged dependencies require reconciliation.
    candidates = set(read(root / "static-dependencies.json")["jobs"])
    fresh = set(current())
    now = time.time()
    for path in sorted((root / "jobs").glob("*-submission.json")):
        if path.name in {"cpu-submission.json", "prepare-submission.json"}:
            continue
        saved = read(path)
        candidates.add(saved["job"])
        age = now - saved.get("acknowledged_at_unix_seconds", float("-inf"))
        if 0 <= age < 240:
            fresh.add(saved["job"])
    aged = sorted(candidates - fresh)
    retained, receipt = ([], None)
    if aged:
        completion = importlib.import_module("evicastkv_completion_v14")
        retained, receipt = completion.resolve_dependencies(aged)
    resolution = dict(
        aged_resolution=receipt,
        fresh_acknowledged_or_current=sorted(fresh),
        all_acknowledged_candidates=sorted(candidates),
    )
    write_once(root / "final-dependency-resolution.json", resolution)
    return dict(
        final=submit(
            root,
            "final",
            "finish",
            dependencies=sorted(set(retained) | fresh),
            dependency_resolution=resolution,
        )
    )


def after(root, lane):
    try:
        if lane == "broad":
            read(root / lane / MODULES[lane][1])
        else:
            module(lane).analyze(root / lane)
    except Exception as error:
        shared.failure(root, f"analyze-{lane}", error)
    if lane == "fetch":
        path = root / lane / MODULES[lane][1]
        if not path.is_file() or read(path).get("status") != "pinned_download_complete":
            return submit_final(root)
        try:
            module("providers").prepare(root / "providers", cpu_gate_root=root)
            job = submit(root, "providers", "providers-worker", array=True, dependencies=current())
            return dict(
                providers=job,
                analysis=submit(root, "after-providers", "after-providers", dependencies=[job]),
            )
        except Exception as error:
            shared.failure(root, "prepare-providers", error)
            return submit_final(root)
    if lane == "providers":
        return submit_final(root)
    return dict(lane=lane, analysis_finished=True)


def finish(root):
    plan = load(root)
    gates = {}
    for lane, (_, name) in MODULES.items():
        path = root / lane / name
        gates[lane] = (
            dict(path=str(path), sha256=sha(path), evidence=read(path))
            if path.is_file()
            else dict(status="missing_gate", path=str(path))
        )
    unacknowledged = [
        str(p)
        for p in sorted((root / "jobs").glob("*-submission-error.json"))
        if not p.with_name(p.name.replace("-submission-error", "-submission")).exists()
    ]
    report = dict(
        status="blocked_unacknowledged_submission"
        if unacknowledged
        else "completed_finite_followup_not_paper_confirmation",
        source_sha=plan["source_sha"],
        gates=gates,
        errors={p.name: read(p) for p in sorted((root / "errors").glob("*.json"))},
        independent_quality_N=0,
        fresh_confirmation_established=False,
        nearest_baseline_superiority_established=False,
        six_component_benefits_established=False,
        paper_campaign_authorized=False,
        unacknowledged_requests=unacknowledged,
    )
    write_once(root / "FINAL_report.json", report)
    return report


def dispatch(root, mode):
    root = Path(root).resolve()
    load(root)
    if mode == "prepare":
        return prepare(root)
    if mode == "finish":
        return finish(root)
    for lane in MODULES:
        if mode == f"{lane}-worker":
            if lane == "timing" and shared.gpu_snapshot()["active_gpus"] > 6:
                raise ValueError("Aggregate six-A100 ceiling exceeded")
            if lane == "broad":
                return module(lane).run(
                    root / lane, parent=V13 / "ownership", reviewed_rules=BROAD_RULES
                )
            return module(lane).worker(root / lane, int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
        if mode == f"after-{lane}":
            return after(root, lane)
    raise ValueError("Unknown finite V14 mode")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = (
        register(args.root, execute=args.execute)
        if args.mode == "register"
        else dispatch(args.root, args.mode)
    )
    print(json.dumps(result, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
