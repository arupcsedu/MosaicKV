#!/usr/bin/env python3
"""Finite source-ownership and historical-evidence repair, never confirmation.

Each lane produces an explicit terminal receipt, including failures. The last
terminal callback submits one report; no log polling or GPU retry loop exists.
"""

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
V14 = Path("/scratch/djy8hg/workdir/fox-review-v14/run-20261111-2")
RECOVERED = Path("/scratch/djy8hg/workdir/fox-trace-recovery-20261111-1")
RULES = Path(
    "/scratch/djy8hg/workdir/fox-kv-only-rules-draft-v4-20261111-1/kv_only_rules_draft.json"
)
LANES = ("ownership", "providers", "derived", "legacy-kv", "history", "completion")
MODULES = {
    "ownership": "evicastkv_streaming_lossless_ownership",
    "providers": "evicastkv_historical_provider_recovery",
    "derived": "evicastkv_derived_artifact_proofs",
    "legacy-kv": "evicastkv_kv_only_refinement",
    "history": "evicastkv_history_segment_recovery",
    "completion": "evicastkv_completion_v14",
}
GATES = {
    "ownership": "GATE_streaming_lossless_ownership.json",
    "providers": "GATE_historical_provider_recovery.json",
    "derived": "GATE_derived_artifact_proofs.json",
    "legacy-kv": "GATE_kv_only_refinement.json",
    "history": "GATE_history_segment_recovery.json",
    "completion": "GATE_completion_v14.json",
}


def read(path):
    return json.loads(Path(path).read_text())


def module(lane):
    return importlib.import_module(MODULES[lane])


def load(root):
    plan = read(root / "registration.json")
    clean_source(SOURCE, plan["source_sha"])
    if (
        plan["execute"] is not True
        or plan["paper_campaign_authorized"] is not False
        or plan["confirmation_authorized"] is not False
    ):
        raise ValueError("Only finite consumed-input and metadata repairs are executable")
    if any(sha(Path(p)) != value for p, value in plan["bindings"].items()):
        raise ValueError("Immutable repair evidence changed")
    return plan


def submit(root, label, mode, *, dependencies=(), array=None, gpu=False, tests=False):
    request = root / f"jobs/{label}-request.json"
    if request.exists():
        raise FileExistsError("Do not repeat a failed/unacknowledged scheduler request")
    cpus = (
        12 if gpu or tests or mode in ("legacy-kv-worker", "after-ownership") else 1 if array else 4
    )
    minutes = (
        20
        if gpu or tests
        else 15
        if mode in ("legacy-kv-worker", "after-ownership", "history-worker")
        else 5
        if array
        else 10
    )
    cmd = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name=fox15-{label}",
        f"--cpus-per-task={cpus}",
        "--mem=64G"
        if gpu
        else "--mem=32G"
        if mode in ("legacy-kv-worker", "after-ownership")
        else "--mem=16G",
        f"--time={minutes // 60:02d}:{minutes % 60:02d}:00",
        f"--output={root}/slurm/{label}-%A_%a.out",
        f"--error={root}/slurm/{label}-%A_%a.err",
        "--partition=bii-gpu" if gpu else "--partition=bii,standard",
    ]
    if gpu:
        cmd += ["--reservation=bi_fox_dgx", "--gres=gpu:a100:1"]
    if array:
        if array not in ("0-1%2", "0-7%8") or (gpu and array != "0-1%2"):
            raise ValueError("Only registered finite two/eight-worker arrays are allowed")
        cmd.append(f"--array={array}")
    if dependencies:
        cmd.append(
            "--dependency=afterany:" + ":".join(sorted({shared.identity(d) for d in dependencies}))
        )
    launcher = "evicastkv_review_closure_v2" if tests else "evicastkv_review_v15"
    cmd += [
        str(SOURCE / f"mosaickv/slurm/{launcher}.sbatch"),
        str(SOURCE),
        str(root / "full-cpu") if tests else str(root),
        "tests" if tests else mode,
    ]
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(request, dict(command=cmd, automatic_retry=False))
    try:
        job = shared.identity(subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE))
    except Exception as error:
        write_once(
            root / f"jobs/{label}-submission-error.json",
            dict(
                error=f"{type(error).__name__}: {error}",
                stderr=str(getattr(error, "stderr", "")),
                automatic_retry=False,
            ),
        )
        raise shared.SubmissionError(
            "V15 scheduler submission failed or was unacknowledged"
        ) from error
    write_once(
        root / f"jobs/{label}-submission.json",
        dict(job=job, command=cmd, acknowledged_at_unix_seconds=time.time()),
    )
    return job


def register(root, *, execute=False):
    root = Path(root).resolve()
    allowed = Path("/scratch/djy8hg/workdir").resolve()
    if root.exists() or root == allowed or not root.is_relative_to(allowed):
        raise ValueError("New immutable scratch root required")
    revision = clean_source(SOURCE)
    selection = read(V12 / "frontier-selection.json")
    frontier = Path(selection["path"]).resolve(strict=True)
    expected = V12 / ("frontier" if selection["index"] == 1 else f"frontier-{selection['index']}")
    gate = frontier / "GATE_freshness_empty_parquet_frontier.json"
    if (
        not 1 <= selection["index"] <= 4
        or frontier != expected.resolve()
        or sha(gate) != selection["gate_sha256"]
    ):
        raise ValueError("Exact final V12 state-bearing frontier required")
    paths = [
        Path(__file__),
        SOURCE / "mosaickv/slurm/evicastkv_review_v15.sbatch",
        SOURCE / "mosaickv/env/common/requirements.lock",
        V12 / "frontier-selection.json",
        gate,
        V14 / "providers/plan.json",
        V14 / "providers/GATE_historical_provider_superset.json",
        Path("/scratch/djy8hg/workdir/fox-review-v13/run-20261111-1/FINAL_report.json"),
        RULES,
        RECOVERED / "recovery.json",
    ]
    paths += [SOURCE / f"mosaickv/scripts/{name}.py" for name in MODULES.values()]
    write_once(
        root / "registration.json",
        dict(
            source_sha=revision,
            source_root=str(SOURCE),
            execute=execute,
            bindings={str(p): sha(p) for p in paths},
            derived_parent=str(frontier),
            maximum_derived_passes=3,
            maximum_a100s=6,
            maximum_new_gpu_workers=2,
            six_a100s_explicitly_authorized=True,
            no_fresh_samples=True,
            guards_unchanged=True,
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


def terminal(root, lane, *, path=None, error=None):
    if lane not in LANES:
        raise ValueError("Unknown finite repair lane")
    write_once(
        root / f"terminal/{lane}.json",
        dict(
            lane=lane,
            gate_path=str(path) if path else None,
            gate_sha256=sha(path) if path and path.is_file() else None,
            error=f"{type(error).__name__}: {error}" if error else None,
            scientific_pass_not_implied=True,
        ),
    )


def barrier(root):
    if not (root / "preparation-complete.json").is_file() or not all(
        (root / f"terminal/{lane}.json").is_file() for lane in LANES
    ):
        return dict(status="waiting_for_terminal_callbacks_no_polling")
    claim = root / "final-submission-claim.json"
    try:
        write_once(claim, dict(callback_job=current(), all_lanes_terminal=True))
    except FileExistsError:
        return dict(status="final_already_claimed_no_duplicate")
    # Even when submitting an analyzer failed, its acknowledged producer must
    # finish before the report. Resolve only aged IDs; scheduler accounting can
    # lag a fresh acknowledgement or the currently executing callback.
    candidates, fresh = set(current()), set(current())
    now = time.time()
    for path in sorted((root / "jobs").glob("*-submission.json")):
        saved = read(path)
        if path.name in {"cpu-submission.json", "prepare-submission.json"}:
            continue
        candidates.add(saved["job"])
        age = now - saved.get("acknowledged_at_unix_seconds", float("-inf"))
        if 0 <= age < 240:
            fresh.add(saved["job"])
    aged = sorted(candidates - fresh)
    retained, receipt = ([], None)
    if aged:
        retained, receipt = importlib.import_module(
            "evicastkv_completion_v14"
        ).resolve_dependencies(aged)
    write_once(
        root / "final-dependency-resolution.json",
        dict(
            aged_resolution=receipt,
            fresh_acknowledged_or_current=sorted(fresh),
            all_acknowledged_candidates=sorted(candidates),
        ),
    )
    return dict(final=submit(root, "final", "finish", dependencies=sorted(set(retained) | fresh)))


def launch_lane(root, lane, *, label=None):
    label = label or lane
    job = submit(
        root,
        label,
        label + "-worker",
        array="0-1%2"
        if lane in ("ownership", "providers")
        else "0-7%8"
        if lane == "derived"
        else None,
        gpu=lane == "ownership",
    )
    return submit(root, "after-" + label, "after-" + label, dependencies=[job, *current()])


def prepare(root):
    plan = load(root)
    try:
        write_once(root / "GATE_cpu_suite.json", shared.cpu_gate(root, plan["source_sha"]))
    except Exception as error:
        for lane in LANES:
            terminal(root, lane, error=error)
        write_once(root / "preparation-complete.json", dict(cpu_suite_pass=False))
        return barrier(root)
    jobs = {}
    for lane in LANES:
        try:
            if lane == "ownership":
                module(lane).prepare(root / lane, cpu_gate_root=root)
                snapshot = shared.gpu_snapshot()
                if snapshot["active_gpus"] + 2 > 6 or snapshot["pending"]:
                    raise ValueError(
                        "New ownership workers need two slots below aggregate six-A100 ceiling"
                    )
                write_once(
                    root / "GPU-admission.json",
                    dict(snapshot=snapshot, maximum_a100s=6, new_workers=2),
                )
            elif lane == "providers":
                module(lane).prepare(root / lane, cpu_gate_root=root, parent=V14 / "providers")
            elif lane == "derived":
                module(lane).prepare(
                    root / "derived-1",
                    cpu_gate_root=root,
                    parent=plan["derived_parent"],
                    parent_gate_filename="GATE_freshness_empty_parquet_frontier.json",
                )
            elif lane == "history":
                module(lane).prepare(root / lane, cpu_gate_root=root)
            jobs[lane] = launch_lane(root, lane, label="derived-1" if lane == "derived" else None)
        except Exception as error:
            shared.failure(root, "prepare-" + lane, error)
            # Acknowledged producers still receive their afterany callback. If
            # callback submission itself failed, report failure without retry.
            terminal(root, lane, error=error)
    write_once(root / "preparation-complete.json", dict(cpu_suite_pass=True, callbacks=jobs))
    return barrier(root)


def after(root, lane, pass_index=None):
    output = root / (f"derived-{pass_index}" if lane == "derived" else lane)
    gate = output / GATES[lane]
    try:
        if lane in ("legacy-kv", "history", "completion"):
            read(gate)
        else:
            module(lane).analyze(output)
        report = read(gate)
        if (
            lane == "derived"
            and report["unattempted_files"] > 0
            and pass_index < load(root)["maximum_derived_passes"]
        ):
            next_index = pass_index + 1
            module(lane).prepare(
                root / f"derived-{next_index}",
                cpu_gate_root=root,
                parent=load(root)["derived_parent"],
                parent_gate_filename="GATE_freshness_empty_parquet_frontier.json",
                previous=[root / f"derived-{pass_index}"],
            )
            return dict(next_callback=launch_lane(root, lane, label=f"derived-{next_index}"))
        terminal(root, lane, path=gate)
    except Exception as error:
        shared.failure(root, f"analyze-{lane}-{pass_index or 0}", error)
        terminal(root, lane, path=gate, error=error)
    return barrier(root)


def finish(root):
    plan = load(root)
    if not all((root / f"terminal/{lane}.json").is_file() for lane in LANES):
        raise ValueError("Every finite producer requires its own terminal analysis receipt")
    lanes = {}
    for lane in LANES:
        item = read(root / f"terminal/{lane}.json")
        if item["gate_path"] and item["gate_sha256"]:
            path = Path(item["gate_path"])
            if sha(path) != item["gate_sha256"]:
                raise ValueError("Terminal evidence changed")
            item["evidence"] = read(path)
        lanes[lane] = item
    report = dict(
        status="finite_repair_completed_not_paper_confirmation",
        source_sha=plan["source_sha"],
        lanes=lanes,
        errors={p.name: read(p) for p in sorted((root / "errors").glob("*.json"))},
        independent_quality_N=0,
        fresh_confirmation_established=False,
        nearest_baseline_superiority_established=False,
        six_component_benefits_established=False,
        paper_campaign_authorized=False,
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
    for lane in LANES:
        labels = [f"derived-{i}" for i in range(1, 4)] if lane == "derived" else [lane]
        for label in labels:
            if mode == "after-" + label:
                return after(root, lane, int(label[-1]) if lane == "derived" else None)
            if mode != label + "-worker":
                continue
            if lane == "ownership" and shared.gpu_snapshot()["active_gpus"] > 6:
                raise ValueError("Aggregate six-A100 ceiling exceeded")
            if lane == "legacy-kv":
                return module(lane).run(
                    root / lane, parent=RECOVERED, reviewed_rules=RULES, cpu_gate_root=root
                )
            if lane == "history":
                return module(lane).run(root / lane)
            if lane == "completion":
                return module(lane).register(root / lane, execute=True, cpu_gate_root=root)
            return module(lane).worker(
                root / label, int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
            )
    raise ValueError("Unknown finite V15 mode")


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
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
