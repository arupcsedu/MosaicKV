#!/usr/bin/env python3
"""Finite independent review experiments in ONE aggregate six-A100 array.

Dry-run by default. Afterany callbacks preserve failed producers and analyze
every lane. No log-polling loop, answer-guided search, or fresh confirmation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import traceback
from pathlib import Path

from evicastkv_review_followups import cpu_gate, gpu_snapshot, parent_controls

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
LANES = {
    "boundary": "evicastkv_boundary_linear_diagnostic",
    "terms": "evicastkv_independent_terms",
    "pooling": "evicastkv_pooling_index_study",
    "ownership": "evicastkv_review_lifetimes",
    "source": "evicastkv_review_source_controls",
}
LANE_GATES = {
    "boundary": "GATE_boundary_linear.json",
    "terms": "analysis/mechanisms/GATE_independent_terms.json",
    "pooling": "analysis/GATE.json",
    "ownership": "analysis/GATE.json",
    "source": "analysis/GATE_source_controls.json",
}


def read(path):
    return json.loads(Path(path).read_text())


def safe_root(path):
    path = Path(path)
    path.resolve().relative_to(Path("/scratch/djy8hg").resolve())
    return path


def job_id(value):
    value = str(value).strip()
    if re.fullmatch(r"\d+(?:;[A-Za-z0-9_-]+)?", value) is None:
        raise ValueError("a concrete scheduler job ID is required")
    return value.split(";", 1)[0]


def array_tasks(prepared):
    tasks = []
    for lane, info in prepared.items():
        if lane not in LANES or info.get("status") != "prepared":
            continue
        count = info["workers"]
        if type(count) is not int or not 1 <= count <= 16:
            raise ValueError("bounded positive worker count required")
        for index in range(count):
            tasks.append(
                dict(
                    lane=lane,
                    module=LANES[lane],
                    index=index,
                    root=info["root"],
                    spec_sha=info.get("spec_sha"),
                    protocol_path=info.get("protocol_path"),
                    protocol_sha=info.get("protocol_sha"),
                )
            )
    return tasks


def submit(root, label, mode, dependencies=(), *, array=None, launcher=None, arguments=None):
    request = root / "jobs" / f"{label}-request.json"
    if request.exists():
        raise FileExistsError("single-attempt submission already recorded")
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name=fox6-{label}",
        f"--output={root}/slurm/{label}-%A_%a.out",
        f"--error={root}/slurm/{label}-%A_%a.err",
    ]
    if array is not None:
        if re.fullmatch(r"0-\d+%6", array) is None:
            raise ValueError("only one aggregate array with throttle six is authorized")
        command += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=64G",
            "--time=02:00:00",
            f"--array={array}",
            "--export=ALL,EVICAST_REVIEW_TASKS_SHA=" + sha(root / "tasks.json"),
        ]
    else:
        command += ["--partition=bii", "--cpus-per-task=4", "--mem=16G", "--time=00:30:00"]
    if dependencies:
        command += ["--dependency=afterany:" + ":".join(job_id(d) for d in dependencies)]
    command += [str(launcher or SOURCE / "mosaickv/slurm/evicastkv_review_parallel.sbatch")]
    command += arguments or [str(SOURCE), str(root), mode]
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(request, dict(command=command, mode=mode, immutable_single_attempt=True))
    identity = job_id(subprocess.check_output(command, text=True))
    write_once(root / "jobs" / f"{label}-submission.json", dict(job=identity, command=command))
    return identity


def register(root, parent, mechanisms, prior, *, execute=False):
    root = safe_root(root)
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("parallel review registration requires new root")
    _, bindings = parent_controls(parent)
    for path in (
        mechanisms / "analysis/mechanisms/GATE_mechanisms.json",
        mechanisms / "analysis/full_graph_pilot/GATE_full_graph_pilot.json",
    ):
        if read(path).get("status") != "passed":
            raise ValueError("completed mechanism and full-routing controls required")
        bindings[str(path)] = sha(path)
    if (
        read(prior / "FINAL_report.json").get("status")
        != "COMPLETED_CONSUMED_A_STUDIES_BOUNDARY_SEPARATE"
    ):
        raise ValueError("completed prior follow-ups required; do not duplicate active studies")
    bindings[str(prior / "FINAL_report.json")] = sha(prior / "FINAL_report.json")
    for module in (*LANES.values(), "evicastkv_review_freshness_audit"):
        path = SOURCE / "mosaickv/scripts" / f"{module}.py"
        bindings[str(path)] = sha(path)
    snapshot = gpu_snapshot() if execute else None
    registration = dict(
        source_root=str(SOURCE),
        source_sha=revision,
        parent=str(parent),
        mechanisms=str(mechanisms),
        prior=str(prior),
        bindings=bindings,
        maximum_a100s=6,
        resource_override="explicit user request supersedes four-A100 repository default",
        execution_authorized=execute,
        consumed_development_only=True,
        new_confirmation_materialization_authorized=False,
        configuration_freeze_authorized=False,
        thresholds_unchanged=True,
        source_and_retention_not_pooled=True,
        at_registration_gpu_snapshot=snapshot,
    )
    write_once(root / "registration.json", registration)
    if not execute:
        return dict(status="DRY_RUN_ONLY", jobs_submitted=0)
    cpu = submit(
        root,
        "cpu",
        "tests",
        launcher=SOURCE / "mosaickv/slurm/evicastkv_review_closure_v2.sbatch",
        arguments=[str(SOURCE), str(root / "full-cpu"), "tests"],
    )
    boot = submit(root, "prepare", "prepare", [cpu])
    return dict(status="submitted_CPU_gate_then_parallel_experiments", cpu=cpu, preparation=boot)


def load(root):
    record = read(root / "registration.json")
    clean_source(SOURCE, record["source_sha"])
    if record["execution_authorized"] is not True:
        raise ValueError("dry-run registration does not authorize execution")
    for path, expected in record["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"registered artifact changed: {path}")
    return record


def prepare(root):
    record = load(root)
    gate = cpu_gate(root, record["source_sha"])
    write_once(root / "GATE_cpu_suite.json", gate)
    prepared = {}
    for lane, module in LANES.items():
        target = root / lane
        try:
            entry = importlib.import_module(module)
            if lane == "boundary":
                info = entry.prepare(
                    Path(record["parent"]),
                    target,
                    prior_boundary=Path(record["prior"]) / "boundary-operators",
                )
            elif lane == "terms":
                info = entry.prepare(target, Path(record["parent"]), Path(record["mechanisms"]))
            elif lane == "source":
                info = entry.prepare(target)
            else:
                info = entry.prepare(target, Path(record["parent"]))
            info = info or {}
            if lane == "source" and info.get("status", "").startswith("blocked"):
                raise ValueError(info)
            count = info.get("workers", 1)
            prepared[lane] = dict(
                status="prepared",
                root=str(target),
                workers=count,
                spec_sha=info.get("spec_sha256", info.get("spec_sha")),
            )
            protocol = target / (
                "plan.json"
                if lane == "source"
                else "protocol.json"
                if lane == "boundary"
                else "protocol_lock.json"
            )
            prepared[lane].update(protocol_path=str(protocol), protocol_sha=sha(protocol))
        except Exception as error:
            prepared[lane] = dict(
                status="blocked_preparation",
                root=str(target),
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
    write_once(root / "preparation.json", prepared)
    tasks = array_tasks(prepared)
    write_once(
        root / "tasks.json",
        dict(source_sha=record["source_sha"], tasks=tasks, maximum_a100s=6, array_concurrency=6),
    )
    # Freshness is CPU work and independent of consumed diagnostic results.
    freshness = submit(root, "freshness", "freshness")
    if tasks:
        snapshot = gpu_snapshot()
        dependencies = sorted(
            {r["job"].split("_", 1)[0] for r in snapshot["active"] + snapshot["pending"]}
        )
        write_once(
            root / "admission.json",
            dict(
                snapshot=snapshot,
                wait_for_existing_own_gpu_jobs=dependencies,
                single_global_array=True,
                maximum_a100s=6,
            ),
        )
        array = submit(root, "gpu", "worker", dependencies, array=f"0-{len(tasks) - 1}%6")
        submit(root, "analyze", "analyze", [array, freshness])
    else:
        submit(root, "analyze", "analyze", [freshness])
    return dict(tasks=len(tasks), prepared=prepared)


def worker(root, index):
    load(root)
    snapshot = gpu_snapshot()
    if snapshot["active_gpus"] > 6:
        raise RuntimeError("aggregate own GPU occupancy exceeds six; refusing model work")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("cuBLAS determinism must be configured before Python starts")
    if os.environ.get("EVICAST_REVIEW_TASKS_SHA") != sha(root / "tasks.json"):
        raise ValueError("global array task ledger changed after immutable submission")
    grid = read(root / "tasks.json")["tasks"]
    if type(index) is not int or not 0 <= index < len(grid):
        raise ValueError("registered array index required")
    task = grid[index]
    if sha(Path(task["protocol_path"])) != task["protocol_sha"]:
        raise ValueError("lane protocol changed after array registration")
    module = importlib.import_module(task["module"])
    if task["lane"] in {"boundary"}:
        return module.worker(Path(task["root"]))
    if task["lane"] == "source":
        return module.worker(Path(task["root"]), task["index"])
    return module.worker(Path(task["root"]), task["index"], task["spec_sha"])


def analyze(root):
    registration = load(root)
    outcomes = {}
    for lane, prepared in read(root / "preparation.json").items():
        if prepared["status"] != "prepared":
            outcomes[lane] = prepared
            continue
        try:
            if sha(Path(prepared["protocol_path"])) != prepared["protocol_sha"]:
                raise ValueError("registered lane protocol changed before analysis")
            result = importlib.import_module(LANES[lane]).analyze(Path(prepared["root"]))
            path = Path(prepared["root"]) / LANE_GATES[lane]
            outcomes[lane] = (
                dict(gate=read(path), gate_path=str(path), gate_sha256=sha(path))
                if path.is_file()
                else result or dict(status="analysis_missing_required_gate")
            )
        except Exception as error:
            outcomes[lane] = dict(
                status="failed_analysis",
                error=f"{type(error).__name__}: {error}",
                traceback=traceback.format_exc(),
            )
    freshness = root / "freshness/GATE_freshness_audit.json"
    outcomes["freshness"] = (
        dict(gate=read(freshness), gate_path=str(freshness), gate_sha256=sha(freshness))
        if freshness.is_file()
        else dict(status="missing_CPU_freshness_gate")
    )
    packet = dict(
        source_sha=registration["source_sha"],
        outcomes=outcomes,
        original_failures_preserved=True,
        complete_current_scope_only=True,
        fresh_confirmation_authorized=False,
        configuration_freeze_authorized=False,
        superiority_established=False,
        source_retention_not_pooled=True,
    )
    write_once(root / "FINAL_report.json", packet)
    return packet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "prepare", "worker", "analyze", "freshness"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--mechanisms", type=Path)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.mode == "register":
        if not all((args.parent, args.mechanisms, args.prior)):
            parser.error("registration requires parent, mechanisms and prior")
        result = register(args.root, args.parent, args.mechanisms, args.prior, execute=args.execute)
    elif args.mode == "worker":
        result = worker(args.root, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    elif args.mode == "freshness":
        load(args.root)
        from evicastkv_review_freshness_audit import run

        result = run(args.root / "freshness")
    else:
        result = globals()[args.mode](args.root)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
