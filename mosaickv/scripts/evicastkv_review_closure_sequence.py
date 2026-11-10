#!/usr/bin/env python3
"""Finite V7 closure: CPU suite, saved-trace audits, then consumed GPU controls.

The three GPU scopes execute sequentially and do not authorize confirmation.
Failed producers receive afterany analysis; there is no polling or retry loop.
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

from evicastkv_review_followups import cpu_gate, gpu_snapshot

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
V6 = Path("/scratch/djy8hg/runs/evicastkv/fox-review-v6-parallel-controls-20261110")
MODULES = {
    "shape": "evicastkv_boundary_shape_validation",
    "ownership": "evicastkv_review_ownership_closure",
    "source-prefix": "evicastkv_review_source_prefix",
    "common-bridge": "evicastkv_review_common_bridge",
    "freshness": "evicastkv_review_freshness_worker",
}
GATES = {
    "shape": "shape/GATE_shape_transport.json",
    "ownership": "ownership/analysis/GATE_ownership_closure.json",
    "source-prefix": "source-prefix/analysis/GATE_source_fixed_prefix.json",
    "common-bridge": "common-bridge/analysis/GATE_common_bridge.json",
    "freshness": "freshness/GATE_freshness_workers.json",
}


def read(path):
    return json.loads(Path(path).read_text())


def identity(value):
    value = str(value).strip()
    if re.fullmatch(r"\d+(?:;[A-Za-z0-9_-]+)?", value) is None:
        raise ValueError("concrete Slurm job ID required")
    return value.split(";", 1)[0]


def safe_root(root):
    root = Path(root)
    root.resolve().relative_to(Path("/scratch/djy8hg/audits").resolve())
    if root.resolve() == Path("/scratch/djy8hg/audits").resolve():
        raise ValueError("a dedicated audit output directory is required")
    return root


def submit(root, label, mode, dependencies=(), *, gpu_count=0, cpu_array=False, tests=False):
    """One immutable attempt; sequential GPU arrays admit at most two A100s."""
    if gpu_count not in (0, 1, 2) or (gpu_count and (cpu_array or tests)):
        raise ValueError("only bounded registered CPU/GPU scopes allowed")
    request = root / f"jobs/{label}-request.json"
    if request.exists():
        raise FileExistsError("submission already attempted; no in-place retry")
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name=fox7-{label}",
        f"--output={root}/slurm/{label}-%A_%a.out",
        f"--error={root}/slurm/{label}-%A_%a.err",
    ]
    if gpu_count:
        command += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=64G",
            "--time=02:00:00",
            f"--array=0-{gpu_count - 1}%{gpu_count}",
        ]
    else:
        command += ["--partition=bii", "--cpus-per-task=4", "--mem=16G", "--time=00:30:00"]
        if cpu_array:
            command += ["--array=0-7%4"]
    if dependencies:
        command += ["--dependency=afterany:" + ":".join(identity(d) for d in dependencies)]
    launcher = "evicastkv_review_closure_v2" if tests else "evicastkv_review_closure_sequence"
    command += [str(SOURCE / f"mosaickv/slurm/{launcher}.sbatch"), str(SOURCE)]
    command += [str(root / "full-cpu"), "tests"] if tests else [str(root), mode]
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(request, dict(command=command, mode=mode, single_attempt=True))
    job = identity(subprocess.check_output(command, text=True))
    write_once(root / f"jobs/{label}-submission.json", dict(job=job, command=command))
    return job


def load(root):
    record = read(root / "registration.json")
    clean_source(SOURCE, record["source_sha"])
    if record["execute"] is not True:
        raise ValueError("dry-run registration cannot execute")
    for path, expected in record["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"immutable binding changed: {path}")
    return record


def register(root, previous=V6, *, execute=False):
    root, previous = safe_root(root), Path(previous)
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new immutable V7 output root required")
    final = previous / "FINAL_report.json"
    prior = read(final)
    registration = read(previous / "registration.json")
    clean_source(Path(registration["source_root"]), registration["source_sha"])
    if prior.get("source_sha") != registration["source_sha"]:
        raise ValueError("completed prior packet/source disagree")
    bindings = {
        str(final): sha(final),
        str(previous / "registration.json"): sha(previous / "registration.json"),
    }
    for module in (*MODULES.values(), "evicastkv_review_freshness_stream"):
        path = SOURCE / f"mosaickv/scripts/{module}.py"
        bindings[str(path)] = sha(path)
    bindings[str(SOURCE / "mosaickv/src/mosaickv/evaluation/fox_shape_transport.py")] = sha(
        SOURCE / "mosaickv/src/mosaickv/evaluation/fox_shape_transport.py"
    )
    snapshot = gpu_snapshot() if execute else None
    write_once(
        root / "registration.json",
        dict(
            source_root=str(SOURCE),
            source_sha=revision,
            previous=str(previous),
            parent=registration["parent"],
            execute=execute,
            bindings=bindings,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            sequential_gpu_worker_counts=[1, 2, 2],
            cpus_per_gpu_worker=12,
            gpu_time_limit_seconds=7200,
            freshness_shards=8,
            freshness_concurrency=4,
            consumed_inputs_only=True,
            new_cohort_materialization_authorized=False,
            configuration_freeze_authorized=False,
            fresh_confirmation_authorized=False,
            all_thresholds_unchanged=True,
            snapshot=snapshot,
            independence="Mistral FP16 shape control and LLaVA BF16 source scopes are separate",
        ),
    )
    if not execute:
        return dict(status="dry_run", submitted_jobs=0)
    cpu = submit(root, "cpu", "tests", tests=True)
    boot = submit(root, "prepare", "prepare", [cpu])
    return dict(status="submitted_cpu_then_bounded_closure", cpu=cpu, preparation=boot)


def module(lane):
    return importlib.import_module(MODULES[lane])


def record_error(root, lane, error):
    record = dict(
        status="blocked", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
    )
    write_once(root / f"errors/{lane}.json", record)
    return record


def prepare(root):
    record = load(root)
    try:
        write_once(root / "GATE_cpu_suite.json", cpu_gate(root, record["source_sha"]))
    except Exception as error:
        record_error(root, "cpu", error)
        return finish(root)
    previous, prepared = Path(record["previous"]), {}
    for lane in ("shape", "ownership", "source-prefix", "common-bridge"):
        try:
            if lane == "shape":
                info = module(lane).prepare(root / lane, Path(record["parent"]), previous)
            else:
                parent = previous if lane == "ownership" else previous / "source"
                info = module(lane).prepare(root / lane, parent)
            prepared[lane] = dict(status="prepared", info=info)
        except Exception as error:
            prepared[lane] = record_error(root, f"prepare-{lane}", error)
    write_once(root / "preparation.json", prepared)
    submit(root, "ownership", "ownership")
    freshness = submit(root, "freshness", "freshness-worker", cpu_array=True)
    submit(root, "freshness-analysis", "freshness-analysis", [freshness])
    if prepared["shape"]["status"] == "prepared":
        snapshot = gpu_snapshot()
        dependencies = sorted(
            {r["job"].split("_")[0] for r in snapshot["active"] + snapshot["pending"]}
        )
        write_once(
            root / "gpu_admission.json",
            dict(
                snapshot=snapshot,
                wait_for_existing_jobs=dependencies,
                maximum_a100s=6,
                own_chain_maximum=2,
            ),
        )
        worker = submit(root, "shape", "shape-worker", dependencies, gpu_count=1)
        submit(root, "shape-analysis", "after-shape", [worker])
    else:
        launch_source(root)
    return prepared


def analyze_lane(root, lane):
    try:
        module(lane).analyze(root / lane)
        return read(root / GATES[lane])
    except Exception as error:
        return record_error(root, f"analyze-{lane}", error)


def launch_source(root):
    prepared = read(root / "preparation.json")
    if prepared["source-prefix"]["status"] != "prepared":
        return submit_final(root)
    workers = submit(root, "source-prefix", "source-worker", gpu_count=2)
    return submit(root, "source-analysis", "after-source", [workers])


def source_controls_allow_bridge(gate):
    # A lossy-source fidelity failure is an outcome, not permission to discard
    # rows. Only complete execution and passed FullKV controls admit the next
    # consumed diagnostic. The Mistral numerical gate is NOT waived or pooled.
    return (
        gate.get("complete") is True and not gate.get("failures") and gate.get("actual_rows") == 14
    )


def after_source(root):
    gate = analyze_lane(root, "source-prefix")
    prepared = read(root / "preparation.json")
    if source_controls_allow_bridge(gate) and prepared["common-bridge"]["status"] == "prepared":
        workers = submit(root, "common-bridge", "bridge-worker", gpu_count=2)
        return submit(root, "bridge-analysis", "after-bridge", [workers])
    write_once(
        root / "GATE_bridge_admission.json", dict(status="blocked", source_gate=gate, no_retry=True)
    )
    return submit_final(root)


def submit_final(root):
    dependencies = [
        read(root / f"jobs/{label}-submission.json")["job"]
        for label in ("ownership", "freshness-analysis")
    ]
    return submit(root, "final", "finish", dependencies)


def finish(root):
    record = load(root)
    gates, errors = {}, {}
    for lane, path in GATES.items():
        target = root / path
        gates[lane] = (
            dict(path=str(target), sha256=sha(target), gate=read(target))
            if target.exists()
            else dict(status="missing_or_blocked")
        )
    for path in sorted((root / "errors").glob("*.json")):
        errors[str(path)] = read(path)
    report = dict(
        status="completed_bounded_closure_not_confirmation",
        source_sha=record["source_sha"],
        registration_sha256=sha(root / "registration.json"),
        gates=gates,
        errors=errors,
        original_artifacts_preserved=True,
        new_fresh_samples=0,
        scientific_completion=False,
        superiority_established=False,
        fresh_confirmation_authorized=False,
        configuration_freeze_authorized=False,
        remaining=(
            "Independent media/source-group freshness, baseline qualification, and independently "
            "confirmed quality/performance cannot be inferred from these consumed controls."
        ),
    )
    write_once(root / "FINAL_report.json", report)
    return report


def run(root, mode):
    load(root)
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if mode in ("shape-worker", "source-worker", "bridge-worker"):
        if index not in ((0,) if mode == "shape-worker" else (0, 1)):
            raise ValueError("worker index exceeds the immutable scope")
        snapshot = gpu_snapshot()
        write_once(root / f"admission/{mode}-{index}.json", snapshot)
        if snapshot["active_gpus"] > 6:
            raise ValueError("aggregate six-GPU ceiling exceeded; admitting no model work")
    if mode == "prepare":
        return prepare(root)
    if mode == "ownership":
        if read(root / "preparation.json")["ownership"]["status"] != "prepared":
            return dict(status="blocked_preparation")
        try:
            return module("ownership").run(root / "ownership")
        except Exception as error:
            return record_error(root, "ownership", error)
    if mode == "freshness-worker":
        return module("freshness").worker(root / "freshness", index)
    if mode == "freshness-analysis":
        try:
            return module("freshness").aggregate(root / "freshness")
        except Exception as error:
            return record_error(root, "freshness-analysis", error)
    if mode == "shape-worker":
        return module("shape").worker(root / "shape")
    if mode == "after-shape":
        analyze_lane(root, "shape")
        return launch_source(root)
    if mode == "source-worker":
        return module("source-prefix").worker(root / "source-prefix", index)
    if mode == "after-source":
        return after_source(root)
    if mode == "bridge-worker":
        return module("common-bridge").worker(root / "common-bridge", index)
    if mode == "after-bridge":
        analyze_lane(root, "common-bridge")
        return submit_final(root)
    if mode == "finish":
        return finish(root)
    raise ValueError("unknown finite closure stage")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "register",
            "prepare",
            "ownership",
            "freshness-worker",
            "freshness-analysis",
            "shape-worker",
            "after-shape",
            "source-worker",
            "after-source",
            "bridge-worker",
            "after-bridge",
            "finish",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=V6)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = safe_root(args.root)
    result = (
        register(root, args.previous, execute=args.execute)
        if args.mode == "register"
        else run(root, args.mode)
    )
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
