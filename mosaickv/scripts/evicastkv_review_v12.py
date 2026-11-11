#!/usr/bin/env python3
"""Finite source-fidelity/baseline/media follow-up, never an automatic claim.

Each producer has an afterany analyzer. Canonical CPU checks precede work;
missing output is a failure, not a pass. No cohort is selected or submitted.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
from pathlib import Path

import evicastkv_review_v8_diagnostics as shared

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v11/run-20261110-1")
PARENT_SHA = "1aab16c63eee2f063194e6db8c9b3b5af466c3a1"
MAX_FRONTIER_SEGMENTS = 4
GPU = {
    "precision": ("evicastkv_source_precision_control", "GATE_source_precision_control.json"),
    "lookm": ("evicastkv_lookm_hf_checkpoint", "GATE_lookm_hf_checkpoint_compat.json"),
    "lossless": ("evicastkv_lossless_source_control", "GATE_lossless_source_control.json"),
    "ownership": ("evicastkv_source_ownership_v12", "GATE_source_ownership_v12.json"),
}
CPU = {
    "grouped": ("evicastkv_freshness_grouped_offsets", "GATE_grouped_offset_recovery.json"),
    "yaml": ("evicastkv_freshness_yaml_events", "GATE_freshness_yaml_events.json"),
    "historical_media": ("evicastkv_historical_media_hashes", "GATE_historical_media_hashes.json"),
    "media": ("evicastkv_coco_media_availability", "GATE_coco_media_availability.json"),
    "frontier": (
        "evicastkv_freshness_empty_parquet_continue",
        "GATE_freshness_empty_parquet_frontier.json",
    ),
}


def read(path):
    return json.loads(Path(path).read_text())


def module(lane):
    return importlib.import_module((GPU | CPU)[lane][0])


def load(root):
    plan = read(Path(root) / "registration.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["execute"] is not True or plan["confirmation_authorized"]:
        raise ValueError("registered finite diagnostics only")
    if plan["maximum_frontier_segments"] != MAX_FRONTIER_SEGMENTS:
        raise ValueError("bounded four-segment metadata protocol changed")
    for path, expected in plan["bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"immutable V12 evidence changed: {path}")
    return plan


def submit(root, label, mode, *, deps=(), gpu=False, array=False, tests=False):
    """One scheduler request; lost acknowledgement never causes blind retry."""
    if gpu and (array or tests):
        raise ValueError("one GPU worker per independent diagnostic")
    cpus = (
        12 if gpu or tests or mode in ("after-precision", "after-lossless") else 1 if array else 4
    )
    memory = "64G" if gpu else "16G"
    minutes = 20 if gpu or tests else 5 if array else 15
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name=fox12-{label}",
        f"--cpus-per-task={cpus}",
        f"--mem={memory}",
        f"--time=00:{minutes:02d}:00",
        f"--output={root}/slurm/{label}-%A_%a.out",
        f"--error={root}/slurm/{label}-%A_%a.err",
        "--partition=bii-gpu" if gpu else "--partition=bii",
    ]
    if gpu:
        command += ["--reservation=bi_fox_dgx", "--gres=gpu:a100:1"]
    if array:
        command += ["--array=0-7%8"]
    if deps:
        command += ["--dependency=afterany:" + ":".join(shared.identity(d) for d in deps)]
    launcher = "evicastkv_review_closure_v2" if tests else "evicastkv_review_v12"
    command += [
        str(SOURCE / f"mosaickv/slurm/{launcher}.sbatch"),
        str(SOURCE),
        str(root / "full-cpu") if tests else str(root),
        "tests" if tests else mode,
    ]
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(root / f"jobs/{label}-request.json", dict(command=command, single_attempt=True))
    try:
        job = shared.identity(subprocess.check_output(command, text=True, stderr=subprocess.PIPE))
    except Exception as error:
        write_once(
            root / f"jobs/{label}-submission-error.json",
            dict(
                error=f"{type(error).__name__}: {error}",
                stderr=str(getattr(error, "stderr", "")),
                automatic_retry=False,
            ),
        )
        raise shared.SubmissionError("failed or unacknowledged V12 submission") from error
    write_once(root / f"jobs/{label}-submission.json", dict(job=job, command=command))
    return job


def register(root, *, execute=False):
    root = Path(root).resolve()
    scratch = Path("/scratch/djy8hg/workdir").resolve()
    if root.exists() or root == scratch or not root.is_relative_to(scratch):
        raise ValueError("new write-once scratch-workspace diagnostic root required")
    revision = clean_source(SOURCE)
    parent = read(PARENT / "registration.json")
    final = read(PARENT / "FINAL_report.json")
    if parent["source_sha"] != PARENT_SHA or final["source_sha"] != PARENT_SHA:
        raise ValueError("wrong V11 parent source")
    paths = [PARENT / "registration.json", PARENT / "FINAL_report.json"]
    ownership_parent = PARENT / "ownership"
    owner_tool = importlib.import_module("evicastkv_source_ownership_v12")
    accounting = dict(status="unavailable", command=[], text="")
    try:
        manifest = read(ownership_parent / "worker/manifest.json")
        job = shared.identity(manifest["slurm_job_id"])
        command = ["sacct", "-j", job, "-P", "--format=JobIDRaw,State,ElapsedRaw,TimelimitRaw"]
        text = subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=30)
        accounting = dict(status="captured_once", command=command, text=text)
        decision = owner_tool.classify_retry(
            ownership_parent, expected_job_id=job, accounting_text=text
        )
    except Exception as error:
        decision = dict(
            rerun=False, status="blocked_retry_proof", error=f"{type(error).__name__}: {error}"
        )
    write_once(root / "parent-ownership-accounting.json", accounting)
    write_once(root / "ownership-retry-decision.json", decision)
    paths += [root / "parent-ownership-accounting.json", root / "ownership-retry-decision.json"]
    paths += [
        path
        for path in [
            ownership_parent / "protocol.json",
            *(
                ownership_parent / "worker" / name
                for name in (
                    "manifest.json",
                    "raw.json",
                    "logits.pt",
                    "allocator_trace.json.gz",
                    "result.json",
                )
            ),
        ]
        if path.is_file()
    ]
    paths += [SOURCE / f"mosaickv/scripts/{name}.py" for name, _ in (GPU | CPU).values()]
    paths += [
        SOURCE / "mosaickv/src/mosaickv/evaluation/precision_source_control.py",
        SOURCE / "mosaickv/src/mosaickv/evaluation/lossless_source_control.py",
        SOURCE / "mosaickv/scripts/audit_lookm_hf_generation_compat.py",
        SOURCE / "mosaickv/scripts/evicastkv_source_trace_adjudication.py",
        SOURCE / "mosaickv/env/common/requirements.lock",
        SOURCE / "mosaickv/docs/baselines/lookm_import_cleanup_ecf0f51.patch",
    ]
    write_once(
        root / "registration.json",
        dict(
            source_sha=revision,
            source_root=str(SOURCE),
            execute=execute,
            bindings={str(p): sha(p) for p in paths},
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            diagnostic_gpu_workers_maximum=4,
            CPU_arrays_each_maximum_parallel_workers=8,
            maximum_frontier_segments=MAX_FRONTIER_SEGMENTS,
            frontier_continuation_requires_complete_state_and_progress=True,
            consumed_model_inputs_only=True,
            images_read_for_hashes_only=True,
            benchmark_labels_forbidden=True,
            confirmation_authorized=False,
            automatic_paper_campaign=False,
            preserve_all_negative_results=True,
            registered_model_guards_unchanged=True,
        ),
    )
    if not execute:
        return dict(status="dry_run", jobs=0)
    cpu = submit(root, "cpu", "tests", tests=True)
    return dict(cpu=cpu, prepare=submit(root, "prepare", "prepare", deps=[cpu]))


def prepare(root):
    lock = load(root)
    try:
        write_once(root / "GATE_cpu_suite.json", shared.cpu_gate(root, lock["source_sha"]))
    except Exception as error:
        shared.failure(root, "cpu", error)
        return finish(root)
    preparations, dependencies, cpu_workers, cpu_analyses = {}, [], {}, {}
    for lane in ("grouped", "frontier", "historical_media"):
        try:
            kwargs = dict(cpu_gate_root=root)
            if lane != "historical_media":
                kwargs.update(parent=PARENT / "freshness-frontier")
            preparations[lane] = module(lane).prepare(root / lane, **kwargs)
            worker = submit(root, lane, f"{lane}-worker", array=True)
            cpu_workers[lane] = worker
            dependencies.append(worker)
            waits = [worker]
            if lane == "frontier":
                waits.extend(cpu_analyses[prior] for prior in ("grouped",) if prior in cpu_analyses)
                if os.environ.get("SLURM_JOB_ID"):
                    waits.append(shared.identity(os.environ["SLURM_JOB_ID"]))
            analyzer = submit(root, f"after-{lane}", f"after-{lane}", deps=waits)
            cpu_analyses[lane] = analyzer
            dependencies.append(analyzer)
        except Exception as error:
            shared.failure(root, f"prepare-{lane}", error)
    ownership_retry = False
    try:
        tool = importlib.import_module("evicastkv_source_trace_adjudication")
        write_once(root / "ownership-review-inventory.json", tool.inventory(PARENT / "ownership"))
        write_once(
            root / "ownership-retry-action.json",
            dict(rerun=False, reason="complete immutable V11 trace is available for CPU review"),
        )
    except Exception as error:
        ownership_retry = read(root / "ownership-retry-decision.json")["rerun"] is True
        write_once(
            root / "ownership-retry-action.json",
            dict(
                rerun=ownership_retry,
                reason=f"{type(error).__name__}: {error}",
                one_retry_only=True,
                codec_policy_unchanged=True,
                raw_trace_archived_before_indexed_attribution=True,
                registered_timeout_rescue_only=True,
            ),
        )
    snapshot = shared.gpu_snapshot()
    external_waits = sorted(
        {r["job"].split("_")[0] for r in snapshot["active"] + snapshot["pending"]}
    )
    for lane in GPU:
        if lane == "ownership" and not ownership_retry:
            continue
        try:
            kwargs = dict(cpu_gate_root=root) if lane in ("lookm", "lossless") else {}
            preparations[lane] = module(lane).prepare(root / lane, **kwargs)
            # At most four new one-A100 workers; wait for existing allocations.
            write_once(
                root / f"admission-{lane}.json", dict(snapshot=snapshot, deps=external_waits)
            )
            worker = submit(root, lane, f"{lane}-worker", gpu=True, deps=external_waits)
            dependencies.append(worker)
            dependencies.append(submit(root, f"after-{lane}", f"after-{lane}", deps=[worker]))
        except Exception as error:
            shared.failure(root, f"prepare-{lane}", error)
    write_once(root / "preparation.json", preparations)
    write_once(root / "stage-jobs.json", dict(dependencies=dependencies))
    if "frontier" in cpu_analyses:
        return dict(
            frontier_controller=cpu_analyses["frontier"], bounded_segments=MAX_FRONTIER_SEGMENTS
        )
    if dependencies:
        return dict(final=submit(root, "final", "final", deps=dependencies))
    return finish(root)


def prepare_media(root):
    load(root)
    dependencies = read(root / "stage-jobs.json")["dependencies"]
    deferred = root / "frontier-dependent-jobs.json"
    if deferred.exists():
        dependencies += read(deferred)["dependencies"]
    try:
        result = module("media").prepare(
            root / "media",
            cpu_gate_root=root,
            history_root=root / "yaml",
            history_gate_filename=CPU["yaml"][1],
            hash_media=True,
        )
        write_once(root / "prepared-media.json", result)
        worker = submit(root, "media", "media-worker", array=True)
        dependencies.append(worker)
        dependencies.append(submit(root, "after-media", "after-media", deps=[worker]))
    except Exception as error:
        shared.failure(root, "prepare-media", error)
    return dict(final=submit(root, "final", "final", deps=dependencies))


def frontier_path(root, index):
    if type(index) is not int or not 1 <= index <= MAX_FRONTIER_SEGMENTS:
        raise ValueError("registered bounded frontier segment index required")
    return root / ("frontier" if index == 1 else f"frontier-{index}")


def frontier_followup(root, index):
    """Finite afterany state continuation, never a restart or a log-poll loop."""
    load(root)
    path = frontier_path(root, index)
    recovery = root / "grouped" / CPU["grouped"][1]
    result = module("frontier").analyze(
        path, recovery_root=root / "grouped" if index == 1 and recovery.exists() else None
    )
    progress = not result["missing_shards"] and any(
        read(path / "workers" / f"shard-{i}.json")["state_sha256"]
        != read(path / "imports" / f"shard-{i}.json")["state_sha256"]
        for i in range(8)
    )
    continue_walk = bool(
        not result["missing_shards"]
        and not result["frontier_exhausted"]
        and progress
        and index < MAX_FRONTIER_SEGMENTS
    )
    write_once(
        root / f"decisions/frontier-{index}.json",
        dict(
            index=index,
            maximum_segments=MAX_FRONTIER_SEGMENTS,
            gate_sha256=sha(path / CPU["frontier"][1]),
            state_progress=progress,
            continue_walk=continue_walk,
            frontier_exhausted=result["frontier_exhausted"],
            missing_shards=result["missing_shards"],
            fresh_confirmation_authorized=False,
        ),
    )
    current = (
        [shared.identity(os.environ["SLURM_JOB_ID"])] if os.environ.get("SLURM_JOB_ID") else []
    )
    if continue_walk:
        next_index = index + 1
        next_path = frontier_path(root, next_index)
        module("frontier").prepare(
            next_path, cpu_gate_root=root, parent=path, parent_gate_filename=CPU["frontier"][1]
        )
        worker = submit(
            root,
            f"frontier-{next_index}",
            f"frontier-{next_index}-worker",
            deps=current,
            array=True,
        )
        analyzer = submit(
            root, f"after-frontier-{next_index}", f"after-frontier-{next_index}", deps=[worker]
        )
        write_once(
            root / f"frontier-segment-{next_index}-jobs.json",
            dict(worker=worker, analyzer=analyzer),
        )
        return dict(next_segment=next_index, worker=worker, analyzer=analyzer)
    write_once(
        root / "frontier-selection.json",
        dict(
            index=index,
            path=str(path),
            gate_sha256=sha(path / CPU["frontier"][1]),
            frontier_exhausted=result["frontier_exhausted"],
        ),
    )
    try:
        module("yaml").prepare(
            root / "yaml",
            cpu_gate_root=root,
            parent=path,
            parent_gate_filename=CPU["frontier"][1],
        )
        worker = submit(root, "yaml", "yaml-worker", deps=current, array=True)
        analyzer = submit(root, "after-yaml", "after-yaml", deps=[worker])
        waits = [analyzer]
        historical = root / "jobs/after-historical_media-submission.json"
        if historical.exists():
            waits.append(read(historical)["job"])
        write_once(root / "frontier-dependent-jobs.json", dict(dependencies=waits))
        return dict(media_preparation=submit(root, "prepare-media", "prepare-media", deps=waits))
    except Exception as error:
        shared.failure(root, "prepare-final-yaml", error)
        return dict(
            final=submit(
                root,
                "final",
                "final",
                deps=read(root / "stage-jobs.json")["dependencies"] + current,
            )
        )


def frontier_event(root, index):
    try:
        return frontier_followup(root, index)
    except Exception as error:
        shared.failure(root, f"frontier-controller-{index}", error)
        dependencies = [read(p)["job"] for p in sorted((root / "jobs").glob("*-submission.json"))]
        current = os.environ.get("SLURM_JOB_ID")
        if current:
            dependencies.append(shared.identity(current))
        return dict(final=submit(root, "final", "final", deps=sorted(set(dependencies))))


def finish(root):
    lock = load(root)
    artifacts = {}
    for lane, (_, filename) in (GPU | CPU).items():
        path = root / lane / filename
        if lane == "frontier" and (root / "frontier-selection.json").exists():
            selected = read(root / "frontier-selection.json")
            path = frontier_path(root, selected["index"]) / filename
            if sha(path) != selected["gate_sha256"]:
                raise ValueError("selected final historical frontier changed")
        recovered = root / "ownership-recovery/GATE_source_trace_recovery.json"
        if lane == "ownership" and not path.exists() and recovered.exists():
            artifacts[lane] = dict(
                status="CPU_recovered_ownership_capture_not_GPU_completion",
                path=str(recovered),
                sha256=sha(recovered),
                value=read(recovered),
            )
            continue
        if (
            lane == "ownership"
            and not path.exists()
            and (root / "ownership-review-inventory.json").exists()
        ):
            inventory = root / "ownership-review-inventory.json"
            artifacts[lane] = dict(
                status="existing_V11_trace_preserved_CPU_adjudication_required",
                path=str(inventory),
                sha256=sha(inventory),
            )
            continue
        artifacts[lane] = (
            dict(path=str(path), sha256=sha(path), value=read(path))
            if path.exists()
            else dict(status="missing_or_blocked")
        )
    result = dict(
        source_sha=lock["source_sha"],
        status="completed_finite_followup_not_confirmation",
        artifacts=artifacts,
        errors={p.name: read(p) for p in (root / "errors").glob("*.json")},
        missing_output_is_not_a_pass=True,
        superiority_established=False,
        fresh_cohort_submitted=False,
        paper_campaign_submitted=False,
    )
    write_once(root / "FINAL_report.json", result)
    lines = [
        "# V12 bounded diagnostic results",
        "",
        f"Immutable source: `{lock['source_sha']}`.",
        "",
        "| Lane | Recorded status | Evidence |",
        "| --- | --- | --- |",
    ]
    for lane, artifact in artifacts.items():
        status = artifact.get("value", {}).get("status", artifact.get("status", "unknown"))
        path = artifact.get("path")
        evidence = f"[gate]({path})" if path else "missing"
        lines.append(f"| {lane} | {str(status).replace('|', '/')} | {evidence} |")
    lines += [
        "",
        f"Recorded orchestration errors: **{len(result['errors'])}**.",
        "",
        "Initial, final, decode-owned peak and whole-request memory are distinct boundaries.",
        "Inspect numerical guards independently of successful execution and artifact completeness.",
        "Freshness requires complete identity/media coverage, not absent hashes alone.",
        "",
        "These diagnostics do not establish independent answer-quality confirmation, latency",
        "superiority, or benefits from all six research mechanisms.",
        "No paper campaign was submitted.",
        "",
        "The complete machine-readable report preserves every recorded gate and error.",
    ]
    with (root / "SUMMARY.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return result


def run(root, mode):
    root = Path(root).resolve()
    load(root)
    if mode == "prepare":
        return prepare(root)
    if mode == "prepare-media":
        return prepare_media(root)
    if mode == "final":
        return finish(root)
    if mode == "after-frontier":
        return frontier_event(root, 1)
    if (mode.startswith("frontier-") and mode != "frontier-worker") or mode.startswith(
        "after-frontier-"
    ):
        value = mode.removeprefix("after-").removeprefix("frontier-").removesuffix("-worker")
        if not value.isascii() or not value.isdecimal():
            raise ValueError("explicit finite frontier segment required")
        index = int(value)
        path = frontier_path(root, index)
        if index < 2:
            raise ValueError("use the registered first-segment entry point")
        if mode.endswith("-worker"):
            return module("frontier").worker(path, int(os.environ["SLURM_ARRAY_TASK_ID"]))
        return frontier_event(root, index)
    lane = mode.removesuffix("-worker").removeprefix("after-")
    if lane not in GPU | CPU:
        raise ValueError("unregistered V12 mode")
    if mode.endswith("-worker"):
        if lane in GPU:
            if shared.gpu_snapshot()["active_gpus"] > 6:
                raise ValueError("aggregate six-A100 ceiling exceeded")
            return module(lane).worker(root / lane, index=0)
        return module(lane).worker(root / lane, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    if not mode.startswith("after-"):
        raise ValueError("unknown V12 stage")
    try:
        if lane == "ownership":
            tool = importlib.import_module("evicastkv_source_trace_adjudication")
            inventory_parent = root / lane
            try:
                result = module(lane).analyze(root / lane)
            except (FileNotFoundError, EOFError) as error:
                shared.failure(root, "standard-ownership-analysis", error)
                inventory_parent = root / "ownership-recovery"
                result = tool.recover_trace_first(
                    inventory_parent, parent=root / lane, cpu_gate_root=root
                )
            write_once(
                root / "ownership-retry-review-inventory.json", tool.inventory(inventory_parent)
            )
            return result
        if lane == "media":
            history = root / "historical_media" / CPU["historical_media"][1]
            return module(lane).analyze(
                root / lane,
                historical_media_root=root / "historical_media" if history.exists() else None,
            )
        return module(lane).analyze(root / lane)
    except Exception as error:
        return shared.failure(root, f"analyze-{lane}", error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = (
        register(args.root, execute=args.execute)
        if args.mode == "register"
        else run(args.root, args.mode)
    )
    print(json.dumps(result, allow_nan=False))
