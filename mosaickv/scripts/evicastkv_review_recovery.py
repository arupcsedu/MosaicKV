#!/usr/bin/env python3
"""Reuse complete COCO correctness and advance a finite, six-GPU development chain.

The parent's failed MileBench gate is preserved. No within-task failures or
examples are filtered, and no numerical or quality threshold is changed.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

from evicastkv_analyze import collect, pairing
from evicastkv_campaign import SOURCE, jobs_for, load_lock
from evicastkv_packet import schedule_watch
from evicastkv_scope import supported_scope

from mosaickv.evaluation.fox_campaign import (
    clean_source,
    digest,
    gate_payload,
    sha,
    write_once,
)


def runtime_files(source):
    """Include every production module/config and the numerical worker/oracle."""
    paths = [
        *sorted((source / "mosaickv/src/mosaickv").rglob("*.py")),
        *sorted((source / "mosaickv/configs/experiments/evicastkv_fox").rglob("*.json")),
        source / "mosaickv/scripts/evicastkv_worker.py",
        source / "mosaickv/scripts/evicastkv_campaign.py",
        source / "mosaickv/scripts/evicastkv_analyze.py",
    ]
    return {str(p.relative_to(source)): sha(p) for p in paths}


def imported_plan(parent_spec, jobs, workers, *, parent, producer_sha):
    """Match complete jobs exactly; a failed worker cannot become an import."""
    by_identity = {}
    for index, job in enumerate(parent_spec["jobs"]):
        identity = digest(job)
        if identity in by_identity:
            raise ValueError("duplicate parent job identity")
        by_identity[identity] = index
    result = {}
    for index, job in enumerate(jobs):
        old_index = by_identity.get(digest(job))
        if old_index is None:
            raise ValueError("no identical parent job for requested controls")
        worker = workers.get(old_index)
        if worker is None or worker["status"] != "passed" or worker["errors"]:
            raise ValueError("cannot import missing or failed correctness worker")
        # collect() looks up imports by the new index. This recovery preserves
        # indices, avoiding accidental lookup of an unrelated parent's worker.
        if old_index != index:
            raise ValueError("parent worker indices must remain identical")
        result[str(index)] = dict(parent_root=str(parent), source_sha=producer_sha)
    return result


def prepare(parent: Path, root: Path):
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("recovery output must be new and immutable")
    parent_lock_path = parent / "protocol_lock.json"
    old = json.loads(parent_lock_path.read_text())
    old_source = Path(old["source_root"])
    clean_source(old_source, old["source_sha"])
    bindings = runtime_files(SOURCE)
    if bindings != runtime_files(old_source):
        raise ValueError("production or numerical runtime differs; reuse is prohibited")
    scope = supported_scope(parent)
    if old.get("arithmetic", {}).get("attention_layout_policy", "native") != "native":
        raise ValueError("this continuation requires the original native COCO transport")
    scope_path = root / "scope_registration.json"
    write_once(scope_path, scope)
    lock = copy.deepcopy(old)
    lock.update(
        name="fox_review_native_coco_reused_controls_v4",
        source_root=str(SOURCE),
        source_sha=revision,
        enabled_workloads=["coco"],
        budgets_mib={"coco": old["budgets_mib"]["coco"]},
        methods={"coco": old["methods"]["coco"]},
        cohorts={"coco": old["cohorts"]["coco"]},
        templates={"coco": old["templates"]["coco"]},
        unexecuted_primary_contrasts=4,
        scope_registration=dict(path=str(scope_path), sha256=sha(scope_path)),
    )
    lock["resources"].update(
        maximum_a100s=6,
        array_concurrency=6,
        six_a100s_explicitly_authorized=True,
    )
    lock["authorization"].update(
        automatic_stage_progression=True,
        confirmation_authorized=False,
        fresh_confirmation_authorized=False,
    )
    for path in (parent_lock_path, scope_path, Path(scope["parent_gate_path"])):
        lock["historical_bindings"][str(path)] = sha(path)
    lock["reused_controls"] = dict(
        parent_root=str(parent),
        producer_sha=old["source_sha"],
        production_and_numerical_files_sha256=digest(bindings),
        reused_stages=["smoke", "representation_canary", "correctness"],
        scope="complete_COCO_workload_not_a_surviving_sample_intersection",
        parent_MileBench_gate="blocked_preserved",
        task_quality_used_for_scope=False,
        threshold_changes=0,
        additional_correctness_GPU_requests=0,
        development_only=True,
    )
    write_once(root / "protocol_lock.json", lock)
    write_once(root / "runtime_bindings.json", bindings)
    load_lock(root)
    for stage in ("smoke", "representation_canary", "correctness"):
        parent_spec, parent_workers, _rows, _errors = collect(parent, stage)
        workers = {w["index"]: w for w in parent_workers}
        jobs = jobs_for(lock, stage)
        imports = imported_plan(
            parent_spec, jobs, workers, parent=parent, producer_sha=old["source_sha"]
        )
        write_once(
            root / f"jobs/{stage}.json",
            dict(
                stage=stage,
                protocol_sha256=sha(root / "protocol_lock.json"),
                jobs=jobs,
                imported_workers=imports,
            ),
        )
        _spec, reused_workers, rows, errors = collect(root, stage)
        expected = sum(len(j["indices"]) * len(j["arms"]) for j in jobs)
        if errors or len(rows) != expected or any(r["status"] != "completed" for r in rows):
            raise ValueError("reused workload must have complete, successful raw controls")
        pairs = pairing(rows)
        if stage != "correctness":
            gate = gate_payload(
                stage,
                source_sha=revision,
                protocol_sha=sha(root / "protocol_lock.json"),
                passed=True,
                tests=dict(
                    expected=expected,
                    actual=len(rows),
                    completed=len(rows),
                    workers_expected=len(jobs),
                    workers_completed=len(reused_workers),
                    pairing_count=len(pairs),
                    new_GPU_requests=0,
                ),
            )
            gate["reused_producer_sha"] = old["source_sha"]
            write_once(root / f"analysis/{stage}/GATE_{stage}.json", gate)
    write_once(
        root / "exclusion_report.json",
        dict(
            status="development_inputs_previously_consumed",
            fresh_confirmation_authorized=False,
            reused_parent=str(parent),
            reason="Fresh identity/media certification is a separate prerequisite.",
        ),
    )
    write_once(
        root / "readiness.json",
        dict(
            status="prepared_CPU_reanalysis_then_scoring",
            reused_COCO_correctness_requests=scope["supported_coco_numerical_requests"],
            source_sha=revision,
            maximum_a100s=6,
            fresh_confirmation_authorized=False,
            superiority_established=False,
        ),
    )
    return lock


def submit_analysis(root):
    load_lock(root)
    (root / "slurm").mkdir(exist_ok=True)
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem=16G",
        "--time=00:20:00",
        "--job-name=fox-reuse-controls",
        f"--output={root}/slurm/reused-correctness-%j.out",
        f"--error={root}/slurm/reused-correctness-%j.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_completion.sbatch"),
        str(SOURCE),
        str(root),
        "correctness",
    ]
    identity = subprocess.check_output(command, text=True).strip().split(";")[0]
    write_once(
        root / "jobs/correctness-submission.json",
        dict(array=None, analysis=identity, GPU_requests=0, callback=command),
    )
    schedule_watch(root, "correctness", identity)
    return identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "submit"))
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    scratch = Path("/scratch/djy8hg").resolve()
    for path in (args.parent, args.root):
        if not path.resolve().is_relative_to(scratch):
            raise ValueError("scratch-only recovery")
    if args.mode == "prepare":
        prepare(args.parent, args.root)
    else:
        prepare(args.parent, args.root)
        print(json.dumps(dict(CPU_analysis_job=submit_analysis(args.root))))


if __name__ == "__main__":
    main()
