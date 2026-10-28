#!/usr/bin/env python3
"""Finite CPU-only statistical reproduction after existing campaign callbacks.

Never mutates runtime, canonical statistics, raw rows, thresholds, or GPU jobs.
Recompute complete cohorts twice; compare numeric contents and report hashes.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

from evicastkv_campaign import SOURCE
from evicastkv_evidence import mechanism_statistics, native_statistics
from evicastkv_finalize import ORDER
from evicastkv_packet import packet, read

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

PROVENANCE_PATHS = frozenset(
    {
        ("reducer_verification", "requested_cpus"),
        ("reducer_verification", "used_cpus"),
    }
)


def compare_numeric(reference, candidate, *, tolerance=1e-10, path=(), provenance=None):
    """Reference keys must all survive; extra implementation metadata is allowed."""
    maximum = 0.0
    if isinstance(reference, dict):
        if not isinstance(candidate, dict) or not set(reference) <= set(candidate):
            raise ValueError("reproduction dropped reference fields")
        for key, value in reference.items():
            child_path = (*path, key)
            # Only these two exact reducer provenance fields may differ. Keep
            # them visible, never ignore a generic metadata subtree or count.
            if tuple(p for p in child_path if not isinstance(p, int)) in PROVENANCE_PATHS:
                other = candidate[key]
                if type(value) is not int or type(other) is not int or min(value, other) < 1:
                    raise ValueError("invalid reducer CPU provenance")
                if provenance is not None:
                    provenance.append(dict(path=list(child_path), reference=value, candidate=other))
                continue
            maximum = max(
                maximum,
                compare_numeric(
                    value,
                    candidate[key],
                    tolerance=tolerance,
                    path=child_path,
                    provenance=provenance,
                ),
            )
    elif isinstance(reference, list):
        if not isinstance(candidate, list) or len(reference) != len(candidate):
            raise ValueError("reproduction changed a denominator or contrast family")
        for index, (left, right) in enumerate(zip(reference, candidate, strict=True)):
            maximum = max(
                maximum,
                compare_numeric(
                    left, right, tolerance=tolerance, path=(*path, index), provenance=provenance
                ),
            )
    elif isinstance(reference, (int, float)) and not isinstance(reference, bool):
        if not isinstance(candidate, (int, float)) or not math.isfinite(candidate):
            raise ValueError("reproduction contains nonfinite or missing measurements")
        maximum = abs(float(reference) - float(candidate))
        if maximum > tolerance:
            raise ValueError(f"reproduction numerical mismatch: {maximum}")
    elif reference != candidate:
        raise ValueError(f"reproduction changed labels: {reference!r} != {candidate!r}")
    return maximum


def reproduce(root, output, *, include_packets=False):
    source_sha = clean_source(SOURCE)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for stage, reducer, name in (
        ("scoring", native_statistics, "native_metrics.json"),
        ("mechanisms", mechanism_statistics, "mechanism_metrics.json"),
    ):
        gate = root / "analysis" / stage / f"GATE_{stage}.json"
        if not gate.exists() or read(gate)["status"] != "passed":
            results.append(dict(stage=stage, status="NOT_RUN_OR_BLOCKED"))
            continue
        started = time.monotonic()
        a, b = output / f"{stage}-A", output / f"{stage}-B"
        if stage == "scoring":
            first = reducer(root, stage, a, workers=1)
            second = reducer(root, stage, b, workers=1)
        else:
            first = reducer(root, a, workers=1)
            second = reducer(root, b, workers=1)
        if read(a / name) != read(b / name) or sha(a / name) != sha(b / name):
            raise ValueError("repeated statistical regeneration is not deterministic")
        original = root / "analysis" / stage / "native" / name
        provenance = []
        error = (
            compare_numeric(read(original), first, provenance=provenance)
            if original.exists()
            else None
        )
        results.append(
            dict(
                stage=stage,
                status="passed",
                repetitions=2,
                resamples_per_repetition=10000,
                wall_seconds=time.monotonic() - started,
                actual_source_equivalence_max_absolute_error=error,
                reducer_provenance_comparison=provenance,
                canonical_source_present=original.exists(),
                canonical_source_sha256=sha(original) if original.exists() else None,
                regenerated_sha256=sha(a / name),
                original_overwritten=False,
                missing_canonical_statistics_not_silently_replaced=True,
            )
        )
        compare_numeric(first, second, tolerance=0)
    if include_packets:
        packet(root, output / "packet-A")
        packet(root, output / "packet-B")
        files_a = {
            str(p.relative_to(output / "packet-A")): sha(p)
            for p in (output / "packet-A").rglob("*")
            if p.is_file()
        }
        files_b = {
            str(p.relative_to(output / "packet-B")): sha(p)
            for p in (output / "packet-B").rglob("*")
            if p.is_file()
        }
        if files_a != files_b:
            raise ValueError("saved-artifact writing packet regeneration changed hashes")
        write_once(output / "packet_reproduction.json", dict(status="passed", files=files_a))
    write_once(
        output / "GATE_statistical_reproduction.json",
        dict(
            status="passed" if any(r["status"] == "passed" for r in results) else "blocked",
            source_sha=source_sha,
            runtime_source_sha=read(root / "protocol_lock.json")["source_sha"],
            protocol_sha256=sha(root / "protocol_lock.json"),
            stages=results,
            GPU_requests=0,
            scientific_gates_changed=False,
            confirmation_authorized=False,
        ),
    )


def schedule(primary, root, stage, dependency, finalizer_source_sha=None):
    revision = clean_source(SOURCE)
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--mem=16G",
        "--time=00:30:00",
        "--job-name=evicast-reproduce",
        f"--dependency=afterany:{dependency}",
        f"--output={primary}/slurm/reproduce-{stage}-%j.out",
        f"--error={primary}/slurm/reproduce-{stage}-%j.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_statistics_recheck.sbatch"),
        str(SOURCE),
        str(primary),
        str(root),
        stage,
        revision,
        finalizer_source_sha or revision,
    ]
    record = root / "jobs" / f"{stage}-reproduction-{revision[:12]}.json"
    if record.exists():
        raise FileExistsError("finite reproduction callback already registered")
    job = subprocess.check_output(command, text=True).strip().split(";")[0]
    write_once(
        record,
        dict(
            job=job,
            command=command,
            source_sha=revision,
            finalizer_source_sha=finalizer_source_sha or revision,
        ),
    )
    print(json.dumps(dict(reproduction_job=job, stage=stage)))


def follow(primary, root, stage, finalizer_source_sha=None):
    """Called after a finite finalizer; follows only its already scheduled job."""
    clean_source(SOURCE)
    decision_path = root / "analysis" / stage / "completion_decision.json"
    decision = read(decision_path) if decision_path.exists() else {}
    successor = (
        decision.get("stage")
        if decision.get("action") == "submitted_registered_successor"
        else None
    )
    if stage == "measurement":
        child = primary.parent / (primary.name + "-mechanisms")
        if (child / "protocol_lock.json").exists():
            root, successor = child, "mechanisms"
    if successor:
        if successor not in ORDER or ORDER.index(successor) <= ORDER.index(stage):
            raise ValueError("reproduction callbacks cannot loop or reorder stages")
        # Finalizer source can differ from the immutable GPU runtime source.
        revision = finalizer_source_sha or clean_source(SOURCE)
        next_job = root / "jobs" / f"{successor}-final-audit-{revision[:12]}.json"
        if next_job.exists():
            if read(next_job)["source_sha"] != revision:
                raise ValueError("unexpected finalizer source revision")
            schedule(primary, root, successor, read(next_job)["job"], revision)
            return
    output = primary / "writing" / f"reproduced-{clean_source(SOURCE)[:12]}"
    for index, candidate in enumerate((primary, primary.parent / (primary.name + "-mechanisms"))):
        if (candidate / "protocol_lock.json").exists():
            reproduce(candidate, output / f"cohort-{index}", include_packets=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reproduce", "schedule", "follow"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--stage", choices=ORDER, default="profiling")
    parser.add_argument("--dependency")
    parser.add_argument("--include-packets", action="store_true")
    parser.add_argument("--finalizer-source-sha")
    args = parser.parse_args()
    if args.mode == "reproduce":
        reproduce(args.root, args.output, include_packets=args.include_packets)
    elif args.mode == "schedule":
        schedule(
            args.primary or args.root,
            args.root,
            args.stage,
            args.dependency,
            args.finalizer_source_sha,
        )
    else:
        follow(args.primary or args.root, args.root, args.stage, args.finalizer_source_sha)


if __name__ == "__main__":
    main()
