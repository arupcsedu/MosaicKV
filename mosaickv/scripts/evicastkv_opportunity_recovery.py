#!/usr/bin/env python3
"""Reuse complete predictions under a new CPU-only derived-analysis root."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

from evicastkv_analyze import collect
from evicastkv_opportunity_development import STAGE, load_analysis_lock, validate_rows

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]


def prepare(parent, root):
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("recovery must use a new immutable derived root")
    lock = load_analysis_lock(parent)
    spec, workers, rows, errors = collect(parent, STAGE)
    if errors or len(workers) != 16:
        raise ValueError("CPU-only recovery requires all 16 workers without errors")
    validate_rows(rows, lock["cohorts"]["coco"]["development"]["sample_ids"])
    root.mkdir(parents=True)
    write_once(root / "protocol_lock.json", lock)
    revised = copy.deepcopy(spec)
    revised["protocol_sha256"] = sha(root / "protocol_lock.json")
    revised["imported_workers"] = {
        str(i): dict(parent_root=str(parent), source_sha=lock["source_sha"]) for i in range(16)
    }
    write_once(root / f"jobs/{STAGE}.json", revised)
    previous_gate = parent / f"analysis/{STAGE}/GATE_evidence_development.json"
    write_once(
        root / "analysis_recovery.json",
        dict(
            producer_root=str(parent),
            producer_source_sha=lock["source_sha"],
            analysis_source_sha=revision,
            original_protocol_sha256=sha(parent / "protocol_lock.json"),
            previous_gate_sha256=sha(previous_gate),
            reused_rows=832,
            new_GPU_requests=0,
            reason="canonical JSON equality replaces Python list-versus-tuple equality only",
            thresholds_and_models_unchanged=True,
        ),
    )
    (root / "slurm").mkdir()
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii",
        "--cpus-per-task=2",
        "--mem=8G",
        "--gres=none",
        "--time=00:10:00",
        f"--output={root}/slurm/analysis-%j.out",
        f"--error={root}/slurm/analysis-%j.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_opportunity_development_analysis.sbatch"),
        str(SOURCE),
        str(root),
    ]
    job = subprocess.check_output(command, text=True).strip().split(";")[0]
    write_once(
        root / "jobs/submission.json",
        dict(
            analysis=job,
            producer_array="22123858",
            new_GPU_requests=0,
            command=command,
        ),
    )
    print(json.dumps(dict(analysis=job, reused_rows=832, new_GPU_requests=0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    a = parser.parse_args()
    if not all(
        p.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()) for p in (a.parent, a.root)
    ):
        raise ValueError("scratch-only recovery")
    prepare(a.parent, a.root)
