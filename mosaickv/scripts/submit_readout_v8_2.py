#!/usr/bin/env python3
"""Submit CPU audit -> afterany decision/Codex review, without polling."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from audit_readout_v8_2 import PROTOCOL, prepare
from bounded_history_pilot import SOURCE, digest, write_once


def dependency(job_ids):
    if not job_ids or any(not re.fullmatch(r"[1-9][0-9]*(?:_[0-9]+)?", j) for j in job_ids):
        raise ValueError("dependency IDs must be exact Slurm job/array-element IDs")
    return "afterany:" + ":".join(dict.fromkeys(job_ids))


def submit(root, parent_jobs=()):
    executable = shutil.which("codex")
    if not executable:
        raise RuntimeError("Codex CLI unavailable; do not claim automatic agent review")
    plan = prepare(root, PROTOCOL)
    plan_sha = digest(root / "plan.json")
    logs = root / "logs"
    logs.mkdir()

    def sbatch(arguments):
        output = (
            subprocess.check_output(
                ["sbatch", "--parsable", "--account=bii_dsc_community", *arguments], text=True
            )
            .strip()
            .split(";")[0]
        )
        if not re.fullmatch(r"[1-9][0-9]*", output):
            raise ValueError("unrecognized sbatch response")
        return output

    worker = sbatch(
        [
            *([f"--dependency={dependency(parent_jobs)}"] if parent_jobs else []),
            f"--output={logs}/audit-%j.out",
            f"--error={logs}/audit-%j.err",
            str(SOURCE / "mosaickv/slurm/readout_audit_cpu.sbatch"),
            str(SOURCE),
            str(root),
            plan_sha,
        ]
    )
    try:
        completion = sbatch(
            [
                f"--dependency={dependency([worker])}",
                f"--output={logs}/completion-%j.out",
                f"--error={logs}/completion-%j.err",
                str(SOURCE / "mosaickv/slurm/readout_completion.sbatch"),
                str(SOURCE),
                str(root),
                plan_sha,
                executable,
            ]
        )
    except Exception as error:
        write_once(
            root / "submission-failure.json",
            dict(
                worker=worker,
                error=type(error).__name__,
                reason="worker preserved; completion submission failed; no retry or silent polling",
            ),
        )
        raise
    record = dict(
        worker=worker,
        completion=completion,
        dependency=dependency([worker]),
        source_sha=plan["source_sha"],
        plan_sha256=plan_sha,
        polling_required=False,
        new_GPU_jobs=0,
        codex_executable=executable,
    )
    write_once(root / "submission.json", record)
    print(json.dumps(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--after", nargs="*", default=[])
    args = parser.parse_args()
    submit(args.root.resolve(), args.after)
