#!/usr/bin/env bash
# Finite consumed-input V21 verification; no polling or scientific promotion.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
cd "$SOURCE_ROOT"
source mosaickv/scripts/cache_env.sh
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$SOURCE_ROOT/mosaickv/src:$SOURCE_ROOT/mosaickv/scripts"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

exec /scratch/djy8hg/env/mosaickv/bin/python - "$SOURCE_ROOT" "$@" <<'PY'
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
import sys

SOURCE = Path(sys.argv[1]).resolve()
BASE_SHA = "2ecd8242504067f1964e61ed2f7cca7c040660a3"
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
SBATCH = SOURCE / "mosaickv/slurm/fox_exact_input_v21.sbatch"
DRIVER = SOURCE / "mosaickv/scripts/evicastkv_exact_input_v21.py"


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read(path):
    if path.stat().st_size > 2 * 2**20:
        raise ValueError("bounded V21 scheduler metadata exceeded two MiB")
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_source(expected=None):
    subprocess.run(
        ["bash", str(SOURCE / "mosaickv/scripts/assert_clean_worktree.sh")],
        check=True,
        env={**os.environ, "MOSAICKV_ALLOW_DIRTY_EXPLORATORY": "0"},
    )
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if expected is not None and revision != expected:
        raise ValueError("V21 immutable source SHA changed")
    subprocess.run(
        ["git", "-C", str(SOURCE), "merge-base", "--is-ancestor", BASE_SHA, revision],
        check=True,
    )
    modules = subprocess.check_output(
        ["git", "-C", str(SOURCE), "submodule", "status", "--recursive"], text=True
    )
    if any(line[:1] in ("-", "+", "U") for line in modules.splitlines()):
        raise ValueError("all pinned submodules must be initialized and unchanged")
    return revision


def submit(root, revision, label, mode, dependencies=()):
    # Request-before-submit prevents an uncertain acknowledgement from becoming
    # a blind duplicate submission. No resubmission path is implemented.
    request = root / "jobs" / f"{label}-request.json"
    if request.exists():
        raise FileExistsError(f"scheduler request already attempted: {label}")
    gpu = mode == "worker"
    cores = 4 if mode in ("tests", "prepare") else 12
    minutes = 30 if mode in ("tests", "worker") else 10
    command = [
        "sbatch", "--parsable", "--account=bii_dsc_community",
        "--nodes=1", "--ntasks=1", f"--job-name=fox21-{label}",
        f"--partition={'bii-gpu' if gpu else 'bii,standard'}",
        f"--cpus-per-task={cores}", f"--mem={'64G' if gpu else '24G'}",
        f"--time=00:{minutes:02d}:00",
        f"--output={root}/slurm/{label}-%A_%a.out",
        f"--error={root}/slurm/{label}-%A_%a.err",
    ]
    if gpu:
        command.extend([
            "--gres=gpu:a100:1", "--reservation=bi_fox_dgx", "--array=0-1%2",
        ])
    if dependencies:
        if any(not str(job).isdigit() for job in dependencies):
            raise ValueError("numeric acknowledged Slurm dependencies required")
        command.append("--dependency=afterany:" + ":".join(map(str, dependencies)))
    command.extend([str(SBATCH), str(SOURCE), str(root), mode, revision])
    write_once(request, dict(command=command, source_sha=revision, automatic_retry=False))
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=30)
        job = output.strip().split(";", 1)[0]
        if not job.isdigit():
            raise ValueError("Slurm acknowledgement did not contain one numeric job ID")
    except Exception as error:
        write_once(root / "jobs" / f"{label}-submission-error.json", dict(
            error=f"{type(error).__name__}: {error}",
            acknowledgement_uncertain=True, automatic_retry=False,
        ))
        raise
    write_once(root / "jobs" / f"{label}-submission.json", dict(
        job=job, command=command, source_sha=revision,
        acknowledged_at_unix_seconds=time.time(), automatic_retry=False,
    ))
    return job


parser = argparse.ArgumentParser(description="Finite clean V21 CPU-to-two-A100 verification")
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--dispatch", action="store_true", help=argparse.SUPPRESS)
parser.add_argument("--source-sha", help=argparse.SUPPRESS)
args = parser.parse_args(sys.argv[2:])
root = args.root.resolve()
allowed = Path("/scratch/djy8hg/workdir").resolve()
if not args.root.is_absolute() or root == allowed or not root.is_relative_to(allowed):
    raise ValueError("an absolute immutable output directory under scratch workdir is required")

if args.dispatch:
    launch = read(root / "launch.json")
    if not args.source_sha or launch["source_sha"] != args.source_sha:
        raise ValueError("callback must bind the registered immutable SHA")
    revision = clean_source(args.source_sha)
    if launch["source_root"] != str(SOURCE):
        raise ValueError("callback source root changed")
    if any(sha(Path(name)) != expected for name, expected in launch["bindings"].items()):
        raise ValueError("registered launch bindings changed")
    completion = read(root / "jobs" / "prepare-completion.json")
    if completion["source_sha"] != revision:
        raise ValueError("preparation completion has the wrong source SHA")
    if completion["returncode"] != 0:
        # Failed producers still receive analysis, but no GPU is requested.
        analysis = submit(root, revision, "analysis", "analyze")
        print(json.dumps(dict(preparation="failed", GPU_requests=0, analysis=analysis)))
    else:
        try:
            workers = submit(root, revision, "gpu", "worker")
        except Exception:
            # Preserve the uncertain GPU request; analyze its missing artifacts
            # without attempting another worker submission.
            submit(root, revision, "analysis", "analyze")
            raise
        analysis = submit(root, revision, "analysis", "analyze", [workers])
        print(json.dumps(dict(workers=workers, array="0-1%2", analysis=analysis)))
else:
    if args.source_sha is not None:
        raise ValueError("--source-sha is reserved for the finite preparation callback")
    revision = clean_source()
    # stdin Python has no source file; bind the actual checked-in shell wrapper.
    bindings = [SOURCE / "mosaickv/scripts/submit_fox_exact_input_v21.sh", SBATCH, DRIVER,
                SOURCE / "mosaickv/env/common/requirements.lock"]
    for path in bindings:
        if not path.is_file():
            raise FileNotFoundError(path)
    root.mkdir(parents=True, exist_ok=False)
    (root / "jobs").mkdir()
    (root / "slurm").mkdir()
    write_once(root / "launch.json", dict(
        source_sha=revision, source_root=str(SOURCE), parent_source_sha=BASE_SHA,
        python=PYTHON, bindings={str(path): sha(path) for path in bindings},
        consumed_inputs_only=True, new_cohort_authorized=False,
        paper_campaign_authorized=False, automatic_retry=False, polling=False,
        maximum_a100s=6, six_a100s_explicitly_authorized=True,
        registered_GPU_workers=2, maximum_concurrent_GPU_workers=2,
        worker_resources=dict(gpus=1, cpus=12, minutes=30, reservation="bi_fox_dgx"),
        CPU_timeout_seconds=1200, analyzer_cpus=12,
        deterministic_controls=dict(CUBLAS_WORKSPACE_CONFIG=":4096:8", ATEN_CPU_CAPABILITY="avx2"),
    ))
    cpu = submit(root, revision, "cpu", "tests")
    try:
        preparation = submit(root, revision, "prepare", "prepare", [cpu])
    except Exception:
        submit(root, revision, "analysis", "analyze", [cpu])
        raise
    print(json.dumps(dict(cpu=cpu, prepare=preparation, root=str(root), source_sha=revision)))
PY
