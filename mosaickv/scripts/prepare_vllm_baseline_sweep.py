#!/usr/bin/env python3
"""Freeze and submit the bounded vLLM projection sweep without changing HF jobs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from prepare_vllm_byte_capped_mirror import sha, write_once
from run_vllm_baseline_sweep import CAPS_MIB, METHODS, TASKS

from mosaickv.manifest import _worktree_fingerprint

PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
SCOPED_FILES = (
    "mosaickv/scripts/prepare_vllm_byte_capped_mirror.py",
    "mosaickv/scripts/prepare_vllm_baseline_sweep.py",
    "mosaickv/scripts/run_vllm_baseline_sweep.py",
    "mosaickv/tests/unit/test_vllm_baseline_sweep.py",
    "mosaickv/docs/vllm_baseline_sweep.md",
)


def text_once(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(content)


def ledger_subset(parent: dict) -> dict:
    ids = parent["sample_ids"]
    if len(ids) < 8 or len(ids) != len(set(ids)):
        raise ValueError("need eight unique parent IDs")
    return {
        **{key: parent[key] for key in ("task", "dataset_id", "dataset_revision", "split")},
        "schema_version": 1,
        "sample_ids": ids[:8],
        "sample_count": 8,
        "classification": "retrospective_development_only",
        "heldout_claim": False,
        "selection_rule": "first eight parent IDs, no labels or outcomes used",
    }


def prepare(root: Path, parent_root: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    root = root.resolve()
    if not root.is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("all artifacts must be under scratch")
    if root.exists():
        raise FileExistsError(root)
    ledgers = {}
    for task in TASKS:
        path = parent_root / "inputs/ledgers" / task / "00.json"
        ledgers[task] = dict(
            ledger_subset(json.loads(path.read_text())),
            parent_path=str(path.resolve()),
            parent_sha256=sha(path),
        )
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    root.mkdir(parents=True)
    source = root / "source"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(source), source_sha], cwd=repo, check=True
    )
    for name in SCOPED_FILES:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, source / name)
    upstream = source / "third_party/PrefixKV"
    subprocess.run(
        ["git", "clone", "--shared", str(repo / "third_party/PrefixKV"), str(upstream)], check=True
    )
    subprocess.run(
        ["git", "checkout", "--detach", "597f1ab032704951550f93bcc8a23f1454b80aa4"],
        cwd=upstream,
        check=True,
    )
    with (root / "source.patch").open("xb") as stream:
        stream.write(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=source))
    for name in subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=source, text=True
    ).splitlines():
        target = root / "untracked_source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    names = subprocess.check_output(["git", "ls-files"], cwd=source, text=True).splitlines()
    names += list(SCOPED_FILES)
    inventory = {
        name: sha(source / name) for name in sorted(set(names)) if (source / name).is_file()
    }
    profile = "third_party/PrefixKV/confs/prefixkv_llava-v1.5-7b_0.8.json"
    inventory[profile] = sha(source / profile)
    write_once(
        root / "source_attestation.json",
        {
            "source_root": str(source),
            "source_git_sha": source_sha,
            "files": inventory,
            "patch_sha": _worktree_fingerprint(source),
            "canonical_eligible": False,
            "source_patch_sha256": sha(root / "source.patch"),
        },
    )
    bound_files = {
        "source_attestation.json": sha(root / "source_attestation.json"),
        "source.patch": sha(root / "source.patch"),
    }
    jobs = []
    # Task-major ordering gives each task a low-cap preflight before larger caps.
    for task in TASKS:
        name = f"ledgers/{task}.json"
        write_once(root / name, ledgers[task])
        bound_files[name] = sha(root / name)
        jobs.extend({"task": task, "cap_bytes": cap * 2**20, "ledger": name} for cap in CAPS_MIB)
    common = f'''set -euo pipefail
module load gcc/11.4.0
source "{source}/mosaickv/scripts/cache_env.sh"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="{source}/mosaickv/src:{source}/mosaickv/scripts:{source}/mosaickv"
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "{source}"
'''
    logs = "/scratch/djy8hg/slurm-logs"
    text_once(
        root / "worker.sbatch",
        f'''#!/bin/bash
#SBATCH --job-name=mk-vllm-baselines
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --account=bii_dsc_community
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output={logs}/%x-%A_%a.out
#SBATCH --error={logs}/%x-%A_%a.err
{common}
exec srun {PYTHON} mosaickv/scripts/run_vllm_baseline_sweep.py --root "{root}" \\
 --plan-sha "${{MOSAICKV_PLAN_SHA:?}}" --index "${{SLURM_ARRAY_TASK_ID:?}}"
''',
    )
    text_once(
        root / "analyze.sbatch",
        f'''#!/bin/bash
#SBATCH --job-name=mk-vllm-sweep-analysis
#SBATCH --partition=bii
#SBATCH --account=bii_dsc_community
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output={logs}/%x-%j.out
#SBATCH --error={logs}/%x-%j.err
{common}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec {PYTHON} mosaickv/scripts/run_vllm_baseline_sweep.py --root "{root}" \\
 --plan-sha "${{MOSAICKV_PLAN_SHA:?}}" --analyze
''',
    )
    shutil.copy2(repo / "mosaickv/scripts/control_a100_schedule.py", root / "controller.py")
    text_once(
        root / "controller.sbatch",
        f'''#!/bin/bash
#SBATCH --job-name=mk-vllm-admission
#SBATCH --partition=bii
#SBATCH --account=bii_dsc_community
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=08:00:00
#SBATCH --output={logs}/%x-%j.out
#SBATCH --error={logs}/%x-%j.err
set -euo pipefail
source "{source}/mosaickv/scripts/cache_env.sh"
export PYTHONDONTWRITEBYTECODE=1
exec {PYTHON} "{root}/controller.py" --policy "{root}/controller_policy.json" \\
 --state "{root}/controller_state.json" --apply --watch
''',
    )
    for name in ("worker.sbatch", "analyze.sbatch", "controller.py", "controller.sbatch"):
        bound_files[name] = sha(root / name)
    write_once(
        root / "plan.json",
        {
            "schema_version": 1,
            "source_root": str(source),
            "source_git_sha": source_sha,
            "bound_files": bound_files,
            "jobs": jobs,
            "methods": list(METHODS),
            "canonical_eligible": False,
            "classification": "development_only",
            "heldout_claim": False,
            "maximum_a100s": 4,
            "samples_per_task": 8,
            "fullkv_latency_excluded": True,
            "online_native_selection_claim": False,
            "quality_gate": "report paired score differences and CIs; never assume parity",
            "execution_gates": [
                "retention_1_tokens",
                "A_B_A_isolation",
                "exact_physical_bytes",
                "telemetry",
                "reference_integrity",
                "complete_rows",
            ],
        },
    )
    print(json.dumps({"root": str(root), "plan_sha256": sha(root / "plan.json")}))


def submit(root: Path) -> None:
    root = root.resolve()
    if (root / "submission.json").exists():
        raise ValueError("already submitted")
    plan = json.loads((root / "plan.json").read_text())
    env = dict(os.environ, MOSAICKV_PLAN_SHA=sha(root / "plan.json"), MOSAICKV_RUN_ROOT=str(root))
    env["PYTHONPATH"] = f"{plan['source_root']}/mosaickv/src:{plan['source_root']}/mosaickv/scripts"
    subprocess.run(
        [
            PYTHON,
            "-c",
            "import os; from pathlib import Path; "
            "from run_vllm_baseline_sweep import check, configurations; "
            "root=Path(os.environ['MOSAICKV_RUN_ROOT']); "
            "p,_=check(root,os.environ['MOSAICKV_PLAN_SHA']); "
            "[configurations(Path(p['source_root']),j['task'],j['cap_bytes']) "
            "for j in p['jobs']]",
        ],
        cwd=plan["source_root"],
        env=env,
        check=True,
    )

    def sbatch(*args):
        return (
            subprocess.check_output(["sbatch", "--parsable", *args], text=True, env=env)
            .strip()
            .split(";")[0]
        )

    gpu = sbatch("--hold", f"--array=0-{len(plan['jobs']) - 1}%4", str(root / "worker.sbatch"))
    write_once(
        root / "submission.json",
        {"array_job": gpu, "plan_sha256": env["MOSAICKV_PLAN_SHA"], "canonical_eligible": False},
    )
    write_once(
        root / "controller_policy.json",
        {
            "arrays": [int(gpu)],
            "adopt_existing_holds": [int(gpu)],
            "blocked_arrays": {},
            "timezone": "America/New_York",
            "maximum_worker_minutes": 120,
            "maximum_a100s": 4,
            "window": "all_times",
        },
    )
    subprocess.run(
        [
            PYTHON,
            str(root / "controller.py"),
            "--policy",
            str(root / "controller_policy.json"),
            "--state",
            str(root / "controller_state.json"),
            "--apply",
        ],
        env=env,
        check=True,
    )
    controller = sbatch(str(root / "controller.sbatch"))
    analysis = sbatch(f"--dependency=afterany:{gpu}", str(root / "analyze.sbatch"))
    deployment = {
        "array_job": gpu,
        "controller_job": controller,
        "analysis_job": analysis,
        "maximum_a100s_global": 4,
        "existing_jobs_modified": False,
    }
    write_once(root / "deployment.json", deployment)
    print(json.dumps(deployment))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.submit:
        submit(args.root)
    else:
        if args.parent_root is None:
            parser.error("prepare requires --parent-root")
        prepare(args.root, args.parent_root)
