#!/usr/bin/env python3
"""Freeze an isolated exploratory snapshot/ID-only plan; submission requires --submit."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from run_prefixkv_decode_preflight import CAP, digest, write_once
from run_prefixkv_evidence_preflight import CLASSIFICATION, ORDER, select_ids

from mosaickv.manifest import _worktree_fingerprint

PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
FILES = (
    "mosaickv/docs/baselines/prefixkv_spec.md",
    "mosaickv/docs/baselines/prefixkv_evidence_ablations.md",
    "mosaickv/scripts/generate_prefixkv_profile.py",
    "mosaickv/scripts/run_prefixkv_allocation_preflight.py",
    "mosaickv/scripts/prepare_prefixkv_evidence_ablations.py",
    "mosaickv/scripts/prepare_prefixkv_evidence_gpu.py",
    "mosaickv/scripts/run_prefixkv_evidence_preflight.py",
    "mosaickv/src/mosaickv/adapters/huggingface/prefixkv_cache.py",
    "mosaickv/src/mosaickv/adapters/huggingface/prompt_evidence.py",
    "mosaickv/src/mosaickv/backends/hf_runtime.py",
    "mosaickv/src/mosaickv/baselines/prefixkv.py",
    "mosaickv/src/mosaickv/config.py",
    "mosaickv/src/mosaickv/evaluation/prefixkv_evidence_ablation.py",
    "mosaickv/tests/unit/test_prefixkv_evidence.py",
    "mosaickv/tests/unit/test_prefixkv_evidence_ablation.py",
    "mosaickv/tests/unit/test_prefixkv_evidence_preflight.py",
    "mosaickv/tests/unit/test_prompt_evidence.py",
    "mosaickv/tests/integration/test_prefixkv_evidence_runtime.py",
)


def text_once(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(content)


def prepare(args):
    repository = Path(__file__).resolve().parents[2]
    root = args.root.resolve()
    if not str(root).startswith("/sfs/weka/scratch/djy8hg/") and not str(root).startswith(
        "/scratch/djy8hg/"
    ):
        raise ValueError("snapshot and artifacts must be under scratch")
    if root.exists():
        raise FileExistsError("write-once root already exists")
    if any(not (repository / path).is_file() for path in FILES):
        raise ValueError("missing scoped source file")
    root.mkdir(parents=True)
    source = root / "source"
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(source), sha], cwd=repository, check=True
    )
    for name in FILES:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / name, source / name)
    upstream = source / "third_party/PrefixKV"
    subprocess.run(
        ["git", "clone", "--shared", str(repository / "third_party/PrefixKV"), str(upstream)],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--detach", "597f1ab032704951550f93bcc8a23f1454b80aa4"],
        cwd=upstream,
        check=True,
    )
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=source)
    with (root / "source.patch").open("xb") as stream:
        stream.write(patch)
    export = root / "untracked_source"
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=source, text=True
    ).splitlines()
    for name in untracked:
        (export / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, export / name)
    jobs, bound = [], {"source.patch": digest(root / "source.patch")}
    for name in untracked:
        bound[str((export / name).relative_to(root))] = digest(export / name)
    for task in ("textvqa", "docvqa"):
        paths = [args.parent_root / "ledgers/prefixkv" / task / f"{i:02d}.json" for i in range(5)]
        parents = [json.loads(p.read_text()) for p in paths]
        keys = ("task", "dataset_id", "dataset_revision", "split")
        if any(any(p[k] != parents[0][k] for k in keys) for p in parents):
            raise ValueError("parent ledger provenance differs")
        prior_paths = [
            args.prior_root / f"{task}_{kind}.json"
            for kind in ("consumed", "calibration", "evaluation")
        ]
        prior_ids = [s for p in prior_paths for s in json.loads(p.read_text())["sample_ids"]]
        cal, ev = select_ids([s for p in parents for s in p["sample_ids"]], prior_ids)
        job = {"task": task}
        for kind, ids in (("calibration", cal), ("evaluation", ev)):
            name = f"{task}_{kind}.json"
            write_once(
                root / name,
                {
                    **{k: parents[0][k] for k in keys},
                    "schema_version": 1,
                    "sample_ids": ids,
                    "sample_count": len(ids),
                    "source_git_sha": sha,
                    "source_git_dirty": True,
                    "classification": CLASSIFICATION,
                    "heldout_claim": False,
                    "selection_rule": (
                        "ID-only parent order, remove prior diagnostic IDs; first10cal,next16eval"
                    ),
                    "parent_ledgers": {str(p): digest(p) for p in paths},
                    "prior_diagnostic_ledgers": {str(p): digest(p) for p in prior_paths},
                },
            )
            job[kind] = name
            bound[name] = digest(root / name)
        jobs.append(job)
        bound.update({str(p.resolve()): digest(p) for p in paths + prior_paths})
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
#SBATCH --job-name=mk-prefix-evidence
#SBATCH --partition=bii-gpu
#SBATCH --reservation=bi_fox_dgx
#SBATCH --account=bii_dsc_community
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output={logs}/%x-%A_%a.out
#SBATCH --error={logs}/%x-%A_%a.err
{common}
exec srun {PYTHON} mosaickv/scripts/run_prefixkv_evidence_preflight.py --root "{root}" \\
 --plan-sha "${{MOSAICKV_PLAN_SHA:?}}" --index "${{SLURM_ARRAY_TASK_ID:?}}"
''',
    )
    text_once(
        root / "analyze.sbatch",
        f'''#!/bin/bash
#SBATCH --job-name=mk-prefix-e-analysis
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
exec {PYTHON} mosaickv/scripts/run_prefixkv_evidence_preflight.py --root "{root}" \\
 --plan-sha "${{MOSAICKV_PLAN_SHA:?}}" --analyze
''',
    )
    # Administrative code is separately snapshotted, not silently included in research changes.
    shutil.copy2(repository / "mosaickv/scripts/control_a100_schedule.py", root / "controller.py")
    text_once(
        root / "controller.sbatch",
        f'''#!/bin/bash
#SBATCH --job-name=mk-evidence-admission
#SBATCH --partition=bii
#SBATCH --account=bii_dsc_community
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256M
#SBATCH --time=02:00:00
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
        bound[name] = digest(root / name)
    plan = dict(
        schema_version=1,
        classification=CLASSIFICATION,
        source_root=str(source.resolve()),
        source_sha=sha,
        source_patch_sha256=_worktree_fingerprint(source),
        source_dirty=True,
        canonical_eligible=False,
        gpu_preflight_authorized=True,
        full_campaign_authorized=False,
        heldout_claim=False,
        jobs=jobs,
        bound_files=bound,
        arms=[list(p) for p in ORDER],
        byte_cap=CAP,
        output_tokens=64,
        seed=20271001,
        warmups=0,
        repetitions=1,
        quality_gate="same-policy Prompt Top-K mean parity, informative FullKV; paired CI reported",
        relative_improvement_claim=(
            "requires positive paired difference; CI shown; no auto-promotion"
        ),
        timing_claim="informational only; initial byte target, not constant decode bytes",
        max_a100s_global=4,
        workers=2,
        gpus_per_worker=1,
        source_files={name: digest(source / name) for name in FILES},
    )
    write_once(root / "plan.json", plan)
    print(
        json.dumps(
            {"status": "prepared", "root": str(root), "plan_sha": digest(root / "plan.json")}
        )
    )


def submit(root):
    root = root.resolve()
    if (root / "submission.json").exists():
        raise RuntimeError("already submitted")
    plan = json.loads((root / "plan.json").read_text())
    env = dict(os.environ, MOSAICKV_PLAN_SHA=digest(root / "plan.json"))
    # Validate from the actual frozen import boundary before any allocation.
    env["PYTHONPATH"] = f"{plan['source_root']}/mosaickv/src:{plan['source_root']}/mosaickv/scripts"
    subprocess.run(
        [
            PYTHON,
            "-c",
            "from pathlib import Path; from run_prefixkv_evidence_preflight "
            "import check_plan; import os; check_plan(Path(os.environ['MOSAICKV_RUN_ROOT']), "
            "os.environ['MOSAICKV_PLAN_SHA'])",
        ],
        cwd=plan["source_root"],
        env=dict(env, MOSAICKV_RUN_ROOT=str(root)),
        check=True,
    )

    def sbatch(*args):
        return (
            subprocess.check_output(["sbatch", "--parsable", *args], env=env, text=True)
            .strip()
            .split(";")[0]
        )

    gpu = sbatch("--hold", "--array=0-1%2", str(root / "worker.sbatch"))
    write_once(
        root / "submission.json",
        dict(
            array_job=gpu,
            max_global_a100s=4,
            plan_sha256=digest(root / "plan.json"),
            source_sha=plan["source_sha"],
            source_patch_sha256=plan["source_patch_sha256"],
            canonical_eligible=False,
        ),
    )
    write_once(
        root / "controller_policy.json",
        dict(
            arrays=[int(gpu)],
            adopt_existing_holds=[int(gpu)],
            blocked_arrays={},
            timezone="America/New_York",
            maximum_worker_minutes=120,
            maximum_a100s=4,
            window="all_times",
        ),
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
        check=True,
        env=env,
    )
    controller = sbatch(str(root / "controller.sbatch"))
    analysis = sbatch(f"--dependency=afterany:{gpu}", str(root / "analyze.sbatch"))
    write_once(
        root / "deployment.json",
        dict(
            array_job=gpu,
            analysis_job=analysis,
            controller_job=controller,
            existing_jobs_modified=False,
            canonical_eligible=False,
        ),
    )
    print(json.dumps(dict(array_job=gpu, analysis_job=analysis, controller_job=controller)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.submit:
        submit(args.root)
    else:
        if args.parent_root is None or args.prior_root is None:
            parser.error("prepare requires --parent-root and --prior-root")
        prepare(args)
