#!/usr/bin/env python3
"""Write-once synthetic operator experiment, never measured task results."""

import argparse
import importlib.metadata
import os
import subprocess
import traceback
from pathlib import Path

import torch

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.native_history_feasibility import study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--frozen", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    revision = clean_source(source)
    if args.root.exists() or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("new immutable root and deterministic workspace before Python required")
    args.root.mkdir(parents=True)
    write_once(
        args.root / "config.json",
        dict(
            seed=20261105,
            rows=[16, 32, 48, 64, 80, 96, 112],
            warmups=5,
            repetitions=20,
            no_model_or_dataset_access=True,
            frozen_previously_audited_algorithm=args.frozen,
        ),
    )
    versions = {}
    for name in ("torch", "transformers", "vllm", "sglang"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    write_once(
        args.root / "manifest.json",
        dict(
            git_sha=revision,
            config_sha=sha(args.root / "config.json"),
            model_id="not_used",
            model_revision="not_used",
            dataset="synthetic_operator",
            dataset_revision=revision,
            cuda=torch.version.cuda,
            driver=subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip(),
            versions=versions,
            gpu=torch.cuda.get_device_name(),
            gpu_count=torch.cuda.device_count(),
            backend="torch_eager_and_cuBLASLt",
            attention_implementation="not_used",
            seed=20261105,
            measurement_type="synthetic_operator_feasibility",
            independent_quality_N=0,
            slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        ),
    )
    try:
        result = study(frozen=args.frozen)
    except Exception as error:
        result = dict(
            complete=False,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            no_runtime_integration=True,
            diagnostic=getattr(error, "diagnostic", None),
        )
    write_once(args.root / "decision.json", result)
    print({k: v for k, v in result.items() if k != "records"})


if __name__ == "__main__":
    main()
