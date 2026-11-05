#!/usr/bin/env python3
"""Write-once, checkpoint-free 192-row projector diagnostic."""

import argparse
import importlib.metadata
import json
import os
import subprocess
import traceback
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--lossless-corrections", action="store_true")
    args = parser.parse_args()
    if (
        args.root.exists()
        or clean_source(SOURCE) != args.source_sha
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("new write-once root and clean immutable source required")
    write_once(
        args.root / "config.json",
        dict(
            source_sha=args.source_sha,
            registered_source_lengths=[600, 604],
            original_rows=128,
            candidate_rows=192,
            source_width=4096,
            original_full_and_joint_algorithm_descriptors_copied_without_search=True,
            added_workspace_bytes=2**20,
            added_repair_bytes_allowed=32768 if args.lossless_corrections else 0,
            lossless_corrections_explicitly_registered=args.lossless_corrections,
            preceding_zero_correction_operator_retained_as_negative=True,
            warmups=5,
            timed_repetitions=20,
            graph_replays_per_interval=5,
            same_consumed_geometry_no_dataset_access=True,
            independent_quality_N=0,
            no_runtime_or_quality_promotion_from_operator_result=True,
        ),
    )
    import torch

    versions = {}
    for name in ("torch", "transformers", "vllm", "sglang"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"

    write_once(
        args.root / "manifest.json",
        dict(
            source_sha=args.source_sha,
            source_dirty=False,
            config_sha256=sha(args.root / "config.json"),
            model="not_used",
            model_revision="not_used",
            dataset="not_used",
            dataset_revision="not_used",
            measurement_type="operator_diagnostic_measured",
            independent_quality_N=0,
            backend="private_hf_eager_operator_fixture",
            software=versions,
            cuda=torch.version.cuda,
            gpu_type=torch.cuda.get_device_name(),
            gpu_count=torch.cuda.device_count(),
            seed=20261105,
            attention_implementation="not_used",
            driver=subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip(),
            job_id=os.environ.get("SLURM_JOB_ID", "not_slurm"),
            nvidia_smi=subprocess.check_output(["nvidia-smi", "-q"], text=True),
            cache_root=os.environ.get("MOSAICKV_CACHE_ROOT"),
        ),
    )
    try:
        from mosaickv.evaluation.source_192_feasibility import study

        result = study(lossless_corrections=args.lossless_corrections)
    except Exception as error:
        result = dict(
            complete=False,
            failures=[f"{type(error).__name__}: {error}"],
            traceback=traceback.format_exc(),
            independent_quality_N=0,
        )
    write_once(args.root / "decision.json", result)
    print(json.dumps(result, allow_nan=False))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
