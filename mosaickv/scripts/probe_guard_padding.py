#!/usr/bin/env python3
"""Finite native padding feasibility; no model integration or quality campaign."""

import argparse
import json
import os
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "worker", "analyze", "controller"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan-sha")
    a = p.parse_args()
    if a.action == "prepare":
        import all_head_native_source as infra

        source = clean_source(SOURCE)
        a.root.mkdir(parents=True, exist_ok=False)
        (a.root / "logs").mkdir()
        write_once(
            a.root / "plan.json",
            dict(
                source_sha=source,
                classification="bounded_synthetic_operator_feasibility_not_pretrained_evidence",
                seeds=[20261107, 20261108],
                maximum_length=731,
                minimum_length=603,
                precision="bf16",
                bitwise_required=True,
                model_integration_authorized=False,
                fresh_or_paper_campaign_authorized=False,
            ),
        )
        infra.SOURCE, infra.SHELL = SOURCE, SOURCE / "mosaickv/slurm/probe_guard_padding.sbatch"
        array = infra.submit(a.root, "worker")
        write_once(
            a.root / "admission.json",
            dict(
                timezone="America/New_York",
                window="all_times",
                maximum_worker_minutes=120,
                maximum_a100s=4,
                arrays=[array],
                adopt_existing_holds=[array],
            ),
        )
        receipt = dict(
            array=array,
            controller=infra.submit(a.root, "controller"),
            analyzer=infra.submit(a.root, "analyze", after=array),
        )
        write_once(a.root / "submission.json", receipt)
        print(json.dumps(receipt), flush=True)
        return
    if sha(a.root / "plan.json") != a.plan_sha:
        raise ValueError("immutable operator plan changed")
    plan = read(a.root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if a.action == "controller":
        return subprocess.call(
            [
                PYTHON,
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(a.root / "admission.json"),
                "--state",
                str(a.root / "admission_state.json"),
                "--apply",
                "--watch",
            ]
        )
    if a.action == "worker":
        import torch
        from evicastkv_campaign import environment

        from mosaickv.evaluation.guard_padding_feasibility import run
        from mosaickv.evaluation.scheduler_identity import scheduler_identity
        from mosaickv.measurements.telemetry import capture_gpu_environment

        i = int(os.environ["SLURM_ARRAY_TASK_ID"])
        folder = a.root / f"workers/{i}"
        folder.mkdir(parents=True, exist_ok=False)
        errors = []
        try:
            if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
                raise ValueError("one isolated A100 required")
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
                raise ValueError("registered deterministic cuBLAS launch required")
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            write_once(
                folder / "manifest.json",
                dict(
                    git_sha=plan["source_sha"],
                    config_sha=digest(plan),
                    source_dirty=False,
                    model="not_loaded",
                    model_revision="not_used",
                    dataset="synthetic_guard_attention_geometry",
                    dataset_revision=plan["source_sha"],
                    seed=plan["seeds"][i],
                    measurement_type="synthetic_operator_conformance",
                    environment=environment(),
                    cuda=torch.version.cuda,
                    gpu=asdict(capture_gpu_environment(torch)),
                    gpu_count=1,
                    backend="pytorch_cuda",
                    attention="eager_native",
                    **scheduler_identity({}),
                ),
            )
            result = run(plan["seeds"][i])
            write_once(folder / "operator.json", result)
        except Exception:
            errors.append(traceback.format_exc())
        write_once(
            folder / "result.json", dict(status="failed" if errors else "completed", errors=errors)
        )
        return int(bool(errors))
    errors, results = {}, {}
    for i in (0, 1):
        folder = a.root / f"workers/{i}"
        try:
            if read(folder / "result.json")["status"] != "completed":
                raise ValueError(read(folder / "result.json")["errors"])
            results[str(i)] = read(folder / "operator.json")
        except Exception:
            errors[str(i)] = traceback.format_exc()
    out = a.root / "analysis"
    out.mkdir(exist_ok=False)
    write_once(
        out / "decision.json",
        dict(
            source_sha=plan["source_sha"],
            complete=not errors,
            errors=errors,
            strict_bitwise_feasible=not errors
            and all(x["strict_bitwise_gate"] for x in results.values()),
            results=results,
            model_integration_authorized=False,
            superiority_established=False,
            fresh_or_paper_campaign_authorized=False,
        ),
    )
    print(json.dumps(dict(complete=not errors)), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
