#!/usr/bin/env python3
"""Finite CPU code-stream screen; no model weights, labels or GPU submission."""

import argparse
import json
import os
import subprocess
import traceback
from pathlib import Path
from time import perf_counter

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
PARENT = Path("/scratch/djy8hg/runs/evicastkv/all-head-source-feasibility-20261107-v1")
SHELL = SOURCE / "mosaickv/slurm/gray_source_study.sbatch"


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, expected):
    if sha(root / "plan.json") != expected:
        raise ValueError("immutable Gray protocol changed")
    p = read(root / "plan.json")
    clean_source(SOURCE, p["source_sha"])
    if any(sha(Path(k)) != v for k, v in p["bound_files"].items()):
        raise ValueError("consumed code streams or controls changed")
    return p


def submit(root, action, after=None):
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem=8G",
        "--time=00:10:00",
        f"--job-name=mk-gray-source-{action}",
    ]
    if action == "worker":
        command += ["--array=0-1%2"]
    if after:
        command += [f"--dependency=afterany:{after}"]
    command += [
        f"--output={root}/logs/{action}-%A_%a.out",
        f"--error={root}/logs/{action}-%A_%a.err",
        str(SHELL),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        action,
    ]
    return subprocess.check_output(command, text=True).strip()


def prepare(root):
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new immutable Gray-code screen root required")
    paths = [PARENT / "plan.json"]
    for i in (0, 1):
        paths += [
            PARENT / f"workers/{i}/{name}"
            for name in ("source_codes.pt", "source_codes.json", "manifest.json")
        ]
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    with (root / "cpu.txt").open("x") as out:
        result = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "mosaickv/tests/unit/test_gray_source_codec.py",
            ],
            cwd=SOURCE,
            stdout=out,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise ValueError("CPU reversible-code tests failed")
    paths.append(root / "cpu.txt")
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            bound_files={str(p): sha(p) for p in paths},
            original_source_archive_root=str(PARENT),
            samples=2,
            layers=32,
            seed=20261107,
            grouped_bytes=384,
            Gray_symbol_bits=3,
            comparator="actual_best_legacy_byte128_and_symbol512_codec",
            minimum_code_stream_reduction_fraction_for_native_operator=0.02,
            new_source_precision_or_model_weights=False,
            benchmark_labels_accessed=False,
            measurement_type="CPU_reversible_code_stream_storage_not_GPU_KV_or_quality",
            fresh_or_paper_campaign_authorized=False,
            native_GPU_automatically_submitted=False,
            native_operator_must_use_existing_replay_destination=True,
        ),
    )
    workers = submit(root, "worker")
    receipt = dict(array=workers, analyzer=submit(root, "analyze", workers), GPUs=0)
    write_once(root / "submission.json", receipt)
    print(json.dumps(receipt), flush=True)


def worker(root, expected):
    import torch
    from evicastkv_campaign import environment

    from mosaickv.evaluation.gray_source_codec import screen
    from mosaickv.evaluation.scheduler_identity import scheduler_identity

    p = checked(root, expected)
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    folder = root / f"workers/{i}"
    folder.mkdir(parents=True, exist_ok=False)
    origin = Path(p["original_source_archive_root"]) / f"workers/{i}"
    old = read(origin / "manifest.json")
    torch.set_num_threads(1)
    write_once(
        folder / "manifest.json",
        dict(
            git_sha=p["source_sha"],
            config_sha=expected,
            source_dirty=False,
            model=old["model"],
            dataset=old["dataset"],
            input_capture_manifest=old,
            environment=environment(),
            CUDA_build=torch.version.cuda,
            CUDA_runtime="not_used_CPU",
            driver="not_used",
            gpu_type="not_used",
            gpu_count=0,
            backend="CPU_code_fixture",
            attention_implementation="not_used",
            vllm="not_used",
            sglang="not_used",
            seed=p["seed"],
            measurement_type=p["measurement_type"],
            **scheduler_identity(old),
        ),
    )
    errors, rows = [], []
    try:
        archive = torch.load(origin / "source_codes.pt", map_location="cpu", weights_only=True)
        if sorted(archive) != list(range(p["layers"])):
            raise ValueError("all original layer source streams required")
        previous = None
        for layer, tensor in sorted(archive.items()):
            start = perf_counter()
            current = tensor.numpy()
            _, proof = screen(previous, current)
            rows.append(
                dict(layer=layer, **proof, CPU_roundtrip_and_screen_seconds=perf_counter() - start)
            )
            previous = current
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        folder / "result.json",
        dict(status="failed" if errors else "completed", errors=errors, rows=rows),
    )
    return int(bool(errors))


def analyze(root, expected):
    p = checked(root, expected)
    results, errors = {}, []
    for i in (0, 1):
        try:
            r = read(root / f"workers/{i}/result.json")
            if r["status"] != "completed" or len(r["rows"]) != p["layers"]:
                raise ValueError(r)
            if not all(v["original_code_bits_exact"] for v in r["rows"]):
                raise ValueError("original source bits differ")
            totals = {
                k: sum(v[k] for v in r["rows"])
                for k in ("common_bytes", "gray384_bytes", "ordinary384_bytes", "selected_bytes")
            }
            results[str(i)] = dict(
                **totals,
                selected_layers=sum(v["gray_selected"] for v in r["rows"]),
                savings_fraction=1 - totals["selected_bytes"] / totals["common_bytes"],
                saved_bytes=totals["common_bytes"] - totals["selected_bytes"],
                original_code_bits_exact=True,
            )
        except Exception:
            errors.append(traceback.format_exc())
    passed = not errors and all(
        v["savings_fraction"] >= p["minimum_code_stream_reduction_fraction_for_native_operator"]
        for v in results.values()
    )
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not errors,
            errors=errors,
            results=results,
            source_sha=p["source_sha"],
            CPU_storage_gate_passed=passed,
            action="bounded_native_operator_only" if passed else "retain_common_codec_no_GPU",
            GPU_KV_latency_quality_or_six_mechanism_superiority_established=False,
            fresh_or_paper_campaign_authorized=False,
        ),
    )
    print(
        json.dumps(dict(complete=not errors, results=results, CPU_storage_gate_passed=passed)),
        flush=True,
    )


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("action", choices=("prepare", "worker", "analyze"))
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--plan-sha")
    args = a.parse_args()
    return {
        "prepare": lambda: prepare(args.root),
        "worker": lambda: worker(args.root, args.plan_sha),
        "analyze": lambda: analyze(args.root, args.plan_sha),
    }[args.action]()


if __name__ == "__main__":
    raise SystemExit(main())
