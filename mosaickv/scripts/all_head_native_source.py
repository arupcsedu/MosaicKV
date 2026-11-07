#!/usr/bin/env python3
"""Finite native-operator verification from hash-bound consumed source streams."""

import argparse
import json
import os
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
CPU = Path("/scratch/djy8hg/runs/evicastkv/all-head-source-feasibility-20261107-v1")
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"


def read(p):
    return json.loads(Path(p).read_text())


def checked(root, expected):
    if sha(root / "plan.json") != expected:
        raise ValueError("immutable native operator plan changed")
    plan = read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    for p, checksum in plan["bound_files"].items():
        if sha(Path(p)) != checksum:
            raise ValueError("bound code stream or CPU evidence changed")
    return plan


def submit(root, action, after=None):
    args = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name=mk-all-head-{action}",
    ]
    if action == "worker":
        args += [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--time=00:30:00",
            "--array=0-1%2",
            "--hold",
        ]
    else:
        args += [
            "--partition=bii",
            "--cpus-per-task=2",
            "--mem=8G",
            "--time=01:00:00" if action == "controller" else "--time=00:10:00",
        ]
    if after:
        args += ["--dependency=afterany:" + after]
    args += [
        f"--output={root}/logs/{action}-%A_%a.out",
        f"--error={root}/logs/{action}-%A_%a.err",
        str(SOURCE / "mosaickv/slurm/all_head_native_source.sbatch"),
        str(SOURCE),
        str(root),
        sha(root / "plan.json"),
        action,
    ]
    return subprocess.check_output(args, text=True).strip()


def prepare(root):
    source = clean_source(SOURCE)
    decision = read(CPU / "analysis/decision.json")
    if not decision["complete"] or not decision["source_factor_feasible_on_both"]:
        raise ValueError("all-head bitwise CPU source-storage screen must pass on both inputs")
    if root.exists():
        raise FileExistsError("new immutable native operator root required")
    bound = [CPU / "plan.json", CPU / "analysis/decision.json"]
    jobs = []
    for i in (0, 1):
        parent = CPU / f"workers/{i}"
        metadata = read(parent / "source_codes.json")
        if sha(parent / "source_codes.pt") != metadata["source_codes_sha256"]:
            raise ValueError("consumed original source-code archive changed")
        bound += [parent / n for n in ("source_codes.pt", "source_codes.json", "manifest.json")]
        jobs.append(
            dict(codes=str(parent / "source_codes.pt"), manifest=str(parent / "manifest.json"))
        )
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    with (root / "cpu.txt").open("x") as stream:
        completed = subprocess.run(
            [
                PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "mosaickv/tests/unit/test_all_head_source_codec.py",
                "mosaickv/tests/unit/test_storage_six_overlay.py",
            ],
            cwd=SOURCE,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        raise ValueError("CPU checks failed; no GPU submission")
    bound.append(root / "cpu.txt")
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            bound_files={str(p): sha(p) for p in bound},
            jobs=jobs,
            baseline="evicastkv_storage_only_existing_byte_XOR_codec",
            selector_bytes_per_layer=1,
            seed=20261107,
            measurement_type="native_lossless_source_operator_validation_not_model_quality",
            original_source_codes_and_logical_positions_unchanged=True,
            prototype_scope="all_source_features_shared_by_all_KV_heads",
            model_weights_loaded=False,
            model_quality_established=False,
            six_mechanism_superiority_established=False,
            fresh_or_paper_campaign_authorized=False,
        ),
    )
    array = submit(root, "worker")
    write_once(
        root / "admission.json",
        dict(
            timezone="America/New_York",
            window="all_times",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[array],
            adopt_existing_holds=[array],
        ),
    )
    receipt = dict(
        array=array,
        controller=submit(root, "controller"),
        analyzer=submit(root, "analyze", after=array),
        maximum_gpu_workers=2,
        gpu_count_per_worker=1,
        cpus_per_worker=12,
        afterany=True,
    )
    write_once(root / "submission.json", receipt)
    print(json.dumps(receipt), flush=True)


def worker(root, expected):
    import numpy as np
    import torch
    from evicastkv_campaign import environment

    from mosaickv.evaluation import all_head_source_gpu as native
    from mosaickv.evaluation.all_head_source_codec import encode as reference
    from mosaickv.evaluation.lossless_code_delta_gpu import encode as legacy
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = checked(root, expected)
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("one isolated A100 required")
    torch.manual_seed(plan["seed"])
    torch.set_num_threads(12)
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    folder = root / f"workers/{i}"
    folder.mkdir(parents=True, exist_ok=False)
    producer = read(plan["jobs"][i]["manifest"])
    write_once(
        folder / "manifest.json",
        dict(
            git_sha=plan["source_sha"],
            config_sha=digest(plan),
            source_dirty=False,
            model=producer["model"],
            dataset=producer["dataset"],
            seed=plan["seed"],
            producer_manifest_sha256=sha(Path(plan["jobs"][i]["manifest"])),
            environment=environment(),
            cuda=torch.version.cuda,
            gpu=asdict(capture_gpu_environment(torch)),
            gpu_count=1,
            backend="native_integer_codec_only",
            attention="not_used",
            measurement_type=plan["measurement_type"],
            model_weights_loaded=False,
            common_lock_sha256=sha(SOURCE / "mosaickv/env/common/requirements.lock"),
            vllm="not_used",
            sglang="not_used",
        ),
    )
    records, errors = [], []
    try:
        codes = torch.load(plan["jobs"][i]["codes"], weights_only=True, map_location="cpu")
        previous_cpu, previous = None, None
        for layer in range(32):
            current = codes[layer].cuda()
            packet = native.encode(previous, current)
            old = legacy(previous, current)
            cpu_packet = reference(previous_cpu, codes[layer].numpy())
            if packet is None:
                if not cpu_packet.raw:
                    raise RuntimeError("native and CPU encoder acceptance differ")
                records.append(
                    dict(
                        layer=layer,
                        native_raw_fallback=True,
                        bitwise=True,
                        original_code_bytes=current.numel(),
                        legacy_serialized_bytes=old.serialized_bytes,
                        native_serialized_bytes=current.numel(),
                    )
                )
            else:
                if (
                    cpu_packet.raw
                    or not np.array_equal(packet.payload.cpu().numpy(), cpu_packet.payload)
                    or not np.array_equal(
                        packet.offsets.cpu().numpy().view(np.uint32), cpu_packet.offsets
                    )
                ):
                    raise RuntimeError("native palette/residual bytes differ from CPU reference")
                actual = native.decode(previous, packet)
                if not torch.equal(actual, current):
                    raise RuntimeError("native decoded source differs from original codes")
                if previous is not None:
                    scratch = previous.clone()
                    native.decode(scratch, packet, scratch)
                    if not torch.equal(scratch, current):
                        raise RuntimeError("in-place source replay differs")
                if not np.array_equal(packet.payload.cpu().numpy(), cpu_packet.payload):
                    raise RuntimeError("replay mutated its immutable source archive")
                records.append(
                    dict(
                        layer=layer,
                        bitwise=True,
                        native_raw_fallback=False,
                        original_code_bytes=current.numel(),
                        legacy_serialized_bytes=old.serialized_bytes,
                        native_serialized_bytes=packet.serialized_bytes,
                        in_place_replay_bitwise=previous is not None,
                        archive_immutable=True,
                    )
                )
                del actual, packet
            previous_cpu, previous = codes[layer].numpy(), current
        # Operator isolation: A-B-A reconstructs different private streams;
        # no mutable source binding escapes this test or serving request.
        a = codes[0].cuda()
        b = codes[1].cuda()
        for source in (a, b, a):
            packet = native.encode(None, source)
            if packet is not None and not torch.equal(native.decode(None, packet), source):
                raise RuntimeError("A-B-A integer-source isolation failed")
        torch.cuda.synchronize()
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        folder / "result.json",
        dict(
            status="failed" if errors else "completed",
            records=records,
            errors=errors,
            source_sha=plan["source_sha"],
            model_or_KV_quality_established=False,
        ),
    )
    return int(bool(errors))


def analyze(root, expected):
    plan = checked(root, expected)
    records = [
        read(root / f"workers/{i}/result.json")
        if (root / f"workers/{i}/result.json").exists()
        else dict(status="missing", errors=["worker missing"], records=[])
        for i in (0, 1)
    ]
    passed = all(r["status"] == "completed" and len(r["records"]) == 32 for r in records)
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    write_once(
        out / "decision.json",
        dict(
            source_sha=plan["source_sha"],
            passed=passed,
            records=records,
            model_level_parity_executed=False,
            source_owner_retirement_in_model_verified=False,
            comparison_baseline=plan["baseline"],
            six_mechanism_superiority_established=False,
            fresh_or_paper_campaign_authorized=False,
            next_scope="pretrained storage-only and hybrid factorial canary"
            if passed
            else "fix only first failed native operator",
        ),
    )
    print(
        json.dumps(dict(passed=passed, actual_records=sum(len(r["records"]) for r in records))),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "worker", "analyze", "controller"))
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--plan-sha")
    a = p.parse_args()
    if a.action == "prepare":
        return prepare(a.root)
    if a.action == "worker":
        return worker(a.root, a.plan_sha)
    if a.action == "analyze":
        return analyze(a.root, a.plan_sha)
    checked(a.root, a.plan_sha)
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


if __name__ == "__main__":
    raise SystemExit(main())
