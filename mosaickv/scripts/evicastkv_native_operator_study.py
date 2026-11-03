#!/usr/bin/env python3
"""Finite consumed-input profile -> operator alternatives -> engineering verdict.

No automatic source modification, fresh identities, or quality campaign. Both
negative native reader policies remain unpromoted. Every producer gets afterany
analysis; only complete measured cases can justify another bounded diagnostic.
"""

import argparse
import gc
import os
import subprocess
import traceback
from contextlib import suppress
from pathlib import Path
from unittest.mock import patch

import evicastkv_full_prompt_native as parent

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = parent.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/full-prompt-native-consumed-20261103-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_native_operator_study.sbatch"
DOC = SOURCE / "mosaickv/docs/native_operator_study_20261103.md"


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable output and clean source required")
    negative = parent.read(PARENT / "engineering_closeout.json")
    if not negative["complete"] or negative["failures"] or negative["paired_latency_gate"]:
        raise ValueError("complete negative two-policy closeout required")
    old = parent.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    for p in (PARENT / "plan.json", PARENT / "engineering_closeout.json", DOC):
        bound[str(p)] = sha(p)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_native_operator_profile",
            chain_root=str(root),
            layers=[0, 16, 31],
            paddings=[0, 64, 127],
            products=["bmm", "wide", "wide_128"],
            warmups=5,
            repetitions=20,
            automatic_full_request_submission=False,
            automatic_quality_submission=False,
            original_reader_retained=True,
            independent_quality_N=0,
        ),
    )
    parent.pair.common.previous.cpu(root, sha(root / "plan.json"))
    if (
        "gpu"
        in subprocess.check_output(
            ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%b"], text=True
        ).lower()
    ):
        raise ValueError("preceding owned GPU jobs must finish")
    (root / "slurm").mkdir(exist_ok=True)
    base = ["sbatch", "--parsable", "--account=bii_dsc_community", "--nodes=1", "--ntasks=1"]
    args = [str(SOURCE), str(root), sha(root / "plan.json")]
    job = subprocess.check_output(
        [
            *base,
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--time=00:10:00",
            "--array=0-1%2",
            "--job-name=mk-native-ops",
            f"--output={root}/slurm/gpu-%A_%a.out",
            f"--error={root}/slurm/gpu-%A_%a.err",
            str(SCRIPT),
            *args,
            "worker",
        ],
        text=True,
    ).strip()
    from control_a100_schedule import tick

    policy = dict(
        timezone="America/New_York",
        maximum_worker_minutes=10,
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
        arrays=[job],
    )
    write_once(root / "scheduler/policy.json", policy)
    write_once(
        root / "scheduler/registration.json", tick(policy, {}, root / "scheduler/state.json", False)
    )
    after = subprocess.check_output(
        [
            *base,
            "--partition=bii,standard",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:10:00",
            f"--dependency=afterany:{job}",
            "--job-name=mk-native-ops-analysis",
            f"--output={root}/slurm/analysis-%j.out",
            f"--error={root}/slurm/analysis-%j.err",
            str(SCRIPT),
            *args,
            "analyze",
        ],
        text=True,
    ).strip()
    write_once(root / "submission.json", dict(gpu_array=job, cpu_analysis=after))


class Captured(Exception):
    pass


def worker(root, expected, index):
    import torch

    from mosaickv.evaluation import full_prompt_source as native
    from mosaickv.evaluation.common_streaming_timing import request
    from mosaickv.evaluation.native_source_profile import LAYERS, PADDINGS, study
    from mosaickv.evaluation.reconstruction_policy import use
    from mosaickv.evaluation.text_source_storage import TextSourceFormat

    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        plan, adapter, refs, messages, profile, _ = parent.setup(
            root, expected, index, operator_study=True
        )
        packets, attentions = [], []

        def capture(values, modules, policy):
            if policy != "kv" or len(values) != 32:
                raise ValueError("frozen 32-layer full prompt capture required")
            actual = parent.pair.common.control.digest_tensors(
                tuple(t for p in values for t in p.tensors())
            )
            if actual != refs["mosaickv_exact"]["payload"]["payload_sha256"]:
                raise RuntimeError("captured pre-conversion source bits changed")
            packets.extend(values)
            attentions.extend(modules)
            raise Captured

        with patch.object(native, "convert", capture), suppress(Captured):
            request(
                adapter,
                messages,
                profile,
                "mosaickv_exact",
                minimum_only=plan["exact_minimum_only"],
                componentwise_fitting=True,
                construction_geometry=plan.get("construction_geometry"),
                full_prompt_native="kv",
            )
        if len(packets) != 32:
            raise RuntimeError("capture must stop before measured decoding")
        cases = []
        for layer in LAYERS:
            packet = (
                packets[layer].base
                if isinstance(packets[layer], TextSourceFormat)
                else packets[layer]
            )
            for padding in PADDINGS:
                with use("carry_literals"):
                    case = study(packet, attentions[layer], layer, padding, seed=plan["seed"])
                cases.append(case)
                write_once(out / f"layer-{layer}-padding-{padding}.json", case)
        write_once(
            out / "profile.json",
            dict(
                cases=cases,
                no_decode_or_benchmark_scoring=True,
                no_labels_or_answers_accessed=True,
                one_original_prefill=True,
                reference_packet_hash_matched=True,
                independent_quality_N=0,
                original_reader_retained=True,
            ),
        )
        packets.clear()
        attentions.clear()
        del adapter
        gc.collect()
        torch.cuda.synchronize()
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    parent.finish_worker(out, result)


def evaluate(cases):
    """All geometry cases must win; arithmetic mean alone cannot promote a mode."""
    if {(c["layer"], c["padding"]) for c in cases} != {
        (layer, padding) for layer in (0, 16, 31) for padding in (0, 64, 127)
    } or len(cases) != 9:
        raise ValueError("complete unique registered geometry grid required")
    findings = {}
    for mode in ("wide", "wide_128"):
        ratios = [
            c["aggregate"][mode]["mean_cuda_ms"] / c["aggregate"]["original"]["mean_cuda_ms"]
            for c in cases
        ]
        findings[mode] = dict(
            operator_improvement_gate=all(r <= 0.9 for r in ratios),
            original_pipeline_ratios=ratios,
            bitwise=all(c["correctness"][mode]["bitwise"] for c in cases),
            maximum_extra_output_buffer_bytes=max(
                c["correctness"][mode]["extra_output_workspace_bytes"] for c in cases
            ),
            maximum_layer_repair_bytes=max(c["correctness"][mode]["repair_bytes"] for c in cases),
        )
    return findings


def analyze(root, expected):
    parent.checked(root, expected)
    failures, findings = [], []
    for i in (0, 1):
        try:
            out = root / f"workers/{i}"
            result = parent.read(out / "result.json")
            if result["status"] != "completed" or result["source_sha"] != clean_source(SOURCE):
                raise ValueError(f"producer failed: {result.get('failures')}")
            for name, value in result["outputs"].items():
                if sha(out / name) != value:
                    raise ValueError("modified producer output")
            findings.append(evaluate(parent.read(out / "profile.json")["cases"]))
        except (OSError, KeyError, ValueError) as error:
            failures.append(f"input {i}: {error}")
    decision = dict(
        complete=not failures,
        failures=failures,
        input_findings=findings,
        operator_candidates=[
            m
            for m in ("wide", "wide_128")
            if not failures
            and all(f[m]["operator_improvement_gate"] and f[m]["bitwise"] for f in findings)
        ],
        native_reader_promotion=False,
        automatic_full_request_submission=False,
        quality_campaign_authorized=False,
        independent_quality_N=0,
        paper_superiority_established=False,
        next_action=(
            "Inspect measured subphases and explicit output-buffer memory "
            "before any full-request run."
        ),
    )
    write_once(root / "analysis/decision.json", decision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        analyze(args.root, args.plan_sha)


if __name__ == "__main__":
    main()
