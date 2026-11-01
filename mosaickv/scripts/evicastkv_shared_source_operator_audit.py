#!/usr/bin/env python3
"""Trace projection accumulation without model inference or benchmark labels."""

import argparse
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import evicastkv_funded_delimiter_diagnostic as parent
import evicastkv_projection_source_cpu as cpu_source
import evicastkv_request_validation as previous
from evicastkv_campaign import environment

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
STUDY = Path("/scratch/djy8hg/runs/evicastkv/protected-source-cpu-20261101-v2")
FAILED = Path("/scratch/djy8hg/runs/evicastkv/shared-source-runtime-consumed-20261101-v1")
PRECISIONS = ("bf16_default", "bf16_no_truncated_reduction", "fp32")


def prepare(root, *, micro=False):
    if root.exists():
        raise ValueError("new immutable operator-audit root required")
    plan = previous.read(STUDY / "plan.json")
    record = previous.read(FAILED / "workers/0/result.json")
    if record["status"] != "failed" or "Tensor-likes are not close" not in record["failures"][0]:
        raise ValueError("preserve the original construction failure")
    bound = dict(plan["bound_files"])
    for path in (
        STUDY / "plan.json",
        FAILED / "workers/0/result.json",
        SOURCE / "mosaickv/docs/shared_source_operator_audit_20261101.md",
    ):
        bound[str(path)] = sha(path)
    if micro:
        numerical = Path(
            "/scratch/djy8hg/runs/evicastkv/compact-source-metadata-cpu-20261101-v1/decision.json"
        )
        model = Path(
            "/scratch/djy8hg/runs/evicastkv/source-transport-corrected-consumed-20261101-v1/analysis/decision.json"
        )
        first, second = previous.read(numerical), previous.read(model)
        eligible = next(g for g in second["gates"] if g["arm"] == "source3_exact_text_source")
        if (
            not first["complete"]
            or first["failures"]
            or not second["complete"]
            or second["failures"]
            or not first["compact_metadata_feasibility_pass"]
            or not eligible["model_gate"]
            or not eligible["all_storage_boundaries_smaller"]
        ):
            raise ValueError("positive numerical and model/storage prerequisites required")
        for path in (
            numerical,
            model,
            SOURCE / "mosaickv/docs/source_operator_microbench_20261101.md",
        ):
            bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            plan,
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            stage="operator_only_projection_accumulation_audit",
            no_model_inference=True,
            no_guard_or_tolerance_change=True,
            independent_quality_N=0,
            microbenchmark=micro,
        ),
    )
    print(sha(root / "plan.json"))


def worker(root, expected, index):
    import torch
    from safetensors import safe_open

    from mosaickv.evaluation.precision_tier import unpack_cpu
    from mosaickv.evaluation.protected_projection_source import (
        metadata_adapter,
        original_text_positions,
    )
    from mosaickv.evaluation.shared_source_format import construct
    from mosaickv.evaluation.source_projection_audit import (
        comparison,
        frozen_projection_weight,
        projected,
    )

    del index
    plan = previous.common.inputs(root, expected)
    out = root / "workers/0"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(source_sha=plan["source_sha"], plan_sha256=expected, failures=[], cases=[])
    telemetry = None
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
            raise ValueError("one A100 operator diagnostic required")
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(20261101)
        from dataclasses import asdict

        from mosaickv.measurements.telemetry import capture_gpu_environment

        telemetry = asdict(capture_gpu_environment(torch))
        metadata = metadata_adapter(plan["checkpoint"])
        checkpoint = Path(plan["checkpoint"])
        weight_map = previous.read(checkpoint / "model.safetensors.index.json")["weight_map"]
        with torch.inference_mode():
            for job in plan["jobs"]:
                capture = torch.load(job["capture"], map_location="cpu", weights_only=True)
                features = cpu_source.metadata_features(previous.read(job["input_metadata"]))
                text, _ = original_text_positions(
                    metadata,
                    previous.load_messages(features),
                    expected_length=capture["phases"][0].shape[1],
                )
                phases = tuple(t.cuda() for t in capture["phases"])
                n = phases[0].shape[1]
                identity = (
                    torch.arange(n, device="cuda", dtype=torch.int32),
                    torch.arange(n, device="cuda", dtype=torch.int16)[None].expand(32, n),
                )
                for layer, x_cpu in sorted(capture["normalized"].items()):
                    projections = {}
                    for kind in ("k", "v"):
                        name = f"language_model.model.layers.{layer}.self_attn.{kind}_proj.weight"
                        with safe_open(
                            checkpoint / weight_map[name], framework="pt", device="cpu"
                        ) as f:
                            weight = frozen_projection_weight(f.get_tensor(name), device="cuda")
                        projections[kind + "_proj"] = SimpleNamespace(weight=weight, bias=None)
                    attention = SimpleNamespace(**projections)
                    x = x_cpu.cuda()
                    original = projected(attention, x, phases, precision="bf16_default")
                    for conditional in (
                        (True,) if plan.get("microbenchmark", False) else (False, True)
                    ):
                        fmt = construct(
                            x,
                            original,
                            capture["source_memberships"][layer],
                            text,
                            phases,
                            identity,
                            exclude_global_exact=conditional,
                        )
                        if plan.get("microbenchmark", False):
                            from mosaickv.evaluation.compact_quantization_metadata import compact
                            from mosaickv.evaluation.source_operator_microbench import run

                            fmt = replace(fmt, source=compact(fmt.source))
                            record = run(fmt, attention)
                            result["cases"].append(
                                dict(
                                    input_index=job["index"],
                                    layer=layer,
                                    conditional=conditional,
                                    comparisons={},
                                    microbenchmark=record,
                                )
                            )
                            continue
                        actual = fmt.materialize(attention)
                        body = replace(
                            fmt.source,
                            codes=fmt.source.codes.cpu(),
                            minimum=fmt.source.minimum.cpu(),
                            scale=fmt.source.scale.cpu(),
                        )
                        decoded = unpack_cpu(body).squeeze(1).cuda()
                        ids = fmt.row_positions.long()
                        head, position = ids // n, ids % n
                        comparisons = {}
                        for precision in PRECISIONS:
                            native = projected(attention, decoded, phases, precision=precision)
                            for value, src in zip(native, original, strict=True):
                                value[0, head, position] = src[0, head, position]
                            comparisons[precision] = [
                                comparison(a, b) for a, b in zip(native, actual, strict=True)
                            ]
                        if not all(
                            c["registered_tolerance_pass"]
                            for pair in comparisons.values()
                            for c in pair
                        ):
                            oracle = projected(attention, decoded, phases, precision="fp64")
                            for value, src in zip(oracle, original, strict=True):
                                value[0, head, position] = src[0, head, position]
                            comparisons["fp64"] = [
                                comparison(a, b) for a, b in zip(oracle, actual, strict=True)
                            ]
                        result["cases"].append(
                            dict(
                                input_index=job["index"],
                                layer=layer,
                                conditional=conditional,
                                comparisons=comparisons,
                            )
                        )
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
    result.update(
        complete=len(result["cases"]) == (64 if plan.get("microbenchmark", False) else 128)
        and not result["failures"],
        environment=environment(),
        gpu_telemetry=telemetry,
        cuda_runtime=torch.version.cuda,
        no_model_inference=True,
        independent_quality_N=0,
        no_guard_or_tolerance_change=True,
        no_fresh_or_large_campaign_authorized=True,
    )
    write_once(out / "result.json", result)
    if not result["complete"]:
        raise SystemExit(1)


def analyze(root, expected):
    plan = previous.common.inputs(root, expected)
    record = previous.read(root / "workers/0/result.json")
    if record["source_sha"] != plan["source_sha"] or record["plan_sha256"] != expected:
        raise ValueError("operator result provenance changed")
    totals = {}
    for case in record["cases"]:
        for name, pair in case["comparisons"].items():
            totals[name] = totals.get(name, 0) + sum(c["mismatched_elements"] for c in pair)
    operator_selection = {}
    if plan.get("microbenchmark", False) and record["complete"]:
        from mosaickv.evaluation.source_operator_microbench import select_group

        operator_selection = select_group(record["cases"])
    write_once(
        root / "analysis/decision.json",
        dict(
            record,
            mismatched_element_totals=totals,
            result_sha256=sha(root / "workers/0/result.json"),
            reference_change_authorized=False,
            superiority_established=False,
            operator_selection=operator_selection,
        ),
    )


def submit(root, expected):
    original = parent.subprocess.check_output

    def scoped(command, **kwargs):
        if command and command[0] == "sbatch":
            command = [
                str(SOURCE / "mosaickv/slurm/evicastkv_shared_source_operator_audit.sbatch")
                if x.endswith("/evicastkv_funded_delimiter_diagnostic.sbatch")
                else "--array=0%1"
                if x == "--array=0-1%2"
                else x
                for x in command
            ]
        return original(command, **kwargs)

    with patch.object(parent.subprocess, "check_output", scoped):
        parent.submit(root, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("prepare", "prepare_micro", "cpu", "submit", "worker", "analyze")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "prepare_micro":
        prepare(args.root, micro=True)
    elif args.mode == "cpu":
        previous.cpu(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.mode](args.root, args.plan_sha)
