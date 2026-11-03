#!/usr/bin/env python3
"""Consumed-model CUDA attribution, NOT a full-request benchmark or promotion.

Resolve the disagreement between repeated layer-local timings and real model
decoding. Preserve the failed operator gates. No new identities or quality run.
"""

import argparse
import os
import traceback
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from unittest.mock import patch

import evicastkv_full_prompt_native as driver

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = driver.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/native-prepared-consumed-20261103-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_native_model_profile.sbatch"
DOC = SOURCE / "mosaickv/docs/native_model_profile_20261103.md"
STEPS = (0, 64, 126)


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    negative = driver.read(PARENT / "analysis/decision.json")
    if not negative["complete"] or negative["failures"] or negative["fused_operator_gate"]:
        raise ValueError("complete negative prepared operator result required")
    old = driver.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    for p in (PARENT / "plan.json", PARENT / "analysis/decision.json", DOC):
        bound[str(p)] = sha(p)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            chain_root=str(root),
            stage="consumed_native_actual_model_kernel_attribution",
            profiled_steps=list(STEPS),
            diagnostic_warmups=1,
            diagnostic_requests_per_policy=1,
            policies=[None, "kv_fused_prepared"],
            automatic_full_request_submission=False,
            automatic_quality_submission=False,
            profiling_failed_policy_is_not_promotion_or_latency_evidence=True,
        ),
    )
    driver.pair.common.previous.cpu(root, sha(root / "plan.json"))
    if (
        "gpu"
        in driver.subprocess.check_output(
            ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%b"], text=True
        ).lower()
    ):
        raise ValueError("finish owned GPU workers first")
    driver.submit_array(root, "model_profile", "model_analysis")


def event_row(event):
    return dict(
        name=event.key,
        device_type=str(event.device_type),
        count=event.count,
        cpu_us=float(event.cpu_time_total),
        self_cpu_us=float(event.self_cpu_time_total),
        device_us=float(event.device_time_total),
        self_device_us=float(event.self_device_time_total),
        input_shapes=event.input_shapes,
    )


def validate_windows(windows):
    if len(windows) != 3 or [x["step"] for x in windows] != list(STEPS):
        raise ValueError("complete unique actual decode windows required")
    for window in windows:
        if not any("CUDA" in x["device_type"] and x["device_us"] > 0 for x in window["operators"]):
            raise ValueError("actual CUPTI CUDA activity required; CPU events are insufficient")


def scoped(function, label, active):
    import torch

    @wraps(function)
    def call(*args, **kwargs):
        if not active["value"]:
            return function(*args, **kwargs)
        with torch.profiler.record_function(label):
            return function(*args, **kwargs)

    return call


def profiled_request(adapter, call, folder):
    import torch

    from mosaickv.evaluation import full_prompt_source as native
    from mosaickv.evaluation import native_execution_plan as plans
    from mosaickv.evaluation import shared_source_kernel as original

    active, captures, counter = dict(value=False), [], 0
    decode = adapter.decode_one_token

    def step(*args, **kwargs):
        nonlocal counter
        index = counter
        counter += 1
        if index not in STEPS:
            return decode(*args, **kwargs)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            active["value"] = True
            try:
                result = decode(*args, **kwargs)
            finally:
                active["value"] = False
        captures.append((index, profiler))
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(adapter, "decode_one_token", step))
        for module, name, label in (
            (original, "project", "evicast.original_fused_projection"),
            (native, "decode_into", "evicast.source_unpack"),
            (native, "project_into", "evicast.native_projection_inclusive"),
            (native, "projection_product", "evicast.native_bmm"),
            (plans, "restore", "evicast.native_restore"),
        ):
            stack.enter_context(
                patch.object(module, name, scoped(getattr(module, name), label, active))
            )
        measured = call()
    if counter != 127:
        raise RuntimeError("full 127-step model decode required")
    windows = []
    for index, profiler in captures:
        path = folder / f"step-{index}.json"
        profiler.export_chrome_trace(str(path))
        windows.append(
            dict(
                step=index,
                trace=str(path),
                trace_sha256=sha(path),
                operators=[event_row(x) for x in profiler.key_averages(group_by_input_shape=True)],
            )
        )
    validate_windows(windows)
    # Remove all instrumented request durations: they cannot enter latency tables.
    return dict(
        windows=windows,
        parent_logits_bitwise=measured["logits_bitwise"],
        source_packet_bits_checked=measured["original_packet_bits_checked"],
        initial_encoded_bytes=measured["initial_encoded_bytes"],
        final_encoded_bytes=measured["final_encoded_bytes"],
        native_plan_restore_hits=measured["native_plan_restore_hits"],
        native_plan_restore_misses=measured["native_plan_restore_misses"],
        instrumentation_not_latency_evidence=True,
        whole_request_GPU_peak_certified=False,
    )


def worker(root, expected, index):
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root, expected, index, model_profile=True
        )
        records = []
        for policy in plan["policies"]:
            folder = out / (policy or "original")
            folder.mkdir()

            def call(policy=policy):
                return driver.execute(
                    adapter,
                    refs,
                    messages,
                    profile,
                    plan,
                    "mosaickv_exact",
                    policy,
                    audit=False,
                    validate_packet=True,
                )

            call()  # Excluded full-request warmup; no trace or timing evidence.
            report = profiled_request(adapter, call, folder)
            report["policy"] = policy
            write_once(folder / "profile.json", report)
            records.append(report)
        write_once(
            out / "profile.json",
            dict(
                records=records,
                independent_quality_N=0,
                no_quality_or_request_latency_claim=True,
                excludes_history_sealing_before_decode_entry=True,
            ),
        )
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def analyze(root, expected):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "profile.json")
    rankings = []
    if not failures:
        for index, row in enumerate(rows):
            try:
                if len(row["records"]) != 2 or {r["policy"] for r in row["records"]} != {
                    None,
                    "kv_fused_prepared",
                }:
                    raise ValueError("both complete diagnostic policies required")
                for record in row["records"]:
                    validate_windows(record["windows"])
                    if (
                        not record["parent_logits_bitwise"]
                        or not record["source_packet_bits_checked"]
                    ):
                        raise RuntimeError("actual model profile failed numerical controls")
                    for window in record["windows"]:
                        trace = Path(window["trace"]).resolve(strict=True)
                        if (
                            not trace.is_relative_to((root / f"workers/{index}").resolve())
                            or sha(trace) != window["trace_sha256"]
                        ):
                            raise ValueError("foreign or modified CUDA trace")
                        for key in ("device_us", "self_cpu_us"):
                            selected = [
                                x
                                for x in window["operators"]
                                if (
                                    "CUDA" in x["device_type"]
                                    if key == "device_us"
                                    else x["name"].startswith("evicast.")
                                )
                            ]
                            rankings.append(
                                dict(
                                    input_index=index,
                                    policy=record["policy"],
                                    step=window["step"],
                                    quantity=key,
                                    top=sorted(selected, key=lambda x: -x[key])[:10],
                                )
                            )
            except Exception as error:
                failures.append(dict(input_index=index, error=str(error)))
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not failures,
            failures=failures,
            kernel_and_host_rankings=rankings,
            no_policy_promoted=True,
            no_full_request_benchmark_submitted=True,
            no_quality_campaign_submitted=True,
            independent_quality_N=0,
            paper_superiority_established=False,
            whole_request_GPU_peak_certified=False,
        ),
    )


def main():
    driver.SCRIPT = SCRIPT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "model_profile", "model_analysis"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "model_profile":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        analyze(args.root, args.plan_sha)


if __name__ == "__main__":
    main()
