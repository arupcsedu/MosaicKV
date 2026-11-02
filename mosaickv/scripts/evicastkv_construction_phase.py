#!/usr/bin/env python3
"""Bounded consumed-input construction profiling; no new quality cohort."""

import argparse
import gc
import json
import os
import random
import statistics
from contextlib import ExitStack
from dataclasses import asdict
from functools import partial
from pathlib import Path
from unittest.mock import patch

import evicastkv_code_delta_gpu as backend
import evicastkv_common_streaming as common

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = common.SOURCE
PARENT = Path(
    "/scratch/djy8hg/runs/evicastkv/aggregate-transport-checks-consumed-20261102-v1/timing"
)


def prepare(root, *, verify_constructor_memory_once=False):
    if root.exists():
        raise FileExistsError("new immutable profile root required")
    plan = common.previous.read(PARENT / "plan.json")
    decision = common.previous.read(PARENT / "analysis/decision.json")
    if not decision["complete"] or decision["failures"] or decision["all_method_similar_time_gate"]:
        raise ValueError("complete failed-latency parent required before phase profiling")
    bound = dict(plan["bound_files"])
    for path in (
        PARENT / "plan.json",
        PARENT / "analysis/decision.json",
        SOURCE / "mosaickv/docs/construction_phase_timing_20261102.md",
    ):
        bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            plan,
            source_sha=clean_source(SOURCE),
            bound_files=bound,
            stage="consumed_nested_constructor_phase_profiling",
            paired_common_streaming_timing=False,
            maximum_worker_minutes=30 if verify_constructor_memory_once else 15,
            maximum_simultaneous_workers=2,
            profile_warmups=2,
            profile_repetitions=3,
            profile_only_no_decode=True,
            verify_constructor_memory_once=verify_constructor_memory_once,
            instrumentation_not_performance_evidence=True,
        ),
    )
    print(sha(root / "plan.json"))


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation import shared_source_format
    from mosaickv.evaluation.common_streaming_timing import METHODS
    from mosaickv.evaluation.construction_phase_timing import PhaseTimer, instrument, phase_totals
    from mosaickv.evaluation.layer_code_delta import convert
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = common.previous.common.inputs(root, expected)
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"], plan_sha256=expected)
    try:
        if (
            torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one isolated deterministic A100 required")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        cfg = RunConfig.from_mapping(common.previous.resolved_mapping(plan, job))
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        manifest = common.previous.read(
            common.REFERENCE / f"workers/{job['reference_worker']}/manifest.json"
        )
        manifest.update(
            run_id=str(out),
            parent_run_id=manifest.get("run_id"),
            source_sha=plan["source_sha"],
            source_dirty=False,
            config_sha256=digest(dict(plan=expected, worker=index)),
            plan_sha256=expected,
            measurement_type="diagnostic_measured",
            evaluator_commit=plan["source_sha"],
            environment=common.control.environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            independent_quality_N=0,
            instrumented_not_end_to_end_performance_evidence=True,
        )
        write_once(out / "manifest.json", manifest)
        refs = {}
        reference_plan = common.previous.read(common.AUDIT / "plan.json")
        for original in reference_plan["jobs"]:
            if original["input_index"] == job["input_index"]:
                folder = common.AUDIT / f"workers/{original['index']}"
                refs[common.method_of(original)] = dict(
                    payload=common.previous.read(folder / "offline_payload.json"),
                    logits=torch.load(
                        folder / "greedy_logits.pt", weights_only=True, map_location="cpu"
                    )[0],
                )
        if set(refs) != set(METHODS):
            raise ValueError("five frozen method packet/logit references required")
        messages = common.previous.load_messages(common.previous.read(job["inputs"][0]))
        profile = tuple(common.previous.read(Path(plan["profile_path"])))
        rng = random.Random(plan["seed"] + index)
        audited_methods = set()
        for trial in range(-plan["profile_warmups"], plan["profile_repetitions"]):
            methods = list(METHODS)
            rng.shuffle(methods)
            for position, method in enumerate(methods):
                gc.collect()
                torch.cuda.synchronize()
                audit_memory = (
                    plan.get("verify_constructor_memory_once", False)
                    and method not in audited_methods
                )
                timer = PhaseTimer(event_factory=lambda: torch.cuda.Event(enable_timing=True))
                with ExitStack() as stack:
                    stack.enter_context(
                        patch.object(
                            shared_source_format,
                            "fit_transport",
                            partial(shared_source_format.fit_transport, componentwise=True),
                        )
                    )
                    stack.enter_context(instrument(timer))
                    with timer.phase("complete_prefill_constructor"):
                        runtime, packets, memory = backend.construct(
                            adapter, messages, profile, method, audit=audit_memory
                        )
                        if audit_memory:
                            audited_methods.add(method)
                    if method == "mosaickv_exact":
                        with timer.phase("lossless_code_conversion"):
                            converted, code_report = convert(packets)
                    else:
                        converted, code_report = packets, None
                torch.cuda.synchronize()
                # Audits are outside every timed phase and use original uncompressed packets.
                payload_hash = common.control.digest_tensors(
                    tuple(t for packet in packets for t in packet.tensors())
                )
                if payload_hash != refs[method]["payload"]["payload_sha256"] or not torch.equal(
                    runtime.logits.cpu().reshape(-1), refs[method]["logits"].reshape(-1)
                ):
                    raise RuntimeError("profile changed a frozen packet or prefill logit")
                if (
                    memory["final_prefill_cache_bytes"] != refs[method]["payload"]["physical_bytes"]
                    or memory["original_KV_tensor_objects_alive"]
                ):
                    raise RuntimeError("profile changed storage or retirement")
                if code_report is not None and (
                    not code_report["enabled"]
                    or code_report["net_active_bytes"]
                    != plan["optimized_source_initial_bytes"][str(job["input_index"])]
                ):
                    raise RuntimeError("profile changed net encoded storage")
                trace = timer.resolve()
                row = dict(
                    status="completed",
                    method=method,
                    input_index=job["input_index"],
                    trial=trial,
                    warmup=trial < 0,
                    order_index=position,
                    trace=trace,
                    phases=phase_totals(trace),
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                    manifest_path=str(out / "manifest.json"),
                    packet_bitwise=True,
                    prefill_logits_bitwise=True,
                    original_packet_bytes=memory["final_prefill_cache_bytes"],
                    constructor_memory=memory if audit_memory else None,
                    code_report=code_report,
                    independent_quality_N=0,
                    instrumentation_not_performance_evidence=True,
                )
                with (out / "raw.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                del runtime, packets, converted, memory, timer, row
        result["status"] = "completed"
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def analyze(root, expected):
    plan = common.previous.common.inputs(root, expected)
    rows, failures = [], []
    for i in range(2):
        folder = root / f"workers/{i}"
        try:
            result = common.previous.read(folder / "result.json")
            if (
                result["status"] != "completed"
                or result["failures"]
                or result["source_sha"] != plan["source_sha"]
                or result["plan_sha256"] != expected
            ):
                raise ValueError("incomplete or mismatched producer")
            if any(sha(folder / name) != value for name, value in result["outputs"].items()):
                raise ValueError("producer artifact changed")
            rows.extend(
                json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()
            )
        except (OSError, KeyError, ValueError) as exc:
            failures.append(f"worker {i}: {exc}")
    summaries = []
    for index in (0, 1):
        for method in common.previous.read(common.AUDIT / "plan.json")["jobs"]:
            name = common.method_of(method)
            if method["input_index"] != index:
                continue
            group = [
                r
                for r in rows
                if r["input_index"] == index and r["method"] == name and not r["warmup"]
            ]
            if len(group) != plan["profile_repetitions"]:
                failures.append(f"incomplete phase repetitions: {index}/{name}")
                continue
            keys = set().union(*(r["phases"] for r in group))
            totals = {
                key: {
                    field: statistics.mean(r["phases"].get(key, {}).get(field, 0) for r in group)
                    for field in (
                        "calls",
                        "inclusive_wall_seconds",
                        "exclusive_wall_seconds",
                        "stream_elapsed_ms",
                    )
                }
                for key in keys
            }
            summaries.append(
                dict(
                    input_index=index,
                    method=name,
                    phases=totals,
                    exclusive_wall_ranking=sorted(
                        totals, key=lambda k: totals[k]["exclusive_wall_seconds"], reverse=True
                    ),
                )
            )
    expected_rows = 2 * 5 * (plan["profile_warmups"] + plan["profile_repetitions"])
    keys = [(r["input_index"], r["method"], r["trial"]) for r in rows]
    memory_gate = None
    if plan.get("verify_constructor_memory_once", False):
        memory_gate = constructor_memory_gate(rows)
        if not memory_gate:
            failures.append(
                "incomplete memory evidence or source constructor CUDA peak "
                "not lower than every baseline"
            )
    complete = len(rows) == expected_rows and len(keys) == len(set(keys)) and not failures
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=complete,
            failures=failures,
            summaries=summaries,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            independent_quality_N=0,
            no_paper_superiority_claim=True,
            instrumentation_not_performance_evidence=True,
            constructor_memory_gate=memory_gate,
            constructor_memory=[
                dict(input_index=r["input_index"], method=r["method"], **r["constructor_memory"])
                for r in rows
                if r.get("constructor_memory") is not None
            ],
        ),
    )


def constructor_memory_gate(rows):
    """Same consumed constructor boundary; never whole-request GPU memory."""
    expected = {
        (i, m)
        for i in (0, 1)
        for m in (
            "mosaickv_exact",
            "prompt_attention_topk",
            "prefixkv_reimpl",
            "vl_cache_reimpl",
            "lookm_reimpl",
        )
    }
    observed = {}
    for row in rows:
        memory = row.get("constructor_memory")
        if memory is None:
            continue
        key = row["input_index"], row["method"]
        if (
            key in observed
            or not row.get("warmup")
            or not row.get("packet_bitwise")
            or not row.get("prefill_logits_bitwise")
            or not memory.get("tensor_results_observed")
            or memory.get("original_KV_tensor_objects_alive") != 0
            or memory.get("original_native_layers_remaining") != 0
            or not isinstance(memory.get("observed_prefill_cuda_peak_bytes"), int)
            or memory["observed_prefill_cuda_peak_bytes"]
            < memory.get("final_prefill_cache_bytes", float("inf"))
        ):
            return False
        observed[key] = memory["observed_prefill_cuda_peak_bytes"]
    return set(observed) == expected and all(
        observed[i, "mosaickv_exact"]
        < min(v for (j, m), v in observed.items() if j == i and m != "mosaickv_exact")
        for i in (0, 1)
    )


def submit(root, expected):
    common.previous.common.inputs(root, expected)
    original = common.control.parent.subprocess.check_output

    def scoped(command, **kwargs):
        command = [
            str(SOURCE / "mosaickv/slurm/evicastkv_construction_phase.sbatch")
            if str(x).endswith("/evicastkv_funded_delimiter_diagnostic.sbatch")
            else x
            for x in command
        ]
        return original(command, **kwargs)

    with patch.object(common.control.parent.subprocess, "check_output", scoped):
        common.control.parent.submit(root, expected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cpu", "submit", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.root)
    elif args.mode == "cpu":
        common.previous.cpu(args.root, args.plan_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.mode](args.root, args.plan_sha)
