#!/usr/bin/env python3
"""Consumed COCO source capture and bounded source-code operator attribution."""

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PYTHON = "/scratch/djy8hg/env/mosaickv/bin/python"
PARENT = Path("/scratch/djy8hg/runs/evicastkv/complete-owner-replacement-fused-20261107-v2")


def read(path):
    return json.loads(Path(path).read_text())


@contextmanager
def capture_source_packets(capture):
    """Observe the existing conversion; never change request/lease arguments."""
    from mosaickv.evaluation import layer_code_delta as runtime

    original = runtime.convert

    def convert(packets):
        capture(packets)
        return original(packets)

    with patch.object(runtime, "convert", convert):
        yield


def capture_and_initialize_graph(function, torch):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function()
    # Stream capture records operations; it does not execute them.
    graph.replay()
    torch.cuda.synchronize()
    return graph


def kernel_summary(trace):
    rows = []
    for event in trace["traceEvents"]:
        if event.get("ph") != "X" or event.get("cat") not in ("kernel", "gpu_memcpy"):
            continue
        duration = event["dur"]
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(duration)
            or duration < 0
        ):
            raise ValueError("finite measured CUDA kernel/copy duration required")
        rows.append(dict(category=event["cat"], name=event["name"], device_ms=duration / 1000))
    if not rows:
        raise ValueError("no actual CUDA kernel/copy events; do not fabricate device timing")
    return rows


def encode_arm(previous, current, arm, native):
    if arm == "direct":
        from mosaickv.evaluation.lossless_code_delta_gpu import DeviceCodes

        packet = native.encode(previous, current, additive=True) if previous is not None else None
        return DeviceCodes(current.numel(), current) if packet is None else packet
    if arm not in ("common", "additive"):
        raise ValueError("registered common/search/direct encoder arm required")
    return native.encode_best(previous, current, additive_candidate=arm == "additive")[0]


def checked(root, expected):
    if sha(root / "plan.json") != expected:
        raise ValueError("immutable operator protocol changed")
    p = read(root / "plan.json")
    clean_source(SOURCE, p["source_sha"])
    if any(sha(Path(k)) != v for k, v in p["bound_files"].items()):
        raise ValueError("consumed controls or frozen outputs changed")
    return p


def infra():
    import all_head_native_source as shared

    shared.SOURCE, shared.SHELL = SOURCE, SOURCE / "mosaickv/slurm/complete_owner_operator.sbatch"
    return shared


def prepare(root, reuse_captures, decoder_policy="bitmap_prefix", timing_only=False):
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new operator diagnostic root required")
    decision = read(PARENT / "analysis/decision.json")
    if not decision["complete"] or decision["engineering_memory_and_latency_gate_passed"]:
        raise ValueError("complete negative-latency result required; no redundant positive rerun")
    parent = read(PARENT / "plan.json")
    paths = [PARENT / "plan.json", PARENT / "analysis/decision.json"]
    paths += [Path(k) for k in parent["bound_files"]]
    for i in (0, 1):
        paths += [PARENT / f"gpu/{i}/{n}" for n in ("manifest.json", "result.json", "raw.jsonl")]
        old = reuse_captures / f"workers/{i}"
        c = read(old / "capture.json")
        if (
            c.get("all_128_logits_bitwise") is not True
            or c.get("layers") != 32
            or c.get("source_is_original_COCO_not_Description_fixture") is not True
            or c["code_archive_sha"] != sha(old / "source_codes.pt")
            or c["sample_id"] != read(PARENT / f"gpu/{i}/manifest.json")["sample_id"]
            or read(old / "manifest.json")["model"]
            != read(PARENT / f"gpu/{i}/manifest.json")["model"]
        ):
            raise ValueError("complete bitwise-verified original COCO source capture required")
        paths += [old / n for n in ("capture.json", "source_codes.pt", "manifest.json")]
    if read(reuse_captures / "plan.json")["parent"]["model_controls"] != parent["model_controls"]:
        raise ValueError("capture model, inputs, and processor controls changed")
    paths.append(reuse_captures / "plan.json")
    root.mkdir(parents=True)
    (root / "logs").mkdir()
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            bound_files={str(p): sha(p) for p in paths},
            parent=parent,
            reuse_captures=str(reuse_captures),
            decoder_policy=decoder_policy,
            arms=["common", "additive", "direct"],
            seed=20261107,
            layers=32,
            trials=20,
            warmups=3,
            profile_layers=[] if timing_only else [1, 16, 31],
            missing_CUPTI_events_are_not_zero_kernel_cost=True,
            cold_cache_flush_bytes=64 * 2**20,
            benchmark_labels_accessed=False,
            fresh_or_paper_campaign_authorized=False,
            measurement_type="consumed_source_encoder_and_captured_decoder_operator_attribution",
            no_new_candidate_or_quality_or_model_latency_claim=True,
        ),
    )
    shared = infra()
    array = shared.submit(root, "worker")
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
    write_once(
        root / "submission.json",
        dict(
            array=array,
            controller=shared.submit(root, "controller"),
            analyzer=shared.submit(root, "analyze", array),
            maximum_A100s=2,
        ),
    )
    print(json.dumps(read(root / "submission.json")), flush=True)


def worker(root, expected):
    import torch
    from evicastkv_campaign import environment

    from mosaickv.config import RunConfig
    from mosaickv.evaluation import all_head_source_gpu as native
    from mosaickv.evaluation.lossless_code_delta_gpu import decode as byte_decode
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.measurements.telemetry import capture_gpu_environment

    p = checked(root, expected)
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise ValueError("one isolated A100 required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("deterministic CUDA setup required")
    torch.set_num_threads(1)
    torch.manual_seed(p["seed"])
    torch.use_deterministic_algorithms(True)
    i = int(os.environ["SLURM_ARRAY_TASK_ID"])
    folder = root / f"workers/{i}"
    folder.mkdir(parents=True, exist_ok=False)
    controls = p["parent"]["model_controls"]
    cfg = RunConfig.from_mapping(controls["parent"]["config"])
    features = read(controls["inputs"])[i]
    write_once(
        folder / "manifest.json",
        dict(
            git_sha=p["source_sha"],
            config_sha=expected,
            model=asdict(cfg.model),
            dataset=dict(
                id="COCO2014 captions",
                revision="c3213b54e173ae86517d481de17f31439d6f8c72",
                split="consumed_operator_diagnostic",
            ),
            sample_id=features["sample_id"],
            environment=environment(),
            cuda=torch.version.cuda,
            gpu=asdict(capture_gpu_environment(torch)),
            gpu_count=1,
            backend="integer_CUDA_operators_reused_COCO_source_capture",
            attention_implementation="not_executed_origin_HF_eager",
            model_weights_loaded=False,
            seed=p["seed"],
            vllm="not_used",
            sglang="not_used",
            measurement_type=p["measurement_type"],
            source_dirty=False,
            **scheduler_identity({}),
        ),
    )
    errors, rows, profiles, captured = [], [], [], {}
    try:
        origin = Path(p["reuse_captures"]) / f"workers/{i}"
        capture = read(origin / "capture.json")
        if capture["code_archive_sha"] != sha(origin / "source_codes.pt"):
            raise ValueError("verified capture changed")
        captured = torch.load(origin / "source_codes.pt", map_location="cpu", weights_only=True)
        if sorted(captured) != list(range(32)):
            raise ValueError("all 32 original source layers required")
        os.link(origin / "source_codes.pt", folder / "source_codes.pt")
        write_once(
            folder / "capture.json",
            dict(
                capture,
                reused_from=str(origin),
                reused_manifest_sha=sha(origin / "manifest.json"),
                no_new_model_inference=True,
            ),
        )
        gc.collect()
        torch.cuda.empty_cache()
        archives = {layer: codes.cuda() for layer, codes in captured.items()}
        flush = torch.empty(p["cold_cache_flush_bytes"], dtype=torch.uint8, device="cuda")
        for layer, current in sorted(archives.items()):
            previous = None if layer == 0 else archives[layer - 1]
            packets = {arm: encode_arm(previous, current, arm, native) for arm in p["arms"]}
            destinations = {arm: torch.empty_like(current) for arm in packets}
            graphs = {}

            def replay(arm, *, packets=packets, destinations=destinations):
                packet, dest = packets[arm], destinations[arm]
                if isinstance(packet, native.DeviceSymbols):
                    return native.decode(
                        dest if packet.has_previous else None,
                        packet,
                        dest,
                        implementation=p["decoder_policy"],
                    )
                return byte_decode(dest if not packet.raw else None, packet, dest)

            def reset(
                arm, cold=False, *, previous=previous, destinations=destinations, flush=flush
            ):
                if previous is not None:
                    destinations[arm].copy_(previous)
                if cold:
                    flush.zero_()
                torch.cuda.synchronize()

            for arm in packets:
                for _ in range(p["warmups"]):
                    reset(arm)
                    replay(arm)
                reset(arm)
                graph = capture_and_initialize_graph(lambda arm=arm: replay(arm), torch)
                if not torch.equal(destinations[arm], current):
                    raise ValueError("captured in-place source decoder changed bits")
                graphs[arm] = graph
            for phase in ("encoder", "decoder_captured_warm", "decoder_captured_cold"):
                for trial in range(p["trials"]):
                    arms = p["arms"]
                    cycles = [arms[j:] + arms[:j] for j in range(3)]
                    orders = [order for cycle in cycles for order in (cycle, cycle[::-1])]
                    order = orders[(trial + 2 * i) % 6]
                    # Stable balanced ordering; no winner-dependent control changes.
                    for arm in order:
                        reset(arm, phase.endswith("cold"))
                        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        start = perf_counter()
                        begin.record()
                        if phase == "encoder":
                            encoded = encode_arm(previous, current, arm, native)
                        else:
                            graphs[arm].replay()
                        end.record()
                        torch.cuda.synchronize()
                        rows.append(
                            dict(
                                layer=layer,
                                arm=arm,
                                phase=phase,
                                trial=trial,
                                wall_seconds=perf_counter() - start,
                                CUDA_stream_interval_ms=begin.elapsed_time(end),
                                code_packet_tensor_bytes=packets[arm].serialized_bytes,
                                additive_selected=getattr(packets[arm], "additive", False),
                            )
                        )
                        if phase == "encoder":
                            if encoded.serialized_bytes != packets[arm].serialized_bytes:
                                raise ValueError(
                                    "encoder microbenchmark changed its selected packet"
                                )
                            del encoded
                        elif not torch.equal(destinations[arm], current):
                            raise ValueError("operator timing changed source bits")
            if layer in p["profile_layers"]:
                for arm in packets:
                    for phase in ("encoder", "decoder_captured_cold"):
                        reset(arm, True)
                        with torch.profiler.profile(
                            activities=[
                                torch.profiler.ProfilerActivity.CPU,
                                torch.profiler.ProfilerActivity.CUDA,
                            ]
                        ) as prof:
                            if phase == "encoder":
                                encoded = encode_arm(previous, current, arm, native)
                            else:
                                graphs[arm].replay()
                            torch.cuda.synchronize()
                        path = folder / f"trace-{layer}-{arm}-{phase}.json"
                        prof.export_chrome_trace(str(path))
                        profiles.append(
                            dict(
                                layer=layer,
                                arm=arm,
                                phase=phase,
                                trace_sha=sha(path),
                                trace_path=str(path),
                                kernels=kernel_summary(read(path)),
                            )
                        )
            for graph in graphs.values():
                graph.reset()
        del flush, archives
    except Exception:
        errors.append(traceback.format_exc())
    write_once(
        folder / "result.json",
        dict(
            status="failed" if errors else "completed",
            errors=errors,
            rows=rows,
            profiles=profiles,
            source_capture_archive_sha=sha(folder / "source_codes.pt")
            if (folder / "source_codes.pt").exists()
            else None,
            not_model_quality_or_full_request_latency_or_KV_peak_evidence=True,
        ),
    )
    return int(bool(errors))


def analyze(root, expected):
    p = checked(root, expected)
    errors, results = [], {}
    for i in (0, 1):
        try:
            data = read(root / f"workers/{i}/result.json")
            if (
                data["status"] != "completed"
                or data["errors"]
                or len(data["rows"]) != 32 * len(p["arms"]) * 3 * p["trials"]
            ):
                raise ValueError(data.get("errors", "incomplete operator rows"))
            if sha(root / f"workers/{i}/source_codes.pt") != data["source_capture_archive_sha"]:
                raise ValueError("source capture archive changed")
            summary = {}
            for phase in ("encoder", "decoder_captured_warm", "decoder_captured_cold"):
                summary[phase] = {}
                for arm in p["arms"]:
                    rr = [r for r in data["rows"] if r["phase"] == phase and r["arm"] == arm]
                    summary[phase][arm] = dict(
                        sum_layer_mean_wall_ms=sum(
                            statistics.mean(r["wall_seconds"] for r in rr if r["layer"] == layer)
                            * 1000
                            for layer in range(32)
                        ),
                        sum_layer_mean_CUDA_stream_ms=sum(
                            statistics.mean(
                                r["CUDA_stream_interval_ms"] for r in rr if r["layer"] == layer
                            )
                            for layer in range(32)
                        ),
                    )
            for profile in data["profiles"]:
                if sha(Path(profile["trace_path"])) != profile["trace_sha"]:
                    raise ValueError("actual CUDA trace changed")
            results[str(i)] = dict(summary=summary, actual_CUDA_profiles=data["profiles"])
        except Exception:
            errors.append(traceback.format_exc())
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not errors,
            errors=errors,
            results=results,
            source_sha=p["source_sha"],
            model_or_KV_superiority_established=False,
            fresh_or_paper_campaign_authorized=False,
        ),
    )
    print(
        json.dumps(
            dict(
                complete=not errors,
                summary={k: v["summary"] for k, v in results.items()},
                errors=errors,
            )
        ),
        flush=True,
    )


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("action", choices=("prepare", "worker", "analyze", "controller"))
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--plan-sha")
    a.add_argument("--reuse-captures", type=Path)
    a.add_argument(
        "--decoder-policy", choices=("bitmap_prefix", "packed_words"), default="bitmap_prefix"
    )
    a.add_argument("--timing-only", action="store_true")
    args = a.parse_args()
    if args.action == "prepare":
        if args.reuse_captures is None:
            raise ValueError("retry requires verified existing captures; no new inference")
        return prepare(args.root, args.reuse_captures, args.decoder_policy, args.timing_only)
    if args.action == "controller":
        checked(args.root, args.plan_sha)
        return subprocess.call(
            [
                PYTHON,
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(args.root / "admission.json"),
                "--state",
                str(args.root / "admission_state.json"),
                "--apply",
                "--watch",
            ]
        )
    return {"worker": worker, "analyze": analyze}[args.action](args.root, args.plan_sha)


if __name__ == "__main__":
    raise SystemExit(main())
