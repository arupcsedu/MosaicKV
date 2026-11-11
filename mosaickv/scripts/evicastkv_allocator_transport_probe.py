#!/usr/bin/env python3
"""One-shot allocator transport capability control, not a model/quality run.

Two flat pools are synthetic allocation domains. They are never called KV
categories. A successful probe cannot certify a model's whole-request KV peak.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import evicastkv_review_v8_diagnostics as shared

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.measurements.native_event_ledger import NativeEventLedger

SOURCE = Path(__file__).resolve().parents[2]
POLICY = dict(
    scope="synthetic_native_allocator_transport_no_model_or_new_identity",
    additional_CPU_scope="678_existing_native_v4_graph_artifacts_exact_producer_proof_only",
    max_events=65536,
    cpu_suite_timeout_seconds=1200,
    GPU_requests=1,
    aggregate_A100_limit=6,
    worker_A100s=1,
    worker_CPUs=12,
    worker_minutes=10,
    pools="two_flat_non_nested_current_thread_domains",
    allocation_roles="unknown_no_KV_role_inference_from_pool",
    one_shot_callback_session=True,
    requested_not_reserved_bytes=True,
    original_numerical_thresholds_changed=False,
    exact_source_KV_peak_certified=False,
    new_fresh_cohorts=0,
    independent_quality_or_superiority_established=False,
    paper_campaign_authorized=False,
)


def read(path):
    path = Path(path)
    if path.stat().st_size > 16 * 2**20:
        raise ValueError("bounded probe metadata limit exceeded")
    return json.loads(path.read_text())


def load(root):
    plan = read(Path(root) / "registration.json")
    clean_source(SOURCE, plan["source_sha"])
    if plan["policy"] != POLICY or plan["execute"] is not True:
        raise ValueError("only the frozen synthetic transport control is authorized")
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError("registered probe source changed")
    return plan


def command(root, label, mode):
    gpu = mode == "worker"
    if mode not in {
        "tests",
        "after-cpu",
        "worker",
        "analyze",
        "provenance-worker",
        "provenance-analyze",
        "finish",
    }:
        raise ValueError("unregistered one-shot probe mode")
    minutes = 25 if mode == "tests" else 15 if mode.startswith("provenance-") else 10
    cores = 1 if mode == "provenance-worker" else 12
    return [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        "--partition=bii-gpu" if gpu else "--partition=bii,standard",
        f"--job-name=fox18-{label}",
        f"--cpus-per-task={cores}",
        "--mem=16G",
        f"--time=00:{minutes:02d}:00",
        *(["--gres=gpu:a100:1", "--reservation=bi_fox_dgx"] if gpu else []),
        *(["--array=0-1%2"] if mode == "provenance-worker" else []),
        f"--output={root}/slurm/{label}-%j.out",
        f"--error={root}/slurm/{label}-%j.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_allocator_transport_probe.sbatch"),
        str(SOURCE),
        str(root),
        mode,
    ]


def submit(root, label, mode, dependencies=()):
    root = Path(root)
    cmd = command(root, label, mode)
    if dependencies:
        cmd.insert(
            2, "--dependency=afterany:" + ":".join(shared.identity(str(v)) for v in dependencies)
        )
    request = root / f"jobs/{label}-request.json"
    if request.exists():
        raise FileExistsError("request already attempted; no automatic retry")
    (root / "slurm").mkdir(parents=True, exist_ok=True)
    write_once(request, dict(command=cmd, automatic_retry=False))
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.PIPE, timeout=30)
        job = shared.identity(output.strip().split(";", 1)[0])
    except Exception as error:
        write_once(
            root / f"jobs/{label}-submission-error.json",
            dict(
                error=f"{type(error).__name__}: {error}",
                acknowledgement_uncertain=True,
                automatic_retry=False,
            ),
        )
        raise
    write_once(root / f"jobs/{label}-submission.json", dict(command=cmd, job=job, time=time.time()))
    return job


def register(root, *, execute=False):
    root = Path(root).resolve()
    allowed = Path("/scratch/djy8hg/workdir").resolve()
    if root.exists() or root == allowed or not root.is_relative_to(allowed):
        raise ValueError("new immutable scratch control directory required")
    revision = clean_source(SOURCE)
    files = [
        Path(__file__),
        SOURCE / "mosaickv/slurm/evicastkv_allocator_transport_probe.sbatch",
        SOURCE / "mosaickv/src/mosaickv/measurements/native_event_ledger.py",
        SOURCE / "mosaickv/src/mosaickv/measurements/native_allocator_events.py",
        SOURCE / "mosaickv/src/mosaickv/measurements/csrc/native_allocator_events.cpp",
        SOURCE / "mosaickv/scripts/evicastkv_native_v4_provenance_recovery.py",
        SOURCE / "mosaickv/env/common/requirements.lock",
        Path(
            "/scratch/djy8hg/env/mosaickv/lib/python3.11/site-packages/torch/include/c10/cuda/CUDACachingAllocator.h"
        ),
    ]
    write_once(
        root / "registration.json",
        dict(
            source_sha=revision,
            source_root=str(SOURCE),
            execute=execute,
            policy=POLICY,
            bindings={str(p): sha(p) for p in files},
        ),
    )
    if not execute:
        return dict(status="disabled", GPU_requests=0)
    cpu = submit(root, "cpu", "tests")
    after = submit(root, "after-cpu", "after-cpu", [cpu])
    return dict(cpu=cpu, after_cpu=after)


def after_cpu(root):
    plan = load(root)
    try:
        gate = shared.cpu_gate(root, plan["source_sha"])
    except Exception as error:
        write_once(
            Path(root) / "errors/CPU-gate.json",
            dict(
                error=f"{type(error).__name__}: {error}",
                GPU_requests_released=0,
            ),
        )
        return finish(root)
    write_once(Path(root) / "GATE_cpu_suite.json", gate)
    # A single device is below the aggregate ceiling; still reject any already
    # running over-limit state rather than assuming a fresh empty scheduler.
    queue = subprocess.check_output(
        ["squeue", "--user", os.environ["USER"], "--states=RUNNING", "--noheader", "--format=%b"],
        text=True,
        timeout=30,
    )
    running_a100s = conservative_gpu_count(queue)
    if running_a100s >= POLICY["aggregate_A100_limit"]:
        raise ValueError("aggregate A100 ceiling leaves no probe slot; do not blind resubmit")
    import evicastkv_native_v4_provenance_recovery as recovery

    try:
        prepared = recovery.prepare(Path(root) / "provenance", cpu_gate_root=root)
    except Exception as error:
        write_once(
            Path(root) / "errors/provenance-preparation.json",
            dict(
                error=f"{type(error).__name__}: {error}",
                GPU_requests_released=0,
            ),
        )
        return finish(root)
    if prepared["workers"] != 2:
        raise ValueError("registered provenance recovery worker count changed")
    provenance = submit(root, "provenance", "provenance-worker")
    provenance_analyzer = submit(root, "provenance-analysis", "provenance-analyze", [provenance])
    job = submit(root, "probe", "worker")
    dependencies = [job]
    if os.environ.get("SLURM_JOB_ID"):
        dependencies.append(os.environ["SLURM_JOB_ID"])
    analyzer = submit(root, "analysis", "analyze", dependencies)
    final = submit(root, "finish", "finish", [analyzer, provenance_analyzer, *dependencies[1:]])
    return dict(
        worker=job,
        analyzer=analyzer,
        provenance=provenance,
        provenance_analyzer=provenance_analyzer,
        final=final,
    )


def provenance_worker(root):
    import evicastkv_native_v4_provenance_recovery as recovery

    plan = load(root)
    shared.cpu_gate(root, plan["source_sha"])
    return recovery.worker(Path(root) / "provenance", int(os.environ["SLURM_ARRAY_TASK_ID"]))


def provenance_analyze(root):
    import evicastkv_native_v4_provenance_recovery as recovery

    load(root)
    return recovery.analyze(Path(root) / "provenance")


def finish(root):
    plan = load(root)
    root = Path(root)
    outputs = {}
    errors = []
    for name in (
        "GATE_cpu_suite.json",
        "GATE_native_allocator_transport.json",
        "provenance/GATE_native_v4_provenance_recovery.json",
    ):
        path = root / name
        try:
            value = read(path)
            outputs[name] = dict(sha256=sha(path), result=value)
            if value.get("errors") or any(
                value.get(key)
                for key in (
                    "missing_files",
                    "missing_shards",
                    "pending_files",
                    "blocked_files",
                )
            ):
                errors.append(f"{name}: diagnostic failed or remains incomplete")
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")
    controller_errors = {}
    for path in sorted((root / "errors").glob("*.json")):
        controller_errors[path.name] = dict(sha256=sha(path), result=read(path))
        errors.append(f"controller: {path.name}")
    value = dict(
        status="finite_diagnostics_complete_research_gates_still_blocked"
        if not errors
        else "finite_diagnostics_incomplete_research_gates_still_blocked",
        errors=errors,
        controller_errors=controller_errors,
        outputs=outputs,
        source_sha=plan["source_sha"],
        exact_source_KV_peak_certified=False,
        fresh_confirmation_established=False,
        matched_cache_quality_superiority_established=False,
        independent_quality_confirmation_established=False,
        strict_latency_superiority_established=False,
        six_component_benefits_established=False,
        paper_campaign_authorized=False,
        original_negative_results_preserved=True,
        automatic_followup_GPU_requests=0,
    )
    write_once(root / "FINAL_diagnostics.json", value)
    return value


def conservative_gpu_count(queue):
    """Count untyped GPU requests too; unknown GPU syntax fails closed."""
    count = 0
    for line in queue.splitlines():
        for entry in line.strip().split(","):
            if not entry or entry in {"N/A", "(null)"}:
                continue
            parts = entry.removeprefix("gres/").split(":")
            if parts[0] != "gpu":
                continue
            if len(parts) not in {2, 3} or not parts[-1].isdigit():
                raise ValueError("unrecognized running GPU allocation syntax")
            # Counting other GPU types is conservative for the A100 ceiling.
            count += int(parts[-1])
    return count


def snapshot_blocks(torch):
    blocks = []
    for segment in torch.cuda.memory_snapshot():
        address = segment["address"]
        for block in segment["blocks"]:
            if block["state"] in {"active_allocated", "active_awaiting_free"}:
                blocks.append(
                    dict(
                        device=segment["device"],
                        addr=address,
                        size=block["requested_size"],
                        pool=list(segment["segment_pool_id"]),
                        state=block["state"],
                    )
                )
            address += block["size"]
    return blocks


def run_operator():
    import gc

    import torch

    from mosaickv.measurements.native_allocator_events import load_recorder

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("one isolated visible CUDA device required")
    if torch.cuda.memory.get_allocator_backend() != "native":
        raise ValueError("native caching allocator transport required")
    if any(
        os.environ.get(name, "").lower() not in {"", "0", "false"}
        for name in (
            "PYTORCH_NO_CUDA_MEMORY_CACHING",
            "PYTORCH_NO_HIP_MEMORY_CACHING",
        )
    ):
        raise ValueError("uncached allocator paths bypass native event transport")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("deterministic CUDA launch configuration must precede Python")
    if "A100" not in torch.cuda.get_device_name(0):
        raise ValueError("registered capability control requires one A100")
    torch.use_deterministic_algorithms(True)
    torch.cuda.init()
    torch.cuda.synchronize()
    recorder = load_recorder()
    before = snapshot_blocks(torch)
    recorder_start = recorder.start(POLICY["max_events"])
    pool_a, pool_b = torch.cuda.MemPool(), torch.cuda.MemPool()
    witnesses = []
    with torch.cuda.use_mem_pool(pool_a):
        first = torch.zeros(1031, dtype=torch.uint8, device="cuda")
    alias = first[3:1000]
    first_address = first.untyped_storage().data_ptr()
    witnesses.append(
        dict(
            name="first_and_alias",
            addr=first_address,
            requested=1031,
            pool=list(pool_a.id),
            alias_same_backing=alias.untyped_storage().data_ptr() == first_address,
        )
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(2_000_000)
        first.add_(1)
    first.record_stream(stream)
    del alias, first
    gc.collect()
    pending_begin = recorder.cursor()
    pending = snapshot_blocks(torch)
    pending_end = recorder.cursor()
    if not any(
        row["addr"] == first_address and row["state"] == "active_awaiting_free" for row in pending
    ):
        raise ValueError("cross-stream pending-free storage control did not engage")
    torch.cuda.synchronize()
    # The installed allocator processes deferred events outside an active
    # use_mem_pool scope, not inside the private-pool allocation below.
    ordinary = torch.ones(17, dtype=torch.uint8, device="cuda")
    # Repeat a bounded same-size allocation to observe actual address reuse.
    reuse = []
    for _ in range(4):
        with torch.cuda.use_mem_pool(pool_a):
            temporary = torch.empty(1031, dtype=torch.uint8, device="cuda")
        reuse.append(temporary.untyped_storage().data_ptr())
        del temporary
    with torch.cuda.use_mem_pool(pool_a):
        retained_a = torch.ones(1031, dtype=torch.uint8, device="cuda")
    with torch.cuda.use_mem_pool(pool_b):
        retained_b = torch.ones(2051, dtype=torch.uint8, device="cuda")
    witnesses.extend(
        [
            dict(
                name="retained_a",
                addr=retained_a.untyped_storage().data_ptr(),
                requested=1031,
                pool=list(pool_a.id),
            ),
            dict(
                name="retained_b",
                addr=retained_b.untyped_storage().data_ptr(),
                requested=2051,
                pool=list(pool_b.id),
            ),
            dict(
                name="ordinary",
                addr=ordinary.untyped_storage().data_ptr(),
                requested=17,
                pool=[0, 0],
            ),
        ]
    )
    torch.cuda.synchronize()
    final = snapshot_blocks(torch)
    capture = recorder.stop()
    ledger = NativeEventLedger(before).ingest(capture)
    result = ledger.verify_snapshot(final)
    for witness in witnesses:
        matching = [
            row
            for row in ledger.records
            if row["addr"] == witness["addr"]
            and row["size"] == witness["requested"]
            and row["pool"] == witness["pool"]
        ]
        if not matching:
            raise ValueError("actual tensor backing lacks a matching native pool event")
    repeated = [row for row in ledger.records if row["addr"] in reuse and row["generation"] > 0]
    if not witnesses[0]["alias_same_backing"] or not repeated:
        raise ValueError("alias/reuse controls did not engage")
    first_generation = next(row for row in ledger.records if row["addr"] == first_address)
    if not (
        first_generation["free_requested"] is not None
        and first_generation["free_requested"] < pending_begin
        and first_generation["end"] is not None
        and first_generation["end"] >= pending_end
    ):
        raise ValueError("pending snapshot does not lie inside the native pending-free interval")
    if not any(
        row["free_requested"] is not None and row["end"] > row["free_requested"]
        for row in ledger.records
        if row["end"] is not None
    ):
        raise ValueError("native requested/completed free ordering was not observed")
    return dict(
        result=result,
        native_recorder_build_provenance=recorder.build_provenance,
        native_recorder_metadata=recorder.metadata(),
        native_recorder_start=recorder_start,
        events=capture,
        baseline=before,
        final=final,
        pending_snapshot=pending,
        pending_snapshot_cursor_interval=[pending_begin, pending_end],
        tensor_witnesses=witnesses,
        reuse_addresses=reuse,
        source_roles_assigned=False,
        torch_version=torch.__version__,
        torch_commit=torch.version.git_version,
        CUDA_runtime=torch.version.cuda,
        GPU=torch.cuda.get_device_name(0),
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        requested_allocator_transport_only=True,
        CUDA_vendor_memory_outside_allocator_covered=False,
    )


def worker(root):
    plan = load(root)
    shared.cpu_gate(root, plan["source_sha"])
    root = Path(root)
    started = time.monotonic()
    try:
        output = run_operator()
        write_once(root / "probe.json", output)
        result = dict(status="complete", errors=[], probe_sha256=sha(root / "probe.json"))
    except Exception as error:
        result = dict(status="failed", errors=[f"{type(error).__name__}: {error}"])
        # Preserve a bounded native trace even when an operator control fails.
        # This cannot turn the failed control into a successful capability run.
        try:
            from mosaickv.measurements.native_allocator_events import load_recorder

            recorder = load_recorder()
            failed_capture = recorder.stop()
            write_once(
                root / "failed_capture.json",
                dict(
                    capture=failed_capture,
                    build_provenance=recorder.build_provenance,
                    failed_control_preserved=True,
                    capability_pass=False,
                ),
            )
            result["failed_capture_sha256"] = sha(root / "failed_capture.json")
        except Exception as capture_error:
            result["failed_capture_error"] = f"{type(capture_error).__name__}: {capture_error}"
    write_once(
        root / "worker.json",
        dict(
            **result,
            policy=POLICY,
            source_sha=plan["source_sha"],
            wall_seconds=time.monotonic() - started,
            Slurm_job=os.environ.get("SLURM_JOB_ID", "not_used"),
        ),
    )
    return result


def analyze(root):
    plan = load(root)
    root = Path(root)
    errors = []
    try:
        result = read(root / "worker.json")
        if result["status"] != "complete" or result["errors"] or result["policy"] != POLICY:
            raise ValueError(f"worker did not pass: {result}")
        if (
            result["source_sha"] != plan["source_sha"]
            or sha(root / "probe.json") != result["probe_sha256"]
        ):
            raise ValueError("worker/source/probe binding mismatch")
        probe = read(root / "probe.json")
        replay = (
            NativeEventLedger(probe["baseline"])
            .ingest(probe["events"])
            .verify_snapshot(probe["final"])
        )
        if replay != probe["result"]:
            raise ValueError("independent native-event replay differs")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    gate = dict(
        status="transport_capability_pass" if not errors else "transport_capability_fail",
        errors=errors,
        source_sha=plan["source_sha"],
        policy=POLICY,
        exact_source_KV_peak_certified=False,
        independent_quality_or_superiority_established=False,
        no_automatic_model_or_campaign_promotion=True,
    )
    write_once(root / "GATE_native_allocator_transport.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "register",
            "after-cpu",
            "worker",
            "analyze",
            "provenance-worker",
            "provenance-analyze",
            "finish",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = (
        register(args.root, execute=args.execute)
        if args.mode == "register"
        else globals()[args.mode.replace("-", "_")](args.root)
    )
    print(json.dumps(result, sort_keys=True))
    if result.get("errors"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
