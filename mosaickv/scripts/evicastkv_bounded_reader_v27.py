#!/usr/bin/env python3
"""Finite consumed-input streaming-prefill and whole-KV lifecycle diagnostic."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from evicastkv_bounded_reader_v26 import parity, read

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT_SHA = "008f330edbf19421f658761efe6349d4198e2487"
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v26/run-20261111-3")
POLICY = dict(
    id="V27_streaming_layer_retirement_whole_published_KV_diagnostic_v1",
    consumed_inputs_only=True,
    fresh_cohort_authorized=False,
    paper_campaign_authorized=False,
    native_prefill="ordinary_eager_per_layer_attention_then_retire_current_native_KV",
    source="original_normalized_BF16_zero_word_patch_or_unsupported",
    raw="original_KV_joint_per_head_same_512_byte_lattice",
    raw_selection="same_prompt_attention_and_mandatory_positions_as_V26",
    native_dispatch="zero_stride_batch32_broadcast",
    shared_workspace_bytes=8650752,
    original_logical_positions=True,
    position_buffer_slots=2048,
    predictions=128,
    dtype="bf16",
    attention="eager",
    allocator_configuration="backend:native,expandable_segments:True",
    maximum_A100s=6,
    model_workers=4,
    GPUs_per_worker=1,
    CPUs_per_worker=12,
    worker_minutes=15,
    labels_accessed=False,
    whole_request_scope="published_KV_and_all_cache_required_storage_prefill_construction_decode",
    excludes="weights,ordinary_projection_and_attention_activations,opaque_vendor_workspace,allocator_reserved",
    every_ownership_observation_device_synchronized=True,
    native_allocator_generation_FREE_COMPLETED_required=True,
    all_constructor_ATen_temporary_owners_audited=True,
    bounded_allocator_trace_clock_diagnostic=True,
    no_latency_or_superiority_claim=True,
    disabled_mechanisms=["forecast", "graph", "utility", "selector", "prototype", "repair"],
)


def validate(root):
    plan = read(Path(root) / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    if digest(POLICY) != plan["policy_sha256"] or digest(plan["policy"]) != digest(POLICY):
        raise ValueError("immutable V27 policy changed")
    for name, expected in plan["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"immutable V27 input changed: {name}")
    return plan


def prepare(root):
    revision = clean_source(SOURCE)
    parent_plan = read(PARENT / "plan.json")
    gate = read(PARENT / "analysis/GATE_bounded_reader_v26.json")
    required = (
        "complete",
        "source_fidelity_128_pass",
        "constructed_decode_boundary_byte_match_pass",
        "allocator_block_byte_match_pass",
    )
    if (
        gate["source_sha"] != PARENT_SHA
        or parent_plan["source_sha"] != PARENT_SHA
        or gate["plan_sha256"] != sha(PARENT / "plan.json")
        or gate["failures"]
        or any(gate[key] is not True for key in required)
    ):
        raise ValueError("passed immutable V26 evidence required")
    bindings = dict(parent_plan["bindings"])
    for path in (PARENT / "plan.json", PARENT / "analysis/GATE_bounded_reader_v26.json"):
        bindings[str(path)] = sha(path)
    for index in range(2):
        out = PARENT / f"workers/{index}"
        result = read(out / "result.json")
        if result["failures"] or result["actual_rows"] != 3:
            raise ValueError("complete passed V26 worker required")
        bindings[str(out / "result.json")] = sha(out / "result.json")
        for name, expected in result["outputs"].items():
            if sha(out / name) != expected:
                raise ValueError("immutable V26 worker output changed")
            bindings[str(out / name)] = expected
    write_once(
        Path(root) / "plan.json",
        dict(
            source_sha=revision,
            policy=POLICY,
            policy_sha256=digest(POLICY),
            jobs=parent_plan["jobs"],
            bindings=bindings,
            expected_workers=4,
            reference_root=str(PARENT),
        ),
    )


def operator(root):
    import torch

    from mosaickv.evaluation.bounded_component_attention import selftest
    from mosaickv.evaluation.bounded_word_equality import selftest as word_selftest
    from mosaickv.evaluation.native_retirement_receipts import selftest as retirement_selftest

    plan = validate(root)
    out = Path(root) / "operator"
    out.mkdir(exist_ok=False)
    errors, report = [], {}
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
            raise ValueError("one A100 operator canary required")
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(13)
        report["attention"] = selftest(device="cuda")
        report["words"] = word_selftest(device="cuda")
        report["retirement"] = retirement_selftest(device="cuda")
        if (
            report["attention"]["all_bitwise_passed"] is not True
            or report["words"]["all_passed"] is not True
        ):
            raise ValueError("native arithmetic or bounded-word comparison failed")
        if report["retirement"]["all_passed"] is not True:
            raise ValueError("native allocator retirement generation proof failed")
    except BaseException as exc:
        errors.append(dict(error=str(exc), traceback=traceback.format_exc()))
    write_once(
        out / "result.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=sha(Path(root) / "plan.json"),
            report=report,
            failures=errors,
            SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID"),
        ),
    )


def clock_probe(root):
    """One metadata diagnostic; no model loading, GPU inference or promotion."""
    import torch

    from mosaickv.evaluation.native_retirement_receipts import trace_clock_diagnostic

    plan = validate(root)
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
        raise ValueError("one A100 metadata diagnostic required")
    report = trace_clock_diagnostic(device="cuda")
    write_once(
        Path(root) / "trace-clock/result.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=sha(Path(root) / "plan.json"),
            report=report,
            model_weights_loaded=False,
            model_workers_authorized=False,
            measurement_type="native_allocator_metadata_failure_diagnostic",
            SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID"),
        ),
    )


def worker(root, index):
    from dataclasses import asdict

    import evicastkv_review_source_prefix as fixed
    import torch
    from evicastkv_campaign import environment
    from evicastkv_request_validation import load_messages

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.bounded_component_attention import ComponentWorkspace
    from mosaickv.evaluation.bounded_component_cache import run as decode
    from mosaickv.evaluation.exact_selected_control import mandatory_positions
    from mosaickv.evaluation.kv_allocator_ledger import verify_ownership_deltas
    from mosaickv.evaluation.native_retirement_receipts import CUDAFreeCompletedTracker
    from mosaickv.evaluation.streaming_component_prefill import run as prefill
    from mosaickv.evaluation.whole_request_kv_ledger import WholeRequestKVLedger
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    root = Path(root)
    plan = validate(root)
    if index not in range(4):
        raise ValueError("exactly four registered consumed input/method pairs")
    job_index, arm_index = divmod(index, 2)
    mode = ("source", "raw")[arm_index]
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    rows, errors, tensors = [], [], {}
    ledger = None
    try:
        if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name():
            raise ValueError("one A100 required")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or any(
            os.environ.get(key) != POLICY["allocator_configuration"]
            for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
        ):
            raise ValueError("determinism/allocator controls must precede Python")
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(13)
        torch.set_num_threads(12)
        job = plan["jobs"][job_index]
        config = json.loads(json.dumps(job["config"]))
        config["execution"]["seed"] = 13
        cfg = RunConfig.from_mapping(config)
        if (cfg.model.id, cfg.model.revision) != fixed.MODEL or cfg.model.precision.value != "bf16":
            raise ValueError("same pinned BF16 LLaVA required")
        payload = fixed.label_blind_payload(job["inputs"][0])
        write_once(
            out / "manifest.json",
            dict(
                source_sha=plan["source_sha"],
                source_dirty=False,
                plan_sha256=sha(root / "plan.json"),
                config_sha256=digest(dict(config=config, policy=POLICY)),
                dataset=config["dataset"],
                model=config["model"],
                environment=environment(),
                GPU=asdict(capture_gpu_environment(torch)),
                seed=13,
                backend="hf",
                attention="eager",
                measurement_type="consumed_input_streaming_prefill_KV_ownership_diagnostic",
                vllm="not_used",
                sglang="not_used",
                benchmark_answers_read=False,
            ),
        )
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        prepared = adapter.prepare_inputs(load_messages(payload))
        references = torch.load(
            Path(plan["reference_root"]) / f"workers/{job_index}/logits.pt",
            weights_only=True,
            map_location="cpu",
        )
        full = references["FullKV_reference"]
        expected = references["original_source_bounded" if mode == "source" else "raw_topk_bounded"]
        workspace = ComponentWorkspace(device=adapter.device)
        positions = torch.arange(
            POLICY["position_buffer_slots"], device=adapter.device, dtype=torch.long
        )
        ledger = WholeRequestKVLedger(
            device=adapter.device,
            layer_count=len(adapter._language_layers()),
            source=mode == "source",
        )
        ledger.observe((*workspace.owned_tensors(), positions), "request_workspace_created")
        n = prepared.logical_sequence_length
        target = n * (len(adapter._language_layers()) * 32 * 128 * 2 + 2 * 128 * 2)
        with CUDAFreeCompletedTracker(device=adapter.device) as retirement_tracker:
            initial, packets, lifecycle = prefill(
                adapter,
                prepared,
                mode=mode,
                workspace=workspace,
                positions=positions,
                observer=ledger.observe,
                target_bytes=target if mode == "raw" else None,
                mandatory=mandatory_positions(prepared, n),
                retirement_tracker=retirement_tracker,
            )
        write_once(out / "prefill-lifecycle.json", lifecycle)
        if mode == "raw":
            frozen_selection = read(
                Path(plan["reference_root"]) / f"workers/{job_index}/raw-selection.json"
            )
            if lifecycle["raw_position_map_sha256"] != frozen_selection["position_map_sha256"]:
                raise ValueError("streaming construction changed frozen raw selected positions")
        if not torch.equal(initial.logits.reshape(-1).float().cpu(), full[0]):
            raise ValueError("streaming prefill changed original FullKV first logits")
        prefix = fixed.forced_prefix(full)
        logits, measured = decode(
            adapter,
            initial,
            packets,
            forced=prefix,
            workspace=workspace,
            positions=positions,
            ownership_observer=ledger.observe,
        )
        whole = ledger.finish(lifecycle)
        delta = whole["allocator"].pop("ownership_deltas")
        delta_name = "traces/whole_allocator_ownership.json"
        write_once(out / delta_name, dict(ownership_deltas=delta))
        whole["allocator"]["ownership_delta_artifact"] = delta_name
        whole["allocator"]["ownership_delta_sha256"] = sha(out / delta_name)
        verify_ownership_deltas(delta, expected_summary=whole["allocator"])
        tensors.update(expected=expected, FullKV=full, actual=logits)
        row = dict(
            worker=index,
            input_index=job_index,
            arm=mode,
            status="completed",
            sample_id=payload["sample_id"],
            task=payload["task"],
            parity=parity(expected, logits),
            FullKV_diagnostic=fixed.comparisons(full, logits),
            whole_KV=whole,
            measured=measured,
            lifecycle=lifecycle,
            instrumented_diagnostic_not_latency_evidence=True,
            fullkv_fallback=False,
        )
        rows.append(row)
        write_once(out / "traces/result.json", row)
    except BaseException as exc:
        errors.append(
            dict(
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
                worker=index,
                arm=mode,
            )
        )
        if ledger is not None:
            try:
                partial = ledger.ledger.summary()
                deltas = partial.pop("ownership_deltas")
                write_once(
                    out / "traces/partial_ownership.json",
                    dict(
                        ownership_deltas=deltas,
                        summary=partial,
                        phases=ledger.phases,
                        certified=False,
                    ),
                )
            except BaseException as trace_error:
                errors.append(
                    dict(
                        error_type=type(trace_error).__name__,
                        error=str(trace_error),
                        stage="partial_trace",
                    )
                )
    finally:
        if tensors:
            torch.save(tensors, out / "logits.pt")
        with (out / "raw.jsonl").open("x") as stream:
            for item in rows + [dict(status="failed", **e) for e in errors]:
                stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
        outputs = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        write_once(
            out / "result.json",
            dict(
                source_sha=plan["source_sha"],
                plan_sha256=sha(root / "plan.json"),
                actual_rows=len(rows),
                failures=errors,
                outputs=outputs,
                wall_seconds=time.monotonic() - start,
                SLURM_JOB_ID=os.environ.get("SLURM_JOB_ID"),
                SLURM_ARRAY_JOB_ID=os.environ.get("SLURM_ARRAY_JOB_ID"),
                SLURM_ARRAY_TASK_ID=os.environ.get("SLURM_ARRAY_TASK_ID"),
            ),
        )


def analyze(root):
    import evicastkv_review_source_prefix as fixed
    import torch

    from mosaickv.evaluation.kv_allocator_ledger import verify_ownership_deltas
    from mosaickv.evaluation.whole_request_kv_ledger import replay_whole_request

    root = Path(root)
    plan = validate(root)
    fixed.analysis_cpu_controls()
    errors, rows = [], []
    for index in range(4):
        try:
            out = root / f"workers/{index}"
            result = read(out / "result.json")
            if result["source_sha"] != plan["source_sha"] or result["plan_sha256"] != sha(
                root / "plan.json"
            ):
                raise ValueError("worker source/plan provenance changed")
            if result["failures"] or result["actual_rows"] != 1:
                raise ValueError(f"incomplete worker: {result['failures']}")
            for name, expected in result["outputs"].items():
                if sha(out / name) != expected:
                    raise ValueError(f"worker immutable output changed: {name}")
            raw = [json.loads(line) for line in (out / "raw.jsonl").read_text().splitlines()]
            if len(raw) != 1 or raw[0]["arm"] != ("source", "raw")[index % 2]:
                raise ValueError("one registered method per worker required")
            row = raw[0]
            saved = torch.load(out / "logits.pt", weights_only=True, map_location="cpu")
            check = parity(saved["expected"], saved["actual"])
            if digest(check) != digest(row["parity"]) or check["pass_128"] is not True:
                raise ValueError("128-prediction numerical fidelity failed")
            fixed.verify_recomputed_diagnostics(
                fixed.comparisons(saved["FullKV"], saved["actual"]), row["FullKV_diagnostic"]
            )
            whole = row["whole_KV"]
            if (
                whole["whole_request_KV_peak_certified"] is not True
                or row["fullkv_fallback"] is not False
            ):
                raise ValueError("whole-request KV lifecycle or no-fallback gate failed")
            allocator = whole["allocator"]
            path = out / allocator["ownership_delta_artifact"]
            if sha(path) != allocator["ownership_delta_sha256"]:
                raise ValueError("whole lifecycle ownership hash changed")
            deltas = read(path)["ownership_deltas"]
            verify_ownership_deltas(deltas, expected_summary=allocator)
            replay_whole_request(whole, deltas, layer_count=32, source=row["arm"] == "source")
            rows.append(row)
        except BaseException as exc:
            errors.append(dict(worker=index, error=f"{type(exc).__name__}: {exc}"))
    pairs = []
    if not errors:
        for i in range(2):
            left, right = rows[2 * i : 2 * i + 2]
            if left["sample_id"] != right["sample_id"] or left["task"] != right["task"]:
                raise ValueError("paired identity mismatch")
            keys = [
                f"{phase}_{kind}_bytes"
                for phase in ("initial", "final", "peak")
                for kind in ("tensor_backing", "allocator_block")
            ]
            matched = all(left["whole_KV"][key] == right["whole_KV"][key] for key in keys)
            pairs.append(
                dict(
                    sample_id=left["sample_id"],
                    exact_bytes_matched=matched,
                    source={k: left["whole_KV"][k] for k in keys},
                    raw={k: right["whole_KV"][k] for k in keys},
                )
            )
    gate = dict(
        source_sha=plan["source_sha"],
        plan_sha256=sha(root / "plan.json"),
        complete=not errors and len(rows) == 4,
        failures=errors,
        observations=pairs,
        source_fidelity_128_pass=not errors and len(rows) == 4,
        whole_request_KV_byte_match_pass=not errors
        and len(pairs) == 2
        and all(p["exact_bytes_matched"] for p in pairs),
        whole_request_GPU_peak_certified=False,
        independent_quality_confirmation=False,
        paper_campaign_authorized=False,
    )
    write_once(root / "analysis/GATE_bounded_reader_v27.json", gate)
    return gate


def submit(root, revision, label, mode, *, dependencies=(), gpu=False, array=False):
    root = Path(root)
    command = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--nodes=1",
        "--ntasks=1",
        f"--job-name=fox27-{label}",
        f"--partition={'bii-gpu' if gpu else 'bii,standard'}",
        f"--cpus-per-task={12 if gpu or mode == 'analyze' else 4}",
        f"--mem={'64G' if gpu else '24G'}",
        f"--time=00:{30 if mode == 'tests' else 15 if array else 5}:00",
        f"--output={root}/slurm/{label}-%A_%a.out",
        f"--error={root}/slurm/{label}-%A_%a.err",
    ]
    if gpu:
        command += ["--gres=gpu:a100:1", "--reservation=bi_fox_dgx"]
    if array:
        command.append("--array=0-3%4")
    if dependencies:
        if any(not str(j).isdigit() for j in dependencies):
            raise ValueError("numeric acknowledged dependencies required")
        command.append("--dependency=afterany:" + ":".join(dependencies))
    command += [
        str(SOURCE / "mosaickv/slurm/fox_bounded_reader_v27.sbatch"),
        str(SOURCE),
        str(root),
        mode,
        revision,
    ]
    write_once(root / f"jobs/{label}-request.json", dict(command=command, source_sha=revision))
    job = subprocess.check_output(command, text=True, timeout=30).strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("uncertain submission acknowledgement; do not duplicate")
    write_once(
        root / f"jobs/{label}-submission.json", dict(job=job, command=command, source_sha=revision)
    )
    return job


def dispatch(root):
    root = Path(root)
    plan = validate(root)
    errors = []
    try:
        cpu = read(root / "full-cpu/cpu-tests.json")
        op = read(root / "operator/result.json")
        if (
            cpu["status"] != "passed"
            or cpu["source_sha"] != plan["source_sha"]
            or op["source_sha"] != plan["source_sha"]
            or op["failures"]
        ):
            raise ValueError("complete CPU suite and bounded GPU operator must pass")
        queue = subprocess.check_output(
            ["squeue", "--me", "--noheader", "--states=RUNNING", "--format=%b"], text=True
        )
        running = sum(
            int(line.strip().split(":")[-1]) for line in queue.splitlines() if "gpu:" in line
        )
        if running > 2:
            raise ValueError("four-worker admission would exceed six aggregate GPUs")
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    workers = (
        None if errors else submit(root, plan["source_sha"], "gpu", "worker", gpu=True, array=True)
    )
    analysis = submit(
        root,
        plan["source_sha"],
        "analysis",
        "analyze",
        dependencies=() if workers is None else (workers,),
    )
    write_once(
        root / "jobs/dispatch-result.json",
        dict(blockers=errors, workers=workers, analysis=analysis),
    )


def launch(root):
    revision = clean_source(SOURCE)
    modules = subprocess.check_output(
        ["git", "submodule", "status", "--recursive"], cwd=SOURCE, text=True
    )
    if any(line[:1] in ("-", "+", "U") for line in modules.splitlines()):
        raise ValueError("all pinned submodules must be initialized unchanged")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT_SHA, revision], cwd=SOURCE, check=True
    )
    root = Path(root).resolve()
    allowed = Path("/scratch/djy8hg/workdir").resolve()
    if not root.is_relative_to(allowed) or root == allowed:
        raise ValueError("new immutable scratch root required")
    root.mkdir(parents=True, exist_ok=False)
    (root / "jobs").mkdir()
    (root / "slurm").mkdir()
    prepare(root)
    cpu = submit(root, revision, "cpu", "tests")
    op = submit(root, revision, "operator", "operator", gpu=True)
    callback = submit(root, revision, "dispatch", "dispatch", dependencies=(cpu, op))
    print(json.dumps(dict(cpu=cpu, operator=op, dispatch=callback, root=str(root))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("prepare", "operator", "worker", "analyze", "dispatch", "launch", "clock_probe"),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "worker":
        if args.index not in range(4):
            parser.error("worker requires index0..3")
        worker(args.root, args.index)
    else:
        globals()[args.mode](args.root)


if __name__ == "__main__":
    main()
