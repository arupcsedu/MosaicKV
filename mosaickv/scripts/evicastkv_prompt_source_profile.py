#!/usr/bin/env python3
"""Two consumed inputs, prompt-operation attribution only; no quality campaign."""

import argparse
import gc
import json
import os
import subprocess
from dataclasses import asdict
from functools import partial
from pathlib import Path

import evicastkv_paired_request_readers as parent

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.prompt_source_subphases import totals
from mosaickv.evaluation.scheduler_identity import scheduler_identity

SOURCE = parent.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/replay-descriptor-request-consumed-20261102-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_prompt_source_profile.sbatch"
DOC = SOURCE / "mosaickv/docs/prompt_source_subphases_20261102.md"
METHODS = ("mosaickv_exact", "prefixkv_reimpl")


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new root and clean immutable source required")
    old = parent.common.previous.read(PARENT / "plan.json")
    decision = parent.common.previous.read(PARENT / "analysis/decision.json")
    if (
        not decision["complete"]
        or decision["failures"]
        or decision["descriptor_end_to_end_improvement_gate"]
    ):
        raise ValueError("complete negative descriptor decision required")
    bound = dict(old["bound_files"])
    for path in (
        PARENT / "plan.json",
        PARENT / "analysis/decision.json",
        PARENT / "engineering_closeout.json",
        DOC,
    ):
        bound[str(path)] = sha(path)
    for i in (0, 1):
        folder = PARENT / f"workers/{i}"
        result = parent.common.previous.read(folder / "result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("both complete descriptor workers required")
        bound[str(folder / "result.json")] = sha(folder / "result.json")
        for name, value in result["outputs"].items():
            path = folder / name
            if sha(path) != value:
                raise ValueError("changed parent output")
            bound[str(path)] = value
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_prompt_source_subphase_attribution",
            study_root=str(root),
            maximum_worker_minutes=15,
            maximum_simultaneous_workers=2,
            profile_repetitions=2,
            profile_warmups=1,
            selected_reader="carry_literals",
            base_dispatch=False,
            reuse_code_descriptors=False,
            paired_common_streaming_timing=False,
            instrumentation_not_performance_evidence=True,
            fresh_materialization_authorized=False,
            paper_campaign_authorized=False,
            automatic_optimization_authorized=False,
            methods=list(METHODS),
            independent_quality_N=0,
        ),
    )
    expected = sha(root / "plan.json")
    parent.common.previous.cpu(root, expected)
    submit(root, expected)


def submit(root, expected):
    from control_a100_schedule import tick

    plan = parent.common.previous.common.inputs(root, expected)
    proof = parent.common.previous.read(root / "cpu.json")
    if (
        proof["source_sha"] != plan["source_sha"]
        or proof["returncode"]
        or proof["failed_or_skipped"]
        or proof["test_count"] < 3600
        or proof["xml_sha256"] != sha(root / "cpu.xml")
    ):
        raise ValueError("complete clean CPU suite required before GPU submission")
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%b"], text=True
    )
    if any("gpu" in row.lower() for row in queue.splitlines()):
        raise ValueError("wait for other owned GPU workers before bounded diagnostic")
    (root / "slurm").mkdir(exist_ok=True)
    args = [str(SOURCE), str(root), expected]
    gpu = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii-gpu",
            "--account=bii_dsc_community",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--time=00:15:00",
            "--array=0-1%2",
            "--job-name=mk-prompt-detail",
            f"--output={root}/slurm/gpu-%A_%a.out",
            f"--error={root}/slurm/gpu-%A_%a.err",
            str(SCRIPT),
            *args,
            "worker",
        ],
        text=True,
    ).strip()
    analysis = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--nodes=1",
            "--ntasks=1",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:10:00",
            f"--dependency=afterany:{gpu}",
            "--job-name=mk-prompt-detail-analysis",
            f"--output={root}/slurm/analysis-%j.out",
            f"--error={root}/slurm/analysis-%j.err",
            str(SCRIPT),
            *args,
            "analyze",
        ],
        text=True,
    ).strip()
    policy = dict(
        timezone="America/New_York",
        maximum_worker_minutes=15,
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
        arrays=[gpu],
    )
    write_once(root / "scheduler/policy.json", policy)
    write_once(
        root / "scheduler/registration.json", tick(policy, {}, root / "scheduler/state.json", False)
    )
    write_once(root / "submission.json", dict(gpu_array=gpu, cpu_analysis=analysis))


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.common_streaming_timing import request
    from mosaickv.evaluation.prompt_source_subphases import use
    from mosaickv.evaluation.reconstruction_controls import gpu_controls
    from mosaickv.evaluation.request_reader_diagnostic import WindowTimer, instrument, trace_summary
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = parent.common.previous.common.inputs(root, expected)
    if type(index) is not int or index not in (0, 1):
        raise ValueError("exactly the two consumed identities")
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"], plan_sha256=expected)
    active = None
    try:
        if (
            torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one deterministic A100 required")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        write_once(out / "primitive_controls.json", dict(controls=gpu_controls("carry_literals")))
        cfg = RunConfig.from_mapping(parent.common.previous.resolved_mapping(plan, job))
        if (
            cfg.model.id,
            cfg.model.revision,
            cfg.model.precision.value,
            cfg.execution.attention_implementation,
        ) != (
            "llava-hf/llava-1.5-7b-hf",
            "b234b804b114d9e37bb655e11cbbb5f5e971b7a9",
            "bf16",
            "eager",
        ):
            raise ValueError("frozen checkpoint and backend required")
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        manifest = parent.common.previous.read(
            parent.common.REFERENCE / f"workers/{job['reference_worker']}/manifest.json"
        )
        manifest.update(scheduler_identity(manifest))
        manifest.update(
            source_sha=plan["source_sha"],
            source_dirty=False,
            run_id=str(out),
            evaluator_commit=plan["source_sha"],
            plan_sha256=expected,
            config_sha256=digest(dict(plan=expected, index=index)),
            environment=parent.common.control.environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            measurement_type="diagnostic_measured",
            result_category="unified_comparison",
            result_origin="locally_measured",
            independent_quality_N=0,
            source_precision_storage_variant_not_six_mechanism_method=True,
            representation_adaptations_not_official_methods=True,
            shared_encoded_byte_budget_claim=False,
            paper_superiority_established=False,
            evaluator="not_used_bitwise_and_storage_checks_only",
            quality_metrics="not_recomputed",
            actual_timing_protocol=dict(
                instrumented_requests_only=True,
                warmups=1,
                repetitions=2,
                fixed_predictions=128,
                no_latency_confidence_interval=True,
                reader="carry_literals",
                dispatch_reuse=False,
                descriptor_reuse=False,
            ),
            windows=dict(early=[0, 15], middle=[16, 95], late=[96, 126]),
            whole_request_GPU_peak_certified=False,
        )
        write_once(out / "manifest.json", manifest)
        refs = parent.references(index)
        messages = parent.common.previous.load_messages(
            parent.common.previous.read(Path(job["inputs"][0]))
        )
        profile = tuple(parent.common.previous.read(Path(plan["profile_path"])))
        historical = parent.common.previous.read(
            PARENT / f"workers/{index}/audited_storage_controls.json"
        )
        counters = {
            x["method"]: x["storage_boundaries"]["reported_counter_peak_bytes"]
            for x in historical["counter_controls"]
        }

        def validate_packet(packets, method):
            actual = parent.common.control.digest_tensors(
                tuple(t for p in packets for t in p.tensors())
            )
            if actual != refs[method]["payload"]["payload_sha256"]:
                raise RuntimeError("original preconversion packet changed")

        def execute(method, audit=False, validator=None):
            values, packets, measured = request(
                adapter,
                messages,
                profile,
                method,
                minimum_only=plan["exact_minimum_only"],
                componentwise_fitting=True,
                code_delta=method in plan["code_delta_methods"],
                input_packet_validator=validator,
                construction_geometry=plan["construction_geometry"],
                source_geometry=plan.get("source_geometry"),
                source_num_warps=plan.get("source_num_warps"),
                reconstruction_readers="carry_literals",
                reuse_kernel_dispatch=False,
                reuse_code_descriptors=False,
                audit_decode_storage=audit,
            )
            parent.assert_request(values, measured, refs[method], expected_counter=counters[method])
            del values, packets
            return measured

        controls = []
        for method in METHODS:
            active = dict(method=method, phase="audited_control")
            controls.append(
                dict(
                    method=method,
                    logits_bitwise=True,
                    packet_bits_identical=True,
                    measured=execute(
                        method, audit=True, validator=partial(validate_packet, method=method)
                    ),
                )
            )
            active = dict(method=method, phase="excluded_warmup")
            execute(method)
        write_once(out / "audited_controls.json", dict(rows=controls, not_latency_evidence=True))
        for repetition in range(plan["profile_repetitions"]):
            methods = METHODS if (index + repetition) % 2 == 0 else METHODS[::-1]
            for method in methods:
                active = dict(method=method, repetition=repetition, phase="profile")
                gc.collect()
                torch.cuda.synchronize()
                timer = WindowTimer(event_factory=lambda: torch.cuda.Event(enable_timing=True))
                with use(timer), instrument(adapter, timer) as evidence:
                    measured = execute(method)
                torch.cuda.synchronize()
                trace = trace_summary(timer, evidence)
                detail = totals(trace["phases"], require_source=method == "mosaickv_exact")
                row = dict(
                    method=method,
                    repetition=repetition,
                    input_index=index,
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                    manifest_path=str(out / "manifest.json"),
                    measurement_type="diagnostic_measured",
                    logits_bitwise=True,
                    accounting_identical=True,
                    audited_controls_sha256=sha(out / "audited_controls.json"),
                    independent_quality_N=0,
                    prompt_detail=detail,
                    **trace,
                    instrumented_request_not_latency_evidence=measured,
                )
                with (out / "profiles.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                del timer, evidence, trace, measured
        result["status"] = "completed"
    except Exception as exc:
        result["failures"].append(dict(error=f"{type(exc).__name__}: {exc}", active=active))
        raise
    finally:
        result["scheduler"] = scheduler_identity({})
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def analyze(root, expected):
    plan = parent.common.previous.common.inputs(root, expected)
    failures, profiles = [], []
    for i in (0, 1):
        try:
            folder = root / f"workers/{i}"
            result = parent.common.previous.read(folder / "result.json")
            if (
                result["status"] != "completed"
                or result["failures"]
                or result["source_sha"] != plan["source_sha"]
                or result["plan_sha256"] != expected
            ):
                raise ValueError("incomplete/foreign producer")
            for name, value in result["outputs"].items():
                if sha(folder / name) != value:
                    raise ValueError("changed producer output")
            rows = [json.loads(x) for x in (folder / "profiles.jsonl").read_text().splitlines()]
            wanted = {(method, rep) for method in METHODS for rep in range(2)}
            if len(rows) != 4 or {(x["method"], x["repetition"]) for x in rows} != wanted:
                raise ValueError("complete unique diagnostic grid required")
            for row in rows:
                if (
                    row["source_sha"] != plan["source_sha"]
                    or row["plan_sha256"] != expected
                    or row["input_index"] != i
                    or row["steps"] != 127
                    or not row["logits_bitwise"]
                    or not row["accounting_identical"]
                ):
                    raise ValueError("numerical/source/window proof failed")
                if row["prompt_detail"] != totals(
                    row["phases"], require_source=row["method"] == "mosaickv_exact"
                ):
                    raise ValueError("changed derived subphases")
            profiles.extend(rows)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append(dict(index=i, error=str(exc)))
    operations = {}
    if not failures:
        for i in (0, 1):
            rows = [
                x for x in profiles if x["input_index"] == i and x["method"] == "mosaickv_exact"
            ]
            values = {
                name: sum(
                    x["prompt_detail"]["operations"][name]["exclusive_wall_seconds"] for x in rows
                )
                / len(rows)
                for name in rows[0]["prompt_detail"]["operations"]
            }
            operations[str(i)] = values
    ranking = sorted(
        operations.get("0", {}), key=lambda name: (-sum(x[name] for x in operations.values()), name)
    )
    decision = dict(
        complete=not failures,
        failures=failures,
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        profiles=profiles,
        mean_exclusive_host_seconds=operations,
        host_operation_ranking=ranking,
        next_target=ranking[0] if ranking else None,
        GPU_active_kernel_ranking="not_measured_event_intervals_include_host_gaps",
        no_optimization_promoted=True,
        no_GPU_retry_or_sweep_submitted=True,
        independent_quality_N=0,
        paper_superiority_established=False,
        fresh_or_large_campaign_submitted=False,
        whole_request_GPU_peak_certified=False,
    )
    write_once(root / "analysis/decision.json", decision)
    lines = [
        "# Consumed-input prompt-source attribution",
        "",
        "Instrumented operation diagnostics; not latency or independent quality evidence.",
        "",
        f"Complete: {not failures}. Failures: {failures}.",
        "",
        "| Operation | Description exclusive host seconds | MM-Vet exclusive host seconds |",
        "|---|---:|---:|",
    ]
    for name in ranking:
        lines.append(f"| {name} | {operations['0'][name]:.6f} | {operations['1'][name]:.6f} |")
    lines.extend(
        [
            "",
            f"Next measured host target: {decision['next_target']}.",
            "Do not add these leaf timings to parent reconstruction or projection intervals.",
            "Only named allocation entrypoints are tracked; zero calls do not certify GPU peak.",
            "No kernel/selection change or new cohort launched. Larger campaigns remain blocked.",
        ]
    )
    path = root / "analysis/report.md"
    with path.open("x") as stream:
        stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "worker", "analyze"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        analyze(args.root, args.plan_sha)
