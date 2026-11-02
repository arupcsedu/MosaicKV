#!/usr/bin/env python3
"""Finite same-engine reader diagnostic, conditional dispatch reuse, closeout."""

import argparse
import gc
import json
import os
import subprocess
from dataclasses import asdict
from functools import partial
from pathlib import Path
from unittest.mock import patch

import evicastkv_common_streaming as common

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.request_reader_diagnostic import METHODS, WindowTimer, order, summarize
from mosaickv.evaluation.request_storage_evidence import audited, timed
from mosaickv.evaluation.scheduler_identity import scheduler_identity

SOURCE = common.SOURCE
PARENT = Path(
    "/scratch/djy8hg/runs/evicastkv/reconstruction-roles-readers-consumed-20261102-v1/verification-1/timing"
)
DOC = SOURCE / "mosaickv/docs/paired_request_readers_20261102.md"
FAILED = Path("/scratch/djy8hg/runs/evicastkv/paired-request-readers-consumed-20261102-v1")
ACCOUNTING = Path(
    "/scratch/djy8hg/runs/evicastkv/paired-request-storage-diagnostic-consumed-20261102-v1"
)
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_paired_request_readers.sbatch"
STORAGE_FIELDS = (
    ("initial_encoded_bytes", "initial_active_kv_storage_bytes"),
    ("final_encoded_bytes", "final_active_kv_storage_bytes"),
    ("decode_owned_peak_bytes", "peak_active_kv_storage_bytes"),
)


def submit(root):
    original = common.control.parent.subprocess.check_output

    def scoped(command, **kwargs):
        return original(
            [
                str(SCRIPT)
                if str(x).endswith("/evicastkv_funded_delimiter_diagnostic.sbatch")
                else x
                for x in command
            ],
            **kwargs,
        )

    with patch.object(common.control.parent.subprocess, "check_output", scoped):
        common.control.parent.submit(root, sha(root / "plan.json"))


def bootstrap(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new write-once root and clean immutable source required")
    parent = common.previous.read(PARENT / "analysis/decision.json")
    if not parent["complete"] or parent["failures"] or parent["all_method_similar_time_gate"]:
        raise ValueError("complete negative original request verification required")
    plan = common.previous.read(PARENT / "plan.json")
    bound = dict(plan["bound_files"])
    for path in (PARENT / "plan.json", PARENT / "analysis/decision.json", DOC):
        bound[str(path)] = sha(path)
    for path in (
        FAILED / "plan.json",
        FAILED / "analysis/decision.json",
        FAILED / "engineering_closeout.json",
        ACCOUNTING / "diagnostic.py",
        ACCOUNTING / "manifest.json",
        *(
            ACCOUNTING / f"audit-{audit}-reader-{reader}.json"
            for audit in (False, True)
            for reader in ("original", "carry_literals")
        ),
    ):
        bound[str(path)] = sha(path)
    for reader in ("original", "carry_literals"):
        diagnosis = common.previous.read(ACCOUNTING / f"audit-True-reader-{reader}.json")
        if not diagnosis["logits_bitwise"] or any(diagnosis["boundary_deltas"].values()):
            raise ValueError("positive same-source AUDITED accounting diagnosis required")
    write_once(
        root / "plan.json",
        dict(
            plan,
            source_sha=revision,
            bound_files=bound,
            paired_reader_stage="reader",
            paired_common_streaming_timing=False,
            maximum_worker_minutes=60,
            maximum_simultaneous_workers=2,
            reader_modes=["original", "carry_literals"],
            study_root=str(root),
            maximum_dispatch_attempts=1,
            stage="consumed_paired_original_literal_full_request",
            fresh_or_large_campaign_authorized=False,
            fresh_materialization_authorized=False,
            paper_campaign_authorized=False,
            physical_peak_evidence="separate_audited_companions_exact_frozen_equality",
            uninstrumented_peak_counters_are_lower_bounds=True,
        ),
    )
    # Preserve historical manifests. Corrections are separate hash-bound records.
    evidence = subprocess.check_output(
        ["sacct", "-j", "22330023", "--format=JobID,State,Start,End", "--noheader", "--parsable2"],
        text=True,
    )
    corrections = []
    for i in (0, 1):
        path = PARENT / f"workers/{i}/manifest.json"
        old = common.previous.read(path)
        job = f"22330023_{i}"
        if not any(line.startswith(job + "|COMPLETED|") for line in evidence.splitlines()):
            raise ValueError("completed scheduler evidence required for provenance correction")
        corrections.append(
            dict(
                original_manifest=str(path),
                original_manifest_sha256=sha(path),
                inherited_incorrect_job_id=old.get("slurm_job_id"),
                actual_array_element=job,
                actual_array_job_id="22330023",
                task_id=str(i),
            )
        )
    write_once(
        root / "historical_scheduler_corrections.json",
        dict(
            corrections=corrections,
            authoritative_sacct=evidence,
            raw_artifacts_unchanged=True,
        ),
    )
    common.previous.cpu(root, sha(root / "plan.json"))
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%b"], text=True
    )
    if any("gpu" in row.lower() for row in queue.splitlines()):
        raise ValueError("wait for preceding owned GPU jobs before diagnostic precheck")
    arguments = [str(SOURCE), str(root), sha(root / "plan.json")]
    (root / "slurm").mkdir(exist_ok=True)
    job = subprocess.check_output(
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
            "--mem=16G",
            "--time=00:10:00",
            "--job-name=mk-reader-precheck",
            f"--output={root}/slurm/precheck-%j.out",
            f"--error={root}/slurm/precheck-%j.err",
            str(SCRIPT),
            *arguments,
            "precheck",
        ],
        text=True,
    ).strip()
    dependent = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:10:00",
            f"--dependency=afterany:{job}",
            "--job-name=mk-reader-begin",
            f"--output={root}/slurm/begin-%j.out",
            f"--error={root}/slurm/begin-%j.err",
            str(SCRIPT),
            *arguments,
            "begin",
        ],
        text=True,
    ).strip()
    write_once(root / "precheck_submission.json", dict(gpu_job=job, cpu_begin=dependent))


def precheck(root, expected):
    import torch

    from mosaickv.evaluation.reconstruction_controls import gpu_controls
    from mosaickv.evaluation.request_kernel_dispatch import use_dispatch
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = common.previous.common.inputs(root, expected)
    result = dict(
        complete=False,
        failures=[],
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        scheduler=scheduler_identity({}),
        synthetic_numerical_control=True,
        independent_quality_N=0,
        model="not_used",
        dataset="not_used",
        measurement_type="diagnostic_measured",
        environment=common.control.environment(),
    )
    try:
        if (
            torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one deterministic A100 required")
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        with use_dispatch(True) as state:
            controls = gpu_controls("carry_literals", repeat_projection=True)
        if state["hits"] <= 0 or state["handles"]:
            raise RuntimeError("compiled hit or request cleanup control missing")
        result.update(
            complete=True,
            controls=controls,
            hits=state["hits"],
            misses=state["misses"],
            gpu=asdict(capture_gpu_environment(torch)),
            request_cache_released=True,
        )
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_once(root / "precheck/result.json", result)


def begin(root, expected):
    plan = common.previous.common.inputs(root, expected)
    path = root / "precheck/result.json"
    try:
        result = common.previous.read(path)
        if (
            not result["complete"]
            or result["failures"]
            or result["plan_sha256"] != expected
            or result["source_sha"] != plan["source_sha"]
            or not result["request_cache_released"]
            or result["hits"] <= 0
        ):
            raise ValueError("compiled reader controls failed")
    except (OSError, ValueError, KeyError) as exc:
        write_once(root / "precheck/blocked.json", dict(error=str(exc), paired_GPU_submitted=False))
        return
    write_once(
        root / "precheck/admitted.json", dict(precheck_sha256=sha(path), independent_quality_N=0)
    )
    submit(root)


def references(index):
    import torch

    result = {}
    for job in common.previous.read(common.AUDIT / "plan.json")["jobs"]:
        if job["input_index"] != index:
            continue
        method = common.method_of(job)
        folder = common.AUDIT / f"workers/{job['index']}"
        rows = [
            json.loads(x)
            for x in (common.CODE_AUDIT / f"workers/{job['index']}/raw.jsonl")
            .read_text()
            .splitlines()
        ]
        result[method] = dict(
            payload=common.previous.read(folder / "offline_payload.json"),
            logits=torch.load(folder / "greedy_logits.pt", weights_only=True, map_location="cpu"),
            storage=rows[0]["generation_storage"],
        )
    return result


def assert_request(logits, measured, reference, *, expected_counter=None):
    import torch

    if not torch.equal(logits, reference["logits"]):
        raise RuntimeError("changed original 128-token vocabulary logits")
    if measured["tensor_result_auditing"]:
        audited(measured, reference["storage"])
    else:
        if expected_counter is None:
            raise RuntimeError("matching uninstrumented counter control required")
        timed(measured, reference["storage"], expected_counter=expected_counter)


def mode_config(plan, mode):
    if (
        plan["paired_reader_stage"] not in ("reader", "dispatch")
        or mode not in plan["reader_modes"]
    ):
        raise ValueError("unregistered request mode")
    return (
        (mode, False)
        if plan["paired_reader_stage"] == "reader"
        else (plan["selected_reader"], mode == "reused_dispatch")
    )


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.common_streaming_timing import request
    from mosaickv.evaluation.reconstruction_controls import gpu_controls
    from mosaickv.evaluation.request_kernel_dispatch import use_dispatch
    from mosaickv.evaluation.request_reader_diagnostic import instrument, trace_summary
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = common.previous.common.inputs(root, expected)
    job = plan["jobs"][index]
    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(status="failed", failures=[], source_sha=plan["source_sha"], plan_sha256=expected)
    active = None
    try:
        base = Path(plan["study_root"])
        admitted = common.previous.read(base / "precheck/admitted.json")
        if sha(base / "precheck/result.json") != admitted["precheck_sha256"]:
            raise ValueError("modified or missing numerical precheck")
        if (
            torch.cuda.device_count() != 1
            or "A100" not in torch.cuda.get_device_name()
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
        ):
            raise ValueError("one isolated deterministic A100 required")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        if plan["paired_reader_stage"] == "dispatch":
            with use_dispatch(True) as dispatch:
                controls = gpu_controls(plan["selected_reader"], repeat_projection=True)
            write_once(
                out / "reader_and_dispatch_controls.json",
                dict(
                    controls=controls,
                    hits=dispatch["hits"],
                    misses=dispatch["misses"],
                    request_cache_released=not dispatch["handles"],
                ),
            )
        cfg = RunConfig.from_mapping(common.previous.resolved_mapping(plan, job))
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
            raise ValueError("frozen model/precision/backend changed")
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        manifest = common.previous.read(
            common.REFERENCE / f"workers/{job['reference_worker']}/manifest.json"
        )
        manifest.update(scheduler_identity(manifest))
        manifest.update(
            source_sha=plan["source_sha"],
            source_dirty=False,
            run_id=str(out),
            plan_sha256=expected,
            config_sha256=digest(dict(plan=expected, index=index)),
            evaluator_commit=plan["source_sha"],
            environment=common.control.environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            measurement_type="diagnostic_measured",
            independent_quality_N=0,
            paired_same_engine_readers=True,
            warmups_per_mode=5,
            repetitions_per_mode=20,
            representation_adaptations_not_official_methods=True,
            shared_encoded_byte_budget_claim=False,
            paper_superiority_established=False,
            reader_modes=plan["reader_modes"],
            windows=dict(early=[0, 15], middle=[16, 95], late=[96, 126]),
            evaluator="not_used_timing_and_bitwise_parent_controls_only",
            quality_metrics="not_recomputed",
            audited_peak_evidence_path=str(out / "audited_storage_controls.json"),
            uninstrumented_peak_counters_are_not_physical_peak_evidence=True,
            actual_timing_protocol=dict(
                order="randomized_methods_and_within_method_reader_pairs",
                warmups=5,
                repetitions=20,
                output_predictions=128,
                profiles_excluded_from_latency=True,
            ),
        )
        write_once(out / "manifest.json", manifest)
        refs = references(job["input_index"])
        messages = common.previous.load_messages(common.previous.read(job["inputs"][0]))
        profile = tuple(common.previous.read(Path(plan["profile_path"])))
        seen, counters = set(), {}

        def validate_packet(packets, *, method):
            actual = common.control.digest_tensors(tuple(t for p in packets for t in p.tensors()))
            if actual != refs[method]["payload"]["payload_sha256"]:
                raise RuntimeError("changed original preconversion packet bits")

        def execute(method, mode, *, validator=None, audit=False, register_counter=False):
            reader, reuse = mode_config(plan, mode)
            logits, packets, measured = request(
                adapter,
                messages,
                profile,
                method,
                minimum_only=plan.get("exact_minimum_only", False),
                componentwise_fitting=True,
                code_delta=method in plan.get("code_delta_methods", []),
                input_packet_validator=validator,
                source_geometry=plan.get("source_geometry"),
                source_num_warps=plan.get("source_num_warps"),
                construction_geometry=plan.get("construction_geometry"),
                reconstruction_readers=reader,
                reuse_kernel_dispatch=reuse,
                audit_decode_storage=audit,
            )
            try:
                counter = (
                    measured["reported_counter_peak_bytes"]
                    if register_counter
                    else counters.get(method)
                )
                if register_counter and mode != plan["reader_modes"][0]:
                    raise ValueError("counter control must use the original registered mode")
                assert_request(logits, measured, refs[method], expected_counter=counter)
            except Exception:
                write_once(
                    out / "validation_failure.json",
                    dict(
                        method=method,
                        mode=mode,
                        audit=audit,
                        active=active,
                        actual_measurements=measured,
                        frozen_storage={v: refs[method]["storage"][v] for _, v in STORAGE_FIELDS},
                        registered_uninstrumented_counter=counters.get(method),
                        logits_bitwise=torch.equal(logits, refs[method]["logits"]),
                    ),
                )
                raise
            if register_counter:
                counters[method] = measured["reported_counter_peak_bytes"]
            del logits, packets
            return measured

        # Physical peak parity is measured with tensor auditing ON, outside
        # latency. Fast counters are separately registered in the same ON/OFF
        # instrumentation category, never substituted for audited peak data.
        control_methods = sorted(refs) if plan["paired_reader_stage"] == "dispatch" else METHODS
        checks, audit_rows = [], []
        for method in control_methods:
            for mode in plan["reader_modes"]:
                active = dict(method=method, reader=mode, phase="audited_storage_control")
                measured = execute(
                    method, mode, validator=partial(validate_packet, method=method), audit=True
                )
                audit_rows.append(
                    dict(
                        method=method,
                        reader=mode,
                        logits_bitwise=True,
                        packet_bits_identical=True,
                        frozen_physical_boundaries_identical=True,
                        measured=measured,
                        frozen_storage={v: refs[method]["storage"][v] for _, v in STORAGE_FIELDS},
                    )
                )
            active = dict(method=method, phase="uninstrumented_counter_control")
            baseline = execute(method, plan["reader_modes"][0], register_counter=True)
            checks.append(
                dict(
                    method=method,
                    logits_bitwise=True,
                    accounting_identical=True,
                    dispatch_hits=audit_rows[-1]["measured"]["dispatch_hits"],
                    audited_storage_boundaries={
                        k: audit_rows[-1]["measured"][k] for k, _ in STORAGE_FIELDS
                    },
                    storage_boundaries={
                        "initial_encoded_bytes": baseline["initial_encoded_bytes"],
                        "final_encoded_bytes": baseline["final_encoded_bytes"],
                        "reported_counter_peak_bytes": counters[method],
                    },
                )
            )
        write_once(
            out / "audited_storage_controls.json",
            dict(
                rows=audit_rows,
                counter_controls=checks,
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                not_latency_evidence=True,
                independent_quality_N=0,
            ),
        )
        audit_sha = sha(out / "audited_storage_controls.json")
        if plan["paired_reader_stage"] == "dispatch":
            write_once(out / "all_method_controls.json", dict(rows=checks))

        for entry in order(plan["seed"] + index, readers=tuple(plan["reader_modes"])):
            active = entry
            method, mode = entry["method"], entry["reader"]
            validator = None
            if (method, mode) not in seen:
                if not entry["warmup"]:
                    raise RuntimeError("packet hashing may only occur during excluded warmup")

                validator = partial(validate_packet, method=method)

            gc.collect()
            torch.cuda.synchronize()
            measured = execute(method, mode, validator=validator)
            seen.add((method, mode))
            row = dict(
                **entry,
                **measured,
                status="completed",
                logits_bitwise=True,
                accounting_identical=True,
                instrumented=False,
                input_index=job["input_index"],
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                manifest_path=str(out / "manifest.json"),
                independent_quality_N=0,
                audited_storage_controls_sha256=audit_sha,
            )
            with (out / "raw.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

        # Instrumented requests are AFTER the complete randomized timing grid.
        # They cannot enter latency confidence intervals.
        for method in METHODS:
            for mode in plan["reader_modes"]:
                active = dict(method=method, reader=mode, phase="instrumented")
                timer = WindowTimer(event_factory=lambda: torch.cuda.Event(enable_timing=True))
                with instrument(adapter, timer) as evidence:
                    measured = execute(method, mode)
                torch.cuda.synchronize()
                trace = trace_summary(timer, evidence)
                record = dict(
                    method=method,
                    reader=mode,
                    input_index=job["input_index"],
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                    logits_bitwise=True,
                    accounting_identical=True,
                    **trace,
                    diagnostic_measurements_not_latency_evidence=measured,
                    audited_storage_controls_sha256=audit_sha,
                )
                with (out / "profiles.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                del timer, evidence, trace, measured
        rows = [json.loads(x) for x in (out / "raw.jsonl").read_text().splitlines()]
        write_once(
            out / "summary.json",
            summarize(rows, plan["seed"] + index, readers=tuple(plan["reader_modes"])),
        )
        result["status"] = "completed"
    except Exception as exc:
        result["failures"].append(dict(error=f"{type(exc).__name__}: {exc}", active=active))
        raise
    finally:
        result["scheduler"] = scheduler_identity({})
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def read_producers(root, plan, expected):
    summaries, profiles, failures = [], [], []
    for i in (0, 1):
        folder = root / f"workers/{i}"
        try:
            result = common.previous.read(folder / "result.json")
            if (
                result["status"] != "completed"
                or result["failures"]
                or result["source_sha"] != plan["source_sha"]
                or result["plan_sha256"] != expected
                or any(sha(folder / f) != h for f, h in result["outputs"].items())
            ):
                raise ValueError("failed/modified paired producer")
            manifest = common.previous.read(folder / "manifest.json")
            if (
                manifest["slurm_job_id"] != result["scheduler"]["slurm_job_id"]
                or manifest["slurm_array_task_id"] != str(i)
                or manifest["scheduler_identity_source"] != "current_process_environment"
            ):
                raise ValueError("incorrect current scheduler provenance")
            rows = [json.loads(x) for x in (folder / "raw.jsonl").read_text().splitlines()]
            controls = common.previous.read(folder / "audited_storage_controls.json")
            expected_methods = (
                set(METHODS)
                if plan["paired_reader_stage"] == "reader"
                else {
                    "mosaickv_exact",
                    "prefixkv_reimpl",
                    "prompt_attention_topk",
                    "vl_cache_reimpl",
                    "lookm_reimpl",
                }
            )
            audit_rows = controls["rows"]
            if len(audit_rows) != len(expected_methods) * 2 or {
                (r["method"], r["reader"]) for r in audit_rows
            } != {(m, p) for m in expected_methods for p in plan["reader_modes"]}:
                raise ValueError("incomplete audited physical-storage companions")
            for control in audit_rows:
                if not all(
                    control[k] is True
                    for k in (
                        "logits_bitwise",
                        "packet_bits_identical",
                        "frozen_physical_boundaries_identical",
                    )
                ):
                    raise ValueError("failed audited original physical-storage parity")
                audited(control["measured"], control["frozen_storage"])
            registered_counters = {
                r["method"]: r["storage_boundaries"]["reported_counter_peak_bytes"]
                for r in controls["counter_controls"]
            }
            if set(registered_counters) != expected_methods or len(
                controls["counter_controls"]
            ) != len(expected_methods):
                raise ValueError("missing matched uninstrumented counter controls")
            for row in rows:
                if row["audited_storage_controls_sha256"] != sha(
                    folder / "audited_storage_controls.json"
                ):
                    raise ValueError("changed physical-storage companion")
                frozen = next(
                    c["frozen_storage"] for c in audit_rows if c["method"] == row["method"]
                )
                timed(row, frozen, expected_counter=registered_counters[row["method"]])
            if any(
                (r["reconstruction_readers"], r["reuse_kernel_dispatch"])
                != mode_config(plan, r["reader"])
                for r in rows
            ):
                raise ValueError("recorded request mode differs from registered policy")
            summaries.append(summarize(rows, plan["seed"] + i, readers=tuple(plan["reader_modes"])))
            group = [json.loads(x) for x in (folder / "profiles.jsonl").read_text().splitlines()]
            if len(group) != 4 or {(r["method"], r["reader"]) for r in group} != {
                (m, p) for m in METHODS for p in plan["reader_modes"]
            }:
                raise ValueError("missing/duplicate profiles")
            for row in group:
                if (
                    row["steps"] != 127
                    or row["logits_bitwise"] is not True
                    or row["accounting_identical"] is not True
                ):
                    raise ValueError("incomplete profile controls")
                for w, count in (("early", 16), ("middle", 80), ("late", 31)):
                    if row["phases"].get(w + "/decode_step", {}).get("calls") != count:
                        raise ValueError("missing decode-window trace")
                    branches = [r for r in row["branch_counts"] if r["window"] == w]
                    reader, _ = mode_config(plan, row["reader"])
                    required = "carried" if reader == "carry_literals" else "original"
                    if not any(r["branch"] == required and r["calls"] > 0 for r in branches):
                        raise ValueError("requested reader did not engage in every window")
                profiles.append(row)
            if plan["paired_reader_stage"] == "dispatch":
                checks = common.previous.read(folder / "all_method_controls.json")["rows"]
                from mosaickv.evaluation.common_streaming_timing import METHODS as ALL_METHODS

                if (
                    len(checks) != 5
                    or {r["method"] for r in checks} != set(ALL_METHODS)
                    or not all(
                        r["logits_bitwise"] and r["accounting_identical"] and r["dispatch_hits"] > 0
                        for r in checks
                    )
                ):
                    raise ValueError("incomplete all-method dispatch controls")
        except (OSError, KeyError, ValueError, TypeError) as exc:
            failures.append(dict(index=i, error=str(exc)))
    return summaries, profiles, failures


def reuse_cpu(source, target):
    for name in ("cpu.json", "cpu.xml"):
        os.link(source / name, target / name)
    write_once(
        target / "cpu_reuse.json",
        dict(
            source_sha=clean_source(SOURCE),
            same_immutable_source=True,
            original_root=str(source),
            cpu_json_sha256=sha(source / "cpu.json"),
            cpu_xml_sha256=sha(source / "cpu.xml"),
        ),
    )


def analyze(root, expected):
    plan = common.previous.common.inputs(root, expected)
    summaries, profiles, failures = read_producers(root, plan, expected)
    selected = (
        "carry_literals"
        if not failures
        and all(
            s["within_method"]["mosaickv_exact"]["familywise_ratio_bounds"][1] <= 1.0
            for s in summaries
        )
        else "original"
    )
    host_projection = [
        sum(
            v["exclusive_wall_seconds"]
            for k, v in row["phases"].items()
            if k.endswith(".projection") and not k.startswith("prefill/")
        )
        for row in profiles
        if row["method"] == "mosaickv_exact"
    ]
    reuse_eligible = (
        plan["paired_reader_stage"] == "reader"
        and not failures
        and len(host_projection) == 4
        and min(host_projection) >= 0.1
    )
    improved = (
        plan["paired_reader_stage"] == "dispatch"
        and not failures
        and len(summaries) == 2
        and all(
            s["within_method"]["mosaickv_exact"]["familywise_ratio_bounds"][1] <= 0.99
            for s in summaries
        )
    )
    decision = dict(
        complete=not failures,
        failures=failures,
        summaries=summaries,
        source_sha=plan["source_sha"],
        plan_sha256=expected,
        profiles=profiles,
        selected_reader=selected
        if plan["paired_reader_stage"] == "reader"
        else plan["selected_reader"],
        label_blind_host_projection_opportunity_seconds=host_projection,
        conditional_dispatch_attempt_authorized=reuse_eligible,
        dispatch_end_to_end_improvement_gate=improved,
        independent_quality_N=0,
        paper_superiority_established=False,
        fresh_or_large_campaign_submitted=False,
    )
    write_once(root / "analysis/decision.json", decision)
    if failures:
        closeout(root, decision)
    elif reuse_eligible:
        target = root / "dispatch"
        bound = dict(plan["bound_files"])
        for p in (root / "plan.json", root / "analysis/decision.json"):
            bound[str(p)] = sha(p)
        write_once(
            target / "plan.json",
            dict(
                plan,
                bound_files=bound,
                paired_reader_stage="dispatch",
                selected_reader=selected,
                reader_modes=["original_dispatch", "reused_dispatch"],
                stage="consumed_conditional_same_binary_dispatch_reuse",
            ),
        )
        reuse_cpu(root, target)
        submit(target)
    elif improved:
        target = root / "timing"
        bound = dict(plan["bound_files"])
        for p in (root / "plan.json", root / "analysis/decision.json"):
            bound[str(p)] = sha(p)
        companions = {}
        for i in (0, 1):
            for name in (
                "all_method_controls.json",
                "audited_storage_controls.json",
                "result.json",
            ):
                path = root / f"workers/{i}/{name}"
                bound[str(path)] = sha(path)
            path = root / f"workers/{i}/audited_storage_controls.json"
            companions[str(i)] = dict(path=str(path), sha256=sha(path))
        write_once(
            target / "plan.json",
            dict(
                plan,
                bound_files=bound,
                paired_common_streaming_timing=True,
                reconstruction_readers=plan["selected_reader"],
                reuse_kernel_dispatch=True,
                audited_storage_companions=companions,
                expected_storage_boundaries={
                    str(i): {
                        row["method"]: row["storage_boundaries"]
                        for row in common.previous.read(
                            root / f"workers/{i}/all_method_controls.json"
                        )["rows"]
                    }
                    for i in (0, 1)
                },
                stage="consumed_all_method_request_verification_after_paired_dispatch_gain",
            ),
        )
        reuse_cpu(root, target)
        common.submit(target, sha(target / "plan.json"))
        dependency = common.previous.read(target / "submission.json")["cpu_analysis"]
        job = subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--partition=bii",
                "--account=bii_dsc_community",
                "--cpus-per-task=2",
                "--mem=16G",
                "--time=00:10:00",
                f"--dependency=afterany:{dependency}",
                "--job-name=mk-paired-reader-closeout",
                f"--output={root}/slurm/closeout-%j.out",
                f"--error={root}/slurm/closeout-%j.err",
                str(SCRIPT),
                str(SOURCE),
                str(root),
                expected,
                "finalize",
            ],
            text=True,
        ).strip()
        write_once(root / "registered_closeout.json", dict(job_id=job, dependency=dependency))
    else:
        closeout(root, decision)


def closeout(root, decision, *, path=None):
    plan = common.previous.read(root / "plan.json")
    write_once(
        Path(plan["study_root"]) / "engineering_closeout.json",
        dict(
            complete=decision["complete"],
            failures=decision["failures"],
            final_decision=str(path or root / "analysis/decision.json"),
            source_sha=plan["source_sha"],
            automatic_dispatch_attempts=1 if plan["paired_reader_stage"] == "dispatch" else 0,
            independent_quality_N=0,
            paper_superiority_established=False,
            fresh_or_large_campaign_submitted=False,
            all_method_similar_time_gate=decision.get("all_method_similar_time_gate", False),
            all_method_no_slower_gate=bool(
                decision.get("all_method_similar_time_gate", False)
                and all(
                    pair["familywise_eight_comparison_ratio_bounds"][1] <= 1.0
                    for summary in decision.get("summaries", [])
                    for pair in summary["pairs"].values()
                )
            ),
        ),
    )


def finalize(root, expected):
    plan = common.previous.common.inputs(root, expected)
    path = root / "timing/analysis/decision.json"
    try:
        decision = common.previous.read(path)
        if decision["source_sha"] != plan["source_sha"] or decision["plan_sha256"] != sha(
            root / "timing/plan.json"
        ):
            raise ValueError("final timing provenance mismatch")
    except (OSError, ValueError) as exc:
        decision = dict(complete=False, failures=[str(exc)])
    closeout(root, decision, path=path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("bootstrap", "precheck", "begin", "worker", "analyze", "finalize")
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan-sha")
    parser.add_argument("--source-sha")
    args = parser.parse_args()
    if args.mode == "bootstrap":
        bootstrap(args.root, args.source_sha)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.mode](args.root, args.plan_sha)
