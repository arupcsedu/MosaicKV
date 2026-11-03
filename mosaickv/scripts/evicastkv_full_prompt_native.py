#!/usr/bin/env python3
"""Finite consumed-input native-source feasibility -> paired -> all-method DAG.

No fresh cohort, answer-based selection, source edits, or paper claim promotion.
Every failure receives afterany analysis; at most two registered policies.
"""

import argparse
import gc
import json
import os
import subprocess
import traceback
from dataclasses import asdict
from pathlib import Path

import evicastkv_paired_request_readers as pair

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = pair.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/projection-opportunity-consumed-20261103-retry1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_full_prompt_native.sbatch"
DOC = SOURCE / "mosaickv/docs/full_prompt_native_20261103.md"
READERS = ("original", "native")
STORAGE = ("initial_encoded_bytes", "final_encoded_bytes", "decode_owned_peak_bytes")


def read(path):
    return json.loads(Path(path).read_text())


def checked(root, expected):
    return pair.common.previous.common.inputs(root, expected)


def copy_cpu(source, target):
    for name in ("cpu.json", "cpu.xml"):
        os.link(source / name, target / name)


def cpu_proof(root, plan):
    proof = read(root / "cpu.json")
    if (
        proof["source_sha"] != plan["source_sha"]
        or proof["returncode"]
        or proof["failed_or_skipped"]
        or proof["test_count"] < 3650
        or proof["xml_sha256"] != sha(root / "cpu.xml")
    ):
        raise ValueError("complete same-clean-source CPU suite required")


def launch(root, action, *, gpu=False, array=False, dependency=None):
    expected = sha(root / "plan.json")
    plan = checked(root, expected)
    cpu_proof(root, plan)
    (root / "slurm").mkdir(exist_ok=True)
    args = ["sbatch", "--parsable", "--account=bii_dsc_community", "--nodes=1", "--ntasks=1"]
    args += (
        [
            "--partition=bii-gpu",
            "--reservation=bi_fox_dgx",
            "--gres=gpu:a100:1",
            "--cpus-per-task=12",
            "--mem=48G",
            "--time=01:00:00",
        ]
        if gpu
        else ["--partition=bii,standard", "--cpus-per-task=2", "--mem=16G", "--time=00:10:00"]
    )
    if array:
        args.append("--array=0-1%2")
    if action in ("primitive", "operator"):
        args = ["--time=00:10:00" if x == "--time=01:00:00" else x for x in args]
    if dependency:
        args.append(f"--dependency=afterany:{dependency}")
    args += [
        f"--job-name=mk-full-native-{action}",
        f"--output={root}/slurm/{action}-%A_%a.out",
        f"--error={root}/slurm/{action}-%A_%a.err",
        str(SCRIPT),
        str(SOURCE),
        str(root),
        expected,
        action,
    ]
    job = subprocess.check_output(args, text=True).strip()
    if gpu and array:
        from control_a100_schedule import tick

        policy = dict(
            timezone="America/New_York",
            maximum_worker_minutes=60,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[job],
        )
        write_once(root / "scheduler/policy.json", policy)
        write_once(
            root / "scheduler/registration.json",
            tick(policy, {}, root / "scheduler/state.json", False),
        )
    return job


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    parent = read(PARENT / "analysis/decision.json")
    if not parent["complete"] or parent["failures"]:
        raise ValueError("complete metadata opportunity audit required")
    old = read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    for path in (PARENT / "plan.json", PARENT / "analysis/decision.json", DOC):
        bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_full_prompt_native_controller",
            chain_root=str(root),
            native_policies=["kv", "v_only"],
            maximum_policy_attempts=2,
            persistent_repair_cap_bytes=3 * 2**20,
            minimum_converted_layers=8,
            maximum_simultaneous_workers=2,
            maximum_worker_minutes=60,
            original_reader="carry_literals",
            independent_quality_N=0,
            fresh_materialization_authorized=False,
            paper_campaign_authorized=False,
        ),
    )
    pair.common.previous.cpu(root, sha(root / "plan.json"))
    queue = subprocess.check_output(
        ["squeue", "-h", "-r", "-u", os.environ["USER"], "-o", "%i|%b"], text=True
    )
    if any("gpu" in r.lower() for r in queue.splitlines()):
        raise ValueError("preceding owned GPU work must finish")
    job = launch(root, "primitive", gpu=True)
    following = launch(root, "begin", dependency=job)
    write_once(root / "primitive_submission.json", dict(gpu=job, cpu_begin=following))


def primitive(root, expected):
    import torch

    from mosaickv.evaluation.full_prompt_source import gpu_controls

    plan = checked(root, expected)
    proof = dict(complete=False, failures=[], source_sha=plan["source_sha"], plan_sha256=expected)
    try:
        torch.set_num_threads(12)
        torch.use_deterministic_algorithms(True)
        proof["controls"] = gpu_controls()
        proof["complete"] = True
    except Exception as error:
        proof["failures"].append(f"{type(error).__name__}: {error}")
        proof["traceback"] = traceback.format_exc()
    write_once(root / "primitive/result.json", proof)
    if not proof["complete"]:
        raise SystemExit(1)


def child(parent, name, **changes):
    plan = read(parent / "plan.json")
    target = parent / name
    bound = dict(plan["bound_files"])
    for path in (
        parent / "plan.json",
        parent / "analysis/decision.json",
        parent / "primitive/result.json",
    ):
        if path.exists():
            bound[str(path)] = sha(path)
    for index in (0, 1):
        path = parent / f"workers/{index}/result.json"
        if path.exists():
            result = read(path)
            bound[str(path)] = sha(path)
            for filename, value in result["outputs"].items():
                artifact = path.parent / filename
                if sha(artifact) != value:
                    raise ValueError("changed parent worker artifact")
                bound[str(artifact)] = value
    write_once(target / "plan.json", dict(plan, bound_files=bound, **changes))
    copy_cpu(parent, target)
    return target


def submit_array(root, worker_action, analyzer):
    job = launch(root, worker_action, gpu=True, array=True)
    after = launch(root, analyzer, dependency=job)
    write_once(root / "submission.json", dict(gpu_array=job, cpu_analysis=after))


def begin(root, expected):
    checked(root, expected)
    try:
        proof = read(root / "primitive/result.json")
        if (
            not proof["complete"]
            or proof["source_sha"] != clean_source(SOURCE)
            or proof["plan_sha256"] != expected
            or len(proof["controls"]["cases"]) != 2
            or any(
                not c["bitwise"] or not c["one_source_decode"] for c in proof["controls"]["cases"]
            )
        ):
            raise ValueError(f"primitive controls failed: {proof.get('failures')}")
    except (OSError, KeyError, ValueError) as error:
        closeout(root, dict(complete=False, failures=[str(error)], summaries=[]))
        return
    target = child(root, "feasibility", stage="native_full_prompt_model_feasibility")
    submit_array(target, "feasibility", "feasibility_analysis")


def setup(root, expected, index, *, operator_study=False, model_profile=False):
    import torch

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation.scheduler_identity import scheduler_identity
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = checked(root, expected)
    cpu_proof(root, plan)
    if (
        torch.cuda.device_count() != 1
        or "A100" not in torch.cuda.get_device_name()
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("one isolated deterministic A100 required")
    torch.set_num_threads(12)
    torch.manual_seed(plan["seed"])
    torch.use_deterministic_algorithms(True)
    job = plan["jobs"][index]
    cfg = RunConfig.from_mapping(pair.common.previous.resolved_mapping(plan, job))
    if (
        cfg.model.id,
        cfg.model.revision,
        cfg.model.precision.value,
        cfg.execution.attention_implementation,
    ) != ("llava-hf/llava-1.5-7b-hf", "b234b804b114d9e37bb655e11cbbb5f5e971b7a9", "bf16", "eager"):
        raise ValueError("frozen model/precision/backend changed")
    refs = pair.references(job["input_index"])
    payload = read(job["inputs"][0])
    messages = pair.common.previous.load_messages(payload)
    profile = tuple(read(plan["profile_path"]))
    manifest = read(pair.common.REFERENCE / f"workers/{job['reference_worker']}/manifest.json")
    manifest.update(scheduler_identity(manifest))
    manifest.update(
        source_sha=plan["source_sha"],
        source_dirty=False,
        execution=dict(manifest["execution"], seed=plan["seed"]),
        seed=plan["seed"],
        execution_commit=plan["source_sha"],
        methods=[
            "mosaickv_exact",
            "prompt_attention_topk",
            "prefixkv_reimpl",
            "vl_cache_reimpl",
            "lookm_reimpl",
        ],
        variant="lossless_full_prompt_native_execution",
        baseline_comparison_admissible=False,
        plan_sha256=expected,
        config_sha256=digest(dict(plan=expected, index=index)),
        run_id=str(root / f"workers/{index}"),
        task=payload["task"],
        sample_id=payload["sample_id"],
        gpu=asdict(capture_gpu_environment(torch)),
        environment=pair.common.control.environment(),
        measurement_type="diagnostic_measured",
        independent_quality_N=0,
        original_six_mechanisms_disabled_in_this_precision_variant=True,
        representation_adaptations_not_official_methods=True,
        shared_encoded_byte_budget_claim=False,
        paper_superiority_established=False,
        whole_request_GPU_peak_certified=False,
        evaluator="not_used_bitwise_parent_controls_and_engineering_only",
        actual_timing_protocol=dict(
            output_predictions=128,
            warmups=5,
            repetitions=20,
            tensor_audits_excluded_from_latency=True,
            conversion_and_workspace_included=True,
        ),
    )
    if operator_study and model_profile:
        raise ValueError("one explicit measurement protocol required")
    if model_profile:
        manifest.update(
            variant="consumed_actual_model_kernel_attribution",
            methods=["mosaickv_exact"],
            actual_timing_protocol=dict(
                output_predictions=128,
                warmups=1,
                instrumented_requests_per_policy=1,
                profiled_decode_steps=[0, 64, 126],
                no_request_latency_or_quality_claim=True,
                history_sealing_outside_profile_windows=True,
                CUDA_kernel_activity_required=True,
            ),
        )
    if operator_study:
        manifest.update(
            variant="consumed_source_reconstruction_operator_study",
            methods=["mosaickv_exact"],
            actual_timing_protocol=dict(
                output_predictions=0,
                warmups=5,
                repetitions=20,
                source_layers=[0, 16, 31],
                history_padding=[0, 64, 127],
                no_model_decode=True,
                no_benchmark_scoring=True,
                operator_intervals_not_complete_request_latency=True,
                intermediates_excluded_from_uninstrumented_pipeline=True,
                source_and_output_workspaces_explicitly_counted=True,
            ),
        )
    write_once(root / f"workers/{index}/manifest.json", manifest)
    adapter = load_hf_adapter(
        cfg.model.id,
        revision=cfg.model.revision,
        model_kwargs=_model_kwargs(cfg, local_files_only=True),
        processor_kwargs={"local_files_only": True},
    )
    return plan, adapter, refs, messages, profile, manifest


def execute(
    adapter, refs, messages, profile, plan, method, policy, *, audit, validate_packet=False
):
    import torch

    from mosaickv.evaluation.common_streaming_timing import request

    def validator(packets):
        actual = pair.common.control.digest_tensors(tuple(t for p in packets for t in p.tensors()))
        if actual != refs[method]["payload"]["payload_sha256"]:
            raise RuntimeError("original pre-conversion packet bits changed")

    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    values, packets, measured = request(
        adapter,
        messages,
        profile,
        method,
        minimum_only=plan["exact_minimum_only"],
        componentwise_fitting=True,
        code_delta=method in plan.get("code_delta_methods", []),
        input_packet_validator=validator if validate_packet else None,
        construction_geometry=plan.get("construction_geometry"),
        reconstruction_readers="carry_literals",
        audit_decode_storage=audit,
        source_hotpath="original",
        full_prompt_native=policy,
    )
    if not torch.equal(values, refs[method]["logits"]):
        raise RuntimeError("native request changed a frozen 128-token vocabulary-logit bit")
    if (
        measured["original_precompression_packet_bytes"]
        != refs[method]["payload"]["physical_bytes"]
    ):
        raise RuntimeError("constructor source storage changed")
    report = measured["native_conversion"]
    if policy is not None and method == "mosaickv_exact":
        selected = len(report["converted_layers"])
        if (
            not report["enabled"]
            or selected < 8
            or report["repair_bytes"] > plan["persistent_repair_cap_bytes"]
            or measured["native_full_source_decodes"] != 127 * selected
            or measured["native_full_component_fills"]
            != 127 * selected * (1 if policy == "v_only" else 2)
        ):
            raise RuntimeError("requested native construction/reuse did not engage")
    elif measured["native_full_source_decodes"] or measured["native_full_component_fills"]:
        raise RuntimeError("ineligible/original packet silently used native projection")
    expected_fused = (
        measured["native_full_component_fills"]
        if policy in ("kv_fused_restore", "kv_fused_prepared") and method == "mosaickv_exact"
        else 0
    )
    if measured["native_fused_restore_fills"] != expected_fused:
        raise RuntimeError("fused native restore activity did not match constructed policy")
    if policy == "kv_fused_prepared" and method == "mosaickv_exact":
        misses = 2 * len(report["converted_layers"])
        if any(
            measured[f"native_plan_{kind}_misses"] != misses
            or measured[f"native_plan_{kind}_hits"] != expected_fused - misses
            for kind in ("weight", "restore")
        ):
            raise RuntimeError("request-private native plans did not engage at every decode step")
    measured.update(
        logits_bitwise=True,
        torch_peak_allocated=torch.cuda.max_memory_allocated(),
        torch_peak_reserved=torch.cuda.max_memory_reserved(),
        original_packet_bits_checked=validate_packet,
    )
    del values, packets
    return measured


def memory_gate(measured, refs):
    competitors = [v["storage"] for k, v in refs.items() if k != "mosaickv_exact"]
    fields = dict(
        zip(
            STORAGE,
            (
                "initial_active_kv_storage_bytes",
                "final_active_kv_storage_bytes",
                "peak_active_kv_storage_bytes",
            ),
            strict=True,
        )
    )
    return all(
        type(measured[k]) is int and measured[k] < min(r[v] for r in competitors)
        for k, v in fields.items()
    )


def verify_fast(measured, audited, counter):
    if (
        measured["tensor_result_auditing"]
        or measured["peak_measurement_admissible"]
        or measured["decode_owned_peak_bytes"] is not None
    ):
        raise RuntimeError("timed counters cannot become physical peak evidence")
    if (
        any(measured[k] != audited[k] for k in STORAGE[:2])
        or measured["reported_counter_peak_bytes"] != counter
    ):
        raise RuntimeError("timed native storage changed its same-policy audited companion")


def registered_bounds(rows, seed, *, all_methods=False):
    """Conservative engineering intervals across BOTH possible policies.

    Keep the ordinary/eight-comparison intervals as diagnostics, never silently
    use them for selection after trying two policies. No image-quality inference.
    """
    import numpy as np

    timed = [r for r in rows if not r["warmup"]]
    ids = np.random.default_rng(seed).integers(0, 20, (10000, 20))
    tail = 0.05 / (2 * 16)

    def array(method, mode=None):
        selected = sorted(
            (r for r in timed if r["method"] == method and (mode is None or r["reader"] == mode)),
            key=lambda r: r["trial"],
        )
        if [r["trial"] for r in selected] != list(range(20)):
            raise ValueError("complete unique twenty-trial contrast required")
        result = np.asarray([r["total_wall_seconds"] for r in selected])
        if not np.isfinite(result).all() or not (result > 0).all():
            raise ValueError("finite positive engineering time required")
        return result

    def bounds(a, b):
        return np.quantile(a[ids].mean(1) / b[ids].mean(1), [tail, 1 - tail]).tolist()

    if all_methods:
        return {
            m: bounds(array("mosaickv_exact"), array(m))
            for m in ("prompt_attention_topk", "prefixkv_reimpl", "vl_cache_reimpl", "lookm_reimpl")
        }
    return dict(
        within_method={
            m: bounds(array(m, "native"), array(m, "original"))
            for m in ("mosaickv_exact", "prefixkv_reimpl")
        },
        versus_prefix={
            mode: bounds(array("mosaickv_exact", mode), array("prefixkv_reimpl", mode))
            for mode in READERS
        },
    )


def finish_worker(out, result):
    if result["status"] != "completed":
        manifest = read(out / "manifest.json") if (out / "manifest.json").exists() else {}
        with (out / "raw.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        status="failed",
                        worker_dir=str(out),
                        errors=result["failures"],
                        source_sha=result["source_sha"],
                        plan_sha256=result["plan_sha256"],
                        measurement_type="diagnostic_measured",
                        task=manifest.get("task", "not_loaded"),
                        sample_id=manifest.get("sample_id", "not_loaded"),
                    ),
                    allow_nan=False,
                )
                + "\n"
            )
    result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    write_once(out / "result.json", result)
    if result["status"] != "completed":
        raise SystemExit(1)


def feasibility(root, expected, index):
    from mosaickv.evaluation.full_prompt_source import FeasibilityDeclined

    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], plan_sha256=expected, source_sha=clean_source(SOURCE)
    )
    try:
        plan, adapter, refs, messages, profile, _manifest = setup(root, expected, index)
        evidence = {}
        for method in ("mosaickv_exact", "prefixkv_reimpl"):
            measured = execute(
                adapter,
                refs,
                messages,
                profile,
                plan,
                method,
                None,
                audit=True,
                validate_packet=True,
            )
            for internal, original in pair.STORAGE_FIELDS:
                if measured[internal] != refs[method]["storage"][original]:
                    raise RuntimeError(
                        "disabled native path changed a frozen audited byte boundary"
                    )
            evidence[method] = measured
        policies = {}
        for policy in plan["native_policies"]:
            try:
                measured = execute(
                    adapter,
                    refs,
                    messages,
                    profile,
                    plan,
                    "mosaickv_exact",
                    policy,
                    audit=True,
                    validate_packet=True,
                )
                policies[policy] = dict(
                    numerical_controls_passed=True,
                    memory_gate=memory_gate(measured, refs),
                    measured=measured,
                )
            except FeasibilityDeclined as error:
                policies[policy] = dict(
                    numerical_controls_passed=False,
                    memory_gate=False,
                    storage_declined=error.report,
                )
        write_once(
            out / "feasibility.json",
            dict(
                original=evidence,
                policies=policies,
                frozen_competitor_bytes={
                    k: v["storage"] for k, v in refs.items() if k != "mosaickv_exact"
                },
            ),
        )
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    finish_worker(out, result)


def load_producers(root, plan, filename):
    rows, failures = [], []
    for index in (0, 1):
        try:
            folder = root / f"workers/{index}"
            result = read(folder / "result.json")
            if (
                result["status"] != "completed"
                or result["failures"]
                or result["source_sha"] != plan["source_sha"]
                or result["plan_sha256"] != sha(root / "plan.json")
                or any(sha(folder / n) != h for n, h in result["outputs"].items())
            ):
                raise ValueError(f"incomplete/changed producer: {result.get('failures')}")
            rows.append(read(folder / filename))
        except Exception as error:
            failures.append(dict(input_index=index, error=str(error)))
    return rows, failures


def feasibility_analysis(root, expected):
    plan = checked(root, expected)
    rows, failures = load_producers(root, plan, "feasibility.json")
    if len(rows) != 2 and not failures:
        failures.append("both consumed-input producers required")
    eligible = [
        p
        for p in plan["native_policies"]
        if not failures
        and all(
            r["policies"].get(p, {}).get("numerical_controls_passed") is True
            and r["policies"].get(p, {}).get("memory_gate") is True
            for r in rows
        )
    ]
    decision = dict(
        complete=not failures,
        failures=failures,
        eligible_policies=eligible,
        selected_policy=eligible[0] if eligible else None,
        summaries=rows,
        independent_quality_N=0,
    )
    write_once(root / "analysis/decision.json", decision)
    if not eligible or failures:
        closeout(root, decision)
        return
    target = child(
        root,
        "paired-0",
        stage="native_full_prompt_paired",
        selected_policy=eligible[0],
        eligible_policies=eligible,
        policy_attempt=0,
        feasibility_root=str(root),
    )
    submit_array(target, "paired", "paired_analysis")


def paired(root, expected, index):
    from mosaickv.evaluation.request_reader_diagnostic import order, summarize

    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], plan_sha256=expected, source_sha=clean_source(SOURCE)
    )
    try:
        plan, adapter, refs, messages, profile, manifest = setup(root, expected, index)
        evidence = read(Path(plan["feasibility_root"]) / f"workers/{index}/feasibility.json")
        controls, counters, rows = {}, {}, []
        for method in ("mosaickv_exact", "prefixkv_reimpl"):
            for mode in READERS:
                native = mode == "native"
                audit = (
                    evidence["policies"][plan["selected_policy"]]["measured"]
                    if native and method == "mosaickv_exact"
                    else evidence["original"][method]
                )
                measured = execute(
                    adapter,
                    refs,
                    messages,
                    profile,
                    plan,
                    method,
                    plan["selected_policy"] if native else None,
                    audit=False,
                    validate_packet=True,
                )
                counter = measured["reported_counter_peak_bytes"]
                verify_fast(measured, audit, counter)
                key = f"{method}:{mode}"
                controls[key], counters[key] = audit, counter
        write_once(
            out / "storage_companions.json",
            dict(
                controls=controls,
                counters=counters,
                same_policy_boundary_not_equal_between_policies=True,
            ),
        )
        for entry in order(plan["seed"] + index, readers=READERS):
            method, mode = entry["method"], entry["reader"]
            measured = execute(
                adapter,
                refs,
                messages,
                profile,
                plan,
                method,
                plan["selected_policy"] if mode == "native" else None,
                audit=False,
            )
            key = f"{method}:{mode}"
            verify_fast(measured, controls[key], counters[key])
            row = dict(
                **entry,
                **measured,
                accounting_identical=True,
                instrumented=False,
                accounting_identical_means="own_registered_policy_companion_not_parent_or_baseline",
                manifest_path=str(out / "manifest.json"),
                run_id=manifest["run_id"],
                task=manifest["task"],
                sample_id=manifest["sample_id"],
                config_sha256=digest(
                    dict(
                        matrix=manifest["config_sha256"],
                        method=method,
                        policy=plan["selected_policy"] if mode == "native" else None,
                    )
                ),
                measurement_type="diagnostic_measured",
                source_sha=plan["source_sha"],
                plan_sha256=expected,
            )
            with (out / "raw.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
        summary = summarize(rows, plan["seed"] + index, readers=READERS)
        adjusted = registered_bounds(rows, plan["seed"] + index)
        for group, contrasts in adjusted.items():
            for name, bounds in contrasts.items():
                summary[group][name]["registered_two_policy_ratio_bounds"] = bounds
        summary["registered_multiple_testing"] = (
            "Bonferroni 16 engineering contrasts including two possible policies"
        )
        write_once(out / "summary.json", summary)
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    finish_worker(out, result)


def paired_passes(rows):
    return len(rows) == 2 and all(
        r["within_method"]["mosaickv_exact"]["registered_two_policy_ratio_bounds"][1] <= 0.99
        and r["versus_prefix"]["native"]["registered_two_policy_ratio_bounds"][1] <= 1.05
        for r in rows
    )


def paired_analysis(root, expected):
    plan = checked(root, expected)
    rows, failures = load_producers(root, plan, "summary.json")
    passed = not failures and paired_passes(rows)
    decision = dict(
        complete=not failures,
        failures=failures,
        summaries=rows,
        paired_latency_gate=passed,
        selected_policy=plan["selected_policy"],
        independent_quality_N=0,
    )
    write_once(root / "analysis/decision.json", decision)
    if failures:
        closeout(root, decision)
    elif passed:
        target = child(root, "all-five", stage="native_full_prompt_all_five")
        submit_array(target, "all_five", "final_analysis")
    elif plan["policy_attempt"] == 0 and len(plan["eligible_policies"]) > 1:
        target = child(
            root, "paired-1", selected_policy=plan["eligible_policies"][1], policy_attempt=1
        )
        submit_array(target, "paired", "paired_analysis")
    else:
        closeout(root, decision)


def all_five(root, expected, index):
    from mosaickv.evaluation.common_streaming_timing import METHODS, order, summarize

    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], plan_sha256=expected, source_sha=clean_source(SOURCE)
    )
    try:
        plan, adapter, refs, messages, profile, manifest = setup(root, expected, index)
        controls, counters, rows = {}, {}, []
        for method in METHODS:
            audited = execute(
                adapter,
                refs,
                messages,
                profile,
                plan,
                method,
                plan["selected_policy"],
                audit=True,
                validate_packet=True,
            )
            fast = execute(
                adapter, refs, messages, profile, plan, method, plan["selected_policy"], audit=False
            )
            verify_fast(fast, audited, fast["reported_counter_peak_bytes"])
            if method == "mosaickv_exact" and not memory_gate(audited, refs):
                raise RuntimeError("final native memory gate no longer passes")
            if method != "mosaickv_exact" and any(
                audited[k] != refs[method]["storage"][v] for k, v in pair.STORAGE_FIELDS
            ):
                raise RuntimeError("ineligible baseline storage changed")
            controls[method], counters[method] = audited, fast["reported_counter_peak_bytes"]
        write_once(
            out / "audited_storage_controls.json", dict(controls=controls, counters=counters)
        )
        for entry in order(plan["seed"] + index):
            method = entry["method"]
            measured = execute(
                adapter, refs, messages, profile, plan, method, plan["selected_policy"], audit=False
            )
            verify_fast(measured, controls[method], counters[method])
            row = dict(
                **entry,
                **measured,
                manifest_path=str(out / "manifest.json"),
                run_id=manifest["run_id"],
                task=manifest["task"],
                sample_id=manifest["sample_id"],
                config_sha256=digest(
                    dict(
                        matrix=manifest["config_sha256"],
                        method=method,
                        policy=plan["selected_policy"],
                    )
                ),
                measurement_type="diagnostic_measured",
                source_sha=plan["source_sha"],
                plan_sha256=expected,
            )
            with (out / "raw.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            rows.append(row)
        summary = summarize(rows, plan["seed"] + index)
        adjusted = registered_bounds(rows, plan["seed"] + index, all_methods=True)
        for method, bounds in adjusted.items():
            summary["pairs"][method]["registered_two_policy_ratio_bounds"] = bounds
        summary["registered_multiple_testing"] = (
            "Bonferroni 16 engineering contrasts including two possible policies"
        )
        for method, distribution in summary["distributions"].items():
            selected = [r for r in rows if not r["warmup"] and r["method"] == method]
            distribution["mean_decode_seconds"] = sum(
                r["decode_wall_seconds"] for r in selected
            ) / len(selected)
            distribution["mean_construction_seconds"] = sum(
                r["prefill_construct_retire_wall_seconds"] for r in selected
            ) / len(selected)
        write_once(out / "summary.json", summary)
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    finish_worker(out, result)


def final_analysis(root, expected):
    plan = checked(root, expected)
    rows, failures = load_producers(root, plan, "summary.json")
    no_slower = (
        not failures
        and len(rows) == 2
        and all(
            p["registered_two_policy_ratio_bounds"][1] <= 1.0
            for r in rows
            for p in r["pairs"].values()
        )
    )
    decision = dict(
        complete=not failures,
        failures=failures,
        summaries=rows,
        all_method_no_slower_gate=no_slower,
        selected_policy=plan["selected_policy"],
        independent_quality_N=0,
    )
    write_once(root / "analysis/decision.json", decision)
    if (
        not failures
        and not no_slower
        and plan["policy_attempt"] == 0
        and len(plan["eligible_policies"]) > 1
    ):
        target = child(
            root,
            "paired-1",
            selected_policy=plan["eligible_policies"][1],
            policy_attempt=1,
            stage="native_full_prompt_paired",
        )
        submit_array(target, "paired", "paired_analysis")
        return
    closeout(root, decision)


def closeout(root, decision):
    plan = read(root / "plan.json")
    terminal = Path(plan["chain_root"])
    write_once(
        terminal / "engineering_closeout.json",
        dict(
            decision,
            final_root=str(root),
            source_sha=plan["source_sha"],
            independent_quality_N=0,
            paper_superiority_established=False,
            fresh_or_large_campaign_submitted=False,
            whole_request_GPU_peak_certified=False,
        ),
    )
    text = "# Full-prompt native reconstruction engineering closeout\n\n"
    text += "Two consumed inputs; no independent quality or byte-matched algorithm claim.\n\n"
    text += "Peak accounting is decode-owned tensor cache storage, NOT whole-request GPU peak.\n\n"
    for index, summary in enumerate(decision.get("summaries", [])):
        distributions = summary.get("distributions", [])
        if isinstance(distributions, dict):
            distributions = [dict(v, method=k) for k, v in distributions.items()]
        if distributions:
            text += f"## Consumed input {index}\n\n"
            text += "| Method/policy | Mean s | p50 s | p95 s |\n|---|---:|---:|---:|\n"
            for d in distributions:
                text += f"| {d['method']} {d.get('reader', '')} | {d['mean']:.6f} "
                text += f"| {d['p50']:.6f} | {d['p95']:.6f} |\n"
    text += "\n```json\n" + json.dumps(decision, indent=2, allow_nan=False) + "\n```\n"
    with (terminal / "engineering_report.md").open("x") as stream:
        stream.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "primitive",
            "begin",
            "feasibility",
            "feasibility_analysis",
            "paired",
            "paired_analysis",
            "all_five",
            "final_analysis",
        ),
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action in ("feasibility", "paired", "all_five"):
        globals()[args.action](args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.action](args.root, args.plan_sha)
