#!/usr/bin/env python3
"""Finite role attribution -> exact reader reuse -> conditional request timing."""

import argparse
import gc
import json
import os
import random
import subprocess
from dataclasses import asdict
from functools import partial
from pathlib import Path
from unittest.mock import patch

import evicastkv_code_delta_gpu as backend
import evicastkv_common_streaming as common
import evicastkv_decode_phase_followup as profiler
import numpy as np

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.reconstruction_roles import choose_order
from mosaickv.evaluation.scheduler_identity import scheduler_identity

SOURCE = common.SOURCE
PARENT = Path(
    "/scratch/djy8hg/runs/evicastkv/staged-source-geometry-consumed-20261102-retry1/timing"
)
DOC = SOURCE / "mosaickv/docs/reconstruction_reader_followup_20261102.md"


def submit(root):
    expected = sha(root / "plan.json")
    original = common.control.parent.subprocess.check_output

    def scoped(command, **kwargs):
        return original(
            [
                str(SOURCE / "mosaickv/slurm/evicastkv_reconstruction_followup.sbatch")
                if str(x).endswith("/evicastkv_funded_delimiter_diagnostic.sbatch")
                else x
                for x in command
            ],
            **kwargs,
        )

    with patch.object(common.control.parent.subprocess, "check_output", scoped):
        common.control.parent.submit(root, expected)


def bootstrap(root, revision):
    if clean_source(SOURCE) != revision or root.exists():
        raise ValueError("clean immutable source and new root required")
    old = common.previous.read(PARENT / "plan.json")
    parent = common.previous.read(PARENT / "analysis/decision.json")
    if (
        not parent.get("complete")
        or parent.get("failures")
        or parent.get("all_method_similar_time_gate")
        or parent.get("plan_sha256") != sha(PARENT / "plan.json")
    ):
        raise ValueError("complete negative request-timing parent required")
    bound = dict(old["bound_files"])
    for p in (PARENT / "plan.json", PARENT / "analysis/decision.json", DOC):
        bound[str(p)] = sha(p)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            reconstruction_stage="roles",
            stage="consumed_prompt_text_history_role_attribution",
            maximum_worker_minutes=15,
            maximum_simultaneous_workers=2,
            paired_common_streaming_timing=False,
            profile_warmups=1,
            profile_repetitions=2,
            reconstruction_readers="original",
            instrumentation_not_performance_evidence=True,
        ),
    )
    common.previous.cpu(root, sha(root / "plan.json"))
    submit(root)


def prepare_verification(root, index, policy):
    if index not in (0, 1):
        raise ValueError("two bounded reader attempts only")
    original = common.previous.read(root / "plan.json")
    target = root / f"verification-{index}"
    bound = dict(original["bound_files"])
    for path in (root / "plan.json", root / "analysis/decision.json", root / "role_order.json"):
        bound[str(path)] = sha(path)
    write_once(
        target / "plan.json",
        dict(
            original,
            bound_files=bound,
            reconstruction_stage="verify",
            stage="consumed_exact_integer_reader_verification",
            reconstruction_readers=policy,
            study_root=str(root),
            attempt_index=index,
            maximum_worker_minutes=45,
            maximum_simultaneous_workers=2,
            instrumented_not_comparative_latency=True,
        ),
    )
    for name in ("cpu.json", "cpu.xml"):
        os.link(root / name, target / name)
    write_once(
        target / "cpu_reuse.json",
        dict(
            source_sha=original["source_sha"],
            producer_root=str(root),
            same_complete_suite_same_immutable_source=True,
            cpu_json_sha256=sha(root / "cpu.json"),
            cpu_xml_sha256=sha(root / "cpu.xml"),
        ),
    )
    submit(target)


def roles_worker(root, expected, index):
    from mosaickv.evaluation import decode_phase_timing, reconstruction_roles

    with patch.object(decode_phase_timing, "instrument", reconstruction_roles.instrument):
        profiler.worker(root, expected, index)


def roles_analyze(root, expected):
    profiler.analyze(root, expected)
    decision = common.previous.read(root / "analysis/decision.json")
    if not decision["complete"] or decision["failures"]:
        return
    order, costs = choose_order(decision["summaries"])
    write_once(
        root / "role_order.json",
        dict(
            order=order,
            projection_stream_ms=costs,
            decision_sha256=sha(root / "analysis/decision.json"),
            not_request_latency_or_quality_evidence=True,
        ),
    )
    prepare_verification(root, 0, order[0])


def verify_worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation import shared_source_format, storage_runs
    from mosaickv.evaluation.common_streaming_timing import METHODS
    from mosaickv.evaluation.layer_code_delta import convert
    from mosaickv.evaluation.reconstruction_controls import gpu_controls
    from mosaickv.evaluation.reconstruction_verification import (
        helper_trials,
        resources,
        spills_pass,
    )
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
            raise ValueError("one deterministic isolated A100 required")
        torch.set_num_threads(12)
        torch.manual_seed(plan["seed"])
        torch.use_deterministic_algorithms(True)
        write_once(
            out / "reader_controls.json", dict(cases=gpu_controls(plan["reconstruction_readers"]))
        )
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
        )
        manifest.update(scheduler_identity(manifest))
        write_once(out / "manifest.json", manifest)
        refs = {}
        for original in common.previous.read(common.AUDIT / "plan.json")["jobs"]:
            if original["input_index"] != job["input_index"]:
                continue
            method = common.method_of(original)
            folder = common.AUDIT / f"workers/{original['index']}"
            old_rows = [
                json.loads(x)
                for x in (common.CODE_AUDIT / f"workers/{original['index']}/raw.jsonl")
                .read_text()
                .splitlines()
            ]
            refs[method] = dict(
                payload=common.previous.read(folder / "offline_payload.json"),
                logits=torch.load(
                    folder / "greedy_logits.pt", weights_only=True, map_location="cpu"
                ),
                storage=old_rows[0]["generation_storage"],
            )
        messages = common.previous.load_messages(common.previous.read(job["inputs"][0]))
        profile = tuple(common.previous.read(Path(plan["profile_path"])))
        methods = list(METHODS)
        random.Random(plan["seed"] + index).shuffle(methods)
        for method in methods:
            with patch.object(
                shared_source_format,
                "fit_transport",
                partial(shared_source_format.fit_transport, componentwise=True),
            ):
                runtime, packets, memory = backend.construct(
                    adapter,
                    messages,
                    profile,
                    method,
                    audit=False,
                    construction_geometry=plan.get("construction_geometry"),
                )
            if (
                common.control.digest_tensors(tuple(t for p in packets for t in p.tensors()))
                != refs[method]["payload"]["payload_sha256"]
            ):
                raise RuntimeError("reader validation changed original immutable packet bits")
            converted, code = convert(packets) if method == "mosaickv_exact" else (packets, None)
            caches = []
            factory = storage_runs.cache_from_formats

            def capture(*args, caches=caches, factory=factory, **kwargs):
                cache = factory(*args, **kwargs)
                caches.append(cache)
                return cache

            with (
                resources(plan["reconstruction_readers"]) as resource_records,
                patch.object(storage_runs, "cache_from_formats", capture),
            ):
                logits, detail = backend.decode_run(adapter, runtime, converted, audit=True)
            if not torch.equal(logits, refs[method]["logits"]):
                raise RuntimeError("reader changed frozen 128-token vocabulary logits")
            accounting = (
                "initial_active_kv_storage_bytes",
                "final_active_kv_storage_bytes",
                "peak_active_kv_storage_bytes",
            )
            if any(detail["storage"][k] != refs[method]["storage"][k] for k in accounting):
                raise RuntimeError("reader changed an audited live-cache storage boundary")
            micro = (
                helper_trials(caches[0], plan["reconstruction_readers"], plan["seed"] + index)
                if method == "mosaickv_exact"
                else None
            )
            row = dict(
                status="completed",
                method=method,
                input_index=job["input_index"],
                sample_id=common.previous.read(job["inputs"][0])["sample_id"],
                manifest_path=str(out / "manifest.json"),
                source_sha=plan["source_sha"],
                plan_sha256=expected,
                reconstruction_readers=plan["reconstruction_readers"],
                generation_128_logits_bitwise=True,
                original_packet_bits_identical=True,
                storage=detail["storage"],
                accounting_identical=True,
                spills_nonincreasing=spills_pass(resource_records),
                resources=list(resource_records.values()),
                helper_trials=micro,
                helper_buffers_outside_primary_storage_audit=True,
                inactive_control_packets_still_alive=True,
                whole_request_GPU_peak_certified=False,
                no_comparative_request_latency_claim=True,
                independent_quality_N=0,
            )
            with (out / "raw.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            del runtime, packets, converted, code, logits, detail, caches, capture, memory, row
            gc.collect()
            torch.cuda.synchronize()
        result["status"] = "completed"
    except Exception as exc:
        result["failures"].append(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        result["outputs"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        write_once(out / "result.json", result)


def helper_gate(trials, policy, seed):
    if len(trials) != 50 or {(r["trial"], r["policy"]) for r in trials} != {
        (t, p) for t in range(-5, 20) for p in ("original", policy)
    }:
        raise ValueError("complete unique five-warmup/twenty-paired-helper grid required")
    arrays = {
        p: np.array(
            [
                next(r["cuda_stream_ms"] for r in trials if r["trial"] == t and r["policy"] == p)
                for t in range(20)
            ]
        )
        for p in ("original", policy)
    }
    if any(not np.isfinite(a).all() or not (a > 0).all() for a in arrays.values()):
        raise ValueError("finite positive helper timings required")
    ids = np.random.default_rng(seed).integers(0, 20, size=(10000, 20))
    ratios = arrays[policy][ids].mean(1) / arrays["original"][ids].mean(1)
    upper = float(np.quantile(ratios, 1 - 0.05 / (2 * 4)))
    return dict(
        mean_ratio=float(arrays[policy].mean() / arrays["original"].mean()),
        familywise_upper_ratio=upper,
        promote=upper <= 0.99,
        not_request_latency=True,
    )


def verify_analyze(root, expected):
    plan = common.previous.common.inputs(root, expected)
    from mosaickv.evaluation.common_streaming_timing import METHODS

    rows, failures = [], []
    for index in (0, 1):
        folder = root / f"workers/{index}"
        try:
            result = common.previous.read(folder / "result.json")
            if (
                result["status"] != "completed"
                or result["failures"]
                or result["source_sha"] != plan["source_sha"]
                or result["plan_sha256"] != expected
                or any(sha(folder / f) != h for f, h in result["outputs"].items())
            ):
                raise ValueError("failed or changed exact-reader producer")
            group = [json.loads(x) for x in (folder / "raw.jsonl").read_text().splitlines()]
            if (
                len(group) != 5
                or {r["method"] for r in group} != set(METHODS)
                or any(
                    not all(
                        r.get(k) is True
                        for k in (
                            "generation_128_logits_bitwise",
                            "original_packet_bits_identical",
                            "accounting_identical",
                        )
                    )
                    for r in group
                )
            ):
                raise ValueError("incomplete original-bit/storage controls")
            rows.extend(group)
        except (OSError, ValueError, KeyError) as exc:
            failures.append(dict(index=index, error=str(exc)))
    summaries = []
    if not failures:
        for r in rows:
            if r["method"] == "mosaickv_exact":
                try:
                    summaries.append(
                        dict(
                            input_index=r["input_index"],
                            **helper_gate(
                                r["helper_trials"],
                                plan["reconstruction_readers"],
                                plan["seed"] + r["input_index"],
                            ),
                        )
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    failures.append(dict(index=r["input_index"], error=str(exc)))
    promoted = (
        not failures
        and all(r["spills_nonincreasing"] for r in rows)
        and len(summaries) == 2
        and all(s["promote"] for s in summaries)
    )
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not failures,
            failures=failures,
            summaries=summaries,
            candidate_promoted=promoted,
            all_method_spills_nonincreasing=bool(rows)
            and all(r["spills_nonincreasing"] for r in rows),
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            independent_quality_N=0,
            paper_superiority_established=False,
        ),
    )
    if failures:
        return
    if not promoted:
        advance(root, expected)
        return
    target = root / "timing"
    parent = common.previous.read(PARENT / "plan.json")
    bound = dict(plan["bound_files"])
    for p in (root / "plan.json", root / "analysis/decision.json"):
        bound[str(p)] = sha(p)
    write_once(
        target / "plan.json",
        dict(
            parent,
            source_sha=plan["source_sha"],
            bound_files=bound,
            reconstruction_readers=plan["reconstruction_readers"],
            reader_verification=str(root / "analysis/decision.json"),
        ),
    )
    for name in ("cpu.json", "cpu.xml"):
        os.link(root / name, target / name)
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
            "--job-name=mk-reader-advance",
            f"--output={root}/slurm/advance-%j.out",
            f"--error={root}/slurm/advance-%j.err",
            str(SOURCE / "mosaickv/slurm/evicastkv_reconstruction_followup.sbatch"),
            str(SOURCE),
            str(root),
            expected,
            "advance",
        ],
        text=True,
    ).strip()
    write_once(root / "registered_advance.json", dict(job_id=job, dependency=dependency))


def advance(root, expected):
    plan = common.previous.common.inputs(root, expected)
    diagnostic = common.previous.read(root / "analysis/decision.json")
    timed = root / "timing/analysis/decision.json"
    timing = common.previous.read(timed) if timed.exists() else None
    safe = diagnostic["complete"] and not diagnostic["failures"]
    if timing is not None:
        safe = safe and timing["complete"] and not timing["failures"]
    passed = bool(safe and timing and timing["all_method_similar_time_gate"])
    study = Path(plan["study_root"])
    if safe and not passed and plan["attempt_index"] == 0:
        order = common.previous.read(study / "role_order.json")["order"]
        candidate = "reuse_both" if diagnostic["candidate_promoted"] else order[1]
        prepare_verification(study, 1, candidate)
    else:
        write_once(
            study / "engineering_closeout.json",
            dict(
                complete=safe,
                source_sha=plan["source_sha"],
                decision_path=str(timed if timing is not None else root / "analysis/decision.json"),
                all_method_similar_time_gate=passed,
                independent_quality_N=0,
                paper_superiority_established=False,
                fresh_or_large_campaign_submitted=False,
                automatic_reader_attempts_completed=plan["attempt_index"] + 1,
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("bootstrap", "worker", "analyze", "advance"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.mode == "bootstrap":
        bootstrap(args.root, args.source_sha)
    elif args.mode == "advance":
        advance(args.root, args.plan_sha)
    else:
        plan = common.previous.common.inputs(args.root, args.plan_sha)
        if args.mode == "worker":
            function = roles_worker if plan["reconstruction_stage"] == "roles" else verify_worker
            function(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
        else:
            function = roles_analyze if plan["reconstruction_stage"] == "roles" else verify_analyze
            function(args.root, args.plan_sha)
