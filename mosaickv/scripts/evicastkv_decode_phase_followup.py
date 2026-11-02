#!/usr/bin/env python3
"""Finite consumed-input attribution after the already registered geometry chain."""

import argparse
import json
import os
import random
import statistics
import subprocess
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import evicastkv_common_streaming as common

from mosaickv.evaluation.decode_phase_timing import next_dependency
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = common.SOURCE
STAGED = Path("/scratch/djy8hg/runs/evicastkv/staged-source-geometry-consumed-20261102-retry1")
ASYNC = Path("/scratch/djy8hg/runs/evicastkv/async-text-bitwise-consumed-20261102-v1/timing")
METHODS = ("mosaickv_exact", "prefixkv_reimpl")


def schedule(root, revision, dependency, step):
    control = root.with_name(root.name + "-launch")
    control.mkdir(exist_ok=True)
    job = subprocess.check_output(
        [
            "sbatch",
            "--parsable",
            "--partition=bii",
            "--account=bii_dsc_community",
            "--cpus-per-task=2",
            "--mem=16G",
            "--time=00:25:00",
            f"--dependency=afterany:{dependency}",
            "--job-name=mk-decode-phase-watch",
            f"--output={control}/watch-{step}-%j.out",
            f"--error={control}/watch-{step}-%j.err",
            str(SOURCE / "mosaickv/slurm/evicastkv_decode_phase_followup.sbatch"),
            str(SOURCE),
            str(root),
            revision,
            "watch",
            str(step),
        ],
        text=True,
    ).strip()
    write_once(
        control / f"watch-{step}.json", dict(job_id=job, dependency=dependency, source_sha=revision)
    )


def watch(root, revision, step):
    if clean_source(SOURCE) != revision or not 0 <= step <= 8:
        raise ValueError("clean finite attribution source required")
    blocked = STAGED.with_name(STAGED.name + "-launch") / "blocked.json"
    if not (STAGED / "engineering_closeout.json").exists() and not blocked.exists():
        dependency = next_dependency(STAGED)
        control = root.with_name(root.name + "-launch")
        prior = [json.loads(p.read_text())["dependency"] for p in control.glob("watch-*.json")]
        if dependency not in prior:
            schedule(root, revision, dependency, step + 1)
            return
        # This callback already waited for the most advanced published node.
        # No new publication means the producer failed; retain the diagnosis.
        write_once(
            control / "upstream_incomplete.json",
            dict(dependency=dependency, no_repeated_polling=True),
        )
    parent = STAGED / "timing"
    if not (parent / "analysis/decision.json").exists():
        parent = ASYNC
    decision = common.previous.read(parent / "analysis/decision.json")
    if (
        not decision.get("complete")
        or decision.get("failures")
        or decision.get("plan_sha256") != sha(parent / "plan.json")
    ):
        raise ValueError("safe complete parent timing evidence required")
    if decision["all_method_similar_time_gate"]:
        write_once(
            root / "not_needed.json",
            dict(
                reason="timing gate passed",
                independent_quality_N=0,
                paper_superiority_established=False,
            ),
        )
        return
    old = common.previous.read(parent / "plan.json")
    bound = dict(old["bound_files"])
    for path in (
        parent / "plan.json",
        parent / "analysis/decision.json",
        SOURCE / "mosaickv/docs/decode_phase_followup_20261102.md",
    ):
        bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_decode_phase_attribution",
            maximum_worker_minutes=15,
            maximum_simultaneous_workers=2,
            paired_common_streaming_timing=False,
            profile_methods=list(METHODS),
            profile_warmups=1,
            profile_repetitions=2,
            parent_timing=str(parent),
            instrumentation_not_performance_evidence=True,
        ),
    )
    expected = sha(root / "plan.json")
    common.previous.cpu(root, expected)
    original = common.control.parent.subprocess.check_output

    def scoped(command, **kwargs):
        return original(
            [
                str(SOURCE / "mosaickv/slurm/evicastkv_decode_phase_followup.sbatch")
                if str(x).endswith("/evicastkv_funded_delimiter_diagnostic.sbatch")
                else x
                for x in command
            ],
            **kwargs,
        )

    with patch.object(common.control.parent.subprocess, "check_output", scoped):
        common.control.parent.submit(root, expected)


def worker(root, expected, index):
    import torch

    from mosaickv.adapters.huggingface.registry import load_hf_adapter
    from mosaickv.config import RunConfig
    from mosaickv.evaluation import storage_runs
    from mosaickv.evaluation.common_streaming_timing import request
    from mosaickv.evaluation.construction_phase_timing import PhaseTimer, phase_totals
    from mosaickv.evaluation.decode_phase_timing import instrument
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
            raise ValueError("one deterministic A100 required")
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
            source_sha=plan["source_sha"],
            source_dirty=False,
            run_id=str(out),
            plan_sha256=expected,
            config_sha256=digest(dict(plan=expected, index=index)),
            measurement_type="diagnostic_measured",
            evaluator_commit=plan["source_sha"],
            environment=common.control.environment(),
            gpu=asdict(capture_gpu_environment(torch)),
            independent_quality_N=0,
            instrumentation_not_performance_evidence=True,
        )
        write_once(out / "manifest.json", manifest)
        original_jobs = common.previous.read(common.AUDIT / "plan.json")["jobs"]
        references = {}
        for original in original_jobs:
            method = common.method_of(original)
            if original["input_index"] == job["input_index"] and method in METHODS:
                folder = common.AUDIT / f"workers/{original['index']}"
                references[method] = dict(
                    logits=torch.load(
                        folder / "greedy_logits.pt", weights_only=True, map_location="cpu"
                    ),
                    payload=common.previous.read(folder / "offline_payload.json"),
                )
        messages = common.previous.load_messages(common.previous.read(job["inputs"][0]))
        profile = tuple(common.previous.read(Path(plan["profile_path"])))
        rng = random.Random(plan["seed"] + index)
        run = storage_runs.run
        for trial in range(-1, 2):
            order = list(METHODS)
            rng.shuffle(order)
            for method in order:
                timer = PhaseTimer(event_factory=lambda: torch.cuda.Event(enable_timing=True))

                def measured_run(*args, timer=timer, **kwargs):
                    with instrument(timer), timer.phase("complete_decode"):
                        return run(*args, **kwargs)

                validator = None
                if trial == -1:

                    def validator(packets, *, method=method):
                        if (
                            common.control.digest_tensors(
                                tuple(t for p in packets for t in p.tensors())
                            )
                            != references[method]["payload"]["payload_sha256"]
                        ):
                            raise RuntimeError("profile changed frozen packet bits")

                with patch.object(storage_runs, "run", measured_run):
                    logits, packets, measured = request(
                        adapter,
                        messages,
                        profile,
                        method,
                        minimum_only=plan.get("exact_minimum_only", False),
                        componentwise_fitting=True,
                        code_delta=method in plan.get("code_delta_methods", []),
                        source_geometry=plan.get("source_geometry"),
                        construction_geometry=plan.get("construction_geometry"),
                        input_packet_validator=validator,
                    )
                torch.cuda.synchronize()
                expected_bytes = (
                    plan["optimized_source_initial_bytes"][str(job["input_index"])]
                    if method == "mosaickv_exact"
                    else references[method]["payload"]["physical_bytes"]
                )
                if (
                    not torch.equal(logits, references[method]["logits"])
                    or measured["initial_encoded_bytes"] != expected_bytes
                ):
                    raise RuntimeError("decode profiling changed frozen logits or storage")
                trace = timer.resolve()
                row = dict(
                    status="completed",
                    method=method,
                    trial=trial,
                    warmup=trial < 0,
                    input_index=job["input_index"],
                    trace=trace,
                    phases=phase_totals(trace),
                    generation_128_logits_bitwise=True,
                    initial_encoded_bytes=expected_bytes,
                    measurements=measured,
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                    manifest_path=str(out / "manifest.json"),
                    independent_quality_N=0,
                    instrumentation_not_performance_evidence=True,
                )
                with (out / "raw.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                del logits, packets, timer, row
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
    for index in (0, 1):
        folder = root / f"workers/{index}"
        try:
            result = common.previous.read(folder / "result.json")
            if (
                result["status"] != "completed"
                or result["failures"]
                or result["source_sha"] != plan["source_sha"]
                or result["plan_sha256"] != expected
                or any(sha(folder / p) != h for p, h in result["outputs"].items())
            ):
                raise ValueError("failed or changed profiling producer")
            group = [json.loads(line) for line in (folder / "raw.jsonl").read_text().splitlines()]
            if (
                len(group) != 6
                or {(r["method"], r["trial"]) for r in group}
                != {(m, t) for m in METHODS for t in range(-1, 2)}
                or any(
                    not r.get("generation_128_logits_bitwise") or not r.get("trace") for r in group
                )
            ):
                raise ValueError("incomplete phase or parity evidence")
            rows.extend(group)
        except (OSError, ValueError, KeyError) as exc:
            failures.append(dict(index=index, error=str(exc)))
    summaries = []
    if not failures:
        for index in (0, 1):
            for method in METHODS:
                group = [
                    r
                    for r in rows
                    if r["input_index"] == index and r["method"] == method and not r["warmup"]
                ]
                phases = {
                    name: dict(
                        mean_exclusive_host_seconds=statistics.mean(
                            r["phases"][name]["exclusive_wall_seconds"] for r in group
                        ),
                        mean_inclusive_host_seconds=statistics.mean(
                            r["phases"][name]["inclusive_wall_seconds"] for r in group
                        ),
                        mean_inclusive_cuda_stream_ms=statistics.mean(
                            r["phases"][name]["stream_elapsed_ms"] for r in group
                        ),
                        calls=group[0]["phases"][name]["calls"],
                    )
                    for name in group[0]["phases"]
                }
                summaries.append(
                    dict(
                        input_index=index,
                        method=method,
                        phases=phases,
                        ranking=sorted(
                            phases,
                            key=lambda p: phases[p]["mean_exclusive_host_seconds"],
                            reverse=True,
                        ),
                    )
                )
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not failures,
            failures=failures,
            summaries=summaries,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            independent_quality_N=0,
            instrumentation_not_performance_evidence=True,
            fresh_or_large_campaign_submitted=False,
            paper_superiority_established=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("watch", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    parser.add_argument("--step", type=int, default=0)
    args = parser.parse_args()
    if args.mode == "watch":
        watch(args.root, args.source_sha, args.step)
    elif args.mode == "worker":
        worker(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        analyze(args.root, args.plan_sha)
