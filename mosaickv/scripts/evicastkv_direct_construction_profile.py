#!/usr/bin/env python3
"""Attribute the remaining constructor gap on the same two consumed inputs.

Instrumentation only: no reader selection, altered math, quality cohort, or
latency promotion. Preserve the strict negative paired-timing result.
"""

import argparse
import json
import os
import random
import statistics
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import evicastkv_direct_text_latency as direct

from mosaickv.evaluation.construction_phase_timing import PhaseTimer, instrument, phase_totals
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

driver, SOURCE = direct.driver, direct.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/direct-text-consumed-20261103-retry1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_direct_construction_profile.sbatch"
DOC = SOURCE / "mosaickv/docs/direct_construction_profile_20261103.md"


@contextmanager
def observe(timer):
    from mosaickv.evaluation import direct_text_construction as construction

    with ExitStack() as stack:
        stack.enter_context(instrument(timer))
        for owner, name, label in (
            (construction, "build", "direct.build"),
            (construction, "text_slots", "direct.text_slots"),
            (construction, "strip_text_overlays", "direct.strip_overlays"),
            (construction, "encode", "direct.text_codec"),
            (construction.Construction, "optimize", "direct.layer_transaction"),
            (construction.Construction, "finish", "direct.storage_validation"),
            (construction.HeadLocalFormat, "materialize", "head_local.materialize"),
        ):
            stack.enter_context(patch.object(owner, name, timer.wrap(label, getattr(owner, name))))
        yield


def submit(root, action, **kwargs):
    with patch.object(direct, "SCRIPT", SCRIPT):
        return direct.submit(root, action, **kwargs)


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable profile root and clean source required")
    decision = driver.read(PARENT / "paired-timing/analysis/decision.json")
    if not decision["complete"] or decision["all_input_latency_gate"]:
        raise ValueError("preserved complete strict negative timing result required")
    old = driver.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    for path in (PARENT / "plan.json", PARENT / "paired-timing/analysis/decision.json", DOC):
        bound[str(path)] = sha(path)
    for index in (0, 1):
        result = driver.read(PARENT / f"workers/{index}/result.json")
        if result["status"] != "completed" or result["failures"]:
            raise ValueError("both exact-packet control producers required")
        for name, checksum in result["outputs"].items():
            path = PARENT / f"workers/{index}" / name
            if sha(path) != checksum:
                raise ValueError("parent control artifact changed")
            bound[str(path)] = checksum
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_direct_constructor_subphase_attribution",
            profile_warmups=1,
            profile_repetitions=2,
            instrumentation_not_performance_evidence=True,
            automatic_optimization_authorized=False,
            automatic_quality_submission=False,
            paper_campaign_authorized=False,
        ),
    )
    write_once(root / "preparation_submission.json", dict(cpu=submit(root, "cpu")))


def cpu(root, expected):
    driver.pair.common.previous.cpu(root, expected)
    driver.cpu_proof(root, driver.checked(root, expected))
    gpu = submit(root, "study", gpu=True)
    analysis = submit(root, "study_analysis", dependency=gpu)
    write_once(root / "submission.json", dict(gpu_array=gpu, analysis=analysis))


def study(root, expected, index):
    import torch

    out = root / f"workers/{index}"
    out.mkdir(parents=True, exist_ok=False)
    result = dict(
        status="failed", failures=[], source_sha=clean_source(SOURCE), plan_sha256=expected
    )
    try:
        plan, adapter, refs, messages, profile, _ = driver.setup(
            root,
            expected,
            index,
            manifest_overrides=dict(
                variant="consumed_direct_constructor_subphase_instrumentation",
                methods=list(direct.common.previous.METHODS),
                instrumentation_not_performance_evidence=True,
                actual_timing_protocol=dict(
                    profile_warmups=1,
                    profile_repetitions=2,
                    output_predictions=128,
                    no_latency_acceptance_from_instrumented_requests=True,
                ),
            ),
        )
        controls = driver.read(PARENT / f"workers/{index}/study.json")["records"]
        rng, rows = random.Random(plan["seed"]), []
        for trial in range(-1, 2):
            methods = list(direct.common.previous.METHODS)
            rng.shuffle(methods)
            for method in methods:
                record = controls[method]["direct"]
                captured = {}

                def verify(packets, method=method, captured=captured):
                    signature = direct.packet_signature(packets)
                    if signature["digest"] != controls[method]["packet_sha256"]:
                        raise RuntimeError("instrumentation changed frozen packet words")
                    captured["packet_sha256"] = signature["digest"]

                timer = PhaseTimer(event_factory=lambda: torch.cuda.Event(enable_timing=True))
                with observe(timer):
                    measured = direct.execute(
                        adapter,
                        refs,
                        messages,
                        profile,
                        plan,
                        method,
                        record,
                        final_validator=verify,
                    )
                torch.cuda.synchronize()
                if any(measured[k] != record["audited"][k] for k in driver.STORAGE[:2]):
                    raise RuntimeError("instrumentation changed initial/final cache storage")
                trace = timer.resolve()
                row = dict(
                    method=method,
                    trial=trial,
                    warmup=trial < 0,
                    trace=trace,
                    phases=phase_totals(trace),
                    **captured,
                    logits_bitwise=measured["logits_bitwise"],
                    initial_encoded_bytes=measured["initial_encoded_bytes"],
                    final_encoded_bytes=measured["final_encoded_bytes"],
                    direct_text_construction=measured["direct_text_construction"],
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                    instrumentation_not_performance_evidence=True,
                )
                rows.append(row)
                with (out / "raw.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
        write_once(out / "profile.json", dict(rows=rows, independent_quality_N=0))
        result["status"] = "completed"
    except Exception as error:
        result["failures"].append(f"{type(error).__name__}: {error}")
        result["traceback"] = traceback.format_exc()
    driver.finish_worker(out, result)


def summarize(rows):
    keys = [(r["method"], r["trial"]) for r in rows]
    methods = direct.common.previous.METHODS
    if len(set(keys)) != len(keys) or set(keys) != {(m, t) for m in methods for t in (-1, 0, 1)}:
        raise ValueError("complete unique one-warmup/two-profile requests required")
    summaries = []
    for method in methods:
        selected = [r for r in rows if r["method"] == method and not r["warmup"]]
        phases = set().union(*(r["phases"] for r in selected))
        required = {"direct.layer_transaction", "direct.build", "direct.storage_validation"}
        if not required <= phases or any(r["logits_bitwise"] is not True for r in selected):
            raise ValueError("complete constructor phases and parent-word controls required")
        aggregate = {
            p: {
                field: statistics.mean(r["phases"].get(p, {}).get(field, 0.0) for r in selected)
                for field in (
                    "calls",
                    "inclusive_wall_seconds",
                    "exclusive_wall_seconds",
                    "stream_elapsed_ms",
                )
            }
            for p in phases
        }
        summaries.append(
            dict(
                method=method,
                phases=aggregate,
                largest_exclusive_phases=sorted(
                    aggregate, key=lambda p: -aggregate[p]["exclusive_wall_seconds"]
                )[:12],
            )
        )
    return summaries


def study_analysis(root, expected):
    plan = driver.checked(root, expected)
    records, failures = driver.load_producers(root, plan, "profile.json")
    summaries = []
    for index, record in enumerate(records):
        try:
            summaries.append(dict(input_index=index, methods=summarize(record["rows"])))
        except Exception as error:
            failures.append(dict(input_index=index, error=str(error)))
    direct.common.previous.closeout(
        root,
        not failures,
        failures,
        summaries=summaries,
        instrumentation_not_performance_evidence=True,
        cuda_stream_intervals_include_host_gaps=True,
        no_reader_promoted=True,
        all_input_latency_gate=False,
        automatic_optimization_submitted=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "cpu", "study", "study_analysis"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "study":
        study(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
    else:
        globals()[args.action](args.root, args.plan_sha)


if __name__ == "__main__":
    main()
