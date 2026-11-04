#!/usr/bin/env python3
"""Same-parent deferred validation controls -> strict paired latency DAG."""

import argparse
import os
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import evicastkv_direct_text_latency as direct
import evicastkv_strict_latency as strict

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, driver = direct.SOURCE, direct.driver
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_deferred_transport_latency.sbatch"
PARENT = Path("/scratch/djy8hg/runs/evicastkv/direct-text-consumed-20261103-retry1")
PROFILE = Path("/scratch/djy8hg/runs/evicastkv/direct-construction-phases-consumed-20261103-retry1")
DOC = SOURCE / "mosaickv/docs/source_validation_barrier_20261104.md"


def bookkeeping_microbenchmark(revision):
    """Same outputs; CPU-only attribution, never a request-latency gate."""
    from mosaickv.evaluation.text_source_storage import text_slots

    def reference(positions, text):
        if (
            not positions
            or len({len(h) for h in positions}) != 1
            or tuple(sorted(set(text))) != tuple(text)
            or any(
                len(set(h)) != len(h) or any(type(p) is not int or p < 0 for p in h)
                for h in positions
            )
        ):
            raise ValueError("invalid fixture")
        lookups = tuple({p: j for j, p in enumerate(h)} for h in positions)
        return tuple(tuple(index.get(t, -1) for t in text) for index in lookups)

    rows = []
    for shared in (True, False):
        head = tuple(range(632))
        positions = (head,) * 32 if shared else tuple(tuple(range(632)) for _ in range(32))
        text = tuple(range(576, 632))
        expected = reference(positions, text)
        if text_slots(positions, text) != expected:
            raise RuntimeError("CPU bookkeeping changed head-slot map")
        trials = {"reference": [], "candidate": []}
        for trial in range(7):
            order = [("reference", reference), ("candidate", text_slots)]
            if trial % 2:
                order.reverse()
            for name, fn in order:
                start = time.perf_counter()
                for _ in range(100):
                    fn(positions, text)
                trials[name].append((time.perf_counter() - start) / 100)
        rows.append(
            dict(
                shared_immutable_head_aliases=shared,
                trials_seconds_per_call=trials,
                median_seconds_per_call={k: statistics.median(v) for k, v in trials.items()},
                head_slots_identical=True,
            )
        )
    return dict(source_sha=revision, rows=rows, CPU_only=True, request_latency_evidence=False)


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    old = driver.read(PARENT / "plan.json")
    decision = driver.read(PARENT / "paired-timing/analysis/decision.json")
    attribution = driver.read(PROFILE / "analysis/decision.json")
    if (
        not decision["complete"]
        or decision["all_input_latency_gate"]
        or not attribution["complete"]
    ):
        raise ValueError("complete preserved strict negative and phase attribution required")
    bound = dict(old["bound_files"])
    for path in (
        PARENT / "plan.json",
        PARENT / "paired-timing/analysis/decision.json",
        PROFILE / "analysis/decision.json",
        DOC,
    ):
        bound[str(path)] = sha(path)
    bookkeeping = Path(
        "/scratch/djy8hg/runs/evicastkv/construction-bookkeeping-consumed-20261104-v1"
    )
    terminal = bookkeeping / "paired-timing/analysis/decision.json"
    full = bookkeeping / "paired-timing/all-baselines/analysis/decision.json"
    if full.exists():
        terminal = full
    last = driver.read(terminal)
    if not last["complete"] or last["all_input_latency_gate"]:
        raise ValueError("complete bookkeeping strict negative required before source follow-up")
    bound[str(terminal)] = sha(terminal)
    bound[str(bookkeeping / "plan.json")] = sha(bookkeeping / "plan.json")
    cpu_benchmark = root / "cpu-bookkeeping.json"
    write_once(cpu_benchmark, bookkeeping_microbenchmark(revision))
    bound[str(cpu_benchmark)] = sha(cpu_benchmark)
    for i in (0, 1):
        path = PARENT / f"workers/{i}/study.json"
        bound[str(path)] = sha(path)
    previous = Path("/scratch/djy8hg/runs/evicastkv/deferred-transport-consumed-20261103-retry1")
    failed_gate = driver.read(previous / "paired-timing/analysis/decision.json")
    if not failed_gate["complete"] or failed_gate["all_input_latency_gate"]:
        raise ValueError("complete preserved deferred-validation negative required")
    for path in (
        previous / "paired-timing/analysis/decision.json",
        previous / "analysis/decision.json",
    ):
        bound[str(path)] = sha(path)
    negative = Path("/scratch/djy8hg/runs/evicastkv/deferred-transport-consumed-20261103-v1")
    if (negative / "analysis/decision.json").exists():
        for path in [
            negative / "analysis/decision.json",
            *(negative / f"workers/{i}/result.json" for i in (0, 1)),
        ]:
            bound[str(path)] = sha(path)
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_unchanged_source_predicates_single_commit_barrier",
            direct_text_root=str(root),
            defer_transport_validation=True,
            coalesce_transport_nonzero=True,
            batch_source_validation=True,
            constructor_mask_hard_bound_bytes=4 * 1024 * 1024,
            immutable_parent_policies=True,
            automatic_quality_submission=False,
            paper_campaign_authorized=False,
            acceptance_ratio_upper_bound=1.0,
        ),
    )
    with patch.object(direct, "SCRIPT", SCRIPT):
        job = direct.submit(root, "cpu")
    write_once(root / "preparation_submission.json", dict(cpu=job))


def require_matching_packet(signature, method, record, frozen, whole_digest=None):
    if record.get("whole_policy") is not None:
        if whole_digest is None:
            raise RuntimeError("matching immediate whole-reader signature required")
        expected = whole_digest
    else:
        expected = frozen[method]["packet_sha256"]
    if signature["digest"] != expected:
        raise RuntimeError("deferred validation changed matching parent packet words")


def study(root, expected, index):
    frozen = driver.read(PARENT / f"workers/{index}/study.json")["records"]
    original = direct.execute
    primitive_controls_done = False

    def checked(*args, **kwargs):
        nonlocal primitive_controls_done
        if not primitive_controls_done:
            from mosaickv.evaluation.deferred_transport_checks import (
                gpu_controls,
                gpu_correction_timing,
                gpu_source_controls,
            )

            write_once(
                root / f"workers/{index}/validation_primitive_controls.json",
                dict(
                    cases=gpu_controls(coalesce_nonzero=True),
                    timing=gpu_correction_timing(),
                    source_controls=gpu_source_controls(),
                    no_task_evidence=True,
                ),
            )
            primitive_controls_done = True
        method, record, validate = args[5], args[6], kwargs.get("final_validator")
        whole_digest = None
        if record.get("whole_policy") is not None:
            # Different frozen reader representations intentionally have
            # different packet words. Compare whole to whole, never to text.
            captured = {}

            def capture(packets):
                captured.update(direct.packet_signature(packets))

            immediate_args = list(args)
            immediate_args[6] = dict(
                record,
                defer_transport_validation=False,
                coalesce_transport_nonzero=False,
                batch_source_validation=False,
            )
            immediate = original(*immediate_args, **dict(kwargs, final_validator=capture))
            previous = frozen[method]["whole-6291456"]["audited"]
            if any(immediate[k] != previous[k] for k in driver.STORAGE):
                raise RuntimeError("immediate whole-reader reference changed audited storage")
            whole_digest = captured["digest"]
            write_once(
                root / f"workers/{index}/immediate-whole-{method}.json",
                dict(
                    signature=captured,
                    measured=immediate,
                    validation_control_only_not_latency_evidence=True,
                ),
            )

        def verify(packets):
            require_matching_packet(
                direct.packet_signature(packets), method, record, frozen, whole_digest
            )
            if validate is not None:
                validate(packets)

        measured = original(*args, **dict(kwargs, final_validator=verify))
        validation = measured["transport_validation"]
        if not validation["enabled"] or not validation["all_checks_before_decode"]:
            raise RuntimeError("validation transaction did not commit before decode")
        if method == "mosaickv_exact" and (
            validation["counts"] != dict(source=32, finite=64, protected=64, close=64)
            or validation["commit_flushes"] != 1
            or validation["validation_flag_peak_tensor_bytes"] > 1025
        ):
            raise RuntimeError("complete bounded source validation transaction required")
        if (
            not validation["coalesced_correction_nonzero"]
            or validation["correction_mask_peak_tensor_bytes"] > 4 * 1024 * 1024
            or validation["correction_mask_bound_fallbacks"] != 0
            or validation["correction_mask_persistent_bytes"] != 0
            or not validation["batched_source_validation"]
        ):
            raise RuntimeError("complete explicitly bounded coalesced construction required")
        return measured

    with patch.object(direct, "execute", checked):
        direct.study(root, expected, index)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "cpu", "study", "study_analysis", "timing", "timing_analysis")
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-sha")
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.root, args.source_sha)
    elif args.action == "timing_analysis":
        strict.analyze(args.root, args.plan_sha, SOURCE, clean_source(SOURCE), "timing", "direct")
    else:
        with patch.object(direct, "SCRIPT", SCRIPT):
            if args.action == "study":
                study(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
            elif args.action == "timing":
                direct.timing(args.root, args.plan_sha, int(os.environ["SLURM_ARRAY_TASK_ID"]))
            else:
                getattr(direct, args.action)(args.root, args.plan_sha)


if __name__ == "__main__":
    main()
