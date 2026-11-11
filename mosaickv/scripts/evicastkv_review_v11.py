#!/usr/bin/env python3
"""Finite causal -> construction -> ownership diagnostics, no automatic claims."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import evicastkv_review_v8_diagnostics as shared

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path("/scratch/djy8hg/workdir/fox-review-v10/run-20261110-3")
PARENT_SHA = "2efa84a4a36624bb685baac68612484e377c6390"
LANES = {
    "causal": ("evicastkv_source_causal_control", "GATE_source_causal_control.json"),
    "bytes": ("evicastkv_construct_shared_byte", "GATE_constructed_shared_bytes.json"),
    "ownership": ("evicastkv_source_ownership_closure", "GATE_source_ownership_closure.json"),
}
FRESHNESS = "evicastkv_freshness_recipe_recovery"
FRONTIER = "evicastkv_freshness_frontier_fork"


def module(lane):
    return importlib.import_module(LANES[lane][0])


def load(root):
    lock = shared.read(root / "registration.json")
    clean_source(SOURCE, lock["source_sha"])
    if not lock["execute"] or lock["fresh_confirmation_authorized"]:
        raise ValueError("registered diagnostic execution only")
    for name, expected in lock["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"immutable V11 binding changed: {name}")
    return lock


def register(root, *, execute=False):
    root = Path(root).resolve()
    workspace = Path("/scratch/djy8hg/workdir").resolve()
    if not root.is_relative_to(workspace) or root == workspace or root.exists():
        raise ValueError("new immutable scratch-workspace diagnostic output required")
    revision = clean_source(SOURCE)
    parent = shared.read(PARENT / "registration.json")
    if parent["source_sha"] != PARENT_SHA:
        raise ValueError("wrong parent diagnostic source")
    final = shared.read(PARENT / "FINAL_report.json")
    if final["errors"] or any("sha256" not in item for item in final["artifacts"].values()):
        raise ValueError("complete V10 evidence required")
    paths = [PARENT / "registration.json", PARENT / "FINAL_report.json"]
    paths += [SOURCE / f"mosaickv/scripts/{name}.py" for name, _ in LANES.values()]
    paths.append(SOURCE / f"mosaickv/scripts/{FRESHNESS}.py")
    paths.append(SOURCE / f"mosaickv/scripts/{FRONTIER}.py")
    paths += [
        SOURCE / "mosaickv/docs/fox_baseline_readiness_20261110.json",
        SOURCE / "mosaickv/docs/baselines/lookm_import_cleanup_ecf0f51.patch",
    ]
    write_once(
        root / "registration.json",
        dict(
            source_sha=revision,
            source_root=str(SOURCE),
            execute=execute,
            bindings={str(path): sha(path) for path in paths},
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            sequential_diagnostic_workers=1,
            consumed_inputs_only=True,
            fresh_confirmation_authorized=False,
            automatic_paper_campaign=False,
            preserve_negative_controls=True,
            unchanged_guard=True,
        ),
    )
    if not execute:
        return dict(status="dry_run", jobs=0)
    try:
        cpu = shared.submit(
            root, "cpu", "tests", tests=True, launcher="evicastkv_review_v11", job_prefix="fox11"
        )
        prep = shared.submit(
            root, "prepare", "prepare", [cpu], launcher="evicastkv_review_v11", job_prefix="fox11"
        )
    except shared.SubmissionError as error:
        return shared.submission_block(root, "registration", error)
    return dict(cpu=cpu, preparation=prep)


def prepare(root):
    lock = load(root)
    try:
        write_once(root / "GATE_cpu_suite.json", shared.cpu_gate(root, lock["source_sha"]))
    except Exception as error:
        shared.failure(root, "cpu", error)
        return finish(root)
    try:
        importlib.import_module(FRESHNESS).run(root / "freshness", cpu_gate_root=root)
    except Exception as error:
        shared.failure(root, "freshness-recovery", error)
    try:
        frontier = importlib.import_module(FRONTIER)
        frontier.prepare(
            root / "freshness-frontier", cpu_gate_root=root, coverage_root=root / "freshness"
        )
        launched = frontier.launch(root / "freshness-frontier", execute=True)
        write_once(root / "freshness-frontier-submission.json", launched)
        if launched.get("status") != "submitted_one_bounded_CPU_segment":
            write_once(root / "errors/freshness-frontier-submission.json", launched)
    except Exception as error:
        shared.failure(root, "freshness-frontier", error)
    prepared = []
    for lane in LANES:
        try:
            result = module(lane).prepare(root / lane)
            write_once(root / f"prepared-{lane}.json", result)
            prepared.append(lane)
        except Exception as error:
            shared.failure(root, f"prepare-{lane}", error)
    return submit_next(root, prepared)


def submit_next(root, pending):
    if not pending:
        return finish_or_defer(root)
    load(root)
    lane = pending[0]
    try:
        snapshot = shared.gpu_snapshot()
        dependencies = sorted(
            {row["job"].split("_")[0] for row in snapshot["active"] + snapshot["pending"]}
        )
        write_once(
            root / f"admission-{lane}.json",
            dict(
                snapshot=snapshot,
                remaining=pending,
                aggregate_ceiling=6,
                worker="one_A100_12_CPUs_30_minutes_maximum",
            ),
        )
        job = shared.submit(
            root,
            lane,
            f"{lane}-worker",
            dependencies,
            gpu=True,
            launcher="evicastkv_review_v11",
            job_prefix="fox11",
        )
        analyzer = shared.submit(
            root,
            f"after-{lane}",
            f"after-{lane}",
            [job],
            launcher="evicastkv_review_v11",
            job_prefix="fox11",
        )
        return dict(lane=lane, worker=job, analyzer=analyzer)
    except shared.SubmissionError as error:
        return shared.submission_block(root, f"submit-{lane}", error)
    except Exception as error:
        shared.failure(root, f"admission-{lane}", error)
        return finish(root)


def finish_or_defer(root):
    gate = root / "freshness-frontier/GATE_freshness_frontier_fork.json"
    submitted = root / "freshness-frontier-submission.json"
    if submitted.exists() and not gate.exists():
        jobs = shared.read(submitted)
        producer = jobs.get("analysis") or jobs.get("workers")
        if producer is not None:
            final = shared.submit(
                root,
                "final",
                "final",
                [producer],
                launcher="evicastkv_review_v11",
                job_prefix="fox11",
            )
            return dict(final_analysis=final, waiting_for_frontier_analysis=producer)
    return finish(root)


def finish(root):
    lock = load(root)
    artifacts = {}
    for lane, (_, gate) in LANES.items():
        path = root / lane / gate
        artifacts[lane] = (
            dict(path=str(path), sha256=sha(path), value=shared.read(path))
            if path.exists()
            else dict(status="missing_or_blocked")
        )
    recovery = root / "freshness/GATE_freshness_recipe_recovery.json"
    artifacts["freshness_recovery"] = (
        dict(path=str(recovery), sha256=sha(recovery), value=shared.read(recovery))
        if recovery.exists()
        else dict(status="missing_or_blocked")
    )
    frontier = root / "freshness-frontier/GATE_freshness_frontier_fork.json"
    artifacts["freshness_frontier"] = (
        dict(path=str(frontier), sha256=sha(frontier), value=shared.read(frontier))
        if frontier.exists()
        else dict(status="missing_or_blocked")
    )
    errors = {p.name: shared.read(p) for p in (root / "errors").glob("*.json")}
    result = dict(
        status="completed_bounded_source_closure_not_confirmation",
        source_sha=lock["source_sha"],
        artifacts=artifacts,
        errors=errors,
        no_missing_result_is_a_pass=True,
        superiority_established=False,
        fresh_confirmation_authorized=False,
        paper_campaign_authorized=False,
    )
    write_once(root / "FINAL_report.json", result)
    return result


def run(root, mode):
    load(root)
    if mode == "final":
        return finish(root)
    if mode == "prepare":
        return prepare(root)
    lane = mode.removesuffix("-worker").removeprefix("after-")
    if lane not in LANES:
        raise ValueError("unregistered diagnostic stage")
    if mode.endswith("-worker"):
        if shared.gpu_snapshot()["active_gpus"] > 6:
            raise ValueError("aggregate six-GPU ceiling exceeded")
        return module(lane).worker(root / lane, index=0)
    if not mode.startswith("after-"):
        raise ValueError("unregistered diagnostic action")
    try:
        module(lane).analyze(root / lane)
    except Exception as error:
        shared.failure(root, f"analysis-{lane}", error)
    order = list(LANES)
    later = order[order.index(lane) + 1 :]
    # Other lanes are independent accounting controls, not a promotion based on
    # this lane's quality. Negative results cannot authorize a fresh campaign.
    pending = [key for key in later if (root / f"prepared-{key}.json").exists()]
    return submit_next(root, pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    value = (
        register(args.root, execute=args.execute)
        if args.mode == "register"
        else run(args.root.resolve(), args.mode)
    )
    print(json.dumps(value, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
