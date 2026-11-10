#!/usr/bin/env python3
"""Finite consumed-input CPU graph diagnostic; no optimization or new cohort.

Prepare only after existing GPU study chains end. This entry point deliberately
does not submit anything; a scheduler callback can invoke prepare, worker, and
analyze with afterany dependencies and aggregate six-A100 admission controls.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from evicastkv_campaign import SOURCE, load_lock
from evicastkv_development import runtime_files

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

STAGE = "graph_cpu_profile"
ARM = ["mosaickv_exact", 40, "complete"]


def registered_plan(parent, *, source=SOURCE):
    """Hash-bound whole consumed ledger; select by registered order, not labels."""
    path = parent / "protocol_lock.json"
    lock = json.loads(path.read_text())
    prior = Path(lock["source_root"])
    clean_source(prior, lock["source_sha"])
    if runtime_files(source) != runtime_files(prior):
        raise ValueError("production runtime changed; graph diagnostic cannot inherit controls")
    controls = (
        "mosaickv/scripts/evicastkv_worker.py",
        "mosaickv/scripts/evicastkv_campaign.py",
        "mosaickv/src/mosaickv/evaluation/fox_numerics.py",
        "mosaickv/src/mosaickv/evaluation/fox_campaign.py",
        "mosaickv/configs/experiments/evicastkv_fox/coco/mosaickv_exact.json",
    )
    if any(sha(source / name) != sha(prior / name) for name in controls):
        raise ValueError("shared model worker, numerical control or policy changed")
    bindings = {str(path): sha(path)}
    for stage in ("correctness", "scoring", "profiling", "measurement"):
        gate = parent / "analysis" / stage / f"GATE_{stage}.json"
        data = json.loads(gate.read_text())
        if (
            data.get("status") != "passed"
            or data.get("protocol_sha256") != sha(path)
            or data.get("source_sha") != lock["source_sha"]
            or data.get("blockers")
        ):
            raise ValueError(f"required parent control not passed: {stage}")
        bindings[str(gate)] = sha(gate)
    cohort = lock["cohorts"]["coco"]["profiler"]
    ledger_path = Path(cohort["path"])
    if sha(ledger_path) != cohort["sha256"]:
        raise ValueError("consumed profiler ledger changed")
    samples = json.loads(ledger_path.read_text())["samples"]
    ids = [sample["sample_id"] for sample in samples]
    if len(ids) < 2 or len(set(ids)) != len(ids) or ids != cohort["sample_ids"]:
        raise ValueError("unique consumed profiler identities required")
    if lock.get("arithmetic", {}).get("attention_layout_policy", "native") != "native":
        raise ValueError("this graph-only diagnostic requires the native COCO transport")
    return copy.deepcopy(lock), bindings, ids[:2]


def prepare(root, parent):
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("graph diagnostic output must be new")
    lock, bindings, ids = registered_plan(parent)
    lock.update(
        name="evicastkv_consumed_graph_cpu_profile_v1",
        source_root=str(SOURCE),
        source_sha=revision,
    )
    lock["authorization"].update(automatic_stage_progression=False, confirmation_authorized=False)
    lock["resources"].update(
        maximum_a100s=6,
        array_concurrency=1,
        six_a100s_explicitly_authorized=True,
        cpus_per_worker=12,
        wall_minutes=120,
    )
    lock.setdefault("inherited_gates", {}).update(
        {path: value for path, value in bindings.items() if "GATE_" in path}
    )
    lock["historical_bindings"].update(bindings)
    lock["graph_cpu_profile"] = dict(
        sample_ids=ids,
        fresh_inputs=False,
        task_scores_used=False,
        maximum_graph_calls=2,
        runtime_changes=False,
        model_worker=sha(SOURCE / "mosaickv/scripts/evicastkv_worker.py"),
        profile_latency_eligible=False,
        reference_root=str(parent),
        admission="only after registered mechanisms and full-graph GPU chains finish",
    )
    write_once(root / "protocol_lock.json", lock)
    spec = dict(
        protocol_sha256=sha(root / "protocol_lock.json"),
        jobs=[dict(workload="coco", subset="profiler", indices=[0, 1], stage=STAGE, arms=[ARM])],
    )
    write_once(root / "jobs" / f"{STAGE}.json", spec)
    load_lock(root)
    return dict(root=str(root), spec_sha256=sha(root / "jobs" / f"{STAGE}.json"))


def worker(root, index, spec_sha):
    from evicastkv_worker import run_worker

    from mosaickv.evaluation.fox_graph_profile import GraphCPUProfiler, profile_bindings

    lock = load_lock(root)
    if index != 0:
        raise ValueError("exactly one two-input graph diagnostic worker is registered")
    profile = GraphCPUProfiler(maximum_graph_calls=2)
    status, error = "failed", None
    try:
        with profile.install():
            run_worker(root, STAGE, index, spec_sha)
        status = "passed"
    except BaseException as exception:
        error = f"{type(exception).__name__}: {exception}"
        raise
    finally:
        write_once(
            root / "diagnostics" / "graph_cpu_profile.json",
            dict(
                profile.report(),
                worker_status=status,
                worker_error=error,
                source_sha=lock["source_sha"],
                source_bindings=profile_bindings(SOURCE),
                sample_ids=lock["graph_cpu_profile"]["sample_ids"],
                # The generic worker's profiler flag concerns torch.profiler.
                # ALL requests here are cProfile-contaminated regardless of it.
                all_raw_rows_cprofile_contaminated=True,
            ),
        )


def analyze(root):
    from evicastkv_analyze import collect

    lock = load_lock(root)
    _spec, _workers, rows, errors = collect(root, STAGE)
    parent = Path(lock["graph_cpu_profile"]["reference_root"])
    _, _, reference_rows, reference_errors = collect(parent, "profiling")
    if reference_errors:
        errors.append(dict(error="original profiler controls incomplete"))
    reference = {
        row["sample_id"]: row
        for row in reference_rows
        if row["method"] == "mosaickv_exact"
        and row["budget_bytes"] == 40 * 2**20
        and row["variant"] == "complete"
    }
    path = root / "diagnostics" / "graph_cpu_profile.json"
    profile = json.loads(path.read_text()) if path.exists() else None
    if not profile or profile["actual_graph_calls"] != 2 or profile["worker_status"] != "passed":
        errors.append(dict(error="two completed graph calls and diagnostic packet required"))
    elif any(call["status"] != "passed" or not call["functions"] for call in profile["calls"]):
        errors.append(dict(error="graph subphase cProfile incomplete"))
    if len(rows) != 2 or sorted(r["sample_id"] for r in rows) != sorted(
        lock["graph_cpu_profile"]["sample_ids"]
    ):
        errors.append(dict(error="consumed identity completeness failed"))
    for row in rows:
        if (
            row["status"] != "completed"
            or row["method"] != "mosaickv_exact"
            or row["budget_bytes"] != 40 * 2**20
            or row.get("initial_kv_bytes") != 40 * 2**20
            or row.get("effective_method") != "mosaickv_exact"
        ):
            errors.append(dict(error="method, fallback or exact byte gate failed"))
        prior = reference.get(row["sample_id"])
        fields = (
            "input_sha",
            "processor_sha",
            "config_sha",
            "token_ids",
            "initial_kv_bytes",
            "final_kv_bytes",
            "peak_kv_bytes",
            "emitted_tokens",
            "forward_steps",
        )
        if prior is None or any(row.get(field) != prior.get(field) for field in fields):
            errors.append(dict(error="tokens, physical bytes or input controls changed"))
    functions = []
    for call in (profile or {}).get("calls", []):
        functions.extend(dict(call_index=call["call_index"], **item) for item in call["functions"])
    output = root / "analysis" / STAGE
    write_once(output / "functions.json", functions)
    write_once(output / "failures.json", errors)
    gate = dict(
        status="blocked" if errors else "passed",
        source_sha=lock["source_sha"],
        protocol_sha256=sha(root / "protocol_lock.json"),
        expected_rows=2,
        actual_rows=len(rows),
        graph_profile_sha256=sha(path) if path.exists() else None,
        blockers=errors,
        headline_latency_eligible=False,
        quality_claim_allowed=False,
        fresh_evaluation_authorized=False,
        next_action="inspect measured exclusive graph CPU functions; no automatic optimization",
    )
    write_once(output / "GATE_graph_cpu_profile.json", gate)
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--spec-sha")
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.parent is None:
            parser.error("prepare requires --parent")
        result = prepare(args.root, args.parent)
    elif args.mode == "worker":
        if args.spec_sha is None:
            parser.error("worker requires --spec-sha")
        result = worker(args.root, args.index, args.spec_sha)
    else:
        result = analyze(args.root)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
