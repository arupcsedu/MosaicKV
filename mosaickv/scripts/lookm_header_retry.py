#!/usr/bin/env python3
"""Retry LOOK-M only, increasing trace capacity without changing the algorithm."""

import json
from pathlib import Path

import common_baseline_headers as baseline
import exact_scale_replay_study as base

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

PARENT = Path("/scratch/djy8hg/runs/evicastkv/common-baseline-headers-20261108-v1")
base.common.WORKER_WRAPPER = base.SOURCE / "mosaickv/slurm/lookm_header_retry.sbatch"


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable retry root required")
    p = base.common.read(PARENT / "plan.json")
    decision = base.common.read(PARENT / "analysis/decision.json")
    if len(decision["pairs"]) != 6 or any(
        not r["all_128_logits_bitwise"] for r in decision["pairs"]
    ):
        raise ValueError("preserve six valid non-LOOK-M pairs before selective retry")
    for i in (0, 1):
        path = PARENT / f"model/{i}/raw.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        failed = [r for r in rows if r["arm"].startswith("lookm_reimpl:")]
        if len(failed) != 2 or any(
            "allocator history overwritten" not in r.get("error", "") for r in failed
        ):
            raise ValueError("retry scope requires the recorded trace-capacity failure only")
        p["bound_files"][str(path)] = sha(path)
    for name in ("plan.json", "analysis/decision.json"):
        path = PARENT / name
        p["bound_files"][str(path)] = sha(path)
    p.update(
        source_sha=clean_source(base.SOURCE),
        arms=baseline.arms(),
        execution_methods={arm: "lookm_reimpl" for arm in baseline.arms()},
        maximum_allocator_events=8000000,
        study="LOOKM_only_complete_allocator_trace_retry_v1",
        retry_changes="500000 -> 8000000 trace entries; no numerical/accounting check disabled",
        successful_non_LOOKM_rows_NOT_rerun=True,
    )
    root.mkdir()
    (root / "logs").mkdir()
    write_once(root / "plan.json", p)
    cpu = base.common.submit(root, "cpu")
    write_once(
        root / "submission.json", dict(cpu=cpu, follow=base.common.submit(root, "after_cpu", cpu))
    )
    print(json.dumps(base.common.read(root / "submission.json")), flush=True)


if __name__ == "__main__":
    baseline.METHODS = ("lookm_reimpl",)
    base.prepare, base.model, base.analyze = prepare, baseline.model, baseline.analyze
    base.main()
