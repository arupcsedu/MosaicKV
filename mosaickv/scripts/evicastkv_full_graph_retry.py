#!/usr/bin/env python3
"""Retry only the failed, consumed-input full-routing pilot under a new SHA."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from evicastkv_campaign import SOURCE, jobs_for, load_lock, submit

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once


def derive_retry_lock(parent: Path, source_sha: str) -> dict:
    old = json.loads((parent / "protocol_lock.json").read_text())
    failure = parent / "analysis/full_graph_pilot/GATE_full_graph_pilot.json"
    gate = json.loads(failure.read_text())
    if gate["status"] != "blocked" or gate["tests_actually_executed"]["completed"] != 0:
        raise ValueError("retry must not rerun any completed full-graph requests")
    errors = []
    for blocker in gate["blockers"]:
        errors.extend(e["error"] for e in blocker.get("errors", []))
        if "error" in blocker:
            errors.append(blocker["error"])
    if not errors or any(
        "UnboundLocalError" not in e or "stable_prompt_ids" not in e for e in errors
    ):
        raise ValueError("unreviewed full-graph failure")
    passed = parent / "analysis/mechanisms/GATE_mechanisms.json"
    if json.loads(passed.read_text())["status"] != "passed":
        raise ValueError("mechanism prerequisite did not pass")
    lock = copy.deepcopy(old)
    lock.update(source_sha=source_sha, source_root=str(SOURCE))
    lock["resources"].update(
        maximum_a100s=6, array_concurrency=4, six_a100s_explicitly_authorized=True
    )
    lock["authorization"].update(
        automatic_stage_progression=False,
        fresh_confirmation_authorized=False,
        retry_only_stage="full_graph_pilot",
    )
    lock.setdefault("inherited_gates", {})[str(passed)] = sha(passed)
    lock["retry_registration"] = dict(
        parent=str(parent),
        parent_protocol_sha256=sha(parent / "protocol_lock.json"),
        failed_gate_sha256=sha(failure),
        unchanged_jobs_sha256=digest(jobs_for(old, "full_graph_pilot")),
        expected_requests=16,
        new_identities=0,
        production_change="initialize stable reservations on the freshly computed reference branch",
        no_parent_rows_overwritten=True,
    )
    jobs = jobs_for(lock, "full_graph_pilot")
    if len(jobs) != 4 or sum(len(j["indices"]) * len(j["arms"]) for j in jobs) != 16:
        raise ValueError("registered retry scope must be four workers and sixteen requests")
    if digest(jobs) != lock["retry_registration"]["unchanged_jobs_sha256"]:
        raise ValueError("retry changed inputs or arms")
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    scratch = Path("/scratch/djy8hg").resolve()
    for path in (args.parent, args.root):
        if not path.resolve().is_relative_to(scratch):
            raise ValueError("scratch-only retry")
    revision = clean_source(SOURCE)
    lock = derive_retry_lock(args.parent, revision)
    if args.root.exists():
        raise FileExistsError("immutable retry root already exists")
    write_once(args.root / "protocol_lock.json", lock)
    load_lock(args.root)
    submit(args.root, "full_graph_pilot")


if __name__ == "__main__":
    main()
