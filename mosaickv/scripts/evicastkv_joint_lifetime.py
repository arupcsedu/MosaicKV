#!/usr/bin/env python3
"""Same consumed inputs, same total history, profile-weighted layer placement."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import evicastkv_lifetime_history as common
import evicastkv_lifetime_v2 as launch
import evicastkv_profile_lifetime as profile

from mosaickv.config import RunConfig, canonical_config, config_sha256
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]


def prepare(root, parent, expected):
    source_sha = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("immutable root required")
    if sha(parent / "plan.json") != expected:
        raise ValueError("parent plan changed")
    gate_path = parent / "analysis/GATE.json"
    gate = common.read(gate_path)
    if gate["status"] != "complete_consumed_diagnostic" or gate["failures"]:
        raise ValueError("fix parent execution before a new design")
    if gate["point_eligible_policies"]:
        raise ValueError("do not revise a candidate that passed the registered point gate")
    old = common.read(parent / "plan.json")
    bound = dict(old["bound_files"])
    for path, digest in bound.items():
        if sha(Path(path)) != digest:
            raise ValueError("parent bound evidence changed")
    bound[str(parent / "plan.json")] = expected
    bound[str(gate_path)] = sha(gate_path)
    jobs = []
    for job in old["jobs"]:
        if job.get("measurement_role") == "remeasured_native_control":
            continue
        native_job = next(
            j
            for j in old["jobs"]
            if j["task"] == job["task"] and j.get("measurement_role") == "remeasured_native_control"
        )
        native = parent / f"workers/{native_job['index']}/rows/prefixkv_reimpl-native-control.json"
        bound[str(native)] = sha(native)
        for path, value in (
            common.read(native)["traces"] | common.read(native)["memory_audit_traces"]
        ).items():
            if sha(Path(path)) != value:
                raise ValueError("native evidence changed")
            bound[path] = value
        arm = job["arms"][0]
        cfg = RunConfig.from_mapping(arm["config"])
        if cfg.bounded_decode.generated_capacity != 128:
            raise ValueError("unexpected parent mean history capacity")
        cfg = replace(
            cfg,
            bounded_decode=replace(
                cfg.bounded_decode,
                incremental_history=replace(
                    cfg.bounded_decode.incremental_history, history_allocation="profile_weighted"
                ),
            ),
        )
        identity = next(
            j["arms"][0]["config"]
            for j in old["jobs"]
            if j["task"] == job["task"]
            and j.get("mandatory_policy") == job["mandatory_policy"]
            and j["arms"][0]["config"]["method"] == "mosaickv_exact"
        )
        jobs.append(
            dict(
                job,
                index=len(jobs),
                native_control_row=str(native),
                retention_one_config=identity,
                arms=[
                    dict(
                        label=arm["label"].removesuffix("-v3") + "-joint-v4",
                        config=canonical_config(cfg),
                        config_sha=config_sha256(cfg),
                    )
                ],
            )
        )
    assert len(jobs) == 8
    write_once(
        root / "plan.json",
        dict(
            old,
            jobs=jobs,
            source_sha=source_sha,
            bound_files=bound,
            protocol="joint_prompt_history_profile_v4_consumed_diagnostic",
            history_budget_mean_per_layer=128,
            minimum_history_positions_per_layer=69,
            native_controls_reused_from=str(parent),
            native_control_producer_sha=old["source_sha"],
            native_latency_not_paired=True,
            source_change_for_native="new_history_mode_disabled_in_native_path_config_hash_unchanged",
            fresh_evaluation_authorized=False,
            paper_campaign_authorized=False,
        ),
    )
    return sha(root / "plan.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "submit", "worker", "analyze"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent", type=Path)
    p.add_argument("--plan-sha")
    p.add_argument("--index", type=int)
    a = p.parse_args()
    if a.mode == "prepare":
        print(prepare(a.root, a.parent, a.plan_sha))
    elif a.mode == "submit":
        launch.submit(
            a.root,
            a.plan_sha,
            script_name="evicastkv_joint_lifetime.sbatch",
            worker_minutes=30,
            memory_gb=16,
        )
    elif a.mode == "worker":
        common.worker(a.root, a.plan_sha, a.index)
    else:
        profile.analyze(a.root, a.plan_sha)


if __name__ == "__main__":
    main()
