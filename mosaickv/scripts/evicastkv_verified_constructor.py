#!/usr/bin/env python3
"""Incremental exact constructor ablations on the two consumed inputs."""

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import evicastkv_history_graph_reader as base
import evicastkv_native_history_reader as history

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

control, SOURCE, parent, driver = base.control, base.SOURCE, base.parent, base.driver
WAIT = base.WAIT
PREVIOUS = Path("/scratch/djy8hg/runs/evicastkv/history-graph-screen-consumed-20261105-v1")
DEBUG_ROOT = Path(
    "/scratch/djy8hg/runs/evicastkv/verified-constructor-correctness-consumed-20261106-v1"
)
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_verified_constructor.sbatch"
DOC = SOURCE / "mosaickv/docs/verified_constructor.md"
ARMS = (
    ("mosaickv_exact", "joint-history-graph"),
    ("mosaickv_exact", "joint-same-buffer"),
    ("mosaickv_exact", "joint-handoff"),
    ("prefixkv_reimpl", "graph-history-graph"),
    ("prefixkv_reimpl", "graph-original-history-graph"),
)
OLD_CONFIGURATIONS, OLD_EXECUTE = base.configurations, base.execute
ORIGINAL_SUMMARY = control.summary


def overrides(method, arm):
    mode = {"joint-same-buffer": "same_buffer", "joint-handoff": "handoff"}.get(arm, "none")
    return dict(
        verified_constructor=mode if method == "mosaickv_exact" else "handoff",
        native_history_reader=True,
        native_history_graph=True,
        overlap_native_fit=mode != "handoff" or method != "mosaickv_exact",
    )


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable output and clean source required")
    prior = driver.read(PREVIOUS / "analysis/decision.json")
    if not prior["complete"] or prior["failures"]:
        raise ValueError("complete prior common-reader screen required")
    old = driver.read(PREVIOUS / "plan.json")
    paths = [DOC, PREVIOUS / "plan.json", PREVIOUS / "analysis/decision.json"]
    if base.PREPARE_SCREEN:
        base.require_debug(driver.read(DEBUG_ROOT / "analysis/decision.json"))
        paths += [DEBUG_ROOT / "plan.json", DEBUG_ROOT / "analysis/decision.json"]
    bound = dict(old["bound_files"])
    bound.update({str(p): sha(p) for p in paths})
    serial = str(driver.read(PREVIOUS / "profile_submission.json")["analysis"])
    if not serial.isdecimal():
        raise ValueError("finite preceding analysis dependency required")
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_verified_constructor_ablation",
            arms=ARMS,
            resolved_arm_overrides=[
                dict(method=m, arm=a, overrides=overrides(m, a)) for m, a in ARMS
            ],
            common_native_history=True,
            common_history_graph=True,
            timing_boundary="original_complete_request_plus_verified_constructor_scope_for_every_arm",
            correctness_only=not base.PREPARE_SCREEN,
            warmups=5 if base.PREPARE_SCREEN else 0,
            repetitions=3 if base.PREPARE_SCREEN else 1,
            same_buffer_then_handoff_separate_ablations=True,
            original_transport_predicates_and_frozen_packet_digest_required=True,
            complete_outer_packet_KV_stride_word_audit_required=True,
            no_full_reference_retention_across_components=True,
            original_per_layer_and_request_correction_caps=True,
            independent_quality_N=0,
            no_automatic_fresh_or_quality_submission=True,
        ),
    )
    cpu = parent.submit(root, "cpu")
    release = parent.submit(root, "release", f"{cpu}:{serial}")
    write_once(
        root / "cpu_submission.json", dict(cpu=cpu, release=release, serialized_after=serial)
    )


def configurations(records):
    original = OLD_CONFIGURATIONS(records)
    values = {}
    for method, arm in ARMS:
        key = "joint-history-graph" if method == "mosaickv_exact" else arm
        values[method, arm] = dict(original[method, key], **overrides(method, arm))
    return values


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    mode = record["verified_constructor"]
    effective = "joint-history-graph" if method == "mosaickv_exact" else arm
    with patch.dict(
        history.POLICIES,
        overlap=record["overlap_native_fit"]
        if method == "mosaickv_exact"
        else history.POLICIES["overlap"],
    ):
        value = OLD_EXECUTE(
            adapter, refs, messages, profile, plan, method, effective, record, **kwargs
        )
    info = value["verified_constructor"]
    active = method == "mosaickv_exact" and mode != "none"
    if info["requested"] != mode or info["eligible"] is not active:
        raise RuntimeError("declared verified constructor did not engage truthfully")
    if active:
        report = info["report"]
        if report["components"] != 64 or not report["retired"]:
            raise RuntimeError("all original components and private retirement required")
        if mode == "handoff" and (
            report["native_word_checks"] != 64
            or report["exact_parent_geometry_component_replays"] != 64
            or report["final_outer_stride_checks"] != (192 if kwargs.get("audit", False) else 0)
            or report["native_workspace_bytes"] != 3 * 2**20 + 8192
            or report["full_reference_tensors_retained_across_components"]
            or not report["converted"]
        ):
            raise RuntimeError(
                "complete bounded native handoff and original outer-word checks required"
            )
    return value


def summary(rows, audited):
    old = audited["mosaickv_exact", "joint-history-graph"]
    same = audited["mosaickv_exact", "joint-same-buffer"]
    new = audited["mosaickv_exact", "joint-handoff"]
    if any(old[key] != same[key] for key in driver.STORAGE):
        raise ValueError("same-buffer correction changed frozen physical storage")
    delta = new[driver.STORAGE[0]] - old[driver.STORAGE[0]]
    if (
        delta > 0
        or any(new[key] - old[key] != delta for key in driver.STORAGE[1:])
        or delta
        != new["lt_source_conversion"]["added_patch_bytes"]
        - old["lt_source_conversion"]["added_patch_bytes"]
    ):
        raise ValueError("handoff storage may change only by measured nonincreasing native repairs")
    return dict(
        ORIGINAL_SUMMARY(rows, audited),
        actual_native_repair_byte_delta=delta,
        physical_storage={
            f"{m}:{a}": {k: v[k] for k in driver.STORAGE} for (m, a), v in audited.items()
        },
        independent_quality_N=0,
        whole_request_GPU_peak_certified=False,
        no_fresh_or_larger_campaign_authorized=True,
    )


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--screen", action="store_true")
    flags, rest = parser.parse_known_args()
    base.PREPARE_SCREEN, sys.argv = flags.screen, [sys.argv[0], *rest]
    base.DEBUG_ROOT, base.SCRIPT, base.DOC = DEBUG_ROOT, SCRIPT, DOC
    base.ARMS, base.DEBUG_ARMS = ARMS, ARMS
    base.DEBUG_VARIANT = "consumed_verified_constructor_correctness"
    base.configurations, base.execute = configurations, execute
    base.prior.prior.correctness.configure()
    parent.WAIT, parent.JOB_NAME = WAIT, "mk-verified-ctor"
    control.WAIT, control.SCRIPT = WAIT, SCRIPT
    control.CANDIDATE_ARM, control.POLICY = "joint-handoff", "kv"
    control.VARIANT, control.DOC = "consumed_verified_constructor_ablation", DOC
    control.ARMS, control.AUXILIARY_PROFILE_ARMS = ARMS, frozenset()
    control.prepare, control.configurations, control.execute, control.summary = (
        prepare,
        configurations,
        execute,
        summary,
    )
    control.worker, control.analyze, control.JOB_NAME = (
        base.worker,
        base.analyze,
        "mk-verified-ctor",
    )
    control.main()


if __name__ == "__main__":
    main()
