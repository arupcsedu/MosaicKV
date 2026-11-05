#!/usr/bin/env python3
"""Target the measured missing-counter exception cost, not a precision sweep."""

from pathlib import Path

import evicastkv_source_192_reader as prior

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

control, SOURCE, parent, driver = prior.control, prior.SOURCE, prior.parent, prior.driver
WAIT = prior.WAIT
PREVIOUS = Path("/scratch/djy8hg/runs/evicastkv/source-192-reader-consumed-20261105-v1")
PROFILE = Path("/scratch/djy8hg/runs/evicastkv/native-history-host-consumed-20261105-retry1")
OPERATOR = Path("/scratch/djy8hg/runs/evicastkv/direct-lease-operator-20261105-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_history_phase_fastpath.sbatch"
DOC = SOURCE / "mosaickv/docs/history_phase_fastpath.md"
ARMS = (
    ("mosaickv_exact", "joint-192"),
    ("mosaickv_exact", "joint-fast-history"),
    ("prefixkv_reimpl", "graph"),
    ("prefixkv_reimpl", "graph-fast-history"),
    ("prefixkv_reimpl", "graph-original"),
    ("prefixkv_reimpl", "graph-original-fast-history"),
)
ORIGINAL_SUMMARY = control.summary


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    for folder in (PREVIOUS, PROFILE):
        decision = driver.read(folder / "analysis/decision.json")
        if not decision["complete"] or decision["failures"]:
            raise ValueError("complete prior numerical and profile evidence required")
    old = driver.read(PREVIOUS / "plan.json")
    bound = dict(old["bound_files"])
    paths = [
        DOC,
        PREVIOUS / "plan.json",
        PREVIOUS / "analysis/decision.json",
        PROFILE / "plan.json",
        PROFILE / "analysis/decision.json",
    ]
    bound.update({str(path): sha(path) for path in paths})
    serial = str(driver.read(OPERATOR / "manifest.json")["job_id"])
    if not serial.isdecimal():
        raise ValueError("finite preceding GPU diagnostic dependency required")
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_common_native_history_missing_counter_fastpath",
            arms=ARMS,
            warmups=5,
            repetitions=3,
            actual_phase_inference_detection_no_exceptions=True,
            original_tracked_tensor_mutation_and_identity_checks=True,
            history_borrows_first_128_rows_without_allocation=True,
            entire_192_row_owner_storage_still_charged=True,
            no_KV_or_positions_or_threshold_changes=True,
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
    base = prior.configurations(records)
    result = {}
    for method, arm in ARMS:
        baseline = "joint-192" if method == "mosaickv_exact" else arm.removesuffix("-fast-history")
        result[method, arm] = dict(
            base[method, baseline], native_history_reader=arm.endswith("-fast-history")
        )
    return result


def execute(adapter, refs, messages, profile, plan, method, arm, record, **kwargs):
    effective = "joint-192" if method == "mosaickv_exact" else arm.removesuffix("-fast-history")
    return prior.execute(
        adapter, refs, messages, profile, plan, method, effective, record, **kwargs
    )


def summary(rows, audited):
    storage = {}
    for (method, arm), value in audited.items():
        storage[f"{method}:{arm}"] = {key: value[key] for key in driver.STORAGE}
        if arm.endswith("-fast-history"):
            old = "joint-192" if method == "mosaickv_exact" else arm.removesuffix("-fast-history")
            if value[driver.STORAGE[0]] != audited[method, old][driver.STORAGE[0]]:
                raise ValueError("native history must preserve exact initial prompt storage")
    return dict(
        ORIGINAL_SUMMARY(rows, audited),
        physical_storage=storage,
        original_prompt_bytes_unchanged=True,
        independent_quality_N=0,
        whole_request_GPU_peak_certified=False,
        no_fresh_or_larger_campaign_authorized=True,
    )


def main():
    prior.correctness.configure()
    parent.WAIT, parent.JOB_NAME = WAIT, "mk-phase-fast"
    control.WAIT, control.SCRIPT = WAIT, SCRIPT
    control.CANDIDATE_ARM, control.POLICY = "joint-fast-history", "kv"
    control.VARIANT, control.DOC = "consumed_history_missing_counter_fastpath", DOC
    control.ARMS, control.AUXILIARY_PROFILE_ARMS = ARMS, frozenset()
    control.prepare, control.configurations, control.execute, control.summary = (
        prepare,
        configurations,
        execute,
        summary,
    )
    control.worker, control.JOB_NAME = prior.worker, "mk-phase-fast"
    control.main()


if __name__ == "__main__":
    main()
