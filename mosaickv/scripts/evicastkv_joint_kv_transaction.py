#!/usr/bin/env python3
"""Value-commit extension only after key-only correctness and timing analysis."""

from pathlib import Path

import evicastkv_joint_key_transaction as control
import evicastkv_joint_row_feasibility as parent

from mosaickv.evaluation.fox_campaign import write_once

SOURCE, driver = parent.SOURCE, parent.driver
WAIT = Path("/scratch/djy8hg/runs/evicastkv/joint-key-latency-consumed-20261105-v1")
CORRECTNESS = Path("/scratch/djy8hg/runs/evicastkv/joint-key-transaction-consumed-20261105-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_joint_kv_transaction.sbatch"
ORIGINAL_PREPARE = parent.prepare
VARIANT = "consumed_joint_K_begin_V_commit_A_B_A"
SEQUENCE = ("graph-lt", "joint-kv", "graph-lt")


def prepare(root, revision):
    gate = driver.read(WAIT / "analysis/decision.json")
    proof = driver.read(CORRECTNESS / "analysis/decision.json")
    if (
        not gate["complete"]
        or gate["failures"]
        or gate["four_percent_POINT_only"]
        or not proof["complete"]
        or proof["failures"]
        or not proof["joint_key_128_vector_parity_and_isolation"]
    ):
        raise ValueError("complete negative key-only timing and positive key-only parity required")
    ORIGINAL_PREPARE(root, revision)


def analyze(root, expected):
    plan = driver.checked(root, expected)
    rows, failures = driver.load_producers(root, plan, "profile.json")
    write_once(
        root / "analysis/decision.json",
        dict(
            complete=not failures,
            failures=failures,
            rows=rows,
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            joint_KV_128_vector_parity_and_isolation=not failures and len(rows) == 2,
            no_timing_or_quality_acceptance=True,
            independent_quality_N=0,
            whole_request_GPU_peak_certified=False,
            no_automatic_larger_or_quality_submission=True,
        ),
    )


def configure():
    control.POLICY, control.CANDIDATE_ARM = "kv", "joint-kv"
    control.VARIANT, control.SEQUENCE = VARIANT, SEQUENCE
    control.EXPECTED_TRANSACTION = dict(
        key_calls=4064,
        folded_keys=4032,
        original_cold_keys=32,
        pending=False,
        all_value_projections_original=False,
        value_commits=4064,
        fake_raw_value_tensors=0,
    )


def main():
    configure()
    parent.WAIT, parent.SCRIPT = WAIT, SCRIPT
    parent.ONLY_LAYER, parent.JOB_NAME = None, "mk-joint-KV"
    parent.STAGE = parent.VARIANT = VARIANT
    parent.DOC = SOURCE / "mosaickv/docs/joint_kv_transaction.md"
    parent.EXTRA_PLAN = dict(
        sequence=SEQUENCE,
        no_fake_raw_values=True,
        original_RoPE_attention_and_history_codec=True,
        extra_incoming_workspace_bytes=8192,
        unchanged_128_vector_bitwise_gate=True,
    )
    parent.EXTRA_BOUND_FILES = (
        WAIT / "analysis/decision.json",
        CORRECTNESS / "analysis/decision.json",
    )
    parent.prepare, parent.analyze = prepare, analyze
    parent.controller.worker = control.worker
    parent.main()


if __name__ == "__main__":
    main()
