#!/usr/bin/env python3
"""Same paired protocol, frozen K/V transaction, bounded excluded attribution."""

from pathlib import Path

import evicastkv_joint_key_latency as control
import evicastkv_joint_kv_transaction as correctness

SOURCE = control.SOURCE
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_joint_kv_latency.sbatch"


def configure():
    correctness.configure()
    control.CANDIDATE_ARM, control.POLICY = "joint-kv", "kv"
    control.GATE_FIELD = "joint_KV_128_vector_parity_and_isolation"
    control.WAIT = Path("/scratch/djy8hg/runs/evicastkv/joint-kv-transaction-consumed-20261105-v1")
    control.SCRIPT = SCRIPT
    control.VARIANT = "consumed_joint_KV_paired_screen_and_excluded_attribution"
    control.JOB_NAME = "mk-joint-KV-time"
    control.DOC = SOURCE / "mosaickv/docs/joint_kv_latency.md"
    control.ARMS = tuple((m, "joint-kv" if a == "joint-keys" else a) for m, a in control.ARMS)
    control.AUXILIARY_PROFILE_ARMS = frozenset(
        {("mosaickv_exact", "joint-kv"), ("prefixkv_reimpl", "graph")}
    )
    control.APPROVED_CONTROLLER_PATHS = {
        "mosaickv/scripts/evicastkv_joint_key_latency.py",
        "mosaickv/scripts/evicastkv_joint_kv_latency.py",
        "mosaickv/slurm/evicastkv_joint_kv_latency.sbatch",
        "mosaickv/docs/joint_kv_latency.md",
        "mosaickv/tests/unit/test_joint_kv_latency.py",
    }


if __name__ == "__main__":
    configure()
    control.main()
