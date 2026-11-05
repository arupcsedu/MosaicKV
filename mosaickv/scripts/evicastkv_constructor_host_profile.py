#!/usr/bin/env python3
"""Profile unique constructor work after the shared counter optimization."""

from pathlib import Path

import evicastkv_lt_model_profile as controller

from mosaickv.evaluation import lt_model_profile
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.host_constructor_profile import collect

SOURCE, PARENT, driver = controller.SOURCE, controller.PARENT, controller.driver
NEGATIVE = Path("/scratch/djy8hg/runs/evicastkv/graph-accounting-consumed-20261105-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_constructor_host_profile.sbatch"
MANIFEST_VARIANT = "consumed_constructor_host_attribution"
MANIFEST_TIMING_PROTOCOL = dict(
    output_predictions=128,
    CPU_profile_start="prepare_inputs",
    CPU_profile_stop="first_decode_entry",
    decode_excluded=True,
    not_pure_compression_time=True,
    not_latency_evidence=True,
)


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    old = driver.read(PARENT / "plan.json")
    gate = driver.read(NEGATIVE / "analysis/decision.json")
    if not gate["complete"] or gate["failures"]:
        raise ValueError("complete counter-ablation screen required")
    bound = dict(old["bound_files"])
    paths = [
        PARENT / "plan.json",
        NEGATIVE / "analysis/decision.json",
        SOURCE / "mosaickv/docs/constructor_host_profile_20261105.md",
    ]
    paths += [PARENT / f"workers/{i}/study.json" for i in (0, 1)]
    bound.update({str(p): sha(p) for p in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_constructor_host_attribution",
            source_runtime_unchanged=True,
            CPU_node="udc-aj36-3c0",
            independent_quality_N=0,
            no_new_identities=True,
            automatic_quality_submission=False,
            automatic_timing_retry=False,
            acceptance_not_established_by_profile=True,
        ),
    )
    cpu = controller.submit(root, "cpu")
    release = controller.submit(root, "release", cpu)
    write_once(root / "cpu_submission.json", dict(cpu=cpu, release=release))


def main():
    controller.MANIFEST_VARIANT = MANIFEST_VARIANT
    controller.MANIFEST_TIMING_PROTOCOL = MANIFEST_TIMING_PROTOCOL
    controller.CPU_NODE = "udc-aj36-3c0"
    controller.SCRIPT, controller.prepare = SCRIPT, prepare
    lt_model_profile.collect = collect
    controller.main()


if __name__ == "__main__":
    main()
