#!/usr/bin/env python3
"""Find actual host-dispatch bottlenecks before any new runtime policy."""

from pathlib import Path

import evicastkv_lt_model_profile as controller

from mosaickv.evaluation import lt_model_profile
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.host_decode_profile import collect

SOURCE, PARENT, driver = controller.SOURCE, controller.PARENT, controller.driver
EVENT = Path("/scratch/djy8hg/runs/evicastkv/event-only-reader-consumed-20261105-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_host_dispatch_profile.sbatch"
MANIFEST_VARIANT = "consumed_host_decode_attribution"


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    old = driver.read(PARENT / "plan.json")
    gate = driver.read(EVENT / "analysis/decision.json")
    if not gate["complete"] or gate["failures"]:
        raise ValueError("complete event-only model attribution required")
    bound = dict(old["bound_files"])
    paths = [
        PARENT / "plan.json",
        PARENT / "paired-timing/analysis/decision.json",
        EVENT / "analysis/decision.json",
        SOURCE / "mosaickv/docs/host_dispatch_profile_20261105.md",
    ]
    paths += [PARENT / f"workers/{i}/study.json" for i in (0, 1)]
    bound.update({str(p): sha(p) for p in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_host_decode_attribution",
            source_runtime_unchanged=True,
            CUPTI_enabled=False,
            cProfile_decode_updates=list(range(1, 127)),
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
    controller.SCRIPT, controller.prepare = SCRIPT, prepare
    lt_model_profile.collect = collect
    controller.main()


if __name__ == "__main__":
    main()
