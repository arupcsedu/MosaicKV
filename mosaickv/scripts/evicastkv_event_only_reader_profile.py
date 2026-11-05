#!/usr/bin/env python3
"""Attribution of the native kernel/request mismatch, not another policy sweep."""

import evicastkv_lt_model_profile as controller

from mosaickv.evaluation import lt_model_profile
from mosaickv.evaluation.event_only_reader_profile import collect
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, PARENT, driver = controller.SOURCE, controller.PARENT, controller.driver
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_event_only_reader_profile.sbatch"
MANIFEST_VARIANT = "consumed_event_only_native_model_attribution"


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    old = driver.read(PARENT / "plan.json")
    gate = driver.read(PARENT / "paired-timing/analysis/decision.json")
    if not gate["complete"] or gate["four_percent_latency_gate"]:
        raise ValueError("complete negative native-reader timing gate required")
    bound = dict(old["bound_files"])
    paths = [
        PARENT / "plan.json",
        PARENT / "paired-timing/analysis/decision.json",
        SOURCE / "mosaickv/docs/event_only_reader_profile_20261105.md",
    ]
    paths += [PARENT / f"workers/{i}/study.json" for i in (0, 1)]
    bound.update({str(p): sha(p) for p in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="event_only_native_model_attribution",
            source_runtime_unchanged=True,
            CUPTI_enabled=False,
            CUDA_event_windows=[1, 64, 126],
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
