#!/usr/bin/env python3
"""Apply an eligible no-extra-byte checkpoint to every consumed baseline path."""

from pathlib import Path

import evicastkv_encoded_baselines as encoded

from mosaickv.evaluation.fox_campaign import sha

base = encoded.base
ROOT = Path("/scratch/djy8hg/runs/evicastkv/packed-checkpoint-baselines-consumed-20261030-v1")
VALIDATED = Path("/scratch/djy8hg/runs/evicastkv/packed-checkpoint-consumed-20261030-v1")


def prepare(root, phase):
    path_decision = VALIDATED / "timing/analysis/packed_checkpoint_decision.json"
    decision = base.common.read(path_decision)
    if not decision["engineering_eligible"]:
        raise ValueError("packed checkpoint must first pass exact bytes and paired latency")
    original = encoded.history.original_writer

    def register(path, value):
        if path.name == "plan.json":
            for stage in ("compact", "timing"):
                for name in ("plan.json", "analysis/GATE.json", "analysis/manifest.json"):
                    bound = VALIDATED / stage / name
                    value["bound_files"][str(bound)] = sha(bound)
            value["bound_files"][str(path_decision)] = sha(path_decision)
            value.update(
                protocol="common_packed_checkpoints_all_baselines_v1",
                packed_literal_counts=True,
                no_extra_checkpoint_bytes=True,
                no_fresh_or_paper_campaign=True,
            )
        original(path, value)

    encoded.history.original_writer = register
    try:
        return encoded.prepare(root, phase)
    finally:
        encoded.history.original_writer = original


def choices(*args, **kwargs):
    # Scope the constructor option to this serialized diagnostic call; do
    # not change normal CLI defaults or any other process's model policy.
    import mosaickv.evaluation.encoded_projection as implementation

    original = implementation.build

    def packed_build(base_format, attention):
        return original(base_format, attention, packed_literal_counts=True)

    implementation.build = packed_build
    try:
        return encoded.fused_choices(*args, **kwargs)
    finally:
        implementation.build = original


if __name__ == "__main__":
    encoded.history.paired.install()
    encoded.history.fused_choices = choices
    base.BASE = ROOT
    base.SCRIPT_NAME = "evicastkv_packed_checkpoint_baselines.sbatch"
    base.prepare, base.baseline_study, base.analyze = (
        prepare,
        encoded.history.study,
        encoded.history.analyze,
    )
    base.main()
