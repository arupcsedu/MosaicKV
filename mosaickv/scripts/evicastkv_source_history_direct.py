#!/usr/bin/env python3
"""Same-bit source-history constructor differential, no new selection/cohort."""

from pathlib import Path

import evicastkv_source_history_runtime as runtime

from mosaickv.evaluation.fox_campaign import sha

base = runtime.base
ROOT = Path("/scratch/djy8hg/runs/evicastkv/source-history-direct-consumed-20261030-v1")
VALIDATED = Path(
    "/scratch/djy8hg/runs/evicastkv/source-history-runtime-consumed-20261030-v1/canary"
)
original_prepare = runtime.prepare


def prepare(root, phase):
    if not base.common.read(VALIDATED / "analysis/source_history_decision.json")[
        "runtime_canary_passed"
    ]:
        raise ValueError("original source-history runtime canary must pass first")
    writer = runtime.encoded.history.original_writer

    def register(path, value):
        if path.name == "plan.json":
            for name in (
                "plan.json",
                "analysis/GATE.json",
                "analysis/manifest.json",
                "analysis/source_history_decision.json",
            ):
                bound = VALIDATED / name
                value["bound_files"][str(bound)] = sha(bound)
            value.update(
                protocol="direct_history_constructor_same_exact_packets_v1",
                source_history_comparison_kind="constructor",
                candidate_constructor="direct",
                reference_constructor="reference",
                original_KV_checks_per_final_packet=3,
                candidate_projection_passes_per_chunk=4,
                reference_projection_passes_per_chunk=9,
                largest_stage_count=2,
                no_selection_or_budget_change=True,
            )
        writer(path, value)

    runtime.encoded.history.original_writer = register
    try:
        return original_prepare(root, phase)
    finally:
        runtime.encoded.history.original_writer = writer


if __name__ == "__main__":
    runtime.encoded.history.paired.install()
    runtime.prepare = prepare
    base.BASE = ROOT
    base.SCRIPT_NAME = "evicastkv_source_history_direct.sbatch"
    base.prepare, base.baseline_study, base.analyze = prepare, runtime.study, runtime.analyze
    base.main()
