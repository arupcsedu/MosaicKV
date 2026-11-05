#!/usr/bin/env python3
"""Bounded consumed-input attribution with event accounting for every reader."""

from pathlib import Path

import evicastkv_lt_model_profile as controller

from mosaickv.evaluation import lt_model_profile
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.evaluation.optimized_runtime_profile import collect

SOURCE, PARENT, driver = controller.SOURCE, controller.PARENT, controller.driver
PROOF = Path("/scratch/djy8hg/runs/evicastkv/graph-accounting-consumed-20261105-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_optimized_runtime_profile.sbatch"
ORIGINAL_EXECUTE = controller.common.execute


def optimized_record(record):
    return dict(
        record,
        graph_accounting="event_sum",
        **({"lt_fit_policy": "direct"} if record.get("lt_source_reader", False) else {}),
    )


def execute(adapter, refs, messages, profile, plan, method, record, **kwargs):
    return ORIGINAL_EXECUTE(
        adapter, refs, messages, profile, plan, method, optimized_record(record), **kwargs
    )


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    gate = driver.read(PROOF / "analysis/decision.json")
    if not gate["complete"] or gate["failures"]:
        raise ValueError("complete optimized-accounting predecessor required")
    old = driver.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    paths = [PROOF / "analysis/decision.json", SOURCE / "mosaickv/docs/optimized_runtime_profile.md"]
    bound.update({str(p): sha(p) for p in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_optimized_runtime_attribution",
            independent_quality_N=0,
            no_new_identities=True,
            automatic_quality_submission=False,
            automatic_timing_retry=False,
            graph_accounting_every_arm="event_sum",
            native_constructor="direct",
            instrumentation_NOT_latency_acceptance=True,
        ),
    )
    cpu = controller.submit(root, "cpu")
    release = controller.submit(root, "release", cpu)
    write_once(root / "cpu_submission.json", dict(cpu=cpu, release=release))


def main():
    controller.MANIFEST_VARIANT = "consumed_optimized_runtime_attribution"
    controller.MANIFEST_TIMING_PROTOCOL = dict(
        output_predictions=128,
        construction="prepare_inputs_to_first_decode_entry",
        decode_updates=list(range(1, 127)),
        initial_capture_excluded=True,
        graph_accounting="event_sum",
        native_constructor="direct",
        not_latency_evidence=True,
    )
    controller.CPU_NODE = "udc-aj36-3c0"
    controller.SCRIPT, controller.prepare = SCRIPT, prepare
    controller.common.execute = execute
    lt_model_profile.collect = collect
    controller.main()


if __name__ == "__main__":
    main()
