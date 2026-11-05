#!/usr/bin/env python3
"""Commit original sparse packets with one bounded constructor-count barrier."""

from pathlib import Path

import evicastkv_prepared_history_dispatch as previous

from mosaickv.evaluation.fox_campaign import clean_source, write_once

SOURCE, controller, base = previous.SOURCE, previous.controller, previous.base
WAIT = Path("/scratch/djy8hg/runs/evicastkv/prepared-history-dispatch-consumed-20261105-retry1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_static_transport_construction.sbatch"
ARMS = tuple(
    (method, arm + suffix)
    for method, arm in (
        ("mosaickv_exact", "graph"),
        ("prefixkv_reimpl", "graph"),
        ("prefixkv_reimpl", "graph-original"),
    )
    for suffix in ("", "-static-transport")
)
CANDIDATE = ("mosaickv_exact", "graph-static-transport")


def variants(records):
    return [
        (
            method,
            arm + "-static-transport",
            dict(records[method][arm], static_transport_patches=True),
        )
        for method, arm in ARMS
        if not arm.endswith("-static-transport")
    ]


def execute(adapter, refs, messages, profile, plan, method, record, **kwargs):
    configured = dict(record, reuse_patch_dispatch=True)
    value = previous.execute(adapter, refs, messages, profile, plan, method, configured, **kwargs)
    proof = value["static_transport_construction"]
    if record.get("static_transport_patches", False):
        expected = 64 if method == "mosaickv_exact" else 0
        if (
            not proof["enabled"]
            or not proof["committed"]
            or proof["constructor_patches"] != expected
            or proof["dynamic_transport_nonzero_calls_avoided"] != expected
            or proof["count_barrier_CPU_bytes"] != expected * 8
            or proof["constructor_temporary_objects_alive"]
            or not proof["canonical_packets_have_no_temporary_patches"]
        ):
            raise RuntimeError(
                "actual bounded canonical commit and complete temporary retirement required"
            )
    elif proof["enabled"]:
        raise RuntimeError("ordinary constructor control must remain ordinary")
    return value


def worker(root, expected, index):
    from mosaickv.evaluation.static_transport_patches import gpu_controls

    try:
        previous.review_predecessor(root)
        previous.admission.actual(at_worker_start=True)
        write_once(root / f"gpu-static-controls-{index}.json", gpu_controls())
    except Exception as error:
        out = root / f"workers/{index}"
        out.mkdir(parents=True, exist_ok=False)
        controller.driver.finish_worker(
            out,
            dict(
                status="failed",
                failures=[f"static GPU controls: {type(error).__name__}: {error}"],
                source_sha=clean_source(SOURCE),
                plan_sha256=expected,
            ),
        )
        return
    previous.worker(root, expected, index)


def summary(rows):
    report = base.screen_summary(rows, arms=ARMS, candidate=CANDIDATE)
    means = {(r["method"], r["arm"]): r["mean"] for r in report["summaries"]}
    ratios = {
        arm: means[CANDIDATE] / means[method, arm]
        for method, arm in ARMS
        if method == "prefixkv_reimpl"
    }
    return dict(
        summaries=report["summaries"],
        ratios_vs_original_and_optimized_PrefixKV=ratios,
        four_percent_POINT_only=max(ratios.values()) <= 0.96,
        formal_latency_acceptance=False,
        independent_quality_N=0,
        full_twenty_repeat_adjusted_ratio_gate_still_required=True,
    )


def main():
    previous.LOOKAHEAD, previous.SCRIPT = WAIT, SCRIPT
    previous.ARMS, previous.CANDIDATE = ARMS, CANDIDATE
    previous.STAGE = "consumed_static_transport_constructor_screen"
    previous.DOC = SOURCE / "mosaickv/docs/static_transport_construction.md"
    previous.EXTRA_PLAN = dict(
        static_transport_capacity=4096,
        method_agnostic_constructor_optimization=True,
        all_arms_prepared_patch_dispatch=True,
    )
    previous.parent.SCRIPT = previous.admission.SCRIPT = SCRIPT
    controller.MANIFEST_VARIANT = previous.STAGE
    controller.SCRIPT, controller.ARMS, controller.CANDIDATE = SCRIPT, ARMS, CANDIDATE
    controller.prepare, controller.variants = previous.prepare, variants
    controller.execute, controller.worker, controller.summary = execute, worker, summary
    controller.analyze = previous.analyze
    base.submit = previous.admission.submit
    controller.main()


if __name__ == "__main__":
    main()
