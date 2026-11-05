#!/usr/bin/env python3
"""Paired consumed-input screen of measured redundant storage-accounting work."""

from pathlib import Path

import evicastkv_selective_text_constructor as controller

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, base, PARENT = controller.SOURCE, controller.base, controller.PARENT
PROFILE = Path("/scratch/djy8hg/runs/evicastkv/host-dispatch-consumed-20261105-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_graph_accounting.sbatch"
CANDIDATE = ("mosaickv_exact", "graph-event-sum")
ARMS = (
    ("mosaickv_exact", "graph"),
    ("mosaickv_exact", "graph-single-pass"),
    CANDIDATE,
    ("prefixkv_reimpl", "graph"),
    ("prefixkv_reimpl", "graph-original"),
    ("prefixkv_reimpl", "graph-event-sum"),
    ("prefixkv_reimpl", "graph-original-event-sum"),
)
ORIGINAL_EXECUTE = controller.execute


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    proof = controller.driver.read(PROFILE / "analysis/decision.json")
    if not proof["complete"] or proof["failures"]:
        raise ValueError("complete actual host-call attribution required")
    old = controller.driver.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    paths = [
        PARENT / "plan.json",
        PROFILE / "analysis/decision.json",
        SOURCE / "mosaickv/docs/graph_accounting_20261105.md",
    ]
    paths += [PARENT / f"workers/{i}/study.json" for i in (0, 1)]
    bound.update({str(p): sha(p) for p in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_graph_accounting_execution_screen",
            warmups=5,
            repetitions=3,
            screen_NOT_acceptance=True,
            no_fresh_identities=True,
            independent_quality_N=0,
            automatic_quality_submission=False,
            automatic_formal_submission=False,
            original_selected_KV_and_GPU_storage_unchanged=True,
            optimization_offered_to_baseline_controls=True,
            arms=ARMS,
        ),
    )
    base.common.stage(root, "cpu")


def variants(records):
    return [
        (
            "mosaickv_exact",
            f"graph-{name.replace('_', '-')}",
            dict(records["mosaickv_exact"]["graph"], graph_accounting=name),
        )
        for name in ("single_pass", "event_sum")
    ] + [
        (
            "prefixkv_reimpl",
            arm + "-event-sum",
            dict(records["prefixkv_reimpl"][arm], graph_accounting="event_sum"),
        )
        for arm in ("graph", "graph-original")
    ]


def execute(adapter, refs, messages, profile, plan, method, record, **kwargs):
    value = ORIGINAL_EXECUTE(adapter, refs, messages, profile, plan, method, record, **kwargs)
    requested = record.get("graph_accounting", "original")
    proof = value["graph_cache_accounting"]
    if (
        proof["policy"] != requested
        or proof["final_independent_tensor_enumeration_preserved"] is not True
    ):
        raise RuntimeError("actual policy and independent tensor enumeration required")
    if requested == "event_sum" and (
        proof["notifications"] < 1
        or (kwargs.get("audit", False) and proof["audited_independent_comparisons"] < 1)
    ):
        raise RuntimeError("actual owner notifications and audited cross-checks required")
    return value


def summary(rows):
    result = base.screen_summary(rows, arms=ARMS, candidate=CANDIDATE)
    means = {(r["method"], r["arm"]): r["mean"] for r in result["summaries"]}
    ratios = {
        arm: means[CANDIDATE] / means[method, arm]
        for method, arm in ARMS
        if method == "prefixkv_reimpl"
    }
    paired = {
        (r["method"], r["arm"], r["trial"]): r["measurement"]["total_wall_seconds"]
        for r in rows
        if not r["warmup"]
    }
    paired_by_control = {
        arm: [
            paired[(*CANDIDATE, trial)] / paired["prefixkv_reimpl", arm, trial]
            for trial in range(3)
        ]
        for arm in ratios
    }
    return dict(
        summaries=result["summaries"],
        paired_ratios_by_original_and_optimized_control=paired_by_control,
        mean_ratios_against_original_and_optimized_PrefixKV=ratios,
        four_percent_POINT_only=bool(
            max(value for group in paired_by_control.values() for value in group) <= 0.96
        ),
        formal_latency_acceptance=False,
        independent_quality_N=0,
        full_twenty_repeat_adjusted_ratio_gate_still_required=True,
    )


def main():
    controller.SCRIPT, controller.CANDIDATE, controller.ARMS = SCRIPT, CANDIDATE, ARMS
    controller.prepare, controller.variants = prepare, variants
    controller.execute, controller.summary = execute, summary
    controller.main()


if __name__ == "__main__":
    main()
