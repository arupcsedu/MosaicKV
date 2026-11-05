#!/usr/bin/env python3
"""Bounded consumed-input constructor screen. Never promotes a paper claim."""

import evicastkv_selective_text_constructor as controller

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, base = controller.SOURCE, controller.base
PARENT = controller.PARENT
CANDIDATE = ("mosaickv_exact", "graph-deferred-exact")
MANIFEST_VARIANT = "consumed_deferred_exact_constructor"
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_deferred_exact_constructor.sbatch"
ARMS = (
    ("mosaickv_exact", "graph"),
    CANDIDATE,
    ("prefixkv_reimpl", "graph"),
    ("prefixkv_reimpl", "graph-original"),
    ("prefixkv_reimpl", "graph-deferred-exact"),
    ("prefixkv_reimpl", "graph-original-deferred-exact"),
)
ORIGINAL_EXECUTE = controller.execute


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    old = controller.driver.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    paths = [
        PARENT / "plan.json",
        PARENT / "analysis/decision.json",
        SOURCE / "mosaickv/docs/deferred_exact_constructor_20261105.md",
    ]
    paths += [PARENT / f"workers/{i}/study.json" for i in (0, 1)]
    bound.update({str(p): sha(p) for p in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="deferred_exact_constructor_screen",
            warmups=5,
            repetitions=3,
            screen_NOT_acceptance=True,
            no_fresh_identities=True,
            independent_quality_N=0,
            automatic_quality_submission=False,
            automatic_formal_submission=False,
            original_selected_KV_and_GPU_storage_unchanged=True,
            same_optimization_offered_to_baselines=True,
            barriers_before_all_byte_dependent_choices=True,
            arms=ARMS,
        ),
    )
    base.common.stage(root, "cpu")


def variants(records):
    return [
        (
            "mosaickv_exact",
            CANDIDATE[1],
            dict(
                records["mosaickv_exact"]["graph"],
                defer_exact_encoding=True,
                selective_text_rows=True,
            ),
        ),
        *[
            (
                "prefixkv_reimpl",
                arm + "-deferred-exact",
                dict(records["prefixkv_reimpl"][arm], defer_exact_encoding=True),
            )
            for arm in ("graph", "graph-original")
        ],
    ]


def execute(adapter, refs, messages, profile, plan, method, record, **kwargs):
    value = ORIGINAL_EXECUTE(adapter, refs, messages, profile, plan, method, record, **kwargs)
    if record.get("defer_exact_encoding", False):
        proof = value["deferred_exact_encoding"]
        if not proof["enabled"] or proof["created"] < 1 or proof["encoded_at_barriers"] < 1:
            raise RuntimeError("actual deferred encoding and surviving-body finalization required")
    return value


def summary(rows):
    proof = base.screen_summary(rows, arms=ARMS, candidate=CANDIDATE)
    means = {(r["method"], r["arm"]): r["mean"] for r in proof["summaries"]}
    ratio = {
        arm: means[CANDIDATE] / means[method, arm]
        for method, arm in ARMS
        if method == "prefixkv_reimpl"
    }
    return dict(
        summaries=proof["summaries"],
        paired_ratios=proof["all_paired_ratios"],
        mean_ratios_against_original_and_optimized_PrefixKV=ratio,
        four_percent_POINT_only=bool(max(ratio.values()) <= 0.96),
        formal_latency_acceptance=False,
        independent_quality_N=0,
        full_twenty_repeat_adjusted_ratio_gate_still_required=True,
    )


def main():
    controller.MANIFEST_VARIANT = MANIFEST_VARIANT
    controller.SCRIPT = SCRIPT
    controller.CANDIDATE, controller.ARMS = CANDIDATE, ARMS
    controller.prepare, controller.variants = prepare, variants
    controller.execute, controller.summary = execute, summary
    controller.main()


if __name__ == "__main__":
    main()
