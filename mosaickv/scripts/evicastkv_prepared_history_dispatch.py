#!/usr/bin/env python3
"""Eliminate measured repeated patch-JIT dispatch; keep the matrix kernels."""

from pathlib import Path

import evicastkv_graph_lookahead as parent
import evicastkv_graph_lookahead_retry as admission

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

controller, base = parent.controller, parent.base
SOURCE, PARENT, PROFILE = parent.SOURCE, parent.PARENT, parent.PROFILE
LOOKAHEAD = Path("/scratch/djy8hg/runs/evicastkv/graph-lookahead-consumed-20261105-retry1")
FAILED_CPU = Path(
    "/scratch/djy8hg/runs/evicastkv/prepared-history-dispatch-consumed-20261105-v1/cpu-analysis/decision.json"
)
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_prepared_history_dispatch.sbatch"
ORIGINAL_EXECUTE = parent.ORIGINAL_EXECUTE
ORIGINAL_ANALYZE = parent.ORIGINAL_ANALYZE
ARMS = tuple(
    (method, arm + suffix)
    for method, arm in (
        ("mosaickv_exact", "graph"),
        ("mosaickv_exact", "graph-lt"),
        ("prefixkv_reimpl", "graph"),
        ("prefixkv_reimpl", "graph-original"),
    )
    for suffix in ("", "-prepared-patch")
)
CANDIDATE = ("mosaickv_exact", "graph-prepared-patch")
STAGE = "consumed_prepared_history_patch_dispatch_screen"
DOC = SOURCE / "mosaickv/docs/prepared_history_dispatch.md"
EXTRA_PLAN = {}


def variants(records):
    return [
        (method, arm + "-prepared-patch", dict(records[method][arm], reuse_patch_dispatch=True))
        for method, arm in ARMS
        if not arm.endswith("-prepared-patch")
    ]


def execute(adapter, refs, messages, profile, plan, method, record, **kwargs):
    configured = dict(record, graph_accounting="event_sum")
    if configured.get("lt_source_reader", False):
        configured["lt_fit_policy"] = "direct"
    measured = ORIGINAL_EXECUTE(
        adapter, refs, messages, profile, plan, method, configured, **kwargs
    )
    proof = measured["prepared_patch_dispatch"]
    if record.get("reuse_patch_dispatch", False):
        if not proof["enabled"] or proof["hits"] < 6000 or proof["misses"] < 1:
            raise RuntimeError("actual common repeated history-patch dispatch required")
    elif proof["enabled"]:
        raise RuntimeError("original controls may not silently receive prepared patch policy")
    if proof["tensor_owners_retained"] or proof["GPU_workspace_bytes"]:
        raise RuntimeError("prepared host plans may not add tensor owners or workspace")
    return measured


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    preceding = controller.driver.read(LOOKAHEAD / "plan.json")
    dependency_file = LOOKAHEAD / "screen_submission.json"
    dependency = str(controller.driver.read(dependency_file)["analysis"])
    if not dependency.isdecimal():
        raise ValueError("explicit preceding terminal analysis job required")
    old = controller.driver.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    paths = [
        PROFILE / "analysis/decision.json",
        LOOKAHEAD / "plan.json",
        dependency_file,
        DOC,
        FAILED_CPU,
    ]
    bound.update({str(p): sha(p) for p in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage=STAGE,
            warmups=5,
            repetitions=3,
            screen_NOT_acceptance=True,
            arms=ARMS,
            independent_quality_N=0,
            no_fresh_identities=True,
            automatic_quality_submission=False,
            automatic_formal_submission=False,
            automatic_bounded_cohort_after_correctness=False,
            automatic_bounded_followup=False,
            automatic_full_request_submission=False,
            original_KV_and_tensor_storage_unchanged=True,
            graph_accounting_all_arms="event_sum",
            native_fit_policy="direct",
            graph_lookahead_all_arms=False,
            optimization_offered_to_all_baseline_controls=True,
            predecessor_source_sha=preceding["source_sha"],
            predecessor_plan_sha256=sha(LOOKAHEAD / "plan.json"),
            **EXTRA_PLAN,
        ),
    )
    cpu = base.common.submit(root, "cpu")
    analysis = base.common.submit(root, "cpu_analysis", dependency=f"{cpu}:{dependency}")
    write_once(
        root / "cpu_submission.json",
        dict(
            worker=cpu,
            analysis=analysis,
            afterany_preceding_analysis=dependency,
            no_overlapping_GPU_screens=True,
        ),
    )


def review_predecessor(root):
    plan = controller.driver.read(root / "plan.json")
    path = LOOKAHEAD / "analysis/decision.json"
    gate = controller.driver.read(path)
    if (
        not gate["complete"]
        or gate["failures"]
        or gate["source_sha"] != plan["predecessor_source_sha"]
        or gate["plan_sha256"] != plan["predecessor_plan_sha256"]
    ):
        raise ValueError("complete source/plan-bound preceding screen required")
    return dict(
        path=str(path),
        sha256=sha(path),
        preceding_screen_complete=True,
        original_arguments_frozen_before_predecessor_finished=True,
    )


def analyze(root, expected, action):
    plan = controller.driver.checked(root, expected)
    if action == "cpu_analysis":
        try:
            review = review_predecessor(root)
        except Exception as error:
            write_once(
                root / "cpu-analysis/decision.json",
                dict(
                    complete=False,
                    failures=[f"predecessor: {type(error).__name__}: {error}"],
                    no_GPU_submitted=True,
                    independent_quality_N=0,
                    source_sha=plan["source_sha"],
                    plan_sha256=expected,
                ),
            )
            return
        write_once(root / "predecessor_review.json", review)
    ORIGINAL_ANALYZE(root, expected, action)


def worker(root, expected, index):
    from mosaickv.evaluation.prepared_patch_dispatch import gpu_controls

    try:
        review = controller.driver.read(root / "predecessor_review.json")
        if review != review_predecessor(root):
            raise ValueError("unchanged reviewed predecessor required")
        admission.actual(at_worker_start=True)
        report = gpu_controls()
        write_once(root / f"gpu-controls-{index}.json", report)
    except Exception as error:
        out = root / f"workers/{index}"
        out.mkdir(parents=True, exist_ok=False)
        controller.driver.finish_worker(
            out,
            dict(
                status="failed",
                failures=[f"GPU controls: {type(error).__name__}: {error}"],
                source_sha=clean_source(SOURCE),
                plan_sha256=expected,
            ),
        )
        return
    parent.ORIGINAL_WORKER(root, expected, index)


def summary(rows):
    result = base.screen_summary(rows, arms=ARMS, candidate=CANDIDATE)
    means = {(r["method"], r["arm"]): r["mean"] for r in result["summaries"]}
    candidates = (CANDIDATE, ("mosaickv_exact", "graph-lt-prepared-patch"))
    controls = [(m, a) for m, a in ARMS if m == "prefixkv_reimpl"]
    ratios = {a: {b: means[m, a] / means[n, b] for n, b in controls} for m, a in candidates}
    return dict(
        summaries=result["summaries"],
        candidate_mean_ratios_vs_all_PrefixKV=ratios,
        four_percent_POINT_only=any(max(r.values()) <= 0.96 for r in ratios.values()),
        formal_latency_acceptance=False,
        independent_quality_N=0,
        full_twenty_repeat_adjusted_ratio_gate_still_required=True,
    )


def main():
    parent.SCRIPT = SCRIPT
    admission.SCRIPT = SCRIPT
    controller.MANIFEST_VARIANT = "consumed_prepared_history_patch_dispatch_screen"
    controller.SCRIPT, controller.ARMS, controller.CANDIDATE = SCRIPT, ARMS, CANDIDATE
    controller.prepare, controller.variants = prepare, variants
    controller.execute, controller.worker, controller.summary = execute, worker, summary
    controller.analyze = analyze
    base.submit = admission.submit
    controller.main()


if __name__ == "__main__":
    main()
