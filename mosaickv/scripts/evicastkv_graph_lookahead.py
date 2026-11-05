#!/usr/bin/env python3
"""Same-buffer producer overlap: finite controls and paired consumed screen."""

import subprocess
from pathlib import Path

import evicastkv_selective_text_constructor as controller

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE, base, PARENT = controller.SOURCE, controller.base, controller.PARENT
PROFILE = Path("/scratch/djy8hg/runs/evicastkv/optimized-runtime-profile-consumed-20261105-v1")
SCRIPT = SOURCE / "mosaickv/slurm/evicastkv_graph_lookahead.sbatch"
ORIGINAL_EXECUTE, ORIGINAL_WORKER, ORIGINAL_ANALYZE = (
    controller.execute,
    controller.worker,
    controller.analyze,
)
ORIGINAL_SUBMIT = base.submit
ARMS = tuple(
    (method, arm + suffix)
    for method, arm in (
        ("mosaickv_exact", "graph"),
        ("mosaickv_exact", "graph-lt"),
        ("prefixkv_reimpl", "graph"),
        ("prefixkv_reimpl", "graph-original"),
    )
    for suffix in ("", "-lookahead")
)
CANDIDATE = ("mosaickv_exact", "graph-lookahead")


def submit(root, action, *, dependency=None):
    if action == "screen":
        return ORIGINAL_SUBMIT(root, action, dependency=dependency)
    controller.driver.checked(root, sha(root / "plan.json"))
    (root / "slurm").mkdir(exist_ok=True)
    args = [
        "sbatch",
        "--parsable",
        "--account=bii_dsc_community",
        "--partition=bii",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=2",
        "--mem=16G",
        "--time=00:25:00",
        "--nodelist=udc-aj36-2c0",
        f"--job-name=mk-lookahead-{action}",
        f"--output={root}/slurm/{action}-%A_%a.out",
        f"--error={root}/slurm/{action}-%A_%a.err",
    ]
    if dependency:
        args.append("--dependency=afterany:" + dependency)
    args += [str(SCRIPT), str(SOURCE), str(root), sha(root / "plan.json"), action]
    return subprocess.check_output(args, text=True).strip()


def variants(records):
    return [
        (method, arm + "-lookahead", dict(records[method][arm], graph_lookahead=True))
        for method, arm in ARMS
        if not arm.endswith("-lookahead")
    ]


def execute(adapter, refs, messages, profile, plan, method, record, **kwargs):
    configured = dict(record, graph_accounting="event_sum")
    if configured.get("lt_source_reader", False):
        configured["lt_fit_policy"] = "direct"
    measured = ORIGINAL_EXECUTE(
        adapter, refs, messages, profile, plan, method, configured, **kwargs
    )
    proof = measured["prompt_graph_reader"]["lookahead"]
    if record.get("graph_lookahead", False):
        if (
            not proof["enabled"]
            or proof["launches"] != 3906
            or proof["hits"] != 3906
            or proof["first_capture_cold_misses"] != 31
            or proof["skipped_token_boundaries"] != 127
            or proof["extra_tensor_workspace_bytes"]
            or proof["prompt_copies"]
            or proof["pending"]
            or not proof["retired"]
        ):
            raise RuntimeError("complete same-buffer 31x126 actual producer overlap required")
    elif proof["enabled"]:
        raise RuntimeError("lookahead may not silently modify original-reader controls")
    return measured


def prepare(root, revision):
    if root.exists() or clean_source(SOURCE) != revision:
        raise ValueError("new immutable root and clean source required")
    decision = controller.driver.read(PROFILE / "analysis/decision.json")
    if not decision["complete"] or decision["failures"]:
        raise ValueError("complete current optimized attribution required")
    old = controller.driver.read(PARENT / "plan.json")
    bound = dict(old["bound_files"])
    paths = [PROFILE / "analysis/decision.json", SOURCE / "mosaickv/docs/graph_lookahead.md"]
    bound.update({str(p): sha(p) for p in paths})
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=revision,
            bound_files=bound,
            stage="consumed_same_buffer_graph_lookahead_screen",
            warmups=5,
            repetitions=3,
            screen_NOT_acceptance=True,
            arms=ARMS,
            independent_quality_N=0,
            no_fresh_identities=True,
            automatic_quality_submission=False,
            automatic_formal_submission=False,
            original_KV_and_tensor_storage_unchanged=True,
            graph_accounting_all_arms="event_sum",
            native_fit_policy="direct",
            optimization_offered_to_all_baseline_controls=True,
        ),
    )
    base.common.stage(root, "cpu")


def worker(root, expected, index):
    from mosaickv.evaluation.graph_lookahead import gpu_controls

    try:
        report = gpu_controls()  # BEFORE large checkpoint loading.
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
    ORIGINAL_WORKER(root, expected, index)


def summary(rows):
    result = base.screen_summary(rows, arms=ARMS, candidate=CANDIDATE)
    means = {(r["method"], r["arm"]): r["mean"] for r in result["summaries"]}
    candidates = (CANDIDATE, ("mosaickv_exact", "graph-lt-lookahead"))
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
    controller.MANIFEST_VARIANT = "consumed_same_buffer_graph_lookahead_screen"
    controller.SCRIPT, controller.ARMS, controller.CANDIDATE = SCRIPT, ARMS, CANDIDATE
    controller.prepare, controller.variants = prepare, variants
    controller.execute, controller.worker, controller.summary = execute, worker, summary
    base.submit = submit
    controller.main()


if __name__ == "__main__":
    main()
