#!/usr/bin/env python3
"""Direct storage/unfused/fused paired timing plus excluded warmup phase audit."""

import json
from collections import Counter
from pathlib import Path
from statistics import median
from unittest.mock import patch

import header_history_timing as timing

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

PARENT = Path("/scratch/djy8hg/runs/evicastkv/mixed-history-projection-20261108-v1/paired-timing")
ORIGINAL = Path("/scratch/djy8hg/runs/evicastkv/selective-history64-20261108-v1")
ARMS = ["storage_common_headers", "selective_unfused", "selective_mixed"]
POLICIES = {
    ARMS[0]: dict(history=False, mixed=False),
    ARMS[1]: dict(history=True, mixed=False),
    ARMS[2]: dict(history=True, mixed=True),
}
COMPARISONS = [(ARMS[0], ARMS[1]), (ARMS[0], ARMS[2]), (ARMS[1], ARMS[2])]


def registered_schedule(seed):
    with patch.object(timing, "ARMS", ARMS):
        return timing.schedule(seed)


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable three-arm root required")
    gate = timing.base.common.read(PARENT / "analysis/decision.json")
    if not gate["complete"] or not all(
        r["end_to_end_seconds"]["equal_or_faster"] for r in gate["results"]
    ):
        raise ValueError("positive same-precision paired timing required before direct comparison")
    p = timing.base.common.read(PARENT / "plan.json")
    original = timing.base.common.read(ORIGINAL / "plan.json")
    p["controls"]["references"] = original["controls"]["references"]
    p["same_precision_references"] = {
        str(i): str(PARENT.parent / f"model/{i}/selective_unfused.logits.pt") for i in (0, 1)
    }
    paths = [PARENT / "plan.json", PARENT / "analysis/decision.json", ORIGINAL / "plan.json"]
    paths += [Path(p["controls"]["references"][str(i)]["storage"]) for i in (0, 1)]
    paths += [Path(path) for path in p["same_precision_references"].values()]
    p["bound_files"].update({str(path): sha(path) for path in paths})
    p.update(
        source_sha=clean_source(timing.SOURCE),
        study="direct_three_arm_history_and_readout_profile_v1",
        measurement_type="consumed_direct_three_arm_paired_timing",
        arms=ARMS,
        arm_policies=POLICIES,
        timing_comparisons=COMPARISONS,
        timing_family_comparisons=12,
        precision_equivalence_required=False,
        profile_first_warmup=True,
        profile_is_excluded_from_timed_trials=True,
        no_multiplication_of_ratios_from_separate_jobs=True,
        fresh_or_paper_campaign_authorized=False,
        independent_quality_validation_authorized=False,
    )
    p["timing_schedules"] = {str(i): registered_schedule(p["seed"] + i) for i in (0, 1)}
    root.mkdir()
    (root / "logs").mkdir()
    write_once(root / "plan.json", p)
    cpu = timing.base.common.submit(root, "cpu")
    write_once(
        root / "submission.json",
        dict(cpu=cpu, follow=timing.base.common.submit(root, "after_cpu", cpu)),
    )
    print(json.dumps(timing.base.common.read(root / "submission.json")), flush=True)


def analyze(root, p):
    results, errors, profiles = [], [], []
    for i in (0, 1):
        path = root / f"model/{i}/raw.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        if len(rows) != 75 or any(r["status"] != "completed" for r in rows):
            errors.append(dict(input=i, error="incomplete/failed registered requests"))
            continue
        timed = [r for r in rows if not r["warmup"]]
        if Counter(r["arm"] for r in timed) != Counter({arm: 20 for arm in ARMS}):
            errors.append(dict(input=i, error="missing/duplicate twenty-trial method pairs"))
            continue
        for baseline, candidate in COMPARISONS:
            stats = {}
            for metric in ("end_to_end_seconds", "decode_seconds"):
                values = [
                    [
                        r[metric]
                        for r in sorted(timed, key=lambda r: r["repetition"])
                        if r["arm"] == arm
                    ]
                    for arm in (baseline, candidate)
                ]
                stats[metric] = timing.paired(
                    values,
                    seed=p["seed"] + i,
                    replicates=p["timing_bootstrap_replicates"],
                    family_comparisons=12,
                )
            results.append(dict(input=i, baseline=baseline, candidate=candidate, **stats))
        profile = next(
            r["mixed_projection"]["phase_profile"]
            for r in rows
            if r["warmup"] and r["repetition"] == 0 and r["arm"] == ARMS[2]
        )
        if len(profile["rows"]) != 72 or len(profile["constructor_host_rows"]) != 192:
            errors.append(dict(input=i, error="missing registered warmup readout profiles"))
        profiles.append(dict(input=i, **profile))
    folder = root / "analysis"
    folder.mkdir(exist_ok=False)
    complete = len(results) == 6 and not errors
    direct = [r for r in results if r["baseline"] == ARMS[0] and r["candidate"] == ARMS[2]]
    write_once(
        folder / "decision.json",
        dict(
            complete=complete,
            errors=errors,
            results=results,
            direct_equal_or_faster_on_both_inputs=complete
            and len(direct) == 2
            and all(r["end_to_end_seconds"]["equal_or_faster"] for r in direct),
            initial_storage_advantage_over_common_storage=False,
            complete_superiority_established=False,
            independent_quality_established=False,
            fresh_cohort_authorized=False,
        ),
    )
    write_once(
        folder / "phase-profiles.json",
        dict(
            diagnostic_excluded_warmup_only=True,
            rows=profiles,
            CUDA_stream_intervals_NOT_pure_kernel_time=True,
        ),
    )
    lines = [
        "# Direct three-arm consumed-input comparison",
        "",
        "Five warmups and twenty randomized paired repetitions, 128 predictions. "
        "All arms get identical common header/predicate policies. Twelve registered "
        "timing endpoints receive Bonferroni-adjusted intervals.",
        "",
        "| Input | Baseline | Candidate | E2E ratio | Adjusted CI | Strict pass |",
        "|---|---|---|---:|---|---|",
    ]
    for r in results:
        v = r["end_to_end_seconds"]
        lines.append(
            f"| {r['input']} | {r['baseline']} | {r['candidate']} | "
            f"{v['mean_latency_ratio']:.6f} | {v['ratio_adjusted_CI']} | {v['equal_or_faster']} |"
        )
    lines += [
        "",
        "## Excluded warmup profiles",
        "",
        "Stream intervals include host dispatch gaps; they are not pure kernel times.",
        "",
        "| Input | Phase | Median host ms | Median stream ms |",
        "|---|---|---:|---:|",
    ]
    for profile in profiles:
        for phase in sorted({r["phase"] for r in profile["rows"]}):
            group = [r for r in profile["rows"] if r["phase"] == phase]
            lines.append(
                f"| {profile['input']} | {phase} | "
                f"{median(r['host_ms'] for r in group):.6f} | "
                f"{median(r['CUDA_stream_ms'] for r in group):.6f} |"
            )
    lines += [
        "",
        "No initial-memory or independent quality superiority is established. "
        "No cross-job latency ratios are multiplied. All failures remain explicit.",
    ]
    (folder / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    timing.ARMS = ARMS
    timing.base.common.WORKER_WRAPPER = timing.SOURCE / "mosaickv/slurm/history_three_arm.sbatch"
    timing.base.EXTRA_TESTS = (
        *timing.base.EXTRA_TESTS,
        "test_mixed_history_projection.py",
        "test_history_three_arm.py",
    )
    timing.prepare, timing.analyze = prepare, analyze
    timing.main()
