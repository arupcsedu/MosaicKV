#!/usr/bin/env python3
"""Fixed 64-input development comparison, never an automatic confirmation."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import subprocess
from pathlib import Path

import yaml
from evicastkv_analyze import collect, export_descriptive, pairing
from evicastkv_campaign import admitted_array_concurrency, environment, load_lock, resolved_config
from evicastkv_opportunity_study import VARIANTS, arms

from mosaickv.config import canonical_config
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.fox_statistics import corpus_cider_statistics, paired_scalar_statistics
from mosaickv.evaluation.metrics.types import MetricSample

SOURCE = Path(__file__).resolve().parents[2]
REGISTRATION = SOURCE / "mosaickv/configs/experiments/evicastkv_opportunity_development_v1.yaml"
CANARY = Path(
    "/scratch/djy8hg/runs/evicastkv/fox-review-20261028-opportunity-demand-canary-v2-vl-tier-retry"
)
PARENT = Path("/scratch/djy8hg/runs/evicastkv/fox-review-20261028-v6-coco-scoped")
STAGE = "evidence_development"
POLICY_PATHS = (
    "mosaickv/src/mosaickv/backends/hf_runtime.py",
    "mosaickv/src/mosaickv/config.py",
    "mosaickv/src/mosaickv/selection/opportunity_demand.py",
    "mosaickv/src/mosaickv/baselines/vl_cache.py",
    "mosaickv/scripts/evicastkv_campaign.py",
    "mosaickv/scripts/evicastkv_worker.py",
)


def point_key(method, cap, variant):
    return f"{method}/{variant}/{cap}"


def registered_comparisons():
    return [
        (
            point_key(baseline, cap, "complete"),
            point_key("mosaickv_exact", cap, variant),
        )
        for cap in (20, 40)
        for variant in VARIANTS[1:]
        for baseline in ("prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact")
    ]


def expected_keys(ids):
    if len(ids) != 64 or len(set(ids)) != 64:
        raise ValueError("use all 64 original unique development identities")
    return {
        (sid, method, None if cap is None else cap * 2**20, variant)
        for sid in ids
        for method, cap, variant in arms()
    }


def validate_rows(rows, ids):
    keys = [(r["sample_id"], r["method"], r["budget_bytes"], r["variant"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys(ids):
        raise ValueError("incomplete, extra, or duplicate registered rows")
    if any(r["status"] != "completed" for r in rows):
        raise ValueError("a failed sample cannot disappear from the denominator")
    for row in rows:
        if row["workload"] != "coco" or row["repeat"] != 0:
            raise ValueError("wrong workload or pseudoreplicated quality observations")
        if row["emitted_tokens"] != 64 or row["forward_steps"] != 63:
            raise ValueError("generation/append schedule changed")
        if row["budget_bytes"] is not None and row["initial_kv_bytes"] != row["budget_bytes"]:
            raise ValueError("measured cache does not fill the shared physical budget")
        for field in ("e2e_wall_seconds", "initial_kv_bytes", "final_kv_bytes", "peak_kv_bytes"):
            if not math.isfinite(row[field]) or row[field] <= 0:
                raise ValueError("missing or nonfinite storage/timing")
        if not row["initial_kv_bytes"] <= row["final_kv_bytes"] <= row["peak_kv_bytes"]:
            raise ValueError("append-only storage boundary is inconsistent")
        expected = row["method"]
        if row["variant"].startswith("evidence-"):
            name = {
                "evidence-routing-v1": "routing_only",
                "evidence-coverage-v1": "coverage_only",
                "evidence-both-v1": "both",
            }[row["variant"]]
            expected += f"__evidence_{name}_v1"
        if row["effective_method"] != expected:
            raise ValueError("unexpected fallback or method relabeling")
    pairs = pairing(rows)
    if len(pairs) != 128:
        raise ValueError("missing input/budget pairs")
    for sid in ids:
        values = [r for r in rows if r["sample_id"] == sid]
        for field in ("references", "media_group_ids", "gpu_uuid", "source_sha"):
            if len({json.dumps(r[field], sort_keys=True) for r in values}) != 1:
                raise ValueError(f"paired {field} changed")
    return pairs


def quality_decision(statistics):
    """Strict improvement, not 'same quality' from a zero-crossing interval."""
    results = {}
    for variant in VARIANTS[1:]:
        required = [
            contrast
            for contrast in statistics["contrasts"]
            if contrast["candidate"].split("/")[1] == variant
            and not contrast["reference"].startswith("mosaickv_exact/")
        ]
        results[variant] = dict(
            required_contrasts=len(required),
            positive_development_signal=len(required) == 4
            and all(c["simultaneous_ci"][0] > 0 for c in required),
        )
    return results


def load_analysis_lock(root):
    """Separate immutable producer provenance from a corrected CPU analyzer."""
    lock = json.loads((root / "protocol_lock.json").read_text())
    producer = Path(lock["source_root"])
    clean_source(producer, lock["source_sha"])
    clean_source(SOURCE)
    if any(sha(SOURCE / p) != sha(producer / p) for p in POLICY_PATHS):
        raise ValueError("historical analysis requires unchanged policy/runtime sources")
    for path, expected in lock["historical_bindings"].items():
        if sha(Path(path)) != expected:
            raise ValueError("historical binding changed")
    for path, expected in lock.get("inherited_gates", {}).items():
        if sha(Path(path)) != expected or json.loads(Path(path).read_text())["status"] != "passed":
            raise ValueError("inherited gate changed or failed")
    for info in lock["templates"].values():
        if sha(Path(info["path"])) != info["sha256"]:
            raise ValueError("template hash changed")
    for groups in lock["cohorts"].values():
        for info in groups.values():
            if sha(Path(info["path"])) != info["sha256"]:
                raise ValueError("cohort hash changed")
    return lock


def validate_config_record(row, config, source_sha, arithmetic):
    """JSON arrays and Python tuples share a canonical JSON identity, not values."""
    path = Path(row["resolved_config_path"])
    if (
        sha(path) != row["resolved_config_file_sha256"]
        or digest(json.loads(path.read_text())) != digest(config)
        or row["config_sha"] != digest(dict(runtime_config=config, arithmetic=arithmetic))
        or row["source_sha"] != source_sha
    ):
        raise ValueError("executed runtime configuration/source changed")


def prepare(root):
    revision = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("use a new immutable development root")
    registration = yaml.safe_load(REGISTRATION.read_text())
    if (
        registration["samples"] != 64
        or registration["budgets_mib"] != [20, 40]
        or registration["variants"] != list(VARIANTS)
        or registration["statistics"]["family_size"] != len(registered_comparisons())
        or any(
            registration[k]
            for k in (
                "fresh_materialization_authorized",
                "confirmation_authorized",
                "paper_campaign_authorized",
            )
        )
    ):
        raise ValueError("registered scope changed")
    canary_lock = json.loads((CANARY / "protocol_lock.json").read_text())
    previous_source = Path(canary_lock["source_root"])
    clean_source(previous_source, registration["policy_source_sha"])
    if any(sha(SOURCE / p) != sha(previous_source / p) for p in POLICY_PATHS):
        raise ValueError("policy changed after the passed canary")
    canary_gate = CANARY / "analysis/evidence_canary/GATE_evidence_canary.json"
    gate = json.loads(canary_gate.read_text())
    if (
        gate["status"] != "passed_consumed_correctness_only"
        or gate["actual_rows"] != 26
        or gate["errors"]
        or gate["protocol_sha256"] != sha(CANARY / "protocol_lock.json")
    ):
        raise ValueError("passed representation canary is required")
    original = json.loads((PARENT / "protocol_lock.json").read_text())
    cohort = original["cohorts"]["coco"]["development"]
    path = Path(cohort["path"])
    if sha(path) != cohort["sha256"]:
        raise ValueError("original development ledger changed")
    ledger = json.loads(path.read_text())
    ids = [s["sample_id"] for s in ledger["samples"]]
    if ids != cohort["sample_ids"]:
        raise ValueError("original identity order changed")
    expected_keys(ids)
    for sample in ledger["samples"]:
        for media in sample["media"]:
            if sha(Path(media["path"])) != media["sha256"]:
                raise ValueError("original development media changed")
    lock = copy.deepcopy(canary_lock)
    lock.pop("evidence_retry", None)
    lock.update(
        name="evicastkv_opportunity_development_v1",
        source_root=str(SOURCE),
        source_sha=revision,
        environment=environment(),
        stages=[STAGE],
        resources=registration["resources"],
        cohorts={"coco": {"development": cohort}},
        development_registration=registration,
    )
    lock["historical_bindings"].update(
        {
            str(REGISTRATION): sha(REGISTRATION),
            str(canary_gate): sha(canary_gate),
            str(CANARY / "protocol_lock.json"): sha(CANARY / "protocol_lock.json"),
            str(PARENT / "protocol_lock.json"): sha(PARENT / "protocol_lock.json"),
            **{str(previous_source / p): sha(previous_source / p) for p in POLICY_PATHS},
        }
    )
    for method, cap, variant in arms():
        resolved_config(lock, "coco", method, cap, variant=variant)
    root.mkdir(parents=True)
    write_once(root / "protocol_lock.json", lock)
    chunk = registration["resources"]["samples_per_worker"]
    jobs = [
        dict(
            workload="coco",
            stage=STAGE,
            subset="development",
            indices=list(range(i, min(i + chunk, 64))),
            arms=arms(),
        )
        for i in range(0, 64, chunk)
    ]
    write_once(
        root / f"jobs/{STAGE}.json",
        dict(
            stage=STAGE,
            protocol_sha256=sha(root / "protocol_lock.json"),
            jobs=jobs,
        ),
    )
    write_once(
        root / "readiness.json",
        dict(
            status="prepared_development_only",
            source_sha=revision,
            requests=832,
            unique_inputs=64,
            already_consumed=True,
            fresh_confirmation=False,
            comparisons=registered_comparisons(),
            family_size=18,
            no_automatic_confirmation=True,
        ),
    )


def analyze(root, *, recorded_source=False):
    lock = load_analysis_lock(root) if recorded_source else load_lock(root)
    output = root / "analysis" / STAGE
    if output.exists():
        raise FileExistsError("analysis is write-once")
    spec, workers, rows, errors = collect(root, STAGE)
    ids = lock["cohorts"]["coco"]["development"]["sample_ids"]
    pairs, activities = [], []
    try:
        if errors or len(workers) != len(spec["jobs"]):
            raise ValueError("incomplete workers; statistical analysis blocked")
        pairs = validate_rows(rows, ids)
        for row in rows:
            cap = None if row["budget_bytes"] is None else row["budget_bytes"] // 2**20
            config = canonical_config(
                resolved_config(lock, "coco", row["method"], cap, variant=row["variant"])
            )
            validate_config_record(row, config, lock["source_sha"], lock["arithmetic"])
            trace = json.loads(Path(row["trace_path"]).read_text())
            activity = trace.get("experimental_evidence_policy")
            if row["variant"].startswith("evidence-") and not activity:
                raise ValueError("experimental activity trace absent")
            if row["method"] in {"mosaickv_exact", "prompt_attention_topk"} and trace.get(
                "packed_padding_kv_bytes", 0
            ):
                raise ValueError("Top-K/MosaicKV unexpectedly uses padding")
            activities.append(
                dict(
                    sample_id=row["sample_id"],
                    method=row["method"],
                    variant=row["variant"],
                    cap=cap,
                    policy=activity,
                )
            )
    except Exception as error:
        errors.append(dict(error=f"{type(error).__name__}: {error}"))
    write_once(
        output / "completeness.json",
        dict(
            expected=832,
            attempted=len(rows),
            completed=sum(r["status"] == "completed" for r in rows),
            workers_expected=len(spec["jobs"]),
            workers_observed=len(workers),
            errors=errors,
        ),
    )
    write_once(output / "pairing_report.json", pairs)
    write_once(output / "configuration_activity.json", activities)
    if errors:
        write_once(
            output / "GATE_evidence_development.json",
            dict(
                status="failed_execution",
                source_sha=lock["source_sha"],
                errors=errors,
                confirmation_authorized=False,
                paper_claim_allowed=False,
            ),
        )
        return
    export_descriptive(root, STAGE, rows)
    points = {}
    groups = {}
    timings = {}
    for row in rows:
        cap = None if row["budget_bytes"] is None else row["budget_bytes"] // 2**20
        key = point_key(row["method"], cap, row["variant"])
        points.setdefault(key, []).append(
            MetricSample(row["sample_id"], row["answer"], tuple(row["references"]))
        )
        groups[row["sample_id"]] = tuple(row["media_group_ids"])
        timings.setdefault(key, {})[row["sample_id"]] = row["e2e_wall_seconds"]
    statistical = lock["development_registration"]["statistics"]
    comparisons = registered_comparisons()
    stats = corpus_cider_statistics(
        {k: tuple(v) for k, v in points.items()},
        comparisons,
        groups,
        resamples=statistical["resamples"],
        seed=statistical["seed"],
        family_size=statistical["family_size"],
        workers=1,
    )
    write_once(output / "quality_statistics.json", stats)
    latency = [
        dict(
            reference=a,
            candidate=b,
            **paired_scalar_statistics(
                timings[a],
                timings[b],
                groups,
                dict.fromkeys(ids, "COCO"),
                resamples=statistical["resamples"],
                seed=statistical["seed"],
                family_size=18,
            ),
        )
        for a, b in comparisons
    ]
    write_once(
        output / "paired_latency_descriptive.json",
        dict(
            headline_claim_allowed=False,
            reason="quality-only runs; no registered warmups/repeated timing",
            contrasts=latency,
        ),
    )
    decisions = quality_decision(stats)
    write_once(
        output / "GATE_evidence_development.json",
        dict(
            status="passed_execution_development_only",
            source_sha=lock["source_sha"],
            analysis_source_sha=clean_source(SOURCE),
            protocol_sha256=sha(root / "protocol_lock.json"),
            actual_rows=len(rows),
            quality_decisions=decisions,
            fresh_confirmation=False,
            confirmation_authorized=False,
            paper_claim_allowed=False,
        ),
    )
    lines = [
        "# Opportunity/demand development comparison",
        "",
        "All 64 previously consumed inputs; development-only, not independent confirmation.",
        "832/832 requests. Same measured initial/final/peak storage within each input/budget.",
        "18 prespecified contrasts; corpus CIDEr recomputed inside every one of "
        "10,000 paired media-cluster replicates.",
        "Latency is descriptive: these quality runs are not repeated headline timing.",
        "",
        "| Method/variant/cap (MiB) | Corpus CIDEr |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {score:.6f} |" for key, score in stats["scores"].items())
    lines += [
        "",
        "| Reference | Candidate | CIDEr difference | Simultaneous interval |",
        "|---|---|---:|---|",
    ]
    lines.extend(
        f"| {c['reference']} | {c['candidate']} | {c['difference']:.6f} | {c['simultaneous_ci']} |"
        for c in stats["contrasts"]
    )
    lines += [
        "",
        "A zero-crossing interval is inconclusive, not equivalence. No automatic cohort expansion.",
        "No selected_config.json is fabricated; independent confirmation requires "
        "a separate frozen protocol.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(status="development_analyzed", decisions=decisions)))


def submit(root):
    lock = load_lock(root)
    if (root / "jobs/submission.json").exists():
        raise FileExistsError("submission already exists")
    active = subprocess.check_output(
        [
            "squeue",
            "-h",
            "-u",
            os.environ["USER"],
            "-t",
            "RUNNING,COMPLETING,PENDING",
            "-o",
            "%i|%b",
        ],
        text=True,
    )
    total = sum(
        int(match[1])
        for line in active.splitlines()
        if (match := re.search(r"gpu(?::[a-zA-Z0-9_-]+)?:(\d+)", line))
    )
    concurrency = admitted_array_concurrency(lock["resources"], total)
    if concurrency < 1:
        raise ValueError("aggregate six-GPU cap reached; preserve existing jobs")
    spec_path = root / f"jobs/{STAGE}.json"
    count = len(json.loads(spec_path.read_text())["jobs"])
    (root / "slurm").mkdir(exist_ok=True)
    common = ["sbatch", "--parsable", "--account=bii_dsc_community"]
    command = [
        *common,
        f"--array=0-{count - 1}%{concurrency}",
        f"--output={root}/slurm/development-%A_%a.out",
        f"--error={root}/slurm/development-%A_%a.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_opportunity_development.sbatch"),
        str(SOURCE),
        str(root),
        "worker",
        sha(spec_path),
    ]
    gpu = subprocess.check_output(command, text=True).strip().split(";")[0]
    callback = [
        *common,
        "--partition=bii",
        "--gres=none",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:10:00",
        f"--dependency=afterany:{gpu}",
        f"--output={root}/slurm/analysis-%j.out",
        f"--error={root}/slurm/analysis-%j.err",
        str(SOURCE / "mosaickv/slurm/evicastkv_opportunity_development_analysis.sbatch"),
        str(SOURCE),
        str(root),
    ]
    cpu = subprocess.check_output(callback, text=True).strip().split(";")[0]
    write_once(
        root / "jobs/submission.json",
        dict(array=gpu, analysis=cpu, command=command, callback=callback),
    )
    print(json.dumps(dict(array=gpu, analysis=cpu, concurrency=concurrency)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "worker", "analyze", "submit"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--spec-sha")
    parser.add_argument("--recorded-source", action="store_true")
    args = parser.parse_args()
    if args.recorded_source and args.mode != "analyze":
        parser.error("recorded-source is CPU analysis only")
    if not args.root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("scratch-only artifacts")
    if args.mode == "worker":
        from evicastkv_worker import run_worker

        run_worker(args.root, STAGE, int(os.environ["SLURM_ARRAY_TASK_ID"]), args.spec_sha)
    elif args.mode == "analyze":
        analyze(args.root, recorded_source=args.recorded_source)
    else:
        if args.recorded_source:
            parser.error("recorded-source is CPU analysis only")
        globals()[args.mode](args.root)
