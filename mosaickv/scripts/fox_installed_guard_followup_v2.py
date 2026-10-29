#!/usr/bin/env python3
"""Finite prompt-only installed-plan study; same model, caps and policy weights."""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
from pathlib import Path

from evicastkv_analyze import collect, export_descriptive, pairing
from evicastkv_campaign import environment, load_lock, resolved_config
from prefixkv_isoquality_development import choose_ids

from mosaickv.config import canonical_config
from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once
from mosaickv.evaluation.fox_statistics import corpus_cider_statistics
from mosaickv.evaluation.metrics.types import MetricSample

SOURCE = Path(__file__).resolve().parents[2]
PARENT = Path(
    "/scratch/djy8hg/runs/evicastkv/fox-review-20261028-opportunity-development-reanalysis-v1"
)
POOL = Path("/scratch/djy8hg/runs/mosaickv/coco-shared-pool-edb6edf-retry1-20261014/ledger.json")
VARIANT = "evidence-output-guarded-v2"
PREFIX_ROOT = Path("/scratch/djy8hg/runs/mosaickv/prefix-native-intermediate-20261029-v2-retry1")


def arms():
    return [("full_kv", None, "complete")] + [
        (method, cap, variant)
        for cap in (20, 40)
        for method, variant in [
            ("prompt_attention_topk", "complete"),
            ("vl_cache_reimpl", "complete"),
            ("mosaickv_exact", "complete"),
            ("mosaickv_exact", VARIANT),
        ]
    ]


def prepare(root):
    revision = clean_source(SOURCE)
    cpu_gate = PREFIX_ROOT / "verification/cpu-tests.json"
    cpu = json.loads(cpu_gate.read_text())
    if cpu["status"] != "passed" or cpu["source_sha"] != revision:
        raise ValueError("clean-source CPU verification must pass first")
    if root.exists():
        raise FileExistsError("new immutable root required")
    old = json.loads((PARENT / "protocol_lock.json").read_text())
    old_cohort = old["cohorts"]["coco"]["development"]
    prior = json.loads(Path(old_cohort["path"]).read_text())
    if sha(Path(old_cohort["path"])) != old_cohort["sha256"]:
        raise ValueError("old consumed identity ledger changed")
    pool = json.loads(POOL.read_text())
    ids = choose_ids(pool["samples"], old_cohort["sample_ids"], count=32, seed=2026102902)
    by_id = {s["sample_id"]: s for s in pool["samples"]}
    lock = copy.deepcopy(old)
    lock.update(
        name="fox_installed_guard_v2",
        source_root=str(SOURCE),
        source_sha=revision,
        environment=environment(),
        warmups=0,
        repeats=1,
        stages=["evidence_canary", "installed_guard_development"],
        cohorts={"coco": {}},
        new_scope=dict(
            user_authorized=True,
            primary_examples=32,
            classification="disjoint_from_64_input_tuning_not_globally_untouched",
            no_draft=True,
            no_prototypes=True,
            no_repair=True,
            no_new_routing_fraction=True,
            confirmation_authorized=False,
        ),
    )
    lock["resources"].update(
        maximum_a100s=6, array_concurrency=3, six_a100s_explicitly_authorized=True, wall_minutes=60
    )
    root.mkdir(parents=True)
    for subset, samples in [
        ("consumed_canary", prior["samples"][:2]),
        ("development", [by_id[sid] for sid in ids]),
    ]:
        path = root / f"manifests/{subset}.json"
        payload = {
            k: pool[k]
            for k in ("schema_version", "task", "dataset_id", "dataset_revision", "split")
        }
        payload["samples"] = samples
        write_once(path, payload)
        lock["cohorts"]["coco"][subset] = dict(
            path=str(path), sha256=sha(path), sample_ids=[s["sample_id"] for s in samples]
        )
        for sample in samples:
            for media in sample["media"]:
                if sha(Path(media["path"])) != media["sha256"]:
                    raise ValueError("input media changed")
    lock["historical_bindings"].update(
        {
            str(POOL): sha(POOL),
            str(PARENT / "protocol_lock.json"): sha(PARENT / "protocol_lock.json"),
        }
    )
    lock.setdefault("inherited_gates", {})[str(cpu_gate)] = sha(cpu_gate)
    # Unchanged entropy 1/256 routing and graph/utility coefficients. New guard
    # compares the actual installed subset to its exact same-route anchor.
    lock["evidence_study"]["policy"] = dict(shortlist_size=8, minimum_query_mass_fraction=0.95)
    for method, cap, variant in arms():
        resolved_config(lock, "coco", method, cap, variant=variant)
    write_once(root / "protocol_lock.json", lock)
    for stage, subset, count, chunk in [
        ("evidence_canary", "consumed_canary", 2, 1),
        ("installed_guard_development", "development", 32, 4),
    ]:
        jobs = [
            dict(
                workload="coco",
                stage=stage,
                subset=subset,
                indices=list(range(i, min(i + chunk, count))),
                arms=arms(),
                identity_variants=[VARIANT],
            )
            for i in range(0, count, chunk)
        ]
        write_once(
            root / f"jobs/{stage}.json",
            dict(stage=stage, protocol_sha256=sha(root / "protocol_lock.json"), jobs=jobs),
        )
    return sha(root / "protocol_lock.json")


def submit(root, stage, mode, dependency=None):
    # PrefixKV reserves three slots independently. A single Fox stage is active
    # at a time and remains capped at three, so aggregate admission is <= six.
    command = [
        "sbatch",
        "--parsable",
        "--job-name=mk-fox-guard-v2",
        "--account=bii_dsc_community",
        f"--output={root}/slurm/{stage}-{mode}-%A_%a.out",
        f"--error={root}/slurm/{stage}-{mode}-%A_%a.err",
    ]
    if mode == "worker":
        spec = root / f"jobs/{stage}.json"
        count = len(json.loads(spec.read_text())["jobs"])
        command += [
            "--partition=bii-gpu",
            "--gres=gpu:a100:1",
            "--reservation=bi_fox_dgx",
            "--cpus-per-task=12",
            "--mem=64G",
            "--time=01:00:00",
            f"--array=0-{count - 1}%3",
        ]
        command += [f"--dependency=afterany:{dependency}"] if dependency else []
        command += [
            str(SOURCE / "mosaickv/slurm/evicastkv_worker.sbatch"),
            str(SOURCE),
            str(root),
            stage,
            sha(spec),
        ]
    else:
        command += ["--partition=bii", "--cpus-per-task=2", "--mem=12G", "--time=00:10:00"]
        command += [f"--dependency=afterany:{dependency}"] if dependency else []
        command += [
            str(SOURCE / "mosaickv/slurm/fox_installed_guard_followup_v2.sbatch"),
            str(SOURCE),
            str(root),
            stage,
        ]
    job = subprocess.check_output(command, text=True, timeout=30).strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("scheduler returned no job ID")
    write_once(root / f"submissions/{stage}-{mode}-{job}.json", dict(command=command, job_id=job))
    if mode == "worker":
        from control_a100_schedule import tick

        policy = dict(
            timezone="America/New_York",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[job],
            per_lane_static_throttle=3,
            other_lane_reserved_slots=3,
        )
        write_once(root / f"scheduler/{job}-policy.json", policy)
        report = tick(policy, {}, root / f"scheduler/{job}-state.json", False)
        write_once(root / f"scheduler/{job}-registration.json", report)
    return job


def validate(root, stage):
    lock = load_lock(root)
    spec, workers, rows, errors = collect(root, stage)
    ids = lock["cohorts"]["coco"][
        "consumed_canary" if stage == "evidence_canary" else "development"
    ]["sample_ids"]
    expected = {
        (sid, m, None if cap is None else cap * 2**20, v) for sid in ids for m, cap, v in arms()
    }
    keys = {(r["sample_id"], r["method"], r["budget_bytes"], r["variant"]) for r in rows}
    if (
        errors
        or len(workers) != len(spec["jobs"])
        or len(rows) != len(expected)
        or keys != expected
    ):
        raise ValueError(f"incomplete matrix: {errors}")
    paired = pairing(rows)
    for row in rows:
        if row["emitted_tokens"] != 64 or row["forward_steps"] != 63:
            raise ValueError("output schedule changed")
        if row["budget_bytes"] is not None and row["initial_kv_bytes"] != row["budget_bytes"]:
            raise ValueError("physical cap not filled")
        for field in ("initial_kv_bytes", "final_kv_bytes", "peak_kv_bytes", "e2e_wall_seconds"):
            if not math.isfinite(row[field]) or row[field] <= 0:
                raise ValueError("missing physical/timing measurement")
        cap = None if row["budget_bytes"] is None else row["budget_bytes"] // 2**20
        config = canonical_config(
            resolved_config(lock, "coco", row["method"], cap, variant=row["variant"])
        )
        if digest(json.loads(Path(row["resolved_config_path"]).read_text())) != digest(config):
            raise ValueError("executed configuration mismatch")
        expected_method = (
            "mosaickv_exact__evidence_output_guarded_v1"
            if row["variant"] == VARIANT
            else row["method"]
        )
        if row["effective_method"] != expected_method:
            raise ValueError("unregistered fallback or effective method")
        if row["variant"] == VARIANT:
            trace = json.loads(Path(row["trace_path"]).read_text())
            activity = trace["experimental_evidence_policy"]
            if activity["grouped_query_heads_averaged"] or not activity.get(
                "installed_output_guard"
            ):
                raise ValueError("missing unpooled installed-selection guard")
            if any(
                r["installed_error"] > r["anchor_error"] + 1e-7
                for r in activity["installed_output_guard"]["routes"]
            ):
                raise ValueError("installed reconstruction worsened")
    return lock, rows, paired


def analyze(root, stage):
    out = root / "analysis" / stage
    try:
        lock, rows, paired = validate(root, stage)
    except Exception as exc:
        write_once(
            out / "GATE.json",
            dict(status="failed_execution", error=str(exc), paper_campaign_authorized=False),
        )
        release_prefix_slots(root)
        return
    export_descriptive(root, stage, rows)
    write_once(out / "pairing.json", paired)
    if stage == "evidence_canary":
        write_once(
            out / "GATE.json",
            dict(
                status="passed_consumed_correctness",
                rows=len(rows),
                not_quality_evidence=True,
                source_sha=lock["source_sha"],
            ),
        )
        job = submit(root, "installed_guard_development", "worker")
        callback = submit(root, "installed_guard_development", "analyze", job)
        write_once(
            root / "decisions/after-canary.json",
            dict(gpu_array=job, cpu_analysis=callback, lane_limit=3),
        )
        return
    points, groups = {}, {}
    for row in rows:
        key = f"{row['method']}/{row['variant']}/{row['budget_bytes']}"
        points.setdefault(key, []).append(
            MetricSample(row["sample_id"], row["answer"], tuple(row["references"]))
        )
        groups[row["sample_id"]] = tuple(row["media_group_ids"])
    comparisons = [
        (f"{baseline}/complete/{cap * 2**20}", f"mosaickv_exact/{VARIANT}/{cap * 2**20}")
        for cap in (20, 40)
        for baseline in ("prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact")
    ]
    stats = corpus_cider_statistics(
        points, comparisons, groups, resamples=10000, workers=2, family_size=6, seed=2026102902
    )
    write_once(out / "quality_statistics.json", stats)
    positive = all(c["simultaneous_ci"][0] > 0 for c in stats["contrasts"])
    write_once(
        out / "GATE.json",
        dict(
            status="complete_development_only",
            source_sha=lock["source_sha"],
            rows=len(rows),
            positive_signal=positive,
            confirmation_authorized=False,
            actual_quality_not_guaranteed_by_prompt_guard=True,
            latency="descriptive; preserve all slower outcomes",
        ),
    )
    write_once(
        out / "cohort_seal.json",
        dict(
            sample_ids=lock["cohorts"]["coco"]["development"]["sample_ids"],
            identities_consumed=True,
            no_outcome_driven_retries=True,
        ),
    )
    if positive:
        write_once(
            out / "frozen_candidate.json",
            dict(
                source_sha=lock["source_sha"],
                variant=VARIANT,
                protocol_sha256=sha(root / "protocol_lock.json"),
                independent_confirmation_authorized=False,
                blocker="historical_media_group_certificate_incomplete",
            ),
        )
    (out / "report.md").write_text(
        "# Installed output-guard development\n\n"
        f"Complete requests: {len(rows)}. Positive registered signal: {positive}.\n"
        "See quality_statistics.json for all scores and adjusted paired intervals.\n"
        "No independent confirmation is claimed. Quality-only timings are descriptive.\n"
        "No numerical or statistical threshold has been relaxed.\n"
    )
    release_prefix_slots(root)


def release_prefix_slots(root):
    """Terminal afterany callback, never an intermediate-canary release."""
    from control_a100_schedule import gpu_count, read_queue

    revision = clean_source(SOURCE)
    queue = read_queue()
    if any(r["name"].startswith("mk-fox-guard-v2") and gpu_count(r["gres"]) > 0 for r in queue):
        raise ValueError("Fox GPU work remains queued/running; six-slot handoff forbidden")
    write_once(
        root / "decisions/terminal.json", dict(source_sha=revision, no_more_fox_gpu_jobs=True)
    )
    plan_path = PREFIX_ROOT / "plan.json"
    if not plan_path.exists() or json.loads(plan_path.read_text())["source_sha"] != revision:
        raise ValueError("cannot release an unregistered PrefixKV campaign")
    registered = {
        json.loads(p.read_text())["job"] for p in (PREFIX_ROOT / "submissions").glob("*.json")
    }
    active = {
        r["array"]
        for r in queue
        if r["array"] in registered
        and r["name"].startswith("mk-prefix-qv2")
        and gpu_count(r["gres"]) > 0
    }
    results = []
    for job in sorted(active):
        run = subprocess.run(
            ["scontrol", "update", f"JobId={job}", "ArrayTaskThrottle=6"],
            capture_output=True,
            text=True,
            check=False,
        )
        results.append(dict(job_id=job, returncode=run.returncode, stderr=run.stderr))
    write_once(root / "decisions/prefix-slot-release.json", dict(results=results, maximum_a100s=6))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "start", "analyze"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", default="evidence_canary")
    a = p.parse_args()
    if not a.root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("scratch-only outputs")
    if a.mode == "prepare":
        print(prepare(a.root))
    elif a.mode == "start":
        load_lock(a.root)
        job = submit(a.root, "evidence_canary", "worker")
        callback = submit(a.root, "evidence_canary", "analyze", job)
        write_once(
            a.root / "decisions/start.json",
            dict(gpu_array=job, cpu_analysis=callback, lane_limit=3),
        )
    else:
        analyze(a.root, a.stage)
