#!/usr/bin/env python3
"""Bounded October-29 follow-up: no outcome-driven retries or fresh claims.

PrefixKV calibration is new to explicit tuning ledgers but historical-native
benchmark reuse is disclosed. A successful calibration freezes a candidate;
it cannot authorize independent confirmation without its separate media gate.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import prefixkv_isoquality_development as native

from mosaickv.config import RunConfig, canonical_config, config_sha256, load_config
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PRIOR = Path("/scratch/djy8hg/runs/mosaickv/prefix-isoquality-development-20261028-retry1")
CAPS = (80, 120, 160)
TASKS = ("llava_description", "mmvet_prefixkv")
METHODS = ("prompt_attention_topk", "prefixkv_reimpl", "mosaickv_exact")
FOX_ROOT = Path("/scratch/djy8hg/runs/evicastkv/fox-installed-guard-20261029-v2")


def prefix_concurrency():
    terminal = FOX_ROOT / "decisions/terminal.json"
    if terminal.exists():
        record = native.read(terminal)
        if (
            record.get("source_sha") == clean_source(SOURCE)
            and record.get("no_more_fox_gpu_jobs") is True
        ):
            return 6
    return 3


def exclude_known_media_groups(samples, blocked):
    """Input-only transitive media exclusion, not ID-only alias checking."""
    from mosaickv.evaluation.fox_statistics import media_components

    ids = [str(s["sample_id"]) for s in samples]
    groups = {str(s["sample_id"]): tuple(m["sha256"] for m in s["media"]) for s in samples}
    expanded = set(blocked)
    for component in media_components(ids, groups):
        members = {ids[i] for i in component}
        if members & expanded:
            expanded.update(members)
    return expanded


def prepare(root):
    revision = clean_source(SOURCE)
    old = native.read(PRIOR / "plan.json")
    if (
        sha(PRIOR / "plan.json")
        != "4e52c33d2dc21b6b7c3e144ef6ca98c4a086b49d53a354bce762e1f3f5f02508"
    ):
        raise ValueError("reviewed parent plan changed")
    for path, expected in old["bound_files"].items():
        if sha(Path(path)) != expected:
            raise ValueError("historical fidelity/configuration binding changed")
    # No silent baseline or candidate change in the PrefixKV lane.
    previous = Path("/scratch/djy8hg/worktrees/mosaickv-prefix-provenance-retry-v1")
    for name in (*native.POLICY_FILES, "mosaickv/src/mosaickv/selection/selector.py"):
        if sha(SOURCE / name) != sha(previous / name):
            raise ValueError("baseline/selector changed since validated prior execution")
    if root.exists():
        raise FileExistsError("immutable output root required")
    bound = {str(PRIOR / "plan.json"): sha(PRIOR / "plan.json")}
    blocked, tuning_bindings = native.exclusions()
    bound.update(tuning_bindings)
    jobs, cohorts = [], {}
    root.mkdir(parents=True)
    for task in TASKS:
        parent_path = Path(
            native.read(
                old["jobs"][next(j["index"] for j in old["jobs"] if j["task"] == task)]["ledger"]
            )["parent_ledger"]
        )
        data = native.read(parent_path)
        bound[str(parent_path)] = sha(parent_path)
        excluded = blocked[task] | set(old["cohorts"][task]["sample_ids"])
        excluded = exclude_known_media_groups(data["samples"], excluded)
        ids = native.choose_ids(data["samples"], excluded, count=32, seed=2026102901)
        by_id = {str(s["sample_id"]): s for s in data["samples"]}
        cohorts[task] = dict(
            sample_ids=ids,
            explicitly_excluded=sorted(excluded),
            globally_untouched=False,
            classification="retrospective_native_development",
        )
        configs = {}
        common = None
        for method in ("full_kv", *METHODS):
            old_job = next(j for j in old["jobs"] if j["task"] == task and j["method"] == method)
            cfg = canonical_config(load_config(old_job["config"]))
            if common is None:
                common = native.controls(cfg)
            if native.controls(cfg) != common:
                raise ValueError("model/processor/generation/block controls differ")
            configs[method] = cfg
            if method == "prefixkv_reimpl":
                profile = Path(cfg["prefixkv"]["profile_path"])
                bound[str(profile)] = sha(profile)
        for chunk in range(4):
            selected = ids[chunk * 8 : (chunk + 1) * 8]
            ledger = root / f"ledgers/{task}-{chunk}.json"
            payload = dict(
                data,
                samples=[by_id[sid] for sid in selected],
                parent_ledger=str(parent_path),
                parent_ledger_sha256=sha(parent_path),
            )
            write_once(ledger, payload)
            bound[str(ledger)] = sha(ledger)
            reference = root / f"references/{task}-{chunk}.json"
            for method in ("full_kv", *METHODS):
                for cap in (None,) if method == "full_kv" else CAPS:
                    cfg = copy.deepcopy(configs[method])
                    if cap is not None:
                        cfg["cache"]["budget_value"] = cap * 2**20
                    parsed = RunConfig.from_mapping(cfg)
                    config = root / f"configs/{task}-{method}-{cap}.json"
                    if not config.exists():
                        write_once(config, canonical_config(parsed))
                    bound[str(config)] = sha(config)
                    index = len(jobs)
                    jobs.append(
                        dict(
                            index=index,
                            task=task,
                            method=method,
                            cap_mib=cap,
                            chunk=chunk,
                            config=str(config),
                            config_sha=config_sha256(parsed),
                            ledger=str(ledger),
                            ledger_sha=sha(ledger),
                            sample_ids=selected,
                            reference=str(reference),
                            output=str(root / f"outputs/{index:03d}"),
                        )
                    )
    protocol = dict(
        tasks=list(TASKS),
        methods=list(METHODS),
        initial_caps_mib=list(CAPS),
        samples_per_task=32,
        output_tokens=1024,
        quality_warmups=0,
        timing_warmups=5,
        timed_repetitions=20,
        seed=2026102901,
        quality_margins={m: 0.0 for m in native.METRICS},
        bootstrap_replicates=10000,
        bootstrap_seed=2026102904,
        classification="retrospective_native_development_not_independent_confirmation",
        resources=dict(prefix_concurrency=3, fox_concurrency=3, aggregate_a100s=6),
    )
    write_once(
        root / "plan.json",
        dict(
            schema_version=2,
            source_sha=revision,
            protocol=protocol,
            bound_files=bound,
            cohorts=cohorts,
            jobs=jobs,
            paper_campaign_authorized=False,
            quality_requests=640,
        ),
    )
    return sha(root / "plan.json")


def inputs(root, plan_sha):
    if sha(root / "plan.json") != plan_sha:
        raise ValueError("plan hash changed")
    plan = native.read(root / "plan.json")
    clean_source(SOURCE, plan["source_sha"])
    p = plan["protocol"]
    if (
        p["initial_caps_mib"] != list(CAPS)
        or p["samples_per_task"] != 32
        or p["output_tokens"] != 1024
        or p["quality_warmups"] != 0
        or p["quality_margins"] != {m: 0.0 for m in native.METRICS}
    ):
        raise ValueError("registered quality controls changed")
    for path, expected in plan["bound_files"].items():
        if sha(Path(path)) != expected:
            raise ValueError(f"bound file changed: {path}")
    return plan


def submit(root, plan_sha, mode, indices=None, dependency=None):
    ceiling = prefix_concurrency()
    command = [
        "sbatch",
        "--parsable",
        "--job-name=mk-prefix-qv2",
        "--account=bii_dsc_community",
        f"--output={root}/slurm/{mode}-%A_%a.out",
        f"--error={root}/slurm/{mode}-%A_%a.err",
    ]
    if mode in {"worker", "timing"}:
        command += [
            "--partition=bii-gpu",
            "--gres=gpu:a100:1",
            "--reservation=bi_fox_dgx",
            "--cpus-per-task=12",
            "--mem=64G",
            "--time=02:00:00" if mode == "timing" else "--time=01:00:00",
            "--array=" + ",".join(map(str, indices)) + f"%{ceiling}",
        ]
    else:
        command += ["--partition=bii", "--cpus-per-task=2", "--mem=12G", "--time=00:10:00"]
    if dependency:
        command += ["--dependency=afterany:" + dependency]
    command += [
        str(SOURCE / "mosaickv/slurm/registered_quality_followup_v2.sbatch"),
        str(SOURCE),
        str(root),
        plan_sha,
        mode,
    ]
    job = subprocess.check_output(command, text=True, timeout=30).strip().split(";")[0]
    if not job.isdigit():
        raise ValueError("no job ID from Slurm")
    write_once(root / f"submissions/{mode}-{job}.json", dict(job=job, command=command))
    if mode in {"worker", "timing"}:
        from control_a100_schedule import tick

        policy = dict(
            timezone="America/New_York",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[job],
            per_lane_static_throttle=ceiling,
            other_lane_reserved_slots=6 - ceiling,
        )
        write_once(root / f"scheduler/{job}-policy.json", policy)
        report = tick(policy, {}, root / f"scheduler/{job}-state.json", False)
        write_once(root / f"scheduler/{job}-registration.json", report)
    return job


def completed(root, plan, plan_sha, indices):
    results = []
    for i in indices:
        record = native.completed_worker(root, plan, plan_sha, i)
        if record is None or record["status"] != "completed":
            raise ValueError(f"missing/failed worker {i}; no automatic scientific retry")
        results.append(record)
    return results


def advance(root, plan_sha, stage):
    plan = inputs(root, plan_sha)
    write_once(
        root / f"decisions/{stage}-started.json", dict(stage=stage, source_sha=plan["source_sha"])
    )
    refs = [j["index"] for j in plan["jobs"] if j["method"] == "full_kv"]
    canary = [
        j["index"]
        for j in plan["jobs"]
        if j["method"] != "full_kv" and j["chunk"] == 0 and j["cap_mib"] == 120
    ]
    try:
        if stage == "start":
            if native.read(root / "verification/cpu-tests.json")["status"] != "passed":
                raise ValueError("complete clean-source CPU suite must pass before GPUs")
            indices, callback = refs, "after-references"
        elif stage == "after-references":
            completed(root, plan, plan_sha, refs)
            indices, callback = canary, "after-canary"
        else:
            records = completed(root, plan, plan_sha, canary)
            # This canary is an execution gate, not an outcome filter.
            for task in TASKS:
                selected = [r for r in records if r["task"] == task]
                for sid in plan["cohorts"][task]["sample_ids"][:8]:
                    rows = [
                        x for r in selected for x in r["record"]["rows"] if x["sample_id"] == sid
                    ]
                    if len(rows) != 3 or any(
                        len({r[f] for r in rows}) != 1 for f in native.STORAGE
                    ):
                        raise ValueError("canary physical storage differs at shared cap")
            indices = [j["index"] for j in plan["jobs"] if j["index"] not in refs + canary]
            callback = "analyze"
        job = submit(root, plan_sha, "worker", indices)
        analysis = submit(root, plan_sha, callback, dependency=job)
        write_once(
            root / f"decisions/{stage}.json",
            dict(gpu_array=job, cpu_callback=analysis, aggregate_limit=6, lane_limit=3),
        )
    except Exception as exc:
        write_once(
            root / f"decisions/{stage}-blocked.json",
            dict(status="blocked", error=str(exc), no_paper_campaign=True),
        )
        raise


def analyze(root, plan_sha):
    plan = inputs(root, plan_sha)
    rows, observations, records = [], {}, []
    failed = []
    for j in plan["jobs"]:
        try:
            w = completed(root, plan, plan_sha, [j["index"]])[0]
            records.append((j, w))
            frame = pd.read_parquet(w["record"]["metric_observations_parquet_output"])
            key = j["task"], j["method"], j["cap_mib"]
            observations.setdefault(key, []).append(frame[frame.metric_scope == "sample"])
            rows.extend(
                dict(**r, task=j["task"], configured_method=j["method"], cap_mib=j["cap_mib"])
                for r in w["record"]["rows"]
            )
        except Exception as exc:
            failed.append(dict(index=j["index"], sample_ids=j["sample_ids"], error=str(exc)))
    out = root / "analysis"
    write_once(
        out / "failures.json", dict(failures=failed, planned_rows=640, completed_rows=len(rows))
    )
    if failed:
        write_once(
            out / "GATE.json",
            dict(status="failed_execution", failures=failed, confirmation_authorized=False),
        )
        return
    frames = {k: pd.concat(v, ignore_index=True) for k, v in observations.items()}
    raw = pd.DataFrame(rows)
    summaries = []
    for (task, method, cap), frame in frames.items():
        for metric in native.METRICS:
            f = frame[frame.metric_name == metric]
            if len(f) != 32 or set(f.sample_id) != set(plan["cohorts"][task]["sample_ids"]):
                raise ValueError("missing/duplicate metric identities")
            value = (
                float(f.metric_value.mean())
                if metric == "rouge_l_f1"
                else math.exp(float(f.numerator.sum() / f.denominator.sum()))
            )
            summaries.append(
                dict(task=task, method=method, cap_mib=cap, metric=metric, value=value, n=32)
            )
    exact = True
    for task in TASKS:
        for cap in CAPS:
            for _, group in raw[(raw.task == task) & (raw.cap_mib == cap)].groupby("sample_id"):
                exact &= len(group) == 3 and all(group[f].nunique() == 1 for f in native.STORAGE)
            checks = [(j, w) for j, w in records if j["task"] == task and j["cap_mib"] == cap]
            for sid in plan["cohorts"][task]["sample_ids"]:
                for phase in ("generation", "teacher_forced"):
                    values = [
                        c["storage_bytes"]
                        for _, w in checks
                        for c in w["phase_checks"]
                        if c["sample_id"] == sid and c["phase"] == phase
                    ]
                    exact &= len(values) == 3 and all(v == values[0] for v in values)
    contrasts = []
    for task in TASKS:
        for cap in (80, 120):
            for baseline in ("prefixkv_reimpl", "prompt_attention_topk"):
                c, r = frames[task, "mosaickv_exact", cap], frames[task, baseline, 160]
                for metric in native.METRICS:
                    stats = native.paired_metrics(
                        c[c.metric_name == metric],
                        r[r.metric_name == metric],
                        metric,
                        family=16,
                        seed=plan["protocol"]["bootstrap_seed"],
                    )
                    contrasts.append(
                        dict(
                            task=task,
                            candidate_cap_mib=cap,
                            baseline=baseline,
                            reference_cap_mib=160,
                            metric_name=metric,
                            **stats,
                        )
                    )
    candidates = []
    reductions = []
    matched = []
    for task in TASKS:
        for cap in CAPS:
            c = frames[task, "mosaickv_exact", cap]
            for baseline in ("prefixkv_reimpl", "prompt_attention_topk"):
                r = frames[task, baseline, cap]
                for metric in native.METRICS:
                    stats = native.paired_metrics(
                        c[c.metric_name == metric],
                        r[r.metric_name == metric],
                        metric,
                        family=24,
                        seed=plan["protocol"]["bootstrap_seed"],
                    )
                    matched.append(
                        dict(
                            task=task,
                            cap_mib=cap,
                            baseline=baseline,
                            metric_name=metric,
                            label="same_measured_byte_budget",
                            **stats,
                        )
                    )
        for cap in (80, 120):
            can = raw[
                (raw.task == task)
                & (raw.configured_method == "mosaickv_exact")
                & (raw.cap_mib == cap)
            ].sort_values("sample_id")
            ref = raw[
                (raw.task == task)
                & (raw.configured_method == "prefixkv_reimpl")
                & (raw.cap_mib == 160)
            ].sort_values("sample_id")
            if can.sample_id.tolist() != ref.sample_id.tolist():
                raise ValueError("cross-budget storage identities differ")
            values = {
                f: float((1 - can[f].to_numpy() / ref[f].to_numpy()).min()) for f in native.STORAGE
            }
            reductions.append(
                dict(
                    task=task,
                    candidate_cap_mib=cap,
                    reference_cap_mib=160,
                    minimum_paired_reductions=values,
                    passes=all(v > 0 for v in values.values()),
                )
            )
    for cap in (80, 120):
        primary = [
            c
            for c in contrasts
            if c["candidate_cap_mib"] == cap and c["baseline"] == "prefixkv_reimpl"
        ]
        memory = all(r["passes"] for r in reductions if r["candidate_cap_mib"] == cap)
        if exact and memory and len(primary) == 4 and all(c["passes"] for c in primary):
            candidates.append(cap)
    raw.to_parquet(out / "sample_rows.parquet", index=False)
    pd.DataFrame(summaries).to_parquet(out / "native_metrics.parquet", index=False)
    pd.DataFrame(contrasts).to_parquet(out / "cross_budget_intervals.parquet", index=False)
    pd.DataFrame(matched).to_parquet(out / "same_budget_intervals.parquet", index=False)
    write_once(out / "memory_reductions.json", reductions)
    gate = dict(
        status="complete_calibration_development_only",
        source_sha=plan["source_sha"],
        quality_memory_gate=bool(candidates),
        byte_matching=bool(exact),
        rows=len(raw),
        confirmation_authorized=False,
        paper_campaign_authorized=False,
        latency="descriptive_only_until_separate_repeated_timing",
    )
    if candidates:
        cap = min(candidates)
        write_once(
            out / "frozen_candidate.json",
            dict(
                source_sha=plan["source_sha"],
                candidate="mosaickv_exact",
                cap_mib=cap,
                reference="prefixkv_selection_only_reimpl",
                reference_cap_mib=160,
                nearest_competitor_claim=all(
                    c["passes"] for c in contrasts if c["candidate_cap_mib"] == cap
                ),
                provenance="calibration_frozen_not_confirmed",
                plan_sha256=plan_sha,
                independent_confirmation_blocker="source_media_disjointness_not_yet_certified",
            ),
        )
    write_once(out / "GATE.json", gate)
    text = [
        "# PrefixKV 32-example development follow-up",
        "",
        "All caps use identical controls. Cross-budget contrasts are NOT byte-matched.",
        "Historical native benchmark reuse is disclosed; no independent confirmation claim.",
        "",
        pd.DataFrame(summaries).to_string(index=False),
        "",
        pd.DataFrame(contrasts).to_string(index=False),
        "",
        "Latency values in quality rows are descriptive. FullKV is a quality reference only.",
        "A failed quality gate stops expansion; a frozen candidate requires all primary gates.",
    ]
    (out / "report.md").write_text("\n".join(text) + "\n")
    # Fixed, descriptive timing study even on a negative calibration: no
    # outcome-dependent choice of method or operating point, no FullKV timing.
    job = submit(root, plan_sha, "timing", [0, 1])
    callback = submit(root, plan_sha, "analyze-timing", dependency=job)
    write_once(
        root / "decisions/timing.json",
        dict(gpu_array=job, cpu_callback=callback, preregistered_120mib_comparison=True),
    )


def timing(root, plan_sha, index):
    import numpy as np
    import torch
    from evicastkv_campaign import environment

    from mosaickv.adapters.huggingface import load_hf_adapter
    from mosaickv.backends.hf_runtime import HuggingFaceMosaicKVModel
    from mosaickv.evaluation.messages import MediaItem, build_multimodal_messages
    from mosaickv.evaluation.model import EvaluationRequest
    from mosaickv.evaluation.paper_native_lanes import _media_payload, load_paper_native_ledger
    from mosaickv.evaluation.tasks import default_task_registry
    from mosaickv.evaluation.worker_support import settle
    from mosaickv.hf_cli import _model_kwargs
    from mosaickv.measurements.telemetry import capture_gpu_environment

    plan = inputs(root, plan_sha)
    if torch.cuda.device_count() != 1 or "A100" not in torch.cuda.get_device_name(0):
        raise ValueError("one allocated A100 is required")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("deterministic workspace missing")
    task = TASKS[index]
    jobs = [
        next(
            j
            for j in plan["jobs"]
            if j["task"] == task
            and j["chunk"] == 0
            and j["method"] == method
            and j["cap_mib"] == cap
        )
        for method, cap in [
            ("prompt_attention_topk", 120),
            ("prefixkv_reimpl", 120),
            ("mosaickv_exact", 120),
            ("prefixkv_reimpl", 160),
        ]
    ]
    cfg = load_config(jobs[0]["config"])
    torch.set_num_threads(12)
    torch.manual_seed(cfg.execution.seed)
    torch.use_deterministic_algorithms(True)
    output = root / f"timing/{index:02d}"
    output.mkdir(parents=True)
    rows = []
    result = dict(
        task=task,
        source_sha=plan["source_sha"],
        plan_sha256=plan_sha,
        scope="one_preregistered_input_technical_repetitions_not_population_latency",
    )
    started = time.monotonic()
    try:
        ledger = load_paper_native_ledger(
            jobs[0]["ledger"],
            expected_sha256=jobs[0]["ledger_sha"],
            task=default_task_registry().resolve(task),
            dataset_revision=cfg.dataset.revision,
        )
        sample = ledger.samples[0]
        messages = build_multimodal_messages(
            sample.prompt, tuple(MediaItem(m.kind, _media_payload(m)) for m in sample.media)
        )
        adapter = load_hf_adapter(
            cfg.model.id,
            revision=cfg.model.revision,
            model_kwargs=_model_kwargs(cfg, local_files_only=True),
            processor_kwargs={"local_files_only": True},
        )
        result["environment"] = environment()
        result["gpu_before"] = asdict(capture_gpu_environment(torch))
        write_once(
            output / "manifest.json",
            dict(**result, configs={j["config"]: sha(Path(j["config"])) for j in jobs}),
        )
        order = np.random.default_rng(2026102903 + index)
        for repeat in range(-5, 20):
            for ordinal, k in enumerate(order.permutation(len(jobs))):
                if time.monotonic() - started > 6500:
                    raise TimeoutError(
                        "timing safety wall budget exhausted; retain completed trials"
                    )
                j = jobs[int(k)]
                config = load_config(j["config"])
                settle(adapter)
                label = f"r{repeat}-{j['method']}-{j['cap_mib']}"
                runtime = HuggingFaceMosaicKVModel(
                    adapter, config, trace_directory=output / f"traces/{label}"
                )
                value = runtime.generate(
                    EvaluationRequest(label, sample.sample_id, task, messages, {})
                )
                row = dict(
                    sample_id=sample.sample_id,
                    method=j["method"],
                    cap_mib=j["cap_mib"],
                    repeat=repeat,
                    order=ordinal,
                    **asdict(value.metrics),
                )
                if (
                    row["generated_tokens"] != 1024
                    or row[native.STORAGE[0]] != j["cap_mib"] * 2**20
                ):
                    raise ValueError("timing output length or physical cache changed")
                row["manifest_path"] = str(output / "manifest.json")
                row["answer"] = value.answer
                write_once(output / f"trials/{label}.json", row)
                if repeat >= 0:
                    rows.append(row)
                del runtime, value
        result.update(
            status="completed", rows=rows, gpu_after=asdict(capture_gpu_environment(torch))
        )
    except Exception as exc:
        result.update(status="failed", error=str(exc), rows=rows)
        raise
    finally:
        result["wall_seconds"] = time.monotonic() - started
        write_once(root / f"timing/workers/{index:02d}.json", result)


def analyze_timing(root, plan_sha):
    import numpy as np

    inputs(root, plan_sha)
    rows, failures = [], []
    for index, task in enumerate(TASKS):
        path = root / f"timing/workers/{index:02d}.json"
        record = native.read(path) if path.exists() else dict(status="missing")
        if record["status"] != "completed":
            failures.append(dict(task=task, record=record))
        rows.extend(dict(task=task, **r) for r in record.get("rows", []))
    out = root / "timing/analysis"
    out.mkdir(parents=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(out / "trials.parquet", index=False)
    summary = []
    if not failures:
        for task in TASKS:
            data = frame[frame.task == task]
            candidate = data[(data.method == "mosaickv_exact") & (data.cap_mib == 120)].sort_values(
                "repeat"
            )
            for method, cap in [
                ("prefixkv_reimpl", 120),
                ("prompt_attention_topk", 120),
                ("prefixkv_reimpl", 160),
            ]:
                reference = data[(data.method == method) & (data.cap_mib == cap)].sort_values(
                    "repeat"
                )
                if len(candidate) != 20 or len(reference) != 20:
                    raise ValueError("timed trial denominator changed")
                interval = native.latency_interval(
                    candidate.end_to_end_time.to_numpy()[None, :],
                    reference.end_to_end_time.to_numpy()[None, :],
                    family_size=6,
                )
                summary.append(
                    dict(
                        task=task,
                        reference=method,
                        reference_cap_mib=cap,
                        candidate_cap_mib=120,
                        label="same_byte" if cap == 120 else "cross_budget_not_byte_matched",
                        candidate_p50=float(np.median(candidate.end_to_end_time)),
                        reference_p50=float(np.median(reference.end_to_end_time)),
                        candidate_p95=float(np.quantile(candidate.end_to_end_time, 0.95)),
                        reference_p95=float(np.quantile(reference.end_to_end_time, 0.95)),
                        paired_technical_interval=interval,
                        scope="one_input_per_task_not_population_latency",
                    )
                )
    write_once(
        out / "report.json",
        dict(
            status="failed" if failures else "completed",
            failures=failures,
            summary=summary,
            no_fullkv_latency_comparison=True,
            quality_gate_independent=native.read(root / "analysis/GATE.json"),
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "mode",
        choices=[
            "prepare",
            "start",
            "after-references",
            "after-canary",
            "worker",
            "analyze",
            "timing",
            "analyze-timing",
            "cpu",
        ],
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan-sha")
    p.add_argument("--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    a = p.parse_args()
    if not a.root.resolve().is_relative_to(Path("/scratch/djy8hg").resolve()):
        raise ValueError("scratch-only outputs")
    if a.mode == "prepare":
        print(prepare(a.root))
    elif not a.plan_sha:
        p.error("plan hash required")
    elif a.mode == "worker":
        native.worker(a.root, a.plan_sha, a.index, plan_loader=inputs)
    elif a.mode == "analyze":
        analyze(a.root, a.plan_sha)
    elif a.mode == "timing":
        timing(a.root, a.plan_sha, a.index)
    elif a.mode == "analyze-timing":
        analyze_timing(a.root, a.plan_sha)
    elif a.mode == "cpu":
        from evicastkv_review_closure_v2 import cpu_suite

        inputs(a.root, a.plan_sha)
        cpu_suite(a.root / "verification")
    else:
        advance(a.root, a.plan_sha, a.mode)


if __name__ == "__main__":
    main()
