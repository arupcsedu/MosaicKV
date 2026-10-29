#!/usr/bin/env python3
"""Finite consumed-input lifetime V2 study; no automatic paper promotion."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import evicastkv_lifetime_history as previous
from control_a100_schedule import tick

from mosaickv.adapters.huggingface.incremental_history_cache import incremental_history_budget
from mosaickv.config import (
    BoundedDecodeConfig,
    IncrementalHistoryConfig,
    RunConfig,
    canonical_config,
    config_sha256,
)
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]


def prepare(root):
    source_sha = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new immutable root required")
    parent_sha = previous.prepare(root / "inputs")
    plan = previous.read(root / "inputs/plan.json")
    for job in plan["jobs"]:
        old = job["arms"]
        full, prefix = old[:2]
        mosaic = RunConfig.from_mapping(old[-1]["config"])
        topk = RunConfig.from_mapping(
            next(a["config"] for a in old if a["label"].startswith("prompt_attention_topk-"))
        )
        arms = [full, prefix]
        for cap in (192, 208):
            for method_base in (topk, mosaic):
                policies = ("sink_recent", "recent_importance") if cap == 208 else ("sink_recent",)
                for policy in policies:
                    cfg = replace(
                        method_base,
                        cache=replace(method_base.cache, budget_value=cap * 2**20),
                        bounded_decode=BoundedDecodeConfig(
                            128,
                            4,
                            incremental_history=IncrementalHistoryConfig(
                                16,
                                64,
                                policy,
                                0.95,
                                fixed_prompt_reservation=True,
                                borrow_prefill_snapshot=True,
                                importance_refresh_tokens=16
                                if policy == "recent_importance"
                                else 1,
                            ),
                        ),
                    )
                    arms.append(
                        dict(
                            label=f"{cfg.method.value}-{policy}-{cap}mib-v2",
                            config=canonical_config(cfg),
                            config_sha=config_sha256(cfg),
                        )
                    )
        # Same prompt capacity, append-only history: separates prompt quality
        # from history eviction. Geometry is checked against measured traces.
        budget = incremental_history_budget(
            208 * 2**20, [16384] * 32, 128, 1023, fixed_prompt_reservation=True
        )
        anchor = replace(
            mosaic,
            bounded_decode=None,
            cache=replace(mosaic.cache, budget_value=budget.prompt_budget_bytes),
        )
        arms.append(
            dict(
                label="mosaickv_exact-append-anchor-137.5mib-v2",
                config=canonical_config(anchor),
                config_sha=config_sha256(anchor),
            )
        )
        # Isolate borrowing from model/selector behavior, at identical capacity.
        source = RunConfig.from_mapping(
            next(a["config"] for a in arms if a["label"] == "mosaickv_exact-sink_recent-208mib-v2")
        )
        clone = replace(
            source,
            bounded_decode=replace(
                source.bounded_decode,
                incremental_history=replace(
                    source.bounded_decode.incremental_history,
                    borrow_prefill_snapshot=False,
                ),
            ),
        )
        arms.append(
            dict(
                label="mosaickv_exact-sink_recent-208mib-cloned-control-v2",
                config=canonical_config(clone),
                config_sha=config_sha256(clone),
            )
        )
        for method in ("prompt_attention_topk", "mosaickv_exact"):
            source = RunConfig.from_mapping(
                next(a["config"] for a in arms if a["label"] == f"{method}-sink_recent-208mib-v2")
            )
            protected = replace(
                source,
                cache=replace(
                    source.cache, physical_budget_mandatory_policy="question_and_boundaries"
                ),
            )
            arms.append(
                dict(
                    label=f"{method}-sink_recent-208mib-question-v2",
                    config=canonical_config(protected),
                    config_sha=config_sha256(protected),
                )
            )
        assert len(arms) == 12
        job["arms"] = arms
    plan.update(
        source_sha=source_sha,
        protocol="incremental_lifetime_v2_consumed_only",
        parent_preparation_sha256=parent_sha,
        separate_memory_replay=True,
        shared_caps_mib=[192, 208],
        history_capacity=128,
        fixed_prompt_reservation=True,
        bounded_followup="at_most_remaining_14_consumed_inputs_only_if_both_metrics_and_memory_pass",
        paper_campaign_authorized=False,
        fresh_evaluation_authorized=False,
        maximum_worker_minutes=120,
        quality_margin_ppl=0.0,
        quality_margin_rouge=0.0,
        stage="two_consumed_diagnostics",
        automatic_consumed_followup=True,
    )
    write_once(root / "plan.json", plan)
    return sha(root / "plan.json")


def analyze(root, expected):
    import pandas as pd

    plan = previous.inputs(root, expected)
    rows, failures = [], []
    for job in plan["jobs"]:
        worker = root / f"workers/{job['index']}"
        if not (worker / "result.json").exists():
            failures.append(dict(worker=job["index"], error="missing result"))
            continue
        failures.extend(previous.read(worker / "result.json")["failures"])
        for arm in job["arms"]:
            p = worker / f"rows/{arm['label']}.json"
            if not p.exists():
                failures.append(dict(worker=job["index"], label=arm["label"], error="missing row"))
                continue
            r = previous.read(p)
            if r["status"] != "completed":
                continue
            if (
                r.get("memory_observer_token_parity") is not True
                or len(r.get("memory_audit_traces", {})) != 1
            ):
                failures.append(dict(label=r["label"], error="memory replay evidence missing"))
                continue
            for path, value in (r["traces"] | r["memory_audit_traces"]).items():
                if sha(Path(path)) != value:
                    raise ValueError("trace hash changed")
            rows.append(
                {
                    k: r[k]
                    for k in (
                        "task",
                        "sample_id",
                        "label",
                        "teacher_forced_ppl",
                        "rouge_l_f1",
                        "request_owned_peak_bytes",
                        "audited_request_owned_peak_bytes",
                        "pytorch_request_peak_allocated_bytes",
                        "memory_observer_token_parity",
                    )
                }
                | {k: r["metrics"][k] for k in (*previous.STORAGE, "end_to_end_time")}
                | {"nll_sum": r.get("nll_sum"), "evaluated_tokens": r.get("evaluated_tokens")}
            )
    if plan.get("reused_summary_path"):
        reused = pd.read_parquet(plan["reused_summary_path"])
        allowed = {a["label"] for a in plan["jobs"][0]["arms"]}
        rows.extend(reused[reused.label.isin(allowed)].to_dict(orient="records"))
    table = pd.DataFrame(rows)
    checks = []
    matched = []
    expected_rows = sum(len(j["arms"]) for j in plan["jobs"]) + plan.get("reused_rows", 0)
    if len(rows) == expected_rows and not failures:
        for task, group in table.groupby("task"):
            for cap in (192, 208):
                for policy in (
                    ("sink_recent", "recent_importance") if cap == 208 else ("sink_recent",)
                ):
                    pair = group[
                        group.label.isin(
                            [
                                f"{method}-{policy}-{cap}mib-v2"
                                for method in ("prompt_attention_topk", "mosaickv_exact")
                            ]
                        )
                    ]
                    if len(pair) == 0:
                        continue
                    same = all(
                        len(p) == 2 and all(p[k].nunique() == 1 for k in previous.STORAGE)
                        for _sample, p in pair.groupby("sample_id")
                    )
                    matched.append(
                        dict(
                            task=task, cap_mib=cap, policy=policy, equal_realized_storage=bool(same)
                        )
                    )
                    if not same:
                        failures.append(
                            dict(task=task, error="matched-cap realized storage mismatch")
                        )
            question = group[
                group.label.isin(
                    [
                        f"{method}-sink_recent-208mib-question-v2"
                        for method in ("prompt_attention_topk", "mosaickv_exact")
                    ]
                )
            ]
            if len(question):
                same = all(
                    len(p) == 2 and all(p[k].nunique() == 1 for k in previous.STORAGE)
                    for _sample, p in question.groupby("sample_id")
                )
                matched.append(
                    dict(
                        task=task,
                        cap_mib=208,
                        policy="sink_recent_question",
                        equal_realized_storage=bool(same),
                    )
                )
                if not same:
                    failures.append(dict(task=task, error="question pair storage mismatch"))
            for candidate in group[
                group.label.str.startswith("mosaickv_exact-")
                & ~group.label.str.contains("anchor|cloned")
            ].itertuples():
                baseline = group[
                    (group.label == "prefixkv_reimpl-native-control")
                    & (group.sample_id == candidate.sample_id)
                ].iloc[0]
                topk = group[
                    (
                        group.label
                        == candidate.label.replace("mosaickv_exact-", "prompt_attention_topk-")
                    )
                    & (group.sample_id == candidate.sample_id)
                ].iloc[0]
                checks.append(
                    dict(
                        task=task,
                        label=candidate.label,
                        sample_id=candidate.sample_id,
                        smaller_initial_final_peak=all(
                            getattr(candidate, k) < baseline[k] for k in previous.STORAGE
                        ),
                        ppl_point_not_worse=bool(
                            candidate.teacher_forced_ppl <= baseline.teacher_forced_ppl
                        ),
                        rouge_point_not_worse=bool(candidate.rouge_l_f1 >= baseline.rouge_l_f1),
                        ppl_not_worse_than_matched_topk=bool(
                            candidate.teacher_forced_ppl <= topk.teacher_forced_ppl
                        ),
                        rouge_not_worse_than_matched_topk=bool(
                            candidate.rouge_l_f1 >= topk.rouge_l_f1
                        ),
                        audited_owners_not_whole_process=True,
                    )
                )
    eligible = []
    for label in sorted({c["label"] for c in checks}):
        matching = [c for c in checks if c["label"] == label]
        if len(matching) == 2 and all(
            c["smaller_initial_final_peak"]
            and c["ppl_point_not_worse"]
            and c["rouge_point_not_worse"]
            and c["ppl_not_worse_than_matched_topk"]
            and c["rouge_not_worse_than_matched_topk"]
            for c in matching
        ):
            eligible.append(label)
    out = root / "analysis"
    if out.exists():
        raise FileExistsError("derived analysis is write-once")
    out.mkdir()
    table.to_parquet(out / "summary.parquet", index=False)
    aggregate = []
    if rows:
        import math

        for (task, label), g in table.groupby(["task", "label"]):
            if g.nll_sum.notna().all() and g.evaluated_tokens.notna().all():
                aggregate.append(
                    dict(
                        task=task,
                        label=label,
                        completed=len(g),
                        nll_sum=float(g.nll_sum.sum()),
                        token_count=int(g.evaluated_tokens.sum()),
                        corpus_ppl=math.exp(float(g.nll_sum.sum() / g.evaluated_tokens.sum())),
                        mean_rouge_l_f1=float(g.rouge_l_f1.mean()),
                        mean_latency_seconds=float(g.end_to_end_time.mean()),
                        no_cross_metric_average=True,
                    )
                )
    pd.DataFrame(aggregate).to_parquet(out / "aggregates.parquet", index=False)
    gate = dict(
        status="complete_consumed_diagnostic"
        if len(rows) == expected_rows and not failures
        else "failed",
        failures=failures,
        checks=checks,
        matched_budget_checks=matched,
        consumed_followup_eligible_labels=eligible if not failures else [],
        paper_campaign_authorized=False,
        fresh_evaluation_authorized=False,
        superiority_established=False,
        whole_request_KV_peak_certified=False,
        memory_replay_timing_excluded_from_primary_latency=True,
        quality_gate="zero_margin_point_checks_only_no_statistical_claim",
    )
    write_once(out / "GATE.json", gate)
    (out / "report.md").write_text(
        "# Lifetime V2 consumed diagnostic\n\n"
        "Consumed development inputs, not independent confirmation. "
        "Native controls are different operating points, not byte-matched. "
        "Within a cap, Top-K/EviCastKV share identical transport and realized budgets. "
        "PPL and generation use identical selected prompts. Memory replay is separate "
        "from latency and must reproduce tokens exactly. CUDA-allocator peak includes "
        "weights/activations; audited KV-owner peak is not whole-process memory.\n\n"
        + table.to_markdown(index=False)
        + "\n\n```json\n"
        + json.dumps(gate, indent=2)
        + "\n```\n"
    )
    write_once(
        out / "manifest.json",
        dict(
            source_sha=plan["source_sha"],
            plan_sha256=expected,
            inputs={str(p): sha(p) for p in (root / "workers").rglob("*.json")},
            outputs={p.name: sha(p) for p in out.iterdir()},
        ),
    )
    print(json.dumps(gate))
    if plan.get("automatic_consumed_followup"):
        if eligible and not failures:
            priority = [
                "mosaickv_exact-sink_recent-192mib-v2",
                "mosaickv_exact-sink_recent-208mib-v2",
                "mosaickv_exact-sink_recent-208mib-question-v2",
                "mosaickv_exact-recent_importance-208mib-v2",
            ]
            selected = next(label for label in priority if label in eligible)
            child = root / "consumed_followup"
            child_sha = prepare_followup(root, expected, selected, child)
            submit(child, child_sha)
            write_once(
                root / "recursive_decision.json",
                dict(
                    decision="bounded_consumed_followup_submitted",
                    selected=selected,
                    child=str(child),
                    child_sha256=child_sha,
                    no_fresh_or_paper_authorization=True,
                ),
            )
        else:
            write_once(
                root / "recursive_decision.json",
                dict(
                    decision="stop_negative_diagnostic",
                    no_further_gpu_work=True,
                    reason="both metrics must pass against native PrefixKV and matched TopK",
                    no_fresh_or_paper_authorization=True,
                ),
            )


def prepare_followup(parent_root, parent_sha, label, child):
    """Only the other seven already-consumed IDs per task; never new data."""
    parent = previous.inputs(parent_root, parent_sha)
    gate_path = parent_root / "analysis/GATE.json"
    gate = previous.read(gate_path)
    if label not in gate["consumed_followup_eligible_labels"] or gate["failures"]:
        raise ValueError("followup has no qualifying diagnostic")
    if child.exists():
        raise FileExistsError("write-once child root")
    bound = dict(parent["bound_files"])
    bound[str(parent_root / "plan.json")] = parent_sha
    bound[str(gate_path)] = sha(gate_path)
    reused = parent_root / "analysis/summary.parquet"
    bound[str(reused)] = sha(reused)
    jobs = []
    for job in parent["jobs"]:
        samples = previous.read(job["ledger"])["samples"]
        selected = [
            a
            for a in job["arms"]
            if a["label"]
            in {
                "full_kv-native-control",
                "prefixkv_reimpl-native-control",
                label,
                label.replace("mosaickv_exact-", "prompt_attention_topk-"),
            }
        ]
        if len(selected) != 4 or len(samples) != 8:
            raise ValueError("followup requires four arms and reviewed eight-ID consumed ledgers")
        for sample in samples[1:]:
            for media in sample["media"]:
                p = Path(media["path"])
                if sha(p) != media["sha256"]:
                    raise ValueError("consumed media changed")
                bound[str(p)] = media["sha256"]
            jobs.append(
                dict(job, index=len(jobs), sample_id=str(sample["sample_id"]), arms=selected)
            )
    if len(jobs) > 14:
        raise ValueError("bounded followup exceeded")
    write_once(
        child / "plan.json",
        dict(
            parent,
            jobs=jobs,
            bound_files=bound,
            automatic_consumed_followup=False,
            stage="remaining_consumed_inputs",
            fixed_candidate=label,
            reused_summary_path=str(reused),
            reused_rows=8,
            source_sha=clean_source(SOURCE),
            fresh_evaluation_authorized=False,
            paper_campaign_authorized=False,
        ),
    )
    return sha(child / "plan.json")


def submit(
    root, expected, *, script_name="evicastkv_lifetime_v2.sbatch", worker_minutes=120, memory_gb=64
):
    if type(worker_minutes) is not int or not 1 <= worker_minutes <= 120:
        raise ValueError("worker duration must be within the registered two-hour maximum")
    plan = previous.inputs(root, expected)
    (root / "slurm").mkdir()
    script = str(SOURCE / "mosaickv/slurm" / script_name)
    args = [str(SOURCE), str(root), expected]
    gpu = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--account=bii_dsc_community",
                "--partition=bii-gpu",
                "--reservation=bi_fox_dgx",
                "--job-name=mk-evicast-life-v2",
                "--nodes=1",
                "--ntasks=1",
                "--gres=gpu:a100:1",
                "--cpus-per-task=12",
                f"--mem={memory_gb}G",
                f"--time={worker_minutes // 60:02d}:{worker_minutes % 60:02d}:00",
                f"--array=0-{len(plan['jobs']) - 1}%{min(6, len(plan['jobs']))}",
                f"--output={root}/slurm/gpu-%A_%a.out",
                f"--error={root}/slurm/gpu-%A_%a.err",
                script,
                *args,
                "worker",
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    write_once(root / "gpu_submission.json", dict(array=gpu))
    cpu = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--account=bii_dsc_community",
                "--partition=bii",
                "--job-name=mk-evicast-life-v2-analysis",
                "--nodes=1",
                "--ntasks=1",
                "--cpus-per-task=2",
                "--mem=8G",
                "--time=00:10:00",
                f"--dependency=afterany:{gpu}",
                f"--output={root}/slurm/analysis-%j.out",
                f"--error={root}/slurm/analysis-%j.err",
                script,
                *args,
                "analyze",
            ],
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    policy = dict(
        timezone="America/New_York",
        maximum_worker_minutes=120,
        maximum_a100s=6,
        six_a100s_explicitly_authorized=True,
        arrays=[gpu],
        per_lane_static_throttle=min(6, len(plan["jobs"])),
        other_lane_reserved_slots=max(0, 6 - len(plan["jobs"])),
    )
    write_once(root / "scheduler/policy.json", policy)
    write_once(
        root / "scheduler/registration.json", tick(policy, {}, root / "scheduler/state.json", False)
    )
    write_once(root / "submission.json", dict(gpu_array=gpu, cpu_analysis=cpu))
    print(json.dumps(dict(gpu_array=gpu, cpu_analysis=cpu)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "submit", "worker", "analyze"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--plan-sha")
    p.add_argument("--index", type=int)
    a = p.parse_args()
    if a.mode == "prepare":
        print(prepare(a.root))
    elif a.mode == "submit":
        submit(a.root, a.plan_sha)
    elif a.mode == "worker":
        previous.worker(a.root, a.plan_sha, a.index)
    else:
        analyze(a.root, a.plan_sha)


if __name__ == "__main__":
    main()
