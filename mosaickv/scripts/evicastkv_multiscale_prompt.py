#!/usr/bin/env python3
"""Consumed-input, bounded multiscale prompt-evidence study."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import evicastkv_lifetime_history as common
import evicastkv_lifetime_v2 as launch

from mosaickv.config import RunConfig, canonical_config, config_sha256
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
WEIGHTS = (0.25, 0.5, 1.0)


def bind_row(bound, path):
    row = common.read(path)
    if row["status"] != "completed":
        raise ValueError("control incomplete")
    bound[str(path)] = sha(path)
    for p, h in (row["traces"] | row["memory_audit_traces"]).items():
        if sha(Path(p)) != h:
            raise ValueError("control evidence changed")
        bound[p] = h


def prepare(root, parent, expected, equivalence):
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("immutable consumed-only root required")
    if sha(parent / "plan.json") != expected:
        raise ValueError("parent plan changed")
    gate_path = parent / "analysis/GATE.json"
    gate = common.read(gate_path)
    proof = equivalence / "analysis/GATE.json"
    if common.read(proof)["status"] != "passed_equivalence_and_owner_reduction":
        raise ValueError("zero-clone numerical verification required first")
    proof_plan = common.read(equivalence / "plan.json")
    if proof_plan["bound_files"].get(str(parent / "plan.json")) != expected:
        raise ValueError("equivalence proof belongs to another parent")
    if gate["status"] != "complete_consumed_diagnostic" or gate["failures"]:
        raise ValueError("parent execution incomplete")
    if gate["point_eligible_history_capacities"]:
        raise ValueError("do not redesign a passing candidate")
    old = common.read(parent / "plan.json")
    bound = dict(old["bound_files"])
    bound[str(parent / "plan.json")] = expected
    bound[str(gate_path)] = sha(gate_path)
    bound[str(proof)] = sha(proof)
    bound[str(equivalence / "plan.json")] = sha(equivalence / "plan.json")
    for p, h in bound.items():
        if sha(Path(p)) != h:
            raise ValueError("bound evidence changed")
    jobs = []
    for job in old["jobs"]:
        if job["diagnostic_history_capacity"] not in {128, 512}:
            continue
        arm = job["arms"][0]
        cfg = RunConfig.from_mapping(arm["config"])
        baseline = next(
            j
            for j in old["jobs"]
            if j["task"] == job["task"]
            and j["diagnostic_history_capacity"] == job["diagnostic_history_capacity"]
            and j["arms"][0]["config"]["method"] == "prompt_attention_topk"
        )
        baseline_path = (
            parent / f"workers/{baseline['index']}/rows/{baseline['arms'][0]['label']}.json"
        )
        bind_row(bound, baseline_path)
        for weight in WEIGHTS if cfg.method.is_mosaickv else (1.0,):
            candidate = replace(
                cfg,
                bounded_decode=replace(
                    cfg.bounded_decode,
                    incremental_history=replace(
                        cfg.bounded_decode.incremental_history, full_prompt_prior_weight=weight
                    ),
                ),
            )
            label = f"{arm['label']}-full-prior-{weight:g}-v7"
            jobs.append(
                dict(
                    job,
                    index=len(jobs),
                    frozen_topk_control_row=str(baseline_path),
                    diagnostic_full_prompt_prior_weight=weight,
                    arms=[
                        dict(
                            label=label,
                            config=canonical_config(candidate),
                            config_sha=config_sha256(candidate),
                        )
                    ],
                )
            )
    if len(jobs) != 16:
        raise ValueError("bounded study must contain exactly sixteen arms")
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=source,
            jobs=jobs,
            bound_files=bound,
            protocol="multiscale_prompt_prior_v7_consumed_only",
            prompt_prior_weights=WEIGHTS,
            history_capacity_grid=[128, 512],
            frozen_topk_window_only_controls_reused=True,
            full_prior_topk_rows_are_explicit_ablations_not_window_baseline=True,
            fresh_evaluation_authorized=False,
            paper_campaign_authorized=False,
        ),
    )
    return sha(root / "plan.json")


def point_check(row, baseline, native):
    fields = (*common.STORAGE[:2], "audited_decode_transaction_peak_bytes")

    def value(r, k):
        return r["metrics"][k] if k in common.STORAGE else r[k]

    return dict(
        matched_realized_bytes=all(value(row, k) == value(baseline, k) for k in fields),
        smaller_initial_final_peak_than_native=all(
            value(row, k) < value(native, k) for k in fields
        ),
        ppl_not_worse_than_native=row["teacher_forced_ppl"] <= native["teacher_forced_ppl"],
        rouge_not_worse_than_native=row["rouge_l_f1"] >= native["rouge_l_f1"],
        ppl_not_worse_than_frozen_topk=row["teacher_forced_ppl"] <= baseline["teacher_forced_ppl"],
        rouge_not_worse_than_frozen_topk=row["rouge_l_f1"] >= baseline["rouge_l_f1"],
    )


def analyze(root, expected):
    import pandas as pd

    plan = common.inputs(root, expected)
    rows, checks, failures = [], [], []
    for job in plan["jobs"]:
        w = root / f"workers/{job['index']}"
        raw = w / f"rows/{job['arms'][0]['label']}.json"
        if not (w / "result.json").exists() or not raw.exists():
            failures.append(dict(worker=job["index"], error="missing worker/row"))
            continue
        failures.extend(common.read(w / "result.json")["failures"])
        r = common.read(raw)
        if r["status"] != "completed":
            failures.append(dict(worker=job["index"], error="failed raw sample"))
            continue
        if not r["memory_observer_token_parity"] or len(r["memory_audit_traces"]) != 1:
            failures.append(dict(worker=job["index"], error="missing replay or changed tokens"))
        for p, h in (r["traces"] | r["memory_audit_traces"]).items():
            if sha(Path(p)) != h:
                raise ValueError("trace evidence changed")
        rows.append(
            {
                k: r[k]
                for k in (
                    "task",
                    "sample_id",
                    "label",
                    "method",
                    "teacher_forced_ppl",
                    "rouge_l_f1",
                    "nll_sum",
                    "evaluated_tokens",
                    "audited_request_owned_peak_bytes",
                    "audited_decode_transaction_peak_bytes",
                    "pytorch_request_peak_allocated_bytes",
                )
            }
            | {k: r["metrics"][k] for k in (*common.STORAGE, "end_to_end_time", "compression_time")}
            | dict(
                history_capacity=job["diagnostic_history_capacity"],
                full_prompt_prior_weight=job["diagnostic_full_prompt_prior_weight"],
            )
        )
        if r["method"] == "mosaickv_exact":
            check = point_check(
                r,
                common.read(job["frozen_topk_control_row"]),
                common.read(job["native_control_row"]),
            )
            checks.append(
                dict(
                    check,
                    task=job["task"],
                    history_capacity=job["diagnostic_history_capacity"],
                    full_prompt_prior_weight=job["diagnostic_full_prompt_prior_weight"],
                )
            )
            if not check["matched_realized_bytes"]:
                failures.append(dict(worker=job["index"], error="frozen Top-K bytes not matched"))
    eligible = []
    if len(rows) == len(plan["jobs"]) and not failures:
        for history in (128, 512):
            for weight in WEIGHTS:
                group = [
                    x
                    for x in checks
                    if x["history_capacity"] == history and x["full_prompt_prior_weight"] == weight
                ]
                if len(group) == 2 and all(
                    all(
                        v
                        for k, v in c.items()
                        if k not in {"task", "history_capacity", "full_prompt_prior_weight"}
                    )
                    for c in group
                ):
                    eligible.append(dict(history_capacity=history, full_prompt_prior_weight=weight))
    out = root / "analysis"
    if out.exists():
        raise FileExistsError("write-once derived analysis")
    out.mkdir()
    table = pd.DataFrame(rows)
    table.to_parquet(out / "summary.parquet", index=False)
    gate = dict(
        status="complete_consumed_diagnostic"
        if len(rows) == len(plan["jobs"]) and not failures
        else "failed",
        failures=failures,
        checks=checks,
        point_eligible_configurations=eligible,
        quality_not_statistically_established=True,
        superiority_established=False,
        whole_request_KV_peak_certified=False,
        native_latency_not_paired=True,
        fresh_evaluation_authorized=False,
        paper_campaign_authorized=False,
    )
    write_once(out / "GATE.json", gate)
    (out / "report.md").write_text(
        "# Multiscale prompt-evidence consumed diagnostic\n\n"
        "Not independent confirmation. All attempted configurations are reported. The "
        "window-only frozen Top-K controls are compared at the exact same physical byte "
        "boundary. Full-prior Top-K is a separately labeled scoring ablation. Native "
        "PrefixKV has a different operating point, not byte matched. Timing is descriptive; "
        "PPL and ROUGE-L remain separate.\n\n"
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("prepare", "submit", "worker", "analyze"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent", type=Path)
    p.add_argument("--equivalence", type=Path)
    p.add_argument("--plan-sha")
    p.add_argument("--index", type=int)
    a = p.parse_args()
    if a.mode == "prepare":
        print(prepare(a.root, a.parent, a.plan_sha, a.equivalence))
    elif a.mode == "submit":
        launch.submit(
            a.root,
            a.plan_sha,
            script_name="evicastkv_multiscale_prompt.sbatch",
            worker_minutes=30,
            memory_gb=16,
        )
    elif a.mode == "worker":
        common.worker(a.root, a.plan_sha, a.index)
    else:
        analyze(a.root, a.plan_sha)


if __name__ == "__main__":
    main()
