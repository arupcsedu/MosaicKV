#!/usr/bin/env python3
"""Consumed-only causal history-capacity frontier, never a paper promoter."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import evicastkv_lifetime_history as common
import evicastkv_lifetime_v2 as launch

from mosaickv.config import RunConfig, canonical_config, config_sha256
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.selection.profile_lifetime import allocate_profile_lifetime

SOURCE = Path(__file__).resolve().parents[2]
HISTORY_GRID = (128, 256, 384, 512)


def capacity_plan(ratios, lengths, mandatory, costs, history):
    """Identical prompt allocation: charge only the extra history and scratch.

    Restrict this diagnostic to equal per-layer K/V position costs, as in the
    pinned LLaVA checkpoint. 223 MiB at G=128 is registered before execution;
    it is geometry planning, not an optimizer over benchmark metrics.
    """
    if len(set(costs)) != 1 or history not in HISTORY_GRID:
        raise ValueError("unsupported frontier geometry/history")
    cap = 223 * 2**20 + (history - 128) * (sum(costs) + max(costs))
    return allocate_profile_lifetime(
        ratios, lengths, mandatory, costs, cap_bytes=cap, generated_capacity=history
    )


def geometry(trace):
    costs, lengths = [], []
    for layer in trace["source_kv_layout"]:
        k, v = layer["key_shape"], layer["value_shape"]
        if len(k) != 4 or len(v) != 4 or k[-2] != v[-2]:
            raise ValueError("unsupported source geometry")
        lengths.append(k[-2])
        costs.append(
            math.prod(k) // k[-2] * layer["key_element_size"]
            + math.prod(v) // v[-2] * layer["value_element_size"]
        )
    return tuple(lengths), tuple(costs)


def generation_trace(row):
    records = []
    for path, digest in row["traces"].items():
        if sha(Path(path)) != digest:
            raise ValueError("trace evidence changed")
        trace = common.read(Path(path))
        if trace["execution_phase"] == "generation":
            records.append((path, digest, trace))
    if len(records) != 1:
        raise ValueError("expected exactly one generation trace")
    return records[0]


def prepare(root, parent, expected):
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("immutable successor root required")
    if sha(parent / "plan.json") != expected:
        raise ValueError("parent plan changed")
    gate = common.read(parent / "analysis/GATE.json")
    if gate["status"] != "complete_consumed_diagnostic" or gate["failures"]:
        raise ValueError("parent execution not valid")
    if gate["point_eligible_policies"]:
        raise ValueError("stop redesign when prior candidate passes")
    old = common.read(parent / "plan.json")
    bound = dict(old["bound_files"])
    for p, h in bound.items():
        if sha(Path(p)) != h:
            raise ValueError("parent evidence changed")
    bound[str(parent / "plan.json")] = expected
    bound[str(parent / "analysis/GATE.json")] = sha(parent / "analysis/GATE.json")
    jobs = []
    for old_job in old["jobs"]:
        if old_job["mandatory_policy"] != "method":
            continue
        row_path = parent / f"workers/{old_job['index']}/rows/{old_job['arms'][0]['label']}.json"
        row = common.read(row_path)
        if row["status"] != "completed":
            raise ValueError("missing consumed source execution")
        bound[str(row_path)] = sha(row_path)
        path, digest, trace = generation_trace(row)
        bound[path] = digest
        lengths, costs = geometry(trace)
        cfg = RunConfig.from_mapping(old_job["arms"][0]["config"])
        ratios = common.read(Path(cfg.bounded_decode.incremental_history.layer_profile_path))
        planned = []
        for history in HISTORY_GRID:
            # Capacity computation uses only geometry and the pinned profile.
            # The runtime independently applies the true mandatory minima.
            allocation = capacity_plan(ratios, lengths, [1] * len(costs), costs, history)
            planned.append(allocation.prompt_positions)
            candidate = replace(
                cfg,
                cache=replace(cfg.cache, budget_value=allocation.cap_bytes),
                bounded_decode=replace(
                    cfg.bounded_decode,
                    generated_capacity=history,
                    incremental_history=replace(
                        cfg.bounded_decode.incremental_history, history_allocation="uniform"
                    ),
                ),
            )
            label = f"{cfg.method.value}-profiled-history-{history}-frontier-v5"
            jobs.append(
                dict(
                    old_job,
                    index=len(jobs),
                    diagnostic_history_capacity=history,
                    planned_prompt_positions=allocation.prompt_positions,
                    geometry_only_planned_cap_bytes=allocation.cap_bytes,
                    arms=[
                        dict(
                            label=label,
                            config=canonical_config(candidate),
                            config_sha=config_sha256(candidate),
                        )
                    ],
                )
            )
        if len(set(planned)) != 1:
            raise ValueError("frontier must preserve the prompt allocation")
    if len(jobs) != 16:
        raise ValueError("expected two consumed tasks, two methods, four capacities")
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=source,
            jobs=jobs,
            bound_files=bound,
            protocol="prompt_protected_history_frontier_v5_consumed_diagnostic",
            history_capacity_grid=HISTORY_GRID,
            no_answer_or_score_driven_budget_selection=True,
            native_operating_points_not_byte_matched=True,
            fresh_evaluation_authorized=False,
            paper_campaign_authorized=False,
        ),
    )
    return sha(root / "plan.json")


def analyze(root, expected):
    import pandas as pd

    plan = common.inputs(root, expected)
    failures, rows, controls = [], [], {}
    for job in plan["jobs"]:
        worker = root / f"workers/{job['index']}"
        result = worker / "result.json"
        raw = worker / f"rows/{job['arms'][0]['label']}.json"
        if not result.exists() or not raw.exists():
            failures.append(dict(worker=job["index"], error="missing worker/row"))
            continue
        failures.extend(common.read(result)["failures"])
        row = common.read(raw)
        if row["status"] != "completed":
            failures.append(dict(worker=job["index"], error="failed raw row"))
            continue
        if not row.get("memory_observer_token_parity") or len(row["memory_audit_traces"]) != 1:
            failures.append(
                dict(worker=job["index"], error="memory replay missing or token mismatch")
            )
        for p, h in (row["traces"] | row["memory_audit_traces"]).items():
            if sha(Path(p)) != h:
                raise ValueError("trace evidence changed")
        _, _, trace = generation_trace(row)
        controls.setdefault(job["task"], common.read(job["native_control_row"]))
        capacity = trace["evidence_layer_allocation"]["capacity"]
        if tuple(capacity["prompt_positions"]) != tuple(job["planned_prompt_positions"]):
            failures.append(
                dict(worker=job["index"], error="mandatory feasibility changed planned prompt")
            )
        rows.append(
            {
                k: row[k]
                for k in (
                    "task",
                    "sample_id",
                    "label",
                    "method",
                    "teacher_forced_ppl",
                    "rouge_l_f1",
                    "nll_sum",
                    "evaluated_tokens",
                    "audited_decode_transaction_peak_bytes",
                    "audited_request_owned_peak_bytes",
                    "pytorch_request_peak_allocated_bytes",
                )
            }
            | {
                k: row["metrics"][k]
                for k in (*common.STORAGE, "end_to_end_time", "compression_time")
            }
            | dict(
                history_capacity=job["diagnostic_history_capacity"],
                prompt_slots_sha256=trace["evidence_layer_allocation"]["selected_positions_sha256"],
            )
        )
    table = pd.DataFrame(rows)
    checks = []
    if len(rows) == len(plan["jobs"]) and not failures:
        for (task, history), group in table.groupby(["task", "history_capacity"]):
            a = group[group.method == "mosaickv_exact"].iloc[0]
            b = group[group.method == "prompt_attention_topk"].iloc[0]
            native = controls[task]
            fields = (*common.STORAGE[:2], "audited_decode_transaction_peak_bytes")
            matched = all(a[k] == b[k] for k in fields)
            checks.append(
                dict(
                    task=task,
                    history_capacity=int(history),
                    matched_realized_bytes=bool(matched),
                    smaller_initial_final_peak_than_native=bool(
                        all(a[k] < native["metrics"][k] for k in common.STORAGE[:2])
                        and a.audited_decode_transaction_peak_bytes
                        < native["audited_decode_transaction_peak_bytes"]
                    ),
                    ppl_not_worse_than_native=bool(
                        a.teacher_forced_ppl <= native["teacher_forced_ppl"]
                    ),
                    rouge_not_worse_than_native=bool(a.rouge_l_f1 >= native["rouge_l_f1"]),
                    ppl_not_worse_than_matched_topk=bool(
                        a.teacher_forced_ppl <= b.teacher_forced_ppl
                    ),
                    rouge_not_worse_than_matched_topk=bool(a.rouge_l_f1 >= b.rouge_l_f1),
                )
            )
            if not matched:
                failures.append(
                    dict(task=task, history=int(history), error="matched byte mismatch")
                )
        for (task, method), group in table.groupby(["task", "method"]):
            if group.prompt_slots_sha256.nunique() != 1:
                failures.append(
                    dict(task=task, method=method, error="history sweep changed prompt slots")
                )
    eligible = []
    for history in HISTORY_GRID:
        c = [x for x in checks if x["history_capacity"] == history]
        if len(c) == 2 and all(
            all(v for k, v in x.items() if k not in {"task", "history_capacity"}) for x in c
        ):
            eligible.append(history)
    out = root / "analysis"
    if out.exists():
        raise FileExistsError("write-once analysis")
    out.mkdir()
    table.to_parquet(out / "summary.parquet", index=False)
    gate = dict(
        status="complete_consumed_diagnostic"
        if len(rows) == len(plan["jobs"]) and not failures
        else "failed",
        failures=failures,
        checks=checks,
        point_eligible_history_capacities=eligible,
        superiority_established=False,
        quality_not_statistically_established=True,
        whole_request_KV_peak_certified=False,
        native_latency_not_paired=True,
        fresh_evaluation_authorized=False,
        paper_campaign_authorized=False,
        decision="preserve_consumed_frontier_no_automatic_fresh_or_paper_promotion",
    )
    write_once(out / "GATE.json", gate)
    (out / "report.md").write_text(
        "# Consumed prompt/history frontier\n\n"
        "Not independent confirmation. Native controls are different operating points; "
        "only same-capacity Top-K/EviCastKV pairs are byte-matched. All capacities, "
        "including failures, are reported. PPL and ROUGE-L remain separate. "
        "Known-owner/decode transaction peaks do not certify unobserved whole-request KV.\n\n"
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "submit", "worker", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--plan-sha")
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.mode == "prepare":
        print(prepare(args.root, args.parent, args.plan_sha))
    elif args.mode == "submit":
        launch.submit(
            args.root,
            args.plan_sha,
            script_name="evicastkv_lifetime_frontier.sbatch",
            worker_minutes=30,
            memory_gb=16,
        )
    elif args.mode == "worker":
        common.worker(args.root, args.plan_sha, args.index)
    else:
        analyze(args.root, args.plan_sha)


if __name__ == "__main__":
    main()
