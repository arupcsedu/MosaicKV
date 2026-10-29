#!/usr/bin/env python3
"""V10: fixed layer-wise evidence schedules on two already consumed inputs."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import evicastkv_lifetime_history as common
import evicastkv_lifetime_v2 as launch
import evicastkv_multiscale_prompt as prior

from mosaickv.config import RunConfig, canonical_config, config_sha256
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
POLICIES = ("early_global", "late_global", "profile_high_global", "profile_low_global")


def layer_schedules(forget):
    """Fixed halves determined only by layer index or pinned profile, no labels."""
    n = len(forget)
    if n < 2 or n % 2 or any(not isinstance(x, (int, float)) or not 0 <= x <= 1 for x in forget):
        raise ValueError("finite even layer profile in [0,1] required")
    early = set(range(n // 2))
    high = set(sorted(range(n), key=lambda i: (forget[i], i))[: n // 2])
    groups = (early, set(range(n)) - early, high, set(range(n)) - high)
    return {
        name: tuple(1.0 if i in group else 0.5 for i in range(n))
        for name, group in zip(POLICIES, groups, strict=True)
    }


def phase_trace(row, phase="generation", replay=False):
    paths = row["memory_audit_traces"] if replay else row["traces"]
    traces = []
    for path, digest in paths.items():
        if sha(Path(path)) != digest:
            raise ValueError("trace evidence changed")
        trace = common.read(path)
        if trace["execution_phase"] == phase:
            traces.append(trace)
    if len(traces) != 1:
        raise ValueError("one hash-valid phase trace required")
    return traces[0]


def audit_saved(parent, plan):
    """Only trace/accounting data; answers, references and scores not inspected."""
    records = []
    for job in plan["jobs"]:
        if job["diagnostic_history_capacity"] != 256:
            continue
        if job["arms"][0]["config"]["method"] != "mosaickv_exact":
            continue
        path = parent / f"workers/{job['index']}/rows/{job['arms'][0]['label']}.json"
        row = common.read(path)
        trace = phase_trace(row)
        replay = phase_trace(row, replay=True)
        native = common.read(job["native_control_row"])
        native_replay = phase_trace(native, replay=True)
        records.append(
            dict(
                task=job["task"],
                sample_id=job["sample_id"],
                row_sha256=sha(path),
                prior_weight=job["diagnostic_full_prompt_prior_weight"],
                selection_digest=trace["selected_blocks"],
                per_layer_selected_positions_available=False,
                layer_prompt_capacities=trace["selected_layer_prompt_capacities"],
                prompt_mass=trace["prompt_mass_audit"],
                initial=row["metrics"][common.STORAGE[0]],
                final=row["metrics"][common.STORAGE[1]],
                working_peak=replay["request_KV_transactions"]["decode_transaction_storage"][
                    "peak_storage_bytes"
                ],
                registered_request_owners=replay["request_KV_ownership"],
                native_registered_request_owners=native_replay["request_KV_ownership"],
                whole_process_allocator_peak=replay["peak_gpu_memory_allocated"],
                source_retirement=trace["source_KV_retirement"],
            )
        )
    return dict(
        status="complete_label_blind_saved_trace_audit",
        records=records,
        inspected_source_sha256={
            str(p.relative_to(SOURCE)): sha(p)
            for p in (
                SOURCE / "mosaickv/src/mosaickv/measurements/kv_ownership.py",
                SOURCE / "mosaickv/src/mosaickv/measurements/kv_transactions.py",
                SOURCE / "mosaickv/src/mosaickv/backends/hf_runtime.py",
                SOURCE / "mosaickv/src/mosaickv/adapters/huggingface/incremental_history_cache.py",
            )
        },
        exact_changed_layers_identifiable=False,
        reason=(
            "Historical selected-position catalogs were omitted; "
            "whole-selection digest is not a per-layer audit."
        ),
        whole_request_KV_peak_exact_certified=False,
        accounting_gaps=[
            "Internal prefill K/V creation and all temporary owners are not fully instrumented.",
            "Registered role byte counts overlap through aliases; do not sum them.",
            "CUDA allocator peak includes model weights and activations; it is not KV-only.",
            "Native reference retains source snapshots unlike bounded variants: "
            "request-owner comparison is confounded.",
        ],
        allowed_peak_comparison=(
            "Same-boundary post-compression active decode transaction; "
            "not exact whole-request peak."
        ),
        fresh_evaluation_authorized=False,
        paper_campaign_authorized=False,
    )


def prepare(root, parent, expected):
    source = clean_source(SOURCE)
    if root.exists() or sha(parent / "plan.json") != expected:
        raise ValueError("new immutable root and exact V9 plan required")
    old = common.read(parent / "plan.json")
    gate = common.read(parent / "analysis/GATE.json")
    if (
        gate["status"] != "complete_consumed_diagnostic"
        or gate["failures"]
        or gate["point_eligible_configurations"]
    ):
        raise ValueError("complete negative V9 required")
    bound = dict(old["bound_files"])
    for p in (parent / "plan.json", parent / "analysis/GATE.json"):
        bound[str(p)] = sha(p)
    for p, digest in bound.items():
        if sha(Path(p)) != digest:
            raise ValueError("bound evidence changed")
    audit = audit_saved(parent, old)
    jobs = []
    tasks = sorted({j["task"] for j in old["jobs"]})
    if tasks != ["llava_description", "mmvet_prefixkv"]:
        raise ValueError("only two existing consumed tasks allowed")
    for task in tasks:
        templates = [
            j for j in old["jobs"] if j["task"] == task and j["diagnostic_history_capacity"] == 256
        ]
        topk = next(
            j
            for j in templates
            if j["arms"][0]["config"]["method"] == "prompt_attention_topk"
            and j["diagnostic_full_prompt_prior_weight"] == 0
        )
        mosaic = next(
            j
            for j in templates
            if j["arms"][0]["config"]["method"] == "mosaickv_exact"
            and j["diagnostic_full_prompt_prior_weight"] == 0.5
        )
        cfg = RunConfig.from_mapping(mosaic["arms"][0]["config"])
        history = cfg.bounded_decode.incremental_history
        profile = Path(history.layer_profile_path)
        if sha(profile) != history.layer_profile_sha256:
            raise ValueError("pinned layer profile changed")
        schedules = layer_schedules(common.read(profile))
        if len({tuple(w) for w in schedules.values()}) != 4:
            raise ValueError("four distinct preregistered schedules required")
        control_path = parent / f"workers/{topk['index']}/rows/{topk['arms'][0]['label']}.json"
        prior.bind_row(bound, control_path)
        variants = [("window_topk_control", topk, (), 0.0)]
        variants += [
            (
                f"uniform_{weight:g}_control",
                next(
                    j
                    for j in templates
                    if j["arms"][0]["config"]["method"] == "mosaickv_exact"
                    and j["diagnostic_full_prompt_prior_weight"] == weight
                ),
                (),
                weight,
            )
            for weight in (0.5, 1.0)
        ]
        variants += [(name, mosaic, weights, 0.5) for name, weights in schedules.items()]
        for name, template, weights, weight in variants:
            base = RunConfig.from_mapping(template["arms"][0]["config"])
            h = base.bounded_decode.incremental_history
            candidate = replace(
                base,
                bounded_decode=replace(
                    base.bounded_decode,
                    incremental_history=replace(h, full_prompt_prior_layer_weights=weights),
                ),
            )
            row = parent / f"workers/{template['index']}/rows/{template['arms'][0]['label']}.json"
            prior.bind_row(bound, row)
            label = f"{candidate.method.value}-{name}-history256-v10"
            jobs.append(
                dict(
                    template,
                    index=len(jobs),
                    diagnostic_layer_policy=name,
                    diagnostic_full_prompt_prior_weight=weight,
                    frozen_topk_control_row=str(control_path),
                    equivalence_control_row=str(row) if not weights else None,
                    arms=[
                        dict(
                            label=label,
                            config=canonical_config(candidate),
                            config_sha=config_sha256(candidate),
                        )
                    ],
                )
            )
    if len(jobs) != 14:
        raise ValueError("bounded fourteen-arm scope required")
    write_once(root / "saved_trace_audit.json", audit)
    bound[str(root / "saved_trace_audit.json")] = sha(root / "saved_trace_audit.json")
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=source,
            jobs=jobs,
            bound_files=bound,
            protocol="layerwise_shared_prior_v10_consumed_only",
            history_capacity_grid=[256],
            layer_policies=POLICIES,
            capture_selection_fingerprints=True,
            fresh_evaluation_authorized=False,
            paper_campaign_authorized=False,
            automatic_consumed_followup=False,
        ),
    )
    return sha(root / "plan.json")


def fingerprints(row):
    records = phase_trace(row, replay=True).get("selected_position_fingerprints")
    if not records or len(records) != 32 * 32:
        raise ValueError("all 32 layers/32 heads require selected-position fingerprints")
    return {(x["layer"], x["kv_head"]): (x["positions"], x["sha256"]) for x in records}


def analyze(root, expected):
    import pandas as pd

    plan = common.inputs(root, expected)
    rows, checks, failures, layer_audit, controls = [], [], [], [], []
    completed = {}
    for job in plan["jobs"]:
        w = root / f"workers/{job['index']}"
        raw = w / f"rows/{job['arms'][0]['label']}.json"
        if not (w / "result.json").exists() or not raw.exists():
            failures.append(dict(worker=job["index"], error="missing worker/row"))
            continue
        failures.extend(common.read(w / "result.json")["failures"])
        r = common.read(raw)
        if r["status"] != "completed" or not r.get("memory_observer_token_parity"):
            failures.append(dict(worker=job["index"], error="failed sample/replay"))
            continue
        fingerprints(r)
        for phase in ("generation", "teacher_forced"):
            phase_trace(r, phase)
        completed[(job["task"], job["diagnostic_layer_policy"])] = (r, job)
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
            | dict(layer_policy=job["diagnostic_layer_policy"])
        )
        if job["equivalence_control_row"]:
            old = common.read(job["equivalence_control_row"])
            fields = (
                "answer",
                "config_sha",
                "nll_sum",
                "evaluated_tokens",
                "teacher_forced_ppl",
                "rouge_l_f1",
            )
            equal = all(r[k] == old[k] for k in fields) and all(
                phase_trace(r, p)["generated_token_ids"]
                == phase_trace(old, p)["generated_token_ids"]
                for p in ("generation", "teacher_forced")
            )
            controls.append(
                dict(
                    task=job["task"], policy=job["diagnostic_layer_policy"], exact_equivalence=equal
                )
            )
            if not equal:
                failures.append(
                    dict(worker=job["index"], error="control numerical equivalence failed")
                )
    for (task, policy), (r, job) in completed.items():
        if policy == "window_topk_control":
            continue
        pair = completed.get((task, "window_topk_control"))
        if pair is None:
            failures.append(dict(task=task, error="missing contemporary Top-K"))
            continue
        check = prior.point_check(r, pair[0], common.read(job["native_control_row"]))
        checks.append(dict(check, task=task, policy=policy))
        if not check["matched_realized_bytes"]:
            failures.append(dict(task=task, policy=policy, error="unmatched realized storage"))
        for control in ("uniform_0.5_control", "uniform_1_control"):
            if (task, control) not in completed:
                continue
            a, b = fingerprints(r), fingerprints(completed[task, control][0])
            changed = sorted({layer for (layer, head), v in a.items() if v != b[layer, head]})
            layer_audit.append(
                dict(
                    task=task,
                    policy=policy,
                    reference=control,
                    changed_layers=changed,
                    changed_heads=sum(v != b[k] for k, v in a.items()),
                    capacities_identical=all(v[0] == b[k][0] for k, v in a.items()),
                )
            )
    eligible = []
    if len(rows) == 14 and not failures:
        for policy in POLICIES:
            group = [c for c in checks if c["policy"] == policy]
            if len(group) == 2 and all(
                all(v for k, v in c.items() if k not in {"task", "policy"}) for c in group
            ):
                eligible.append(policy)
    out = root / "analysis"
    if out.exists():
        raise FileExistsError("immutable derived analysis required")
    out.mkdir()
    table = pd.DataFrame(rows)
    table.to_parquet(out / "summary.parquet", index=False)
    gate = dict(
        status="complete_consumed_diagnostic" if len(rows) == 14 and not failures else "failed",
        failures=failures,
        checks=checks,
        control_checks=controls,
        point_eligible_schedules=eligible,
        actual_rows=len(rows),
        expected_rows=14,
        quality_not_statistically_established=True,
        superiority_established=False,
        whole_request_KV_peak_certified=False,
        fresh_evaluation_authorized=False,
        paper_campaign_authorized=False,
    )
    write_once(out / "GATE.json", gate)
    write_once(out / "layer_selection_audit.json", dict(records=layer_audit))
    (out / "report.md").write_text(
        "# V10 layer-wise evidence diagnostic\n\n"
        "Two consumed inputs only; not independent confirmation. "
        "Native PrefixKV is a different operating point. Initial/final/working-transaction bytes "
        "must match contemporary Top-K. Whole-request KV peak is not certified. "
        "PPL and ROUGE-L are separate; timing is descriptive.\n\n"
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
    p.add_argument("--plan-sha", required=True)
    p.add_argument("--index", type=int)
    a = p.parse_args()
    if a.mode == "prepare":
        print(prepare(a.root, a.parent, a.plan_sha))
    elif a.mode == "submit":
        launch.submit(
            a.root,
            a.plan_sha,
            script_name="evicastkv_layerwise_prior.sbatch",
            worker_minutes=30,
            memory_gb=16,
        )
    elif a.mode == "worker":
        common.worker(a.root, a.plan_sha, a.index)
    else:
        analyze(a.root, a.plan_sha)


if __name__ == "__main__":
    main()
