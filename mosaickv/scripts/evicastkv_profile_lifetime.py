#!/usr/bin/env python3
"""Conditional, consumed-only profile-shape diagnostic, never a paper promoter."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import evicastkv_lifetime_history as common
import evicastkv_lifetime_v2 as v2

from mosaickv.config import RunConfig, canonical_config, config_sha256
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]


def conditional_submit(root, parent, expected):
    source_sha = clean_source(SOURCE)
    if sha(parent / "plan.json") != expected:
        raise ValueError("parent plan changed")
    gate_path = parent / "analysis/GATE.json"
    if not gate_path.exists():
        write_once(
            root / "DECISION.json",
            dict(decision="stop_missing_parent_analysis", gpu_submitted=False),
        )
        return
    gate = common.read(gate_path)
    if gate["status"] != "complete_consumed_diagnostic" or gate["failures"]:
        write_once(
            root / "DECISION.json",
            dict(decision="stop_parent_execution_failure", gpu_submitted=False),
        )
        return
    if gate["consumed_followup_eligible_labels"]:
        write_once(
            root / "DECISION.json",
            dict(decision="skip_prior_candidate_advanced", gpu_submitted=False),
        )
        return
    if root.exists():
        raise FileExistsError("write-once successor root")
    plan = common.read(parent / "plan.json")
    bound = dict(plan["bound_files"])
    bound[str(parent / "plan.json")] = expected
    bound[str(gate_path)] = sha(gate_path)
    jobs = []
    for old in plan["jobs"]:
        reference = parent / f"workers/{old['index']}/fullkv_reference.json"
        native = parent / f"workers/{old['index']}/rows/prefixkv_reimpl-native-control.json"
        bound[str(reference)] = sha(reference)
        bound[str(native)] = sha(native)
        native_row = common.read(native)
        for p, h in native_row["memory_audit_traces"].items():
            if sha(Path(p)) != h:
                raise ValueError("parent native memory evidence changed")
            bound[p] = h
        profile = Path(
            next(
                a["config"]["prefixkv"]["profile_path"]
                for a in old["arms"]
                if a["label"] == "prefixkv_reimpl-native-control"
            )
        )
        bound[str(profile)] = sha(profile)
        native_arm = next(a for a in old["arms"] if a["label"] == "prefixkv_reimpl-native-control")
        identity_config = next(
            a["config"] for a in old["arms"] if a["label"] == "mosaickv_exact-sink_recent-208mib-v2"
        )
        jobs.append(
            dict(
                index=len(jobs),
                task=old["task"],
                ledger=old["ledger"],
                sample_id=old["sample_id"],
                mandatory_policy="native_control",
                frozen_fullkv_reference=str(reference),
                native_control_row=str(native),
                retention_one_config=identity_config,
                arms=[native_arm],
                measurement_role="remeasured_native_control",
            )
        )
        for policy in ("method", "question_and_boundaries"):
            for method in ("prompt_attention_topk", "mosaickv_exact"):
                cfg = RunConfig.from_mapping(
                    next(
                        a["config"]
                        for a in old["arms"]
                        if a["label"] == f"{method}-sink_recent-208mib-v2"
                    )
                )
                cfg = replace(
                    cfg,
                    cache=replace(cfg.cache, physical_budget_mandatory_policy=policy),
                    bounded_decode=replace(
                        cfg.bounded_decode,
                        incremental_history=replace(
                            cfg.bounded_decode.incremental_history,
                            layer_profile_path=str(profile),
                            layer_profile_sha256=sha(profile),
                        ),
                    ),
                )
                label = f"{method}-profiled-208mib-{policy}-v3"
                jobs.append(
                    dict(
                        index=len(jobs),
                        task=old["task"],
                        ledger=old["ledger"],
                        sample_id=old["sample_id"],
                        mandatory_policy=policy,
                        frozen_fullkv_reference=str(reference),
                        native_control_row=str(native),
                        arms=[
                            dict(
                                label=label,
                                config=canonical_config(cfg),
                                config_sha=config_sha256(cfg),
                            )
                        ],
                    )
                )
    write_once(
        root / "plan.json",
        dict(
            source_sha=source_sha,
            bound_files=bound,
            jobs=jobs,
            protocol="profile_shape_lifetime_v3_consumed_only",
            stage="same_consumed_inputs_no_fresh_cohort",
            separate_memory_replay=True,
            result_category="unified_comparison",
            result_origin="locally_measured",
            prior_native_operating_points_not_byte_matched=True,
            native_latency_not_paired=True,
            upstream_profile_calibration_identities_not_certified=True,
            maximum_a100s=6,
            cpus_per_worker=12,
            worker_memory_gb=16,
            worker_minutes=30,
            fresh_evaluation_authorized=False,
            paper_campaign_authorized=False,
            quality_margin_ppl=0.0,
            quality_margin_rouge=0.0,
        ),
    )
    h = sha(root / "plan.json")
    v2.submit(
        root, h, script_name="evicastkv_profile_lifetime.sbatch", worker_minutes=30, memory_gb=16
    )
    write_once(
        root / "DECISION.json",
        dict(
            decision="bounded_profile_diagnostic_submitted",
            source_sha=source_sha,
            plan_sha256=h,
            gpu_submitted=True,
        ),
    )


def analyze(root, expected):
    import pandas as pd

    plan = common.inputs(root, expected)
    rows = []
    failures = []
    controls = {}
    for job in plan["jobs"]:
        w = root / f"workers/{job['index']}"
        if not (w / "result.json").exists():
            failures.append(dict(worker=job["index"], error="missing worker"))
            continue
        failures.extend(common.read(w / "result.json")["failures"])
        p = w / f"rows/{job['arms'][0]['label']}.json"
        if not p.exists():
            failures.append(dict(worker=job["index"], error="missing raw row"))
            continue
        r = common.read(p)
        if r["status"] != "completed":
            continue
        for p, h in (r["traces"] | r["memory_audit_traces"]).items():
            if sha(Path(p)) != h:
                raise ValueError("result trace changed")
        prior_native = common.read(job["native_control_row"])
        controls.setdefault(job["task"], prior_native)
        if job.get("measurement_role") == "remeasured_native_control":
            if (
                r["answer"] != prior_native["answer"]
                or r["nll_sum"] != prior_native["nll_sum"]
                or r["evaluated_tokens"] != prior_native["evaluated_tokens"]
            ):
                failures.append(
                    dict(task=job["task"], error="native control changed after metadata hooks")
                )
            controls[job["task"]] = r
        rows.append(
            {
                k: r[k]
                for k in (
                    "task",
                    "sample_id",
                    "label",
                    "teacher_forced_ppl",
                    "rouge_l_f1",
                    "nll_sum",
                    "evaluated_tokens",
                    "audited_request_owned_peak_bytes",
                    "pytorch_request_peak_allocated_bytes",
                    "memory_observer_token_parity",
                    "audited_decode_transaction_peak_bytes",
                )
            }
            | dict(method=r["method"], mandatory_policy=job["mandatory_policy"])
            | {k: r["metrics"][k] for k in (*common.STORAGE, "end_to_end_time")}
        )
    table = pd.DataFrame(rows)
    checks = []
    expected_rows = sum(len(j["arms"]) for j in plan["jobs"])
    if len(rows) == expected_rows and not failures:
        for (task, policy), g in table.groupby(["task", "mandatory_policy"]):
            if policy == "native_control":
                continue
            a = g[g.method == "mosaickv_exact"].iloc[0]
            b = g[g.method == "prompt_attention_topk"].iloc[0]
            native = controls[task]
            storage_fields = (*common.STORAGE[:2], "audited_decode_transaction_peak_bytes")
            same = all(a[k] == b[k] for k in storage_fields)
            checks.append(
                dict(
                    task=task,
                    mandatory_policy=policy,
                    matched_realized_bytes=bool(same),
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
            if not same:
                failures.append(dict(task=task, policy=policy, error="realized bytes not matched"))
    eligible = []
    for policy in ("method", "question_and_boundaries"):
        c = [r for r in checks if r["mandatory_policy"] == policy]
        if len(c) == 2 and all(
            all(v for k, v in x.items() if k not in {"task", "mandatory_policy"}) for x in c
        ):
            eligible.append(policy)
    out = root / "analysis"
    if out.exists():
        raise FileExistsError("write-once derived analysis")
    out.mkdir()
    table.to_parquet(out / "summary.parquet", index=False)
    gate = dict(
        status="complete_consumed_diagnostic"
        if len(rows) == expected_rows and not failures
        else "failed",
        failures=failures,
        checks=checks,
        point_eligible_policies=eligible,
        quality_not_statistically_established=True,
        superiority_established=False,
        whole_request_KV_peak_certified=False,
        native_latency_not_paired=True,
        peak_comparison_field="audited_decode_transaction_peak_bytes",
        historical_phase_sampled_peaks_not_used_for_peak_gate=True,
        official_PrefixKV_reproduction=False,
        fresh_evaluation_authorized=False,
        paper_campaign_authorized=False,
        decision="stop_and_preserve_development_result_no_automatic_paper_promotion",
    )
    write_once(out / "GATE.json", gate)
    (out / "report.md").write_text(
        f"# {plan.get('protocol', 'Profile-shape lifetime diagnostic')}\n\n"
        "Same consumed inputs, not fresh confirmation. The pinned upstream profile's "
        "calibration identities are not certified. This is an EviCastKV/Top-K "
        "profile-resizing adaptation, not official PrefixKV. Matched-policy pairs "
        "require identical realized bytes. Reused native controls are separate "
        "operating points; their timing is not paired with these jobs. Whole-request "
        "CUDA-allocator peak and audited KV ownership have separate boundaries.\n\n"
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
    p.add_argument("mode", choices=("conditional_submit", "worker", "analyze"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--parent", type=Path)
    p.add_argument("--plan-sha")
    p.add_argument("--index", type=int)
    a = p.parse_args()
    if a.mode == "conditional_submit":
        conditional_submit(a.root, a.parent, a.plan_sha)
    elif a.mode == "worker":
        common.worker(a.root, a.plan_sha, a.index)
    else:
        analyze(a.root, a.plan_sha)


if __name__ == "__main__":
    main()
