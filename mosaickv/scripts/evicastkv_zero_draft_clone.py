#!/usr/bin/env python3
"""Numerical-equivalence and ownership verification for zero-step metadata reads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evicastkv_lifetime_history as common
import evicastkv_lifetime_v2 as launch

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]


def prepare(root, parent, expected, target="kv_owner"):
    if target not in {"kv_owner", "cuda_allocator"}:
        raise ValueError("unsupported equivalence target")
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("immutable equivalence root required")
    if sha(parent / "plan.json") != expected:
        raise ValueError("parent plan changed")
    gate = common.read(parent / "analysis/GATE.json")
    required = (
        "complete_consumed_diagnostic"
        if target == "kv_owner"
        else "passed_equivalence_and_owner_reduction"
    )
    if gate["status"] != required or gate["failures"]:
        raise ValueError("equivalence requires complete parent execution")
    old = common.read(parent / "plan.json")
    bound = dict(old["bound_files"])
    bound[str(parent / "plan.json")] = expected
    bound[str(parent / "analysis/GATE.json")] = sha(parent / "analysis/GATE.json")
    for p, h in bound.items():
        if sha(Path(p)) != h:
            raise ValueError("parent evidence changed")
    jobs = []
    for old_job in old["jobs"]:
        arm = old_job["arms"][0]
        if arm["config"]["method"] != "mosaickv_exact" or old_job[
            "diagnostic_history_capacity"
        ] not in {128, 512}:
            continue
        row_path = parent / f"workers/{old_job['index']}/rows/{arm['label']}.json"
        row = common.read(row_path)
        if row["status"] != "completed":
            raise ValueError("parent row incomplete")
        bound[str(row_path)] = sha(row_path)
        for p, h in (row["traces"] | row["memory_audit_traces"]).items():
            if sha(Path(p)) != h:
                raise ValueError("parent trace changed")
            bound[p] = h
        jobs.append(
            dict(
                old_job,
                index=len(jobs),
                prior_equivalence_row=str(row_path),
                arms=[
                    dict(
                        arm,
                        label=arm["label"]
                        + ("-metadata-only-v6" if target == "kv_owner" else "-hook-collector-v8"),
                    )
                ],
            )
        )
    if len(jobs) != 4:
        raise ValueError("expected four consumed endpoint controls")
    write_once(
        root / "plan.json",
        dict(
            old,
            source_sha=source,
            jobs=jobs,
            bound_files=bound,
            protocol=(
                "zero_step_clone_removal_v6_equivalence_consumed_only"
                if target == "kv_owner"
                else "hook_only_attention_collection_v8_equivalence"
            ),
            equivalence_target_metric=target,
            exact_numerical_equivalence_required=True,
            fresh_evaluation_authorized=False,
            paper_campaign_authorized=False,
        ),
    )
    return sha(root / "plan.json")


def compare_rows(new, old, target="kv_owner"):
    fields = (
        "answer",
        "nll_sum",
        "evaluated_tokens",
        "teacher_forced_ppl",
        "rouge_l_f1",
        "config_sha",
    )
    quality = all(new[k] == old[k] for k in fields)
    storage = all(new["metrics"][k] == old["metrics"][k] for k in common.STORAGE[:2])
    storage = (
        storage
        and new["audited_decode_transaction_peak_bytes"]
        == old["audited_decode_transaction_peak_bytes"]
    )
    result = dict(
        exact_quality_and_config_equivalence=bool(quality),
        exact_live_storage_equivalence=bool(storage),
        registered_owner_peak_reduced=bool(
            new["audited_request_owned_peak_bytes"] < old["audited_request_owned_peak_bytes"]
        ),
        new_registered_owner_peak_bytes=new["audited_request_owned_peak_bytes"],
        old_registered_owner_peak_bytes=old["audited_request_owned_peak_bytes"],
        new_end_to_end_seconds=new["metrics"]["end_to_end_time"],
        old_end_to_end_seconds=old["metrics"]["end_to_end_time"],
    )
    if target == "cuda_allocator":
        result.update(
            cuda_allocator_peak_reduced=bool(
                new["pytorch_request_peak_allocated_bytes"]
                < old["pytorch_request_peak_allocated_bytes"]
            ),
            new_cuda_allocator_peak_bytes=new["pytorch_request_peak_allocated_bytes"],
            old_cuda_allocator_peak_bytes=old["pytorch_request_peak_allocated_bytes"],
            registered_owner_peak_unchanged=bool(
                new["audited_request_owned_peak_bytes"] == old["audited_request_owned_peak_bytes"]
            ),
        )
    return result


def analyze(root, expected):
    plan = common.inputs(root, expected)
    target = plan.get("equivalence_target_metric", "kv_owner")
    checks, failures = [], []
    for job in plan["jobs"]:
        worker = root / f"workers/{job['index']}"
        result = worker / "result.json"
        raw = worker / f"rows/{job['arms'][0]['label']}.json"
        if not result.exists() or not raw.exists():
            failures.append(dict(worker=job["index"], error="missing result or row"))
            continue
        failures.extend(common.read(result)["failures"])
        new, old = common.read(raw), common.read(job["prior_equivalence_row"])
        if new["status"] != "completed":
            failures.append(dict(worker=job["index"], error="failed sample"))
            continue
        for p, h in (new["traces"] | new["memory_audit_traces"]).items():
            if sha(Path(p)) != h:
                raise ValueError("new trace changed")
        check = compare_rows(new, old, target)

        def phases(row):
            return {common.read(p)["execution_phase"]: common.read(p) for p in row["traces"]}

        a, b = phases(new), phases(old)
        check["all_phase_tokens_and_selected_blocks_equal"] = bool(
            set(a) == set(b) == {"generation", "teacher_forced"}
            and all(
                a[p]["generated_token_ids"] == b[p]["generated_token_ids"]
                and a[p]["selected_blocks"] == b[p]["selected_blocks"]
                for p in a
            )
        )
        check.update(task=job["task"], history_capacity=job["diagnostic_history_capacity"])
        checks.append(check)
        for key in (
            "exact_quality_and_config_equivalence",
            "exact_live_storage_equivalence",
            "registered_owner_peak_reduced"
            if target == "kv_owner"
            else "cuda_allocator_peak_reduced",
            "all_phase_tokens_and_selected_blocks_equal",
        ):
            if not check[key]:
                failures.append(dict(worker=job["index"], error=key))
        if target == "cuda_allocator" and not check["registered_owner_peak_unchanged"]:
            failures.append(dict(worker=job["index"], error="unexpected KV owner change"))
    out = root / "analysis"
    if out.exists():
        raise FileExistsError("write-once analysis")
    out.mkdir()
    gate = dict(
        status=(
            "passed_equivalence_and_owner_reduction"
            if target == "kv_owner"
            else "passed_equivalence_and_allocator_reduction"
        )
        if len(checks) == 4 and not failures
        else "failed",
        checks=checks,
        failures=failures,
        superiority_established=False,
        latency_improvement_statistically_established=False,
        whole_request_KV_peak_certified=False,
        fresh_evaluation_authorized=False,
        paper_campaign_authorized=False,
    )
    write_once(out / "GATE.json", gate)
    (out / "report.md").write_text(
        f"# Equivalence and {target} allocation verification\n\n"
        "Consumed-input engineering verification, not a new quality result. All phase "
        "tokens, NLL/counts, selected blocks and live storage must match. Registered "
        "owners and whole-request KV certification are different scopes. Timing is "
        "descriptive, not an interleaved repeated benchmark.\n\n```json\n"
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
    parser.add_argument("--target", choices=("kv_owner", "cuda_allocator"), default="kv_owner")
    args = parser.parse_args()
    if args.mode == "prepare":
        print(prepare(args.root, args.parent, args.plan_sha, args.target))
    elif args.mode == "submit":
        launch.submit(
            args.root,
            args.plan_sha,
            script_name="evicastkv_zero_draft_clone.sbatch",
            worker_minutes=30,
            memory_gb=16,
        )
    elif args.mode == "worker":
        common.worker(args.root, args.plan_sha, args.index)
    else:
        analyze(args.root, args.plan_sha)


if __name__ == "__main__":
    main()
