#!/usr/bin/env python3
"""Selective inactive-control retry, then same-consumed-panel safety check."""

import argparse
import json
import subprocess
from pathlib import Path

import storage_six_ablation as base

from mosaickv.evaluation.fox_campaign import clean_source, digest, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
GATE = Path("/scratch/djy8hg/runs/evicastkv/live-anchor-guard-20261107-v1")
PANEL = Path("/scratch/djy8hg/runs/evicastkv/six-component-consumed-panel-20261107-v1")
RETRY = Path("/scratch/djy8hg/runs/evicastkv/live-anchor-control-retry-20261107-v1")
GUARDED_PANEL = Path("/scratch/djy8hg/runs/evicastkv/live-anchor-consumed-panel-20261107-v1")
RISK_GATE = Path("/scratch/djy8hg/runs/evicastkv/live-anchor-risk-repair-20261107-v1")


def prepare(root, *, panel=False, risk_repair=False):
    if risk_repair and not panel:
        raise ValueError("decision-risk extension is restricted to the fixed consumed panel")
    source = clean_source(SOURCE)
    if root.exists():
        raise FileExistsError("new immutable selective-retry root required")
    parent_root = PANEL if panel else GATE
    parent = base.read(parent_root / "plan.json")
    inputs = base.read(parent_root / "inputs.json")
    gate_root = RISK_GATE if risk_repair else GATE
    gate = base.read(gate_root / "plan.json")
    paths = [parent_root / "plan.json", parent_root / "inputs.json", gate_root / "plan.json"]
    if digest(parent["parent"]["config"]) != digest(gate["parent"]["config"]):
        raise ValueError("unchanged pinned model and controls required")
    if panel:
        repaired = RETRY / "analysis/repaired_gate.json"
        if base.read(repaired).get("passed") is not True or len(inputs) != 16:
            raise ValueError(
                "actual corrected control gate required before consumed-panel extension"
            )
        paths.append(repaired)
        if risk_repair:
            risk = RISK_GATE / "analysis/decision.json"
            safety = GUARDED_PANEL / "analysis/safety_decision.json"
            if (
                base.read(risk).get("execution_passed") is not True
                or base.read(safety).get("passed") is not True
            ):
                raise ValueError("actual repair/preloaded parity and previous safety must pass")
            paths += [risk, safety]
        selected = (
            ("anchor_risk_repair", "anchor_repair_disabled")
            if risk_repair
            else ("guarded_full", "guarded_parent_reserved")
        )
        policies = {name: gate["component_policies"][name] for name in selected}
    else:
        if len(inputs) != 2:
            raise ValueError("only two existing missing inactive-control rows may be retried")
        policies = {
            "guarded_parent_reserved": gate["component_policies"]["guarded_parent_reserved"]
        }
        for i in (0, 1):
            old = base.read(next((GATE / f"workers/{i}").glob("[0-9][0-9]-guarded_full.json")))
            if not (
                old.get("all_parent_counterfactual_logits_bitwise")
                and old.get("all_128_emitted_parent_tokens_identical")
            ):
                raise ValueError("retain already valid actual live-parent controls")
    reused = {}
    for i, sample in enumerate(inputs):
        reused[str(i)] = {}
        for arm in ("full_kv", "parent"):
            p = next((parent_root / f"workers/{i}").glob(f"[0-9][0-9]-{arm}.json"))
            row = base.read(p)
            if row["status"] != "completed" or row["sample_id"] != sample["sample_id"]:
                raise ValueError("completed same-ID quality control required")
            if sha(Path(row["logits_path"])) != row["logits_sha256"]:
                raise ValueError("original control prediction hash changed")
            reused[str(i)][arm] = str(p)
            paths += [p, Path(row["logits_path"]), Path(row["manifest_path"])]
    for p, expected in parent["bound_files"].items():
        if sha(Path(p)) != expected:
            raise ValueError("original immutable input changed")
        paths.append(Path(p))
    root.mkdir(parents=True)
    write_once(root / "inputs.json", inputs)
    with (root / "cpu.txt").open("x") as output:
        result = subprocess.run(
            [
                base.PYTHON,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "mosaickv/tests/unit/test_live_anchor_guard.py",
                "mosaickv/tests/unit/test_skip_discarded_exact.py",
                "mosaickv/tests/unit/test_storage_six_overlay.py",
                "mosaickv/tests/unit/test_anchor_risk_repair.py",
            ],
            cwd=SOURCE,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    write_once(
        root / "cpu.json", dict(passed=result.returncode == 0, log_sha=sha(root / "cpu.txt"))
    )
    if result.returncode:
        raise RuntimeError("CPU gate failed; no GPU submission")
    paths += [root / "inputs.json", root / "cpu.json"]
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            parent=parent["parent"],
            options=parent["options"],
            bound_files={str(p): sha(p) for p in paths},
            reused_controls=reused,
            component_policies=policies,
            arms=["full_kv", "parent", *policies],
            component_reference_arm="anchor_risk_repair"
            if risk_repair
            else "guarded_full"
            if panel
            else "full",
            live_parent_decision_risk_repair_study=risk_repair,
            risk_repair=risk_repair,
            storage_boundary_version="shared_initial_final_peak_v2",
            guard_predictions=32,
            guard_kl=0.02,
            guard_logit=1.0,
            output_predictions=128,
            source_replacement_study=False,
            fresh_evaluation_authorized=False,
            larger_campaign_authorized=False,
            original_compaction_design_reproduction=False,
            classification="same_consumed_panel_live_anchor_safety"
            if panel
            else "selective_control_retry",
            panel=panel,
            parent_root=str(parent_root),
            quality_controls_reused_with_original_manifest=True,
            current_parent_warmup_requires_all_128_logits_bitwise=True,
            stopping_rule="after_same_consumed_panel_analysis_no_new_cohort_or_paper_campaign",
        ),
    )
    import six_mechanism_feasibility as infra

    infra.SOURCE, infra.SHELL = SOURCE, SOURCE / "mosaickv/slurm/storage_six_anchor_followup.sbatch"
    array = infra.submit(root, "worker", array=len(inputs), phase="anchor")
    write_once(
        root / "anchor_admission.json",
        dict(
            timezone="America/New_York",
            window="all_times",
            maximum_worker_minutes=120,
            maximum_a100s=6,
            six_a100s_explicitly_authorized=True,
            arrays=[array],
            adopt_existing_holds=[array],
        ),
    )
    controller = infra.submit(root, "controller", phase="anchor")
    analyzer = infra.submit(root, "analyze", after=array, phase="anchor")
    receipt = dict(
        array=array,
        controller=controller,
        analyzer=analyzer,
        maximum_aggregate_a100s=6,
        gpu_count_per_worker=1,
        cpus_per_worker=12,
        worker_limit_minutes=30,
    )
    write_once(root / "anchor_submission.json", receipt)
    print(json.dumps(receipt), flush=True)


def analyze(root, expected):
    base.analyze(root, expected)
    plan = base.checked(root, expected)
    decision = base.read(root / "analysis/decision.json")
    rows = base.read(root / "analysis/rows.json")
    guards = [r for r in rows if r["arm"].startswith(("guarded_", "anchor_"))]
    passed = decision["execution_passed"] and all(
        r.get("all_parent_counterfactual_logits_bitwise")
        and r.get("all_128_emitted_parent_tokens_identical")
        for r in guards
    )
    proof = dict(
        passed=passed,
        controls_reused_not_remeasured=True,
        guard_records=len(guards),
        attempts_retained_in_original_roots=True,
        fresh_evaluation_authorized=False,
        larger_campaign_authorized=False,
        all_six_superiority_established=False,
        whole_request_peak_certified=False,
    )
    if not plan["panel"]:
        complete = base.read(GATE / "analysis/rows.json")
        valid = [r for r in complete if r["arm"] != "guarded_parent_reserved"]
        proof["retained_valid_original_records"] = len(valid)
        proof["original_failed_attempts"] = 2
        proof["passed"] = (
            passed and len(valid) == 28 and all(r["status"] == "completed" for r in valid)
        )
        from mosaickv.evaluation.storage_component_gate import storage_match

        pairs = {}
        for sample in base.read(root / "inputs.json"):
            a = next(
                r
                for r in valid
                if r["sample_id"] == sample["sample_id"] and r["arm"] == "guarded_full"
            )
            b = next(r for r in guards if r["sample_id"] == sample["sample_id"])
            pairs[sample["sample_id"]] = storage_match([a, b])
        proof["new_guard_storage_matching"] = pairs
        proof["passed"] &= all(r["all_boundaries_matched"] for r in pairs.values())
        write_once(root / "analysis/repaired_gate.json", proof)
        if proof["passed"]:
            prepare(GUARDED_PANEL, panel=True)
    else:
        proof["storage_matching"] = decision["storage_matching"]
        proof["diagnostic_cider"] = decision["diagnostic_cider"]
        proof["veto_counts"] = [
            r["mechanism_evidence"]["live_anchor_guard"]["prototype_or_exact_changes_vetoed"]
            for r in guards
            if r["arm"] == ("anchor_risk_repair" if plan.get("risk_repair") else "guarded_full")
        ]
        if plan.get("risk_repair"):
            proof["natural_repairs"] = [
                event
                for r in guards
                if r["arm"] == "anchor_risk_repair"
                for event in r["mechanism_evidence"]["repair_events"]
                if not event["stress_control"]
            ]
            proof["no_repair_control_events"] = sum(
                len(r["mechanism_evidence"]["repair_events"])
                for r in guards
                if r["arm"] == "anchor_repair_disabled"
            )
            proof["passed"] &= proof["no_repair_control_events"] == 0
        proof["passed"] &= all(
            x["all_boundaries_matched"] for x in decision["storage_matching"].values()
        )
        write_once(root / "analysis/safety_decision.json", proof)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("prepare", "worker", "analyze", "controller"))
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--plan-sha")
    p.add_argument("--phase", default="anchor")
    p.add_argument("--risk-repair", action="store_true")
    a = p.parse_args()
    base.SOURCE = SOURCE
    if a.action == "prepare":
        prepare(a.root, panel=a.risk_repair, risk_repair=a.risk_repair)
    elif a.action == "worker":
        return base.worker(a.root, a.plan_sha)
    elif a.action == "analyze":
        analyze(a.root, a.plan_sha)
    else:
        base.checked(a.root, a.plan_sha)
        return subprocess.call(
            [
                base.PYTHON,
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(a.root / "anchor_admission.json"),
                "--state",
                str(a.root / "anchor_admission_state.json"),
                "--apply",
                "--watch",
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
