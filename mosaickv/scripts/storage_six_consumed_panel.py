#!/usr/bin/env python3
"""Frozen sixteen-consumed-image component diagnostic, not paper confirmation."""

import argparse
import json
import subprocess
from pathlib import Path

import storage_six_ablation as base

from mosaickv.evaluation.consumed_component_panel import frozen_policies, select_inputs
from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
PYTHON = base.PYTHON


def prepare(root, gate_root, after):
    source = clean_source(SOURCE)
    if root.exists() or not after or not after.isdigit():
        raise ValueError("new immutable root and explicit predecessor timing-array ID required")
    gate = base.read(gate_root / "plan.json")
    decision = base.read(gate_root / "analysis/decision.json")
    if not decision["execution_passed"] or decision["actual_rows"] != decision["expected_rows"]:
        raise ValueError("complete graph and pretrained logit parity required")
    seed = gate["component_policies"]["inline_indexed"]
    policies = frozen_policies(seed)
    ledger = base.read(base.PARENT / "ledger.json")
    excluded = [s["sample_id"] for s in base.read(gate_root / "inputs.json")]
    # Read only identity/status columns; no historical predictions or scores
    # may influence membership. A ledger entry alone does not prove consumption.
    import pyarrow.parquet as pq

    prior_rows = base.PARENT / "memory/analysis/measurements.parquet"
    prior_decision = base.PARENT / "memory/analysis/decision.json"
    if base.read(prior_decision).get("complete") is not True:
        raise ValueError("completed prior storage study required")
    observed = pq.read_table(prior_rows, columns=["sample_id", "method", "status"]).to_pylist()
    consumed = {
        r["sample_id"]
        for r in observed
        if r["method"] == "mosaickv_exact" and r["status"] == "completed"
    }
    # The parent ledger contains 258 entries, but its complete measured cohort
    # has 256. Select from that audited ID-only membership, not warmup extras.
    pool = [s for s in ledger["samples"] if s["sample_id"] in consumed]
    if not set(excluded) <= consumed:
        raise ValueError("heavily diagnosed exclusions must belong to the consumed pool")
    inputs = select_inputs(pool, excluded)
    if not {s["sample_id"] for s in inputs} <= consumed:
        raise ValueError("every diagnostic identity must be proven previously consumed")
    paths = [
        base.PARENT / "ledger.json",
        gate_root / "plan.json",
        gate_root / "analysis/decision.json",
    ]
    paths += [gate_root / "inputs.json"]
    paths += [prior_rows, prior_decision]
    paths += [Path(m["path"]) for s in inputs for m in s["media"]]
    paths += [Path(p) for p in gate["bound_files"]]
    root.mkdir(parents=True)
    write_once(root / "inputs.json", inputs)
    tests = [
        "mosaickv/tests/unit/test_consumed_component_panel.py",
        "mosaickv/tests/unit/test_storage_six_overlay.py",
        "mosaickv/tests/unit/test_inline_first_step_repair.py",
        "mosaickv/tests/unit/test_partition_graph_topk.py",
        "mosaickv/tests/unit/test_storage_error_aware.py",
    ]
    with (root / "cpu.txt").open("x") as stream:
        done = subprocess.run(
            [PYTHON, "-m", "pytest", "-q", "-o", "addopts=", *tests],
            cwd=SOURCE,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    write_once(root / "cpu.json", dict(passed=done.returncode == 0, log_sha=sha(root / "cpu.txt")))
    if done.returncode:
        raise RuntimeError("CPU gate failed; no GPU submission")
    paths += [root / "cpu.json", root / "inputs.json"]
    write_once(
        root / "plan.json",
        dict(
            source_sha=source,
            parent=gate["parent"],
            options=gate["options"],
            component_policies=policies,
            arms=["full_kv", "parent", *policies],
            bound_files={str(p): sha(p) for p in paths},
            storage_boundary_version="shared_initial_final_peak_v2",
            classification="six_component_consumed_development_panel_not_confirmation",
            previous_consumption_ledger_sha256=sha(base.PARENT / "ledger.json"),
            cohort_selection="first_sixteen_in_sha256_six-panel-v1_plus_sample_id_order",
            component_reference_arm="full",
            output_predictions=128,
            guard_predictions=32,
            guard_kl=0.02,
            guard_logit=1.0,
            fresh_evaluation_authorized=False,
            larger_campaign_authorized=False,
            source_replacement_study=False,
            gpu_start_after_any_job=after,
            stopping_rule="analysis_only_no_automatic_new_candidate_or_cohort",
            source_codec="validated_original_default_no_optional_replacement",
            all_baseline_superiority_claim_authorized=False,
        ),
    )
    import six_mechanism_feasibility as infra

    infra.SOURCE, infra.SHELL = SOURCE, SOURCE / "mosaickv/slurm/storage_six_consumed_panel.sbatch"
    array = infra.submit(root, "worker", array=len(inputs), after=after, phase="panel")
    write_once(
        root / "panel_admission.json",
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
    controller = infra.submit(root, "controller", phase="panel")
    analyzer = infra.submit(root, "analyze", after=array, phase="panel")
    receipt = dict(
        array=array,
        controller=controller,
        analyzer=analyzer,
        gpu_count_per_worker=1,
        cpus_per_worker=12,
        maximum_aggregate_a100s=6,
        worker_limit_minutes=30,
    )
    write_once(root / "panel_submission.json", receipt)
    print(json.dumps(receipt), flush=True)


def analyze(root, expected):
    # Existing complete-corpus reducer, failure preservation, storage and
    # original-parent numerical checks; never average shard CIDEr.
    base.analyze(root, expected)
    import numpy as np

    from mosaickv.evaluation.cider_resampling import ResampledCider, paired_draw_counts
    from mosaickv.evaluation.coco_all_methods import caption_text
    from mosaickv.evaluation.metrics.types import MetricSample

    plan = base.checked(root, expected)
    decision = base.read(root / "analysis/decision.json")
    rows = base.read(root / "analysis/rows.json")
    if not decision["execution_passed"]:
        write_once(
            root / "analysis/panel_stopping_decision.json",
            dict(execution_passed=False, followup_authorized=False, errors=decision["errors"]),
        )
        return
    refs = {
        s["sample_id"]: tuple(s["references"])
        for s in base.read(base.PARENT / "ledger.json")["samples"]
    }
    groups = {
        (arm, None): [
            MetricSample(r["sample_id"], caption_text(r["detail"]), refs[r["sample_id"]])
            for r in rows
            if r["arm"] == arm
        ]
        for arm in plan["arms"]
    }
    scorer = ResampledCider(groups)
    draws = paired_draw_counts(16, 10000, 20261107)
    sampled = scorer.score(draws)
    official_error = scorer.validate_official(draws[:2], sampled[:2])
    full = scorer.points.index(("full", None))
    effects = {}
    for column, (arm, _) in enumerate(scorer.points):
        delta = sampled[:, full] - sampled[:, column]
        effects[arm] = dict(
            full_minus_arm_cider=decision["diagnostic_cider"]["full"]
            - decision["diagnostic_cider"][arm],
            retrospective_unadjusted_95_interval=np.quantile(delta, [0.025, 0.975]).tolist(),
            full_minus_arm_caption_changes=sum(
                a.prediction != b.prediction
                for a, b in zip(scorer.ordered[full], scorer.ordered[column], strict=True)
            ),
        )
    write_once(
        root / "analysis/panel_stopping_decision.json",
        dict(
            execution_passed=True,
            consumed_development_only=True,
            complete_corpus_bootstrap_replicates=10000,
            official_reducer_maximum_fixture_difference=official_error,
            effects=effects,
            natural_repair_events=sum(
                len(r.get("mechanism_evidence", {}).get("repair_events", []))
                for r in rows
                if r["arm"] == "full"
            ),
            followup_authorized=False,
            superiority_established=False,
            all_six_quality_contribution_established=False,
            whole_request_peak_certified=False,
            reason=(
                "Diagnostic only; do not tune using answers/failure identities "
                "or select a claim from ablations."
            ),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "worker", "analyze", "controller"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--plan-sha")
    parser.add_argument("--graph-gate-root", type=Path)
    parser.add_argument("--after-gpu-job")
    parser.add_argument("--phase", default="panel")
    args = parser.parse_args()
    base.SOURCE = SOURCE
    if args.action == "prepare":
        prepare(args.root, args.graph_gate_root, args.after_gpu_job)
    elif args.action == "worker":
        return base.worker(args.root, args.plan_sha)
    elif args.action == "analyze":
        analyze(args.root, args.plan_sha)
    else:
        base.checked(args.root, args.plan_sha)
        return subprocess.call(
            [
                PYTHON,
                str(SOURCE / "mosaickv/scripts/control_a100_schedule.py"),
                "--policy",
                str(args.root / "panel_admission.json"),
                "--state",
                str(args.root / "panel_admission_state.json"),
                "--apply",
                "--watch",
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
