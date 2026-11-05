#!/usr/bin/env python3
"""Hash-bound descriptive corrections; never rewrite producer measurements."""

import argparse
import json
from pathlib import Path

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
RUNS = Path("/scratch/djy8hg/runs/evicastkv")
SPECS = (
    (
        "deferred-exact",
        "deferred_exact_constructor_screen",
        "consumed_exact_selective_text_constructor",
        "consumed_deferred_exact_constructor",
    ),
    (
        "graph-accounting",
        "consumed_graph_accounting_execution_screen",
        "consumed_exact_selective_text_constructor",
        "consumed_graph_accounting_execution_screen",
    ),
    (
        "event-only-reader",
        "event_only_native_model_attribution",
        "consumed_native_actual_model_attribution",
        "consumed_event_only_native_model_attribution",
    ),
    (
        "host-dispatch",
        "consumed_host_decode_attribution",
        "consumed_native_actual_model_attribution",
        "consumed_host_decode_attribution",
    ),
)


def annotate(root, stage, original, corrected):
    plan_path, decision_path = root / "plan.json", root / "analysis/decision.json"
    plan = json.loads(plan_path.read_text())
    decision = json.loads(decision_path.read_text())
    if (
        plan["stage"] != stage
        or decision["complete"] is not True
        or decision["failures"]
        or decision["source_sha"] != plan["source_sha"]
        or decision["plan_sha256"] != sha(plan_path)
    ):
        raise ValueError("complete hash-bound producer decision and exact protocol stage required")
    manifests = []
    for i in (0, 1):
        path = root / f"workers/{i}/manifest.json"
        data = json.loads(path.read_text())
        if (
            data["variant"] != original
            or data["source_sha"] != plan["source_sha"]
            or data["plan_sha256"] != sha(plan_path)
            or data["independent_quality_N"] != 0
        ):
            raise ValueError("original consumed-only manifest and unchanged source/plan required")
        manifests.append(dict(path=str(path), sha256=sha(path), producer_variant=original))
    return dict(
        root=str(root),
        measurement_source_sha=plan["source_sha"],
        plan_sha256=sha(plan_path),
        decision_sha256=sha(decision_path),
        manifests=manifests,
        descriptive_variant_correction=corrected,
        raw_artifacts_and_producer_config_hashes_unchanged=True,
        numerical_controls_or_acceptance_NOT_changed=True,
        retrospective_provenance_annotation_NOT_new_measurement=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-proof-root", type=Path)
    parser.add_argument("--plan-sha")
    args = parser.parse_args()
    revision = clean_source(SOURCE)
    if args.cpu_proof_root is not None:
        if args.plan_sha is None:
            raise ValueError("immutable plan hash required for CPU-only analysis")
        import evicastkv_lt_latency

        failures = []
        try:
            plan = evicastkv_lt_latency.driver.checked(args.cpu_proof_root, args.plan_sha)
            evicastkv_lt_latency.driver.cpu_proof(args.cpu_proof_root, plan)
        except Exception as error:
            failures.append(f"{type(error).__name__}: {error}")
        write_once(
            args.output,
            dict(
                complete=not failures,
                failures=failures,
                source_sha=revision,
                plan_sha256=args.plan_sha,
                CPU_only=True,
                no_GPU_submission=True,
            ),
        )
        return
    if args.plan_sha is not None:
        raise ValueError("CPU plan hash supplied without CPU proof root")
    rows = [
        annotate(RUNS / f"{name}-consumed-20261105-v1", stage, old, new)
        for name, stage, old, new in SPECS
    ]
    write_once(
        args.output, dict(annotation_source_sha=revision, rows=rows, independent_quality_N=0)
    )


if __name__ == "__main__":
    main()
