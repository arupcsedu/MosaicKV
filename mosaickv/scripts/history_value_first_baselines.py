#!/usr/bin/env python3
"""Apply the schedule-only optimization to every compatible local baseline."""

import copy
import json

import private_prefill_baselines_study as baseline

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

base, SOURCE, PARENT, ARMS = baseline.base, baseline.SOURCE, baseline.PARENT, baseline.ARMS
base.common.WORKER_WRAPPER = SOURCE / "mosaickv/slurm/history_value_first_baselines.sbatch"
base.EXTRA_TESTS = (*base.EXTRA_TESTS, "test_history_value_first.py")


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable baseline schedule root required")
    gate = root.parent / "analysis/value-order-decision.json"
    if not base.common.read(gate)["bitwise_and_storage_gate"]:
        raise ValueError("consumed pretrained schedule parity must pass first")
    p = base.common.read(PARENT / "plan.json")
    for path in (PARENT / "plan.json", root.parent / "plan.json", gate):
        p["bound_files"][str(path)] = sha(path)
    p.update(
        source_sha=clean_source(SOURCE),
        study="common_history_value_first_four_baselines_v1",
        measurement_type="consumed_baseline_common_operator_parity",
        arms=ARMS,
        execution_methods={a: a.split(":")[0] for a in ARMS},
        maximum_allocator_events=8000000,
        comparison="within each local baseline: identical values; reconstruction order only",
        private_prefill_optimization_identical_in_both_arms=True,
        cross_method_byte_matched_claim=False,
        fresh_or_paper_campaign_authorized=False,
        independent_quality_validation_authorized=False,
    )
    root.mkdir()
    (root / "logs").mkdir()
    write_once(root / "plan.json", p)
    cpu = base.common.submit(root, "cpu")
    write_once(
        root / "submission.json", dict(cpu=cpu, follow=base.common.submit(root, "after_cpu", cpu))
    )


def operator(root, p, folder, i):
    from compact_hybrid_study import manifest

    manifest(p, base.sha(root / "plan.json"), folder, i, gpu=True)
    gate = base.common.read(root.parent / "analysis/value-order-decision.json")
    write_once(
        folder / "result.json",
        dict(
            status="completed",
            passed=gate["bitwise_and_storage_gate"],
            actual_operator_evidence=str(root.parent / "operator"),
            operator_kernels_NOT_reexecuted=True,
        ),
    )


def model(root, p, folder, i):
    from compact_hybrid_study import fidelity

    from mosaickv.evaluation.exact_scale_replay import use as headers
    from mosaickv.evaluation.history_value_first import use as schedule
    from mosaickv.evaluation.paired_history_source import use as pairs
    from mosaickv.evaluation.tiled_softmax import use as tile

    references = {}

    def context(stack, arm):
        return dict(
            late=stack.enter_context(headers()),
            paired_source=stack.enter_context(pairs()),
            private_scores=stack.enter_context(tile(rows=p["tiled_softmax_rows"], inplace=True)),
            history_value_first=stack.enter_context(schedule(arm.endswith(":private"))),
        )

    def compare(actual, unused, plan, arm):
        method, flag = arm.split(":")
        if flag == "reference":
            references[method] = actual.clone()
        if method not in references:
            raise ValueError("matching original-schedule baseline required")
        proof = fidelity(actual, references[method], strict=True, p=plan)
        proof["within_same_local_baseline_precision_variant"] = True
        proof["NOT_official_reproduction_or_independent_quality"] = True
        return proof

    base.common.extra_model_context, base.common.assess_fidelity = context, compare
    p = copy.deepcopy(p)
    p["controls"]["options"] = p["baseline_options_by_input"][str(i)]
    base.common.model(root, p, folder, i)


def analyze(root, p):
    baseline.analyze(root, p)
    proof = base.common.read(root / "analysis/decision.json")
    errors = []
    for i in (0, 1):
        path = root / f"model/{i}/raw.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        for row in rows:
            if not row["arm"].endswith(":private") or row["status"] != "completed":
                continue
            info = row["common_header"]["history_value_first"]
            if not (
                info["early_values"] >= 32
                and info["additional_GPU_storage_bytes"] == 0
                and info["early_values"] == info["skipped_source_unpacks"]
            ):
                errors.append(dict(input=i, arm=row["arm"], error="common schedule not engaged"))
    passed = proof["complete"] and not errors
    write_once(
        root / "analysis/value-order-baseline-decision.json",
        dict(complete=passed, errors=errors, independent_quality_established=False),
    )
    if passed:
        from history_value_first_timing import prepare as timing_prepare

        timing_prepare(root.parent / "paired-timing")


if __name__ == "__main__":
    base.prepare, base.operator, base.model, base.analyze = prepare, operator, model, analyze
    base.main()
