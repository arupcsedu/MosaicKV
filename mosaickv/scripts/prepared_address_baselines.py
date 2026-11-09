#!/usr/bin/env python3
"""Same address validation on compatible local baselines, before paired timing."""

import copy
import json

import private_prefill_baselines_study as baseline

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

base = baseline.base
SOURCE = baseline.SOURCE
PARENT = baseline.PARENT
METHODS = baseline.METHODS
ARMS = baseline.ARMS
base.common.WORKER_WRAPPER = SOURCE / "mosaickv/slurm/prepared_address_baselines.sbatch"
base.EXTRA_TESTS = (*base.EXTRA_TESTS, "test_prepared_address_baselines.py")


def policy(arm):
    return dict(private_prefill_scores=True, prepared_addresses=arm.endswith(":private"))


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable common-address control root required")
    gate = root.parent / "analysis/address-decision.json"
    if not base.common.read(gate)["bitwise_and_storage_gate"]:
        raise ValueError("same-input pretrained address parity required")
    p = base.common.read(PARENT / "plan.json")
    for path in (PARENT / "plan.json", root.parent / "plan.json", gate):
        p["bound_files"][str(path)] = sha(path)
    p.update(
        source_sha=clean_source(SOURCE),
        study="common_bulk_address_four_baseline_controls_v2",
        measurement_type="consumed_baseline_common_operator_parity",
        arms=ARMS,
        execution_methods={a: a.split(":")[0] for a in ARMS},
        maximum_allocator_events=8000000,
        comparison="within-method original Tensor versus bulk-validated address transport",
        common_prefill_optimization_identical_in_both_arms=True,
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
    gate = base.common.read(root.parent / "analysis/address-decision.json")
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
    from mosaickv.evaluation.paired_history_source import use as pairs
    from mosaickv.evaluation.prepared_address_dispatch import use as addresses
    from mosaickv.evaluation.tiled_softmax import use as tile

    references = {}

    def context(stack, arm):
        return dict(
            late=stack.enter_context(headers()),
            paired_source=stack.enter_context(pairs()),
            private_scores=stack.enter_context(tile(rows=p["tiled_softmax_rows"], inplace=True)),
            prepared_addresses=stack.enter_context(
                addresses(policy(arm)["prepared_addresses"], backend="cpp")
            ),
        )

    def compare(actual, unused, plan, arm):
        method, flag = arm.split(":")
        if flag == "reference":
            references[method] = actual.clone()
        if method not in references:
            raise ValueError("same-baseline Tensor-argument control required")
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
            report = row["common_header"]["prepared_addresses"]
            if not (
                report["patch_replays"] > 0
                and report["validation_backend"] == "cpp"
                and report["additional_GPU_storage_bytes"] == 0
            ):
                errors.append(dict(input=i, arm=row["arm"], error="common dispatch not engaged"))
    passed = proof["complete"] and not errors
    write_once(
        root / "analysis/address-baseline-decision.json",
        dict(complete=passed, errors=errors, independent_quality_established=False),
    )
    if passed:
        from prepared_address_timing import prepare as timing_prepare

        timing_prepare(root.parent / "paired-timing")


if __name__ == "__main__":
    base.prepare, base.operator, base.model, base.analyze = prepare, operator, model, analyze
    base.main()
