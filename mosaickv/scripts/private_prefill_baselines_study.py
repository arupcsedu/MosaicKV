#!/usr/bin/env python3
"""Apply common private-score optimization to all four compatible local baselines."""

import copy
import gzip
import json
from pathlib import Path

import exact_scale_replay_study as base

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once
from mosaickv.measurements.kv_timeline import valid_whole_request_certificate

SOURCE = base.SOURCE
PARENT = Path("/scratch/djy8hg/runs/evicastkv/common-baseline-headers-20261108-v1")
METHODS = ("prompt_attention_topk", "prefixkv_reimpl", "vl_cache_reimpl", "lookm_reimpl")
ARMS = [method + ":" + flag for method in METHODS for flag in ("reference", "private")]
base.common.WORKER_WRAPPER = SOURCE / "mosaickv/slurm/private_prefill_baselines.sbatch"
base.EXTRA_TESTS = (
    *base.EXTRA_TESTS,
    "test_private_prefill_scores.py",
    "test_tiled_softmax.py",
    "test_streaming_baselines.py",
    "test_request_ownership.py",
)


def prepare(root):
    if root.exists():
        raise FileExistsError("new immutable baseline control root required")
    if not base.common.read(root.parent / "analysis/private-score-decision.json")[
        "bitwise_and_peak_gate"
    ]:
        raise ValueError("FullKV and pretrained-reader numerical/peak gate required")
    p = base.common.read(PARENT / "plan.json")
    for path in (
        PARENT / "plan.json",
        root.parent / "plan.json",
        root.parent / "analysis/private-score-decision.json",
        root.parent / "analysis/decision.json",
        *(root.parent / f"model/{i}/fullkv-identity.json" for i in (0, 1)),
    ):
        p["bound_files"][str(path)] = sha(path)
    p.update(
        source_sha=clean_source(SOURCE),
        study="common_private_prefill_four_baseline_controls_v1",
        measurement_type="consumed_baseline_common_operator_parity",
        arms=ARMS,
        execution_methods={a: a.split(":")[0] for a in ARMS},
        maximum_allocator_events=8000000,
        maximum_allocator_entries_increased_not_safety_disabled=True,
        all_methods_receive_same_common_optimization=True,
        comparison=(
            "within-method before/after private prefill scores; not cross-method byte matching"
        ),
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
    # The same source/root already executed actual operator tests and FullKV
    # parity. This stage verifies immutable evidence only; no synthetic claim.
    gate = base.common.read(root.parent / "analysis/private-score-decision.json")
    write_once(
        folder / "result.json",
        dict(
            status="completed",
            passed=gate["bitwise_and_peak_gate"],
            actual_operator_evidence=str(root.parent / "operator"),
            operator_kernels_NOT_reexecuted=True,
        ),
    )


def model(root, p, folder, i):
    from compact_hybrid_study import fidelity

    from mosaickv.evaluation.exact_scale_replay import use as headers
    from mosaickv.evaluation.mixed_verification_barrier import use as barrier
    from mosaickv.evaluation.paired_history_source import use as pairs
    from mosaickv.evaluation.tiled_softmax import use as tile

    references = {}

    def context(stack, arm):
        return dict(
            late=stack.enter_context(headers()),
            paired_source=stack.enter_context(pairs()),
            verification_barrier=stack.enter_context(barrier()),
            private_scores=stack.enter_context(
                tile(rows=p["tiled_softmax_rows"], inplace=arm.endswith(":private"))
            ),
        )

    def compare(actual, unused, plan, arm):
        method, flag = arm.split(":")
        if flag == "reference":
            references[method] = actual.clone()
        if method not in references:
            raise ValueError("matching baseline reference required")
        proof = fidelity(actual, references[method], strict=True, p=plan)
        proof["within_same_local_baseline_precision_variant"] = True
        proof["NOT_official_reproduction_or_independent_quality"] = True
        return proof

    base.common.extra_model_context, base.common.assess_fidelity = context, compare
    p = copy.deepcopy(p)
    p["controls"]["options"] = p["baseline_options_by_input"][str(i)]
    base.common.model(root, p, folder, i)


def analyze(root, p):
    errors = []
    pairs = []
    for i in (0, 1):
        path = root / f"model/{i}/raw.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        if [r["arm"] for r in rows] != ARMS:
            errors.append(dict(input=i, error="incomplete or duplicate registered baseline arms"))
        indexed = {r["arm"]: r for r in rows}
        for method in METHODS:
            local = [indexed.get(method + ":" + flag) for flag in ("reference", "private")]
            if any(
                not r or r["status"] != "completed" or not r.get("numerically_admissible")
                for r in local
            ):
                errors.append(
                    dict(
                        input=i,
                        method=method,
                        error="incomplete or numerical failure",
                        details=[r.get("error") if r else "missing" for r in local],
                    )
                )
                continue
            a, b = local
            for row in local:
                path = Path(row["ledger_path"])
                with gzip.open(path, "rt") as stream:
                    if sha(path) != row["ledger_sha"] or not valid_whole_request_certificate(
                        json.load(stream)
                    ):
                        errors.append(
                            dict(input=i, method=method, error="incomplete ownership certification")
                        )
            info = b["common_header"]["private_scores"]
            passed = bool(
                a["initial_GPU_KV_bytes"] == b["initial_GPU_KV_bytes"]
                and a["final_GPU_KV_bytes"] == b["final_GPU_KV_bytes"]
                and info["private_score_buffer_reused"]
                and info["calls"] >= 32
                and a["whole_request_KV_plus_workspace_peak_bytes"] is not None
                and b["whole_request_KV_plus_workspace_peak_bytes"] is not None
                and b["whole_request_KV_plus_workspace_peak_bytes"]
                <= a["whole_request_KV_plus_workspace_peak_bytes"]
            )
            if not passed:
                errors.append(
                    dict(
                        input=i,
                        method=method,
                        error="storage equality/private operator engagement failed",
                    )
                )
            pairs.append(
                dict(
                    input=i,
                    method=method,
                    gate=passed,
                    all_128_logits_bitwise=True,
                    initial_bytes=b["initial_GPU_KV_bytes"],
                    final_bytes=b["final_GPU_KV_bytes"],
                    peak_before_bytes=a["whole_request_KV_plus_workspace_peak_bytes"],
                    peak_after_bytes=b["whole_request_KV_plus_workspace_peak_bytes"],
                )
            )
    folder = root / "analysis"
    folder.mkdir()
    complete = len(pairs) == 8 and not errors
    write_once(
        folder / "decision.json",
        dict(
            complete=complete,
            pairs=pairs,
            errors=errors,
            fresh_cohort_authorized=False,
            independent_quality_established=False,
            cross_method_byte_matching_established=False,
            superiority_established=False,
        ),
    )
    if complete:
        from private_prefill_scores_timing import prepare as prepare_timing

        prepare_timing(root.parent / "paired-timing")


if __name__ == "__main__":
    original_launch = base.common.launch
    # Parent operator/FullKV gates already passed and are hash-bound above.
    # Never reserve GPUs merely to reread CPU evidence.
    base.common.launch = lambda root, stage: original_launch(
        root, "model" if stage == "operator" else stage
    )
    base.prepare, base.operator, base.model, base.analyze = prepare, operator, model, analyze
    base.main()
