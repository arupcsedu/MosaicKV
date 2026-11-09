#!/usr/bin/env python3
"""One preregistered older-history variant; both arms get the common headers."""

from pathlib import Path

import early_scale_header_study as early
import exact_scale_replay_study as base

from mosaickv.evaluation.fox_campaign import sha, write_once

PARENT = Path("/scratch/djy8hg/runs/evicastkv/early-exact-headers-20261108-v1")
ARMS = ["storage_common_headers", "selective_history64"]
base.ARMS = ARMS
base.common.WORKER_WRAPPER = base.SOURCE / "mosaickv/slurm/selective_history.sbatch"
base.EXTRA_TESTS = (
    "test_early_scale_headers.py",
    "test_async_text_handoff.py",
    "test_history_source_precision.py",
    "test_selective_history_study.py",
)
original_operator = base.operator
original_analyze = base.analyze


def prepare(root):
    gate = base.common.read(PARENT / "analysis/decision.json")
    if not gate["complete"] or not gate["all_initial_final_peak_savings"]:
        raise ValueError("passed complete common header gate required before history test")
    early.PARENT = PARENT
    early.prepare(
        root,
        overrides=dict(
            study="selective_older_history_exact_recent64_v1",
            measurement_type="consumed_selective_history_component_ablation",
            arms=ARMS,
            selective_history=dict(
                feature_group=32, exact_recent_tokens=64, all_128_numerical_guard_required=True
            ),
        ),
    )


def fidelity(actual, reference, p, arm):
    from compact_hybrid_study import fidelity as original

    proof = original(actual, reference, strict=arm == ARMS[0], p=p)
    a, b = actual.double(), reference.double()
    full_kl = float((b.softmax(-1) * (b.log_softmax(-1) - a.log_softmax(-1))).sum(-1).max())
    full_difference = float((a - b).abs().max())
    proof.update(
        maximum_128_token_KL_fp64=full_kl,
        maximum_128_logit_difference=full_difference,
        all_128_KL_is_same_prefix_evidence=proof["all_128_tokens_match"],
        history_deliberately_lossy=arm == ARMS[1],
    )
    proof["passed"] &= (
        proof["all_128_tokens_match"]
        and full_kl <= p["maximum_KL"]
        and full_difference <= p["maximum_logit_difference"]
    )
    return proof


def operator(root, p, folder, i):
    import torch

    from mosaickv.evaluation.history_source_precision import pack_source, unpack_into

    source = torch.randn(1, 1, 16, 4096, device="cuda").bfloat16()
    body = pack_source(source, group=32)
    grouped = body.codes.float().view(1, 1, 16, 128, 32)
    expected = ((grouped * body.scale[..., None]) + body.minimum[..., None]).bfloat16()
    output = torch.full((32, 4096), -91, device="cuda", dtype=torch.bfloat16)
    unpack_into(body, output)
    if not torch.equal(output[:16].flatten(), expected.flatten()) or not bool(
        (output[16:] == -91).all()
    ):
        raise RuntimeError("group32 history CUDA oracle or sentinel failure")
    write_once(
        folder / "history-operator.json",
        dict(
            passed=True,
            group32_original_order_oracle=True,
            sentinels_intact=True,
            model_weights_loaded=False,
        ),
    )
    original_operator(root, p, folder, i)


def model(root, p, folder, i):
    from mosaickv.evaluation.early_scale_headers import use as early_headers
    from mosaickv.evaluation.exact_scale_replay import use as late_headers
    from mosaickv.evaluation.history_source_precision import use as history

    def context(stack, arm):
        result = dict(
            early=stack.enter_context(early_headers()), late=stack.enter_context(late_headers())
        )
        result["selective_history"] = (
            stack.enter_context(
                history(
                    **{
                        k: p["selective_history"][k]
                        for k in ("feature_group", "exact_recent_tokens")
                    }
                )
            )
            if arm == ARMS[1]
            else None
        )
        return result

    base.common.extra_model_context = context
    base.common.assess_fidelity = fidelity
    base.common.model(root, p, folder, i)


def analyze(root, p):
    original_analyze(root, p)
    proof = base.common.read(root / "analysis/decision.json")
    passed = proof["complete"]
    pairs = []
    for i in (0, 1):
        rows = [r for r in proof["rows"] if r["input"] == i]
        if len(rows) != 2 or any(r["status"] != "completed" for r in rows):
            passed = False
            continue
        a, b = rows
        h = b.get("common_header", {}).get("selective_history") or {}
        gate = bool(
            a["initial_GPU_KV_bytes"] == b["initial_GPU_KV_bytes"]
            and b["final_GPU_KV_bytes"] < a["final_GPU_KV_bytes"]
            and b["whole_request_KV_plus_workspace_peak_bytes"]
            <= a["whole_request_KV_plus_workspace_peak_bytes"]
            and b["numerically_admissible"]
            and h.get("conversions", 0) > 0
        )
        passed &= gate
        pairs.append(dict(input=i, component_gate=gate, history=h))
    write_once(
        root / "analysis/history-component-decision.json",
        dict(
            component_gate_passed=passed,
            pairs=pairs,
            initial_storage_advantage_over_common_storage_comparator=False,
            complete_superiority_established=False,
            independent_quality_established=False,
            strict_latency_established=False,
            fresh_cohort_authorized=False,
            no_threshold_changes=True,
            registered_128_guard=True,
            plan_sha=sha(root / "plan.json"),
        ),
    )


if __name__ == "__main__":
    base.prepare, base.operator, base.model, base.analyze = prepare, operator, model, analyze
    base.main()
