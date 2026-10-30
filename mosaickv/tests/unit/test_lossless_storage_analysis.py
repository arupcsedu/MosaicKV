"""Synthetic-only analysis fixture; no measured results or benchmark scores."""

import json
from dataclasses import asdict

import evicastkv_lossless_storage as runner
import pytest
import torch

from mosaickv.config import DraftAllocationGuardConfig
from mosaickv.evaluation.fox_campaign import sha


@pytest.mark.parametrize("reuse,greedy", [(False, False), (True, False), (False, True)])
def test_lossless_analysis_keeps_common_storage_distinct_from_selector_claim(
    tmp_path, monkeypatch, reuse, greedy
):
    jobs = []
    for index in range(2):
        worker, old, full = (tmp_path / f"{kind}/{index}" for kind in ("workers", "old", "full"))
        for folder in (worker, old, full):
            folder.mkdir(parents=True)
        logits = torch.zeros((32, 5))
        measurements, captured, previous = {}, {}, {}
        for role in ("topk_synthetic_fixture", "evicast_synthetic_fixture"):
            measurements[role] = {
                mode: dict(
                    storage={
                        k: 64 if mode == "lossless_storage" else 110 for k in runner.common.STORAGE
                    },
                    construction_seconds=0.0,
                    diagnostic_decode_wall_seconds=0.0,
                )
                for mode in runner.MODES
            }
            captured[role] = {mode: logits for mode in runner.MODES}
            if greedy:
                captured[role]["standard_exact"] = logits
            previous[role] = {"anchor": {"storage": {k: 100 for k in runner.common.STORAGE}}}
        (old / "transactions.json").write_text(json.dumps(previous))
        torch.save({"logits": {"full_kv_reference": logits}}, full / "captures.pt")
        torch.save(captured, worker / "logits.pt")
        (worker / "measurements.json").write_text(json.dumps(measurements))
        if greedy:
            (worker / "lossless_retention_one.json").write_text(
                json.dumps(dict(tokens=32, bitwise=True, maximum_logit_difference=0.0))
            )
        (worker / "result.json").write_text(
            json.dumps(
                dict(
                    status="completed",
                    failures=[],
                    source_sha="c" * 40 if reuse and index == 0 else "a" * 40,
                    plan_sha256="d" * 64 if reuse and index == 0 else "b" * 64,
                    outputs={
                        name: sha(worker / name) for name in ("logits.pt", "measurements.json")
                    },
                )
            )
        )
        jobs.append(
            dict(
                index=index,
                task=f"synthetic{index}",
                delimiter_worker=str(old),
                parent_worker=str(full),
            )
        )
        if reuse and index == 0:
            jobs[-1].update(
                reused_worker=str(worker),
                producer_source_sha="c" * 40,
                producer_plan_sha256="d" * 64,
            )
    plan = dict(
        jobs=jobs,
        source_sha="a" * 40,
        guard=asdict(DraftAllocationGuardConfig()),
        maximum_reference_kl=0.02,
        maximum_logit_difference=1.0,
    )
    if greedy:
        strict = tmp_path / "prior_gate.json"
        strict.write_text(
            json.dumps(
                dict(
                    comparisons=[
                        dict(
                            task=f"synthetic{i}",
                            role=role,
                            unchanged_relative_guard_passed=True,
                            base_selection_absolute_limits_passed=False,
                        )
                        for i in range(2)
                        for role in ("topk_synthetic_fixture", "evicast_synthetic_fixture")
                    ]
                )
            )
        )
        plan.update(
            gpu_constructor=True,
            trajectory="own_greedy_32_tokens_compared_to_standard_exact_cache",
            strict_prefix_guard_reference=str(strict),
        )
    monkeypatch.setattr(runner.common, "inputs", lambda *_args: plan)
    runner.analyze(tmp_path, "b" * 64)
    gate = json.loads((tmp_path / "analysis/GATE.json").read_text())
    assert gate["storage_feasibility_passed"]
    assert gate["common_optimization_applied_to_both_methods"]
    assert not gate["selector_superiority_established"]
    assert not gate["paper_campaign_authorized"]
    assert gate["reused_worker_count"] == int(reuse)
    if greedy:
        assert all(c["free_greedy_32_token_and_logit_parity"] for c in gate["comparisons"])
        assert all(not c["base_selection_absolute_limits_passed"] for c in gate["comparisons"])
