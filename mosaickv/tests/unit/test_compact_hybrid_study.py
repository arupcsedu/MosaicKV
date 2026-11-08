import importlib.util
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from mosaickv.evaluation.storage_six_overlay import OverlayPolicy


def driver():
    path = Path(__file__).resolve().parents[2] / "scripts/compact_hybrid_study.py"
    spec = importlib.util.spec_from_file_location("compact_hybrid_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_policies_resolve_without_changing_original_defaults():
    module = driver()
    original = asdict(OverlayPolicy(inline_repair=True, live_anchor_guard=True))
    policies = module.policies(original)
    assert policies["hybrid_compact"].planner_similarity_limit == 0
    assert policies["hybrid_shortlist"].planner_similarity_limit == 128
    assert policies["hybrid_shortlist"].live_anchor_guard
    assert policies["hybrid_score_risk"].repair_execution == "score_risk"
    assert not policies["hybrid_score_risk"].live_anchor_guard
    assert not policies["storage_compact"].enabled
    assert original == asdict(OverlayPolicy(inline_repair=True, live_anchor_guard=True))


def test_offline_fidelity_keeps_absolute_token_and_logit_thresholds():
    module = driver()
    p = dict(predictions=128, guard_tokens=32, maximum_KL=0.02, maximum_logit_difference=1.0)
    original = torch.zeros(128, 8)
    original[:, 1] = 10
    assert module.fidelity(original, original, strict=True, p=p)["passed"]
    changed = original.clone()
    changed[3, 2] = 11
    proof = module.fidelity(changed, original, strict=False, p=p)
    assert not proof["passed"] and not proof["KL_is_same_prefix_evidence"]
    changed = original + 2  # identical probabilities, but registered absolute logit gate fails
    assert not module.fidelity(changed, original, strict=False, p=p)["passed"]
    changed = original + 0.01
    assert module.fidelity(changed, original, strict=False, p=p)["passed"]
    assert not module.fidelity(changed, original, strict=True, p=p)["passed"]
    with pytest.raises(ValueError, match="complete finite"):
        module.fidelity(original[:32], original, strict=False, p=p)


def test_submission_retains_resource_and_fail_open_dependency_controls(monkeypatch, tmp_path):
    module = driver()
    (tmp_path / "plan.json").write_text("{}")
    calls = []

    def submit(args, **kwargs):
        calls.append(args)
        return "123\n"

    monkeypatch.setattr(module.subprocess, "check_output", submit)
    module.submit(tmp_path, "native")
    assert "--gres=gpu:a100:1" in calls[0]
    assert "--reservation=bi_fox_dgx" in calls[0]
    assert "--cpus-per-task=12" in calls[0] and "--hold" in calls[0]
    assert "--array=0-1%2" in calls[0]
    module.submit(tmp_path, "follow-native", "123")
    assert "--dependency=afterany:123" in calls[1]
