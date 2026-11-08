import json

import header_predicate_study as study


def test_prepare_only_cpu_and_retains_no_fresh_scope(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    (parent / "analysis").mkdir(parents=True)
    (parent / "plan.json").write_text(
        json.dumps(
            dict(
                bound_files={},
                guard_tokens=32,
                predictions=128,
                maximum_KL=0.02,
                maximum_logit_difference=1.0,
            )
        )
    )
    (parent / "analysis/decision.json").write_text(
        json.dumps(dict(complete=True, all_registered_component_gates_passed=True))
    )
    monkeypatch.setattr(study, "PARENT", parent)
    monkeypatch.setattr(study, "clean_source", lambda *_: "a" * 40)
    calls = []
    monkeypatch.setattr(
        study,
        "submit",
        lambda root, action, after=None, **_: calls.append((action, after)) or "fixture",
    )
    root = tmp_path / "output"
    study.prepare(root)
    p = json.loads((root / "plan.json").read_text())
    assert calls == [("cpu", None), ("after_cpu", "fixture")]
    assert p["arms"] == ["control_common", "complete_predicate"]
    assert p["guard_tokens"] == 32 and p["predictions"] == 128
    assert (
        not p["independent_quality_validation_authorized"]
        and not p["fresh_or_paper_campaign_authorized"]
    )


def test_missing_model_rows_preserve_negative_analysis(tmp_path):
    study.analyze(tmp_path, dict(arms=["control_common", "complete_predicate"]))
    result = json.loads((tmp_path / "analysis/decision.json").read_text())
    assert not result["complete"] and len(result["errors"]) == 2
    assert not (tmp_path / "header-costs.json").exists()


def test_gpu_submission_uses_registered_one_a100_twelve_cpu_and_afterany(tmp_path, monkeypatch):
    (tmp_path / "plan.json").write_text("{}")
    calls = []
    monkeypatch.setattr(
        study.subprocess, "check_output", lambda args, **kwargs: calls.append(args) or "fixture"
    )
    study.submit(tmp_path, "model", after="dependency")
    args = calls[0]
    assert "--gres=gpu:a100:1" in args and "--cpus-per-task=12" in args
    assert "--reservation=bi_fox_dgx" in args and "--hold" in args
    assert "--dependency=afterany:dependency" in args
