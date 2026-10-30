"""Fail-closed finite protocol checks, no real data access or submission."""

import json

import evicastkv_exact_storage as study
import pytest


def fixtures(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    (parent / "analysis").mkdir(parents=True)
    (parent / "plan.json").write_text(
        json.dumps(
            {
                "jobs": [],
                "bound_files": {},
                "source_timing": True,
                "fresh_evaluation_authorized": True,
            }
        )
    )
    (parent / "analysis/GATE.json").write_text(
        json.dumps(
            {
                "status": "complete_consumed_component_diagnostic",
                "failures": [],
            }
        )
    )
    (parent / "analysis/manifest.json").write_text("{}")
    monkeypatch.setattr(study, "PARENT", parent)
    monkeypatch.setattr(study, "PARENT_SHA", study.sha(parent / "plan.json"))
    monkeypatch.setattr(study, "clean_source", lambda _: "a" * 40)
    return parent


def test_protocol_does_not_inherit_obsolete_or_broader_permissions(tmp_path, monkeypatch):
    fixtures(tmp_path, monkeypatch)
    root = tmp_path / "stage-1"
    study.prepare(root, 1)
    plan = study.common.read(root / "plan.json")
    assert "source_timing" not in plan
    assert plan["fresh_evaluation_authorized"] is False
    assert plan["paper_campaign_authorized"] is False
    assert plan["calibration_authorized"] is False
    assert plan["selector_changes_authorized"] is False
    assert plan["finite_next_stage_authorized"] is True
    assert plan["output_lengths"] == [32]
    with pytest.raises(ValueError):
        study.prepare(root, 1)
    with pytest.raises(ValueError):
        study.prepare(tmp_path / "invalid", 5)


def test_next_stage_requires_success_and_final_stage_cannot_advance(tmp_path, monkeypatch):
    fixtures(tmp_path, monkeypatch)
    prior = tmp_path / "stage-3"
    (prior / "analysis").mkdir(parents=True)
    gate = prior / "analysis/GATE.json"
    gate.write_text(json.dumps({"status": "failed", "failures": ["bad bits"]}))
    root = tmp_path / "stage-4"
    with pytest.raises(ValueError, match="previous stage"):
        study.prepare(root, 4)
    gate.write_text(json.dumps({"status": "complete_exact_storage_diagnostic", "failures": []}))
    (prior / "plan.json").write_text("{}")
    (prior / "analysis/manifest.json").write_text("{}")
    study.prepare(root, 4)
    plan = study.common.read(root / "plan.json")
    assert plan["finite_next_stage_authorized"] is False
    assert plan["output_lengths"] == [32, 128]
