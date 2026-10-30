"""No data access or real submission in finite-scope protocol fixtures."""

import json

import evicastkv_storage_successor as study
import pytest


def test_preparation_cannot_inherit_old_timing_or_fresh_authorizations(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    (parent / "analysis").mkdir(parents=True)
    (parent / "plan.json").write_text(
        json.dumps(
            dict(jobs=[], bound_files={}, fresh_evaluation_authorized=True, obsolete_timing=True)
        )
    )
    (parent / "analysis/GATE.json").write_text(
        json.dumps(dict(status="complete_exact_storage_diagnostic", failures=[]))
    )
    (parent / "analysis/manifest.json").write_text("{}")
    monkeypatch.setattr(study, "PARENT", parent)
    compact = tmp_path / "prior_compact"
    (compact / "analysis").mkdir(parents=True)
    (compact / "analysis/GATE.json").write_text(
        json.dumps(dict(status="complete_consumed_engineering_diagnostic", failures=[]))
    )
    (compact / "plan.json").write_text("{}")
    (compact / "analysis/manifest.json").write_text("{}")
    monkeypatch.setattr(study, "COMPACT_PARENT", compact)
    monkeypatch.setattr(study, "clean_source", lambda _: "a" * 40)
    root = tmp_path / "compact"
    study.prepare(root, "compact")
    plan = study.common.read(root / "plan.json")
    assert not plan["fresh_evaluation_authorized"]
    assert not plan["calibration_authorized"]
    assert not plan["paper_campaign_authorized"]
    assert "obsolete_timing" not in plan
    assert plan["finite_next_phase_authorized"]
    assert plan["topup_initial_cap_bytes"] == 62 * 2**20
    assert plan["guard"]["absolute_tolerance"] == 1e-7
    with pytest.raises(ValueError):
        study.prepare(root, "compact")
    with pytest.raises(ValueError):
        study.prepare(tmp_path / "unknown", "other")
