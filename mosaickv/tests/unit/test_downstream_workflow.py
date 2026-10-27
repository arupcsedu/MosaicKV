"""Synthetic CPU scope/afterany failure reporting, never submit Slurm jobs."""

from copy import deepcopy

import pytest
import yaml
from bounded_history_pilot import digest, write_once
from downstream_v9 import BLOCKED, POLICY, analyze, prepare_downstream


def fixture_scope(tmp_path, monkeypatch):
    import downstream_v9 as module

    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump(dict(output_replay_from_v8="fixture-consumed-only")))
    report = tmp_path / "negative.json"
    write_once(
        report,
        dict(useful_prospective_signals=[], tasks=[dict(registered_accepted_proposals=0)] * 2),
    )
    monkeypatch.setattr(module, "SOURCE", tmp_path)
    monkeypatch.setattr(module, "prepare_output_replay", lambda _root, _sha, _path, p, **_kw: p)
    return dict(
        base_protocol="base.yaml",
        base_protocol_sha256=digest(base),
        negative_readout_report=str(report),
        negative_readout_report_sha256=digest(report),
        downstream_jacobian=deepcopy(POLICY),
        **BLOCKED,
    )


def test_preparation_reuses_only_frozen_inputs_and_fails_closed_on_scope_changes(
    tmp_path, monkeypatch
):
    scope = fixture_scope(tmp_path, monkeypatch)
    merged = prepare_downstream(tmp_path / "run", "0" * 40, tmp_path / "scope.yaml", scope)
    assert merged["output_replay_from_v8"] == "fixture-consumed-only"
    assert all(merged[key] is False for key in BLOCKED)
    for key in BLOCKED:
        with pytest.raises(ValueError, match="scope"):
            prepare_downstream(
                tmp_path / "run", "0" * 40, tmp_path / "scope.yaml", dict(scope, **{key: True})
            )
    changed = deepcopy(scope)
    changed["downstream_jacobian"]["prefixes"] = [1, 2]
    with pytest.raises(ValueError, match="controls"):
        prepare_downstream(tmp_path / "run", "0" * 40, tmp_path / "scope.yaml", changed)
    scope["negative_readout_report_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        prepare_downstream(tmp_path / "run", "0" * 40, tmp_path / "scope.yaml", scope)


def test_afterany_analysis_records_missing_workers_and_keeps_all_work_blocked(
    tmp_path, monkeypatch
):
    import head_substitution_v7

    monkeypatch.setattr(head_substitution_v7, "validate", lambda *_args: None)
    write_once(
        tmp_path / "plan.json",
        dict(
            source_sha="0" * 40,
            protocol=dict(downstream_jacobian=POLICY),
            jobs=[dict(index=i, task=f"synthetic-{i}") for i in range(2)],
            **BLOCKED,
        ),
    )
    analyze(tmp_path)
    import json

    result = json.loads((tmp_path / "analysis/report.json").read_text())
    assert result["status"] == "incomplete" and len(result["errors"]) == 2
    assert not result["diagnostic_passed"]
    assert result["action"] == "stop_substitution_strategy_no_automatic_successor"
    assert all(result[key] is False for key in BLOCKED)
    with pytest.raises(FileExistsError):
        analyze(tmp_path)
