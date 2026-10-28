import json
from pathlib import Path

import pytest


def fixture(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    import evicastkv_full_graph_retry as retry

    from mosaickv.evaluation.fox_campaign import write_once

    write_once(
        tmp_path / "protocol_lock.json",
        dict(
            source_sha="old",
            resources={},
            authorization={},
            budgets_mib={"coco": [20, 40]},
            cohorts={"coco": {"profiler": {"sample_ids": [str(i) for i in range(8)]}}},
        ),
    )
    write_once(tmp_path / "analysis/mechanisms/GATE_mechanisms.json", dict(status="passed"))
    write_once(
        tmp_path / "analysis/full_graph_pilot/GATE_full_graph_pilot.json",
        dict(
            status="blocked",
            tests_actually_executed=dict(completed=0),
            blockers=[dict(error="UnboundLocalError: stable_prompt_ids")],
        ),
    )
    return retry


def test_retry_exactly_sixteen_requests_and_no_new_cohort(tmp_path, monkeypatch):
    retry = fixture(tmp_path, monkeypatch)
    lock = retry.derive_retry_lock(tmp_path, "new")
    assert lock["retry_registration"]["expected_requests"] == 16
    assert lock["retry_registration"]["new_identities"] == 0
    assert not lock["authorization"]["automatic_stage_progression"]
    assert not lock["authorization"]["fresh_confirmation_authorized"]
    assert lock["source_sha"] == "new"


@pytest.mark.parametrize(
    "completed,error", [(1, "UnboundLocalError: stable_prompt_ids"), (0, "OOM")]
)
def test_completed_or_other_failure_cannot_be_blindly_retried(
    tmp_path, monkeypatch, completed, error
):
    retry = fixture(tmp_path, monkeypatch)
    path = tmp_path / "analysis/full_graph_pilot/GATE_full_graph_pilot.json"
    gate = json.loads(path.read_text())
    gate["tests_actually_executed"]["completed"] = completed
    gate["blockers"] = [dict(error=error)]
    path.write_text(json.dumps(gate))
    with pytest.raises(ValueError):
        retry.derive_retry_lock(tmp_path, "new")
