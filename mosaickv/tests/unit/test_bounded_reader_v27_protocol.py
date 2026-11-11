"""CPU protocol checks; never submits Slurm or opens a model/dataset."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def protocol():
    path = Path(__file__).resolve().parents[2] / "scripts/evicastkv_bounded_reader_v27.py"
    spec = importlib.util.spec_from_file_location("bounded_reader_v27_protocol_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scope_is_consumed_inputs_only_and_no_superiority_claim(protocol):
    assert protocol.POLICY["consumed_inputs_only"] is True
    assert protocol.POLICY["fresh_cohort_authorized"] is False
    assert protocol.POLICY["paper_campaign_authorized"] is False
    assert protocol.POLICY["no_latency_or_superiority_claim"] is True
    assert protocol.POLICY["predictions"] == 128
    assert protocol.POLICY["maximum_A100s"] == 6
    assert protocol.POLICY["model_workers"] == 4
    assert len(protocol.POLICY["disabled_mechanisms"]) == 6


def test_submission_resources_and_finite_afterany_receipt(protocol, monkeypatch, tmp_path):
    submit = Mock(return_value="12345;cluster\n")
    monkeypatch.setattr(protocol.subprocess, "check_output", submit)
    job = protocol.submit(
        tmp_path, "a" * 40, "gpu", "worker", gpu=True, array=True, dependencies=("12000",)
    )
    assert job == "12345"
    command = submit.call_args.args[0]
    for option in (
        "--nodes=1",
        "--cpus-per-task=12",
        "--time=00:15:00",
        "--gres=gpu:a100:1",
        "--array=0-3%4",
        "--reservation=bi_fox_dgx",
        "--dependency=afterany:12000",
    ):
        assert option in command
    receipt = json.loads((tmp_path / "jobs/gpu-submission.json").read_text())
    assert receipt["job"] == job
    assert receipt["source_sha"] == "a" * 40
    with pytest.raises(FileExistsError):
        protocol.submit(tmp_path, "a" * 40, "gpu", "worker", gpu=True)
    assert submit.call_count == 1


@pytest.mark.parametrize("acknowledgement", ["", "error", "123 bad"])
def test_uncertain_scheduler_acknowledgement_never_fabricates_job(
    protocol, monkeypatch, tmp_path, acknowledgement
):
    monkeypatch.setattr(protocol.subprocess, "check_output", Mock(return_value=acknowledgement))
    with pytest.raises(ValueError, match="uncertain"):
        protocol.submit(tmp_path, "a" * 40, "gpu", "worker", gpu=True)
    assert not (tmp_path / "jobs/gpu-submission.json").exists()


def test_bad_dependency_is_rejected_before_submission(protocol, monkeypatch, tmp_path):
    submit = Mock()
    monkeypatch.setattr(protocol.subprocess, "check_output", submit)
    with pytest.raises(ValueError, match="numeric"):
        protocol.submit(tmp_path, "a" * 40, "gpu", "worker", dependencies=("bad",))
    submit.assert_not_called()


def test_complete_failed_worker_analysis_is_write_once_without_gpu(protocol, monkeypatch, tmp_path):
    monkeypatch.setattr(protocol, "validate", lambda root: {"source_sha": "a" * 40})
    import evicastkv_review_source_prefix as fixed

    monkeypatch.setattr(fixed, "analysis_cpu_controls", lambda: None)
    (tmp_path / "plan.json").write_text("{}")
    gate = protocol.analyze(tmp_path)
    assert gate["complete"] is False
    assert gate["source_fidelity_128_pass"] is False
    assert gate["whole_request_KV_byte_match_pass"] is False
    assert gate["paper_campaign_authorized"] is False
    assert len(gate["failures"]) == 4
    assert (tmp_path / "analysis/GATE_bounded_reader_v27.json").is_file()
