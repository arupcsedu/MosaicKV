"""Mocked event-driven completion; never launch an agent or a Slurm job."""

import json
import subprocess
from pathlib import Path

import pytest
from complete_readout_v8_2 import codex_command, decide, review_with_codex
from submit_readout_v8_2 import dependency


def report(useful=False):
    names = ["projected_residual"] if useful else []
    return dict(
        status="completed",
        original_guard_unchanged=True,
        useful_prospective_signals=names,
        fresh_cohort_authorized=False,
        calibration_authorized=False,
        paper_campaign_authorized=False,
        tasks=[
            dict(
                registered_accepted_proposals=0,
                screening={"projected_residual": {"descriptive_signal": useful}},
            )
            for _ in range(2)
        ],
    )


def test_negative_decision_stops_without_new_inference():
    result = decide(report())
    assert result["action"] == "stop_no_prospective_signal"
    assert not result["cpu_candidate_design_authorized"] and not result["gpu_submission_authorized"]
    assert not result["paper_campaign_authorized"] and not result["calibration_authorized"]


def test_descriptive_signal_opens_only_bounded_cpu_design_not_gpu_or_promotion():
    result = decide(report(True))
    assert result["action"] == "cpu_candidate_design_only"
    assert (
        result["cpu_candidate_design_authorized"] and result["registered_accepted_proposals"] == 0
    )
    assert not result["gpu_submission_authorized"] and not result["fresh_cohort_authorized"]


@pytest.mark.parametrize(
    "change",
    [
        dict(status="failed"),
        dict(paper_campaign_authorized=True),
        dict(original_guard_unchanged=False),
        dict(useful_prospective_signals=["actual_readout_linear"]),
        dict(tasks=[]),
    ],
)
def test_bad_evidence_cannot_promote(change):
    assert decide(dict(report(True), **change))["action"] == "stop_integrity_failure"


def test_dependencies_trigger_once_after_all_ids_finish_including_failures():
    assert dependency(["123", "456_7", "123"]) == "afterany:123:456_7"
    for ids in ([], ["123;touch"], ["--all"], ["0"], ["123_1-8"]):
        with pytest.raises(ValueError):
            dependency(ids)


def test_codex_review_is_ephemeral_read_only_and_scratch_logged(tmp_path):
    command = codex_command("/mock/codex", tmp_path, tmp_path / "schema.json")
    assert command[:2] == ["/mock/codex", "exec"]
    assert "--ephemeral" in command and "--ignore-user-config" in command
    assert command[command.index("--sandbox") + 1] == "read-only"
    assert "features.shell_tool=false" in command
    assert f"log_dir={json.dumps(str(tmp_path / 'codex/log'))}" in command
    assert f"sqlite_home={json.dumps(str(tmp_path / 'codex/state'))}" in command
    assert "--dangerously-bypass-approvals-and-sandbox" not in command
    assert not any("API_KEY" in item or "CODEX_HOME=" in item for item in command)


@pytest.mark.parametrize("outcome", ["completed", "timeout", "error", "wrong_action"])
def test_agent_review_never_overrides_deterministic_decision(tmp_path, monkeypatch, outcome):
    import complete_readout_v8_2 as module

    deterministic = decide(report())

    def fake_run(command, **kwargs):
        assert command[:6] == [
            "/mock/bwrap",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/",
            "/",
        ]
        assert "--bind" in command and "/mock/codex" in command
        assert kwargs["timeout"] == 180 and kwargs["check"] is False
        assert "do not use tools" in kwargs["input"].lower()
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, 180)
        if outcome == "error":
            return subprocess.CompletedProcess(command, 1)
        (tmp_path / "codex/review.json").write_text(
            json.dumps(
                dict(
                    summary="synthetic fixture",
                    action="promote" if outcome == "wrong_action" else deterministic["action"],
                    gpu_submission_authorized=False,
                    paper_campaign_authorized=False,
                )
            )
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    monkeypatch.setattr(module.shutil, "which", lambda _name: "/mock/bwrap")
    result = review_with_codex(tmp_path, "/mock/codex", deterministic, report())
    assert (
        result["status"]
        == dict(
            completed="completed",
            timeout="unavailable",
            error="failed",
            wrong_action="rejected_scope_mismatch",
        )[outcome]
    )
    assert not result["automatic_code_or_GPU_execution"]
    assert deterministic["action"] == "stop_no_prospective_signal"


def test_codex_is_not_started_without_read_only_home_protection(tmp_path, monkeypatch):
    import complete_readout_v8_2 as module

    monkeypatch.setattr(module.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_a, **_kw: pytest.fail("must not launch unprotected Codex"),
    )
    result = review_with_codex(tmp_path, "/mock/codex", decide(report()), report())
    assert result["status"] == "unavailable"
    assert result["error"] == "read_only_home_mount_unavailable"


def test_batch_scripts_do_not_poll_or_request_gpus():
    root = Path(__file__).resolve().parents[3]
    for name in ("readout_audit_cpu.sbatch", "readout_completion.sbatch"):
        content = (root / "mosaickv/slurm" / name).read_text()
        assert 'export CUDA_VISIBLE_DEVICES=""' in content
        assert "#SBATCH --gres" not in content
        assert "squeue" not in content and "sleep" not in content and "while " not in content
