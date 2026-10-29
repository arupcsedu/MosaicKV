from pathlib import Path

import evicastkv_campaign as campaign
import evicastkv_worker
import fox_guard_launch_retry_controller as controller
import pytest


@pytest.mark.parametrize("stage", ["evidence_canary", "installed_guard_development"])
def test_actual_campaign_cli_dispatches_registered_custom_stage_without_gpu(stage, monkeypatch):
    seen = []
    monkeypatch.setattr(evicastkv_worker, "run_worker", lambda *args: seen.append(args))
    campaign.main(
        [
            "worker",
            "--root",
            "/scratch/djy8hg/mock",
            "--stage",
            stage,
            "--index",
            "1",
            "--spec-sha",
            "bound-hash",
        ]
    )
    assert seen == [(Path("/scratch/djy8hg/mock"), stage, 1, "bound-hash")]


def test_custom_stage_does_not_authorize_generic_submission():
    with pytest.raises(SystemExit) as error:
        campaign.main(
            ["development", "--root", "/scratch/djy8hg/mock", "--stage", "evidence_canary"]
        )
    assert error.value.code == 2


def test_retry_cannot_overlap_prefix_quality_lease(monkeypatch, tmp_path):
    import control_a100_schedule

    monkeypatch.setattr(controller, "PREFIX_ROOT", tmp_path)
    monkeypatch.setattr(
        control_a100_schedule, "read_queue", lambda: [dict(array="quality", gres="gres/gpu:a100:1")]
    )
    with pytest.raises(ValueError, match="quality or unknown"):
        controller.assert_gpu_lease()


def test_empty_queue_is_safe_for_bounded_retry(monkeypatch, tmp_path):
    import control_a100_schedule

    monkeypatch.setattr(controller, "PREFIX_ROOT", tmp_path)
    monkeypatch.setattr(control_a100_schedule, "read_queue", lambda: [])
    controller.assert_gpu_lease()
