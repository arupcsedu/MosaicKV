"""V10 does not widen the diagnostic protocol or silently submit duplicate work."""

import evicastkv_review_v10 as review
import pytest


def test_output_root_is_workspace_scoped():
    root = review.WORKSPACE / "fox-review-v10/test"
    assert review.validate_root(root) == root
    with pytest.raises(ValueError, match="scratch workspace"):
        review.validate_root(review.WORKSPACE)


def test_hash_binding_rejects_changed_evidence(tmp_path):
    path = tmp_path / "artifact"
    path.write_text("wrong")
    with pytest.raises(ValueError, match="artifact changed"):
        review.verify_bindings({str(path): "0" * 64})


def test_missing_preparation_never_contacts_slurm(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "load", lambda root: {})
    monkeypatch.setattr(review.shared, "gpu_snapshot", lambda: pytest.fail("queried Slurm"))
    with pytest.raises(ValueError, match="before preparation"):
        review.submit(tmp_path)


def test_scheduler_chain_is_one_gpu_and_afterany_cpu(tmp_path, monkeypatch):
    (tmp_path / "source-length-prepared.json").touch()
    monkeypatch.setattr(review, "load", lambda root: {"source_sha": "a" * 40})
    monkeypatch.setattr(review.shared, "cpu_gate", lambda *args: {})
    monkeypatch.setattr(
        review.shared,
        "gpu_snapshot",
        lambda: {"active": [{"job": "12_0"}, {"job": "12_1"}], "pending": []},
    )
    commands = []

    def capture(command, **kwargs):
        commands.append(command)
        return str(100 + len(commands))

    monkeypatch.setattr(review.shared.subprocess, "check_output", capture)
    assert review.submit(tmp_path) == {"worker": "101", "analysis": "102"}
    assert "--gres=gpu:a100:1" in commands[0]
    assert "--time=00:20:00" in commands[0]
    assert "--dependency=afterany:12" in commands[0]
    assert "--dependency=afterany:101" in commands[1]
    assert not any("gres=" in flag for flag in commands[1])
