"""Synthetic controls test orchestration, not measured scientific outcomes."""

import fox_installed_guard_followup_v2 as fox
import numpy as np
import pytest
import registered_quality_followup_v2 as prefix

from mosaickv.evaluation.fox_campaign import sha, write_once


def test_fox_complete_same_grid_has_nine_arms_and_no_new_budget():
    arms = fox.arms()
    assert len(arms) == 9 and len(set(arms)) == 9
    for cap in (20, 40):
        assert [(m, v) for m, c, v in arms if c == cap] == [
            ("prompt_attention_topk", "complete"),
            ("vl_cache_reimpl", "complete"),
            ("mosaickv_exact", "complete"),
            ("mosaickv_exact", fox.VARIANT),
        ]


def test_prefix_registered_caps_and_three_slot_ceiling(tmp_path, monkeypatch):
    import control_a100_schedule

    assert prefix.CAPS == (80, 120, 160)
    assert prefix.METHODS == ("prompt_attention_topk", "prefixkv_reimpl", "mosaickv_exact")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return "12345\n"

    monkeypatch.setattr(prefix.subprocess, "check_output", run)
    monkeypatch.setattr(prefix, "prefix_concurrency", lambda: 3)
    monkeypatch.setattr(control_a100_schedule, "tick", lambda *args: dict(dry_run=True))
    job = prefix.submit(tmp_path, "hash", "worker", [0, 1, 2])
    assert job == "12345"
    assert "--array=0,1,2%3" in commands[0]
    assert "--reservation=bi_fox_dgx" in commands[0]
    assert "--cpus-per-task=12" in commands[0]
    assert "--time=01:00:00" in commands[0]
    assert prefix.native.read(tmp_path / "scheduler/12345-policy.json")["maximum_a100s"] == 6


def test_protocol_rejects_generation_change_before_gpu(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    write_once(
        path,
        dict(
            source_sha="frozen",
            protocol=dict(
                initial_caps_mib=[80, 120, 160],
                samples_per_task=32,
                output_tokens=64,
                quality_warmups=0,
                quality_margins={m: 0.0 for m in prefix.native.METRICS},
            ),
        ),
    )
    monkeypatch.setattr(prefix, "clean_source", lambda *args: "frozen")
    with pytest.raises(ValueError, match="quality controls"):
        prefix.inputs(tmp_path, sha(path))


def test_larger_latency_family_preserves_slower_outcome():
    from prefixkv_isoquality_development import latency_interval

    result = latency_interval(
        np.full((1, 20), 1.2), np.ones((1, 20)), replicates=100, family_size=6
    )
    assert result["family_size"] == 6
    assert result["mean_latency_ratio"] == pytest.approx(1.2)
    assert result["no_slower_gate"] is False


def test_fox_scheduler_uses_only_three_slots(tmp_path, monkeypatch):
    import json

    import control_a100_schedule

    write_once(tmp_path / "jobs/evidence_canary.json", dict(jobs=[{}, {}]))
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return "54321\n"

    monkeypatch.setattr(fox.subprocess, "check_output", run)
    monkeypatch.setattr(control_a100_schedule, "tick", lambda *args: dict(dry_run=True))
    assert fox.submit(tmp_path, "evidence_canary", "worker") == "54321"
    assert "--array=0-1%3" in commands[0]
    policy = json.loads((tmp_path / "scheduler/54321-policy.json").read_text())
    assert policy["other_lane_reserved_slots"] == 3


def test_six_slot_handoff_requires_source_bound_terminal_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(prefix, "FOX_ROOT", tmp_path)
    monkeypatch.setattr(prefix, "clean_source", lambda *args: "frozen")
    assert prefix.prefix_concurrency() == 3
    write_once(
        tmp_path / "decisions/terminal.json", dict(source_sha="frozen", no_more_fox_gpu_jobs=True)
    )
    assert prefix.prefix_concurrency() == 6


def test_queued_fox_gpu_forbids_early_handoff(tmp_path, monkeypatch):
    import control_a100_schedule

    monkeypatch.setattr(fox, "clean_source", lambda *args: "frozen")
    monkeypatch.setattr(
        control_a100_schedule,
        "read_queue",
        lambda: [dict(name="mk-fox-guard-v2", gres="gres/gpu:a100:1")],
    )
    with pytest.raises(ValueError, match="handoff forbidden"):
        fox.release_prefix_slots(tmp_path)
    assert not (tmp_path / "decisions/terminal.json").exists()


def test_transitive_media_aliases_are_excluded_before_id_selection():
    samples = [
        dict(sample_id=sid, media=[dict(sha256=h) for h in hashes])
        for sid, hashes in [
            ("old", ["x"]),
            ("alias", ["x", "y"]),
            ("alias2", ["y"]),
            ("unused", ["z"]),
        ]
    ]
    excluded = prefix.exclude_known_media_groups(samples, {"old"})
    assert excluded == {"old", "alias", "alias2"}
    assert prefix.native.choose_ids(samples, excluded, count=1, seed=42) == ["unused"]
