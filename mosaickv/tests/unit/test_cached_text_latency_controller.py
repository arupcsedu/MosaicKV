from pathlib import Path

import evicastkv_cached_text_latency as controller
import pytest


def rows():
    observations = []
    for method, arm, seconds in (
        ("mosaickv_exact", "reuse", 12),
        ("mosaickv_exact", "text-6291456", 11),
        ("prefixkv_reimpl", "reuse", 11),
        ("prefixkv_reimpl", "whole-6291456", 10),
    ):
        observations.extend(
            dict(method=method, arm=arm, trial=t, measurement=dict(total_wall_seconds=seconds))
            for t in range(-5, 20)
        )
    return observations


def test_fastest_control_cannot_be_hidden_by_slower_prefix_reuse():
    result = controller.contrasts(rows(), 3)
    comparison = result["comparisons"][0]
    assert comparison["own_speedup_gate"]
    assert not comparison["strongest_prefix_similar_gate"]
    assert not comparison["strongest_prefix_no_slower_gate"]
    assert len(comparison["prefixes"]) == 2
    assert result["family_maximum_engineering_contrasts"] == 48


def test_missing_or_duplicate_trials_and_missing_control_fail_closed():
    with pytest.raises(ValueError, match="complete unique"):
        controller.contrasts(rows()[:-1], 3)
    with pytest.raises(ValueError, match="complete unique"):
        controller.contrasts([*rows(), rows()[0]], 3)
    with pytest.raises(ValueError, match="strongest registered"):
        controller.contrasts([r for r in rows() if r["arm"] != "whole-6291456"], 3)


def test_gpu_launcher_is_bound_to_new_finite_controller():
    assert controller.SCRIPT.name == "evicastkv_cached_text_latency.sbatch"
    assert controller.previous.SCRIPT.name == "evicastkv_hybrid_latency.sbatch"
    script = Path(controller.SCRIPT).read_text()
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in script
    assert "assert_clean_worktree.sh" in script
    assert "evicastkv_cached_text_latency.py" in script
    assert "HF_HUB_OFFLINE=1" in script
