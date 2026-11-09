from collections import Counter
from unittest.mock import patch

import private_prefill_baselines_study as baselines
import private_prefill_scores_timing as timing
import pytest


def test_every_compatible_baseline_gets_its_own_identical_control():
    assert baselines.METHODS == (
        "prompt_attention_topk",
        "prefixkv_reimpl",
        "vl_cache_reimpl",
        "lookm_reimpl",
    )
    assert len(baselines.ARMS) == 8
    assert all(
        baselines.ARMS.count(m + ":" + f) == 1
        for m in baselines.METHODS
        for f in ("reference", "private")
    )


def test_failed_parent_gate_cannot_submit_baselines(tmp_path, monkeypatch):
    monkeypatch.setattr(baselines.base.common, "read", lambda _: dict(bitwise_and_peak_gate=False))
    monkeypatch.setattr(
        baselines.base.common, "submit", lambda *a: pytest.fail("unexpected GPU submission")
    )
    root = tmp_path / "baselines"
    with pytest.raises(ValueError, match="FullKV"):
        baselines.prepare(root)
    assert not root.exists()


def test_timing_applies_common_optimizations_to_storage_comparator_too():
    control, candidate = timing.POLICIES[timing.ARMS[0]], timing.POLICIES[timing.ARMS[2]]
    for key in ("private_prefill_scores", "paired_source", "verification_barrier"):
        assert control[key] is candidate[key] is True
    assert not control["history"] and candidate["history"]
    with patch.object(timing.timing, "ARMS", timing.ARMS):
        rows = timing.timing.schedule(3)
    assert Counter(r["arm"] for r in rows) == Counter(dict.fromkeys(timing.ARMS, 25))
