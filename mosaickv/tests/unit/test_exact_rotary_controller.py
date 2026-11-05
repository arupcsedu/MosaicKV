from copy import deepcopy

import evicastkv_exact_rotary_dispatch as study
import pytest


def segment_decision():
    means = [
        dict(method=m, arm=a, total_wall_seconds=9.0 if a.endswith("-segment") else 10.0)
        for m, a in (
            ("mosaickv_exact", "joint-kv-prepared"),
            ("mosaickv_exact", "joint-kv-segment"),
            ("prefixkv_reimpl", "graph-prepared"),
            ("prefixkv_reimpl", "graph-segment"),
            ("prefixkv_reimpl", "graph-original-prepared"),
            ("prefixkv_reimpl", "graph-original-segment"),
        )
    ]
    return dict(
        complete=True,
        failures=[],
        rows=[dict(means=deepcopy(means), exact_common_workspace_charge=True) for _ in (0, 1)],
    )


def test_common_segment_policy_needs_improvement_for_every_reader_and_input():
    decision = segment_decision()
    assert study.segment_policy(decision)
    decision["rows"][1]["means"][-1]["total_wall_seconds"] = 10.01
    assert not study.segment_policy(decision)
    decision = segment_decision()
    decision["rows"][0]["exact_common_workspace_charge"] = False
    assert not study.segment_policy(decision)
    decision = segment_decision()
    decision["complete"] = False
    assert not study.segment_policy(decision)


@pytest.mark.parametrize("segment", [False, True])
def test_both_common_optimizations_apply_identically_to_every_method(monkeypatch, segment):
    monkeypatch.setattr(study, "USE_POST_SEGMENT", segment)
    records = {
        m: {a: dict(reader=a) for a in ("graph", "graph-lt", "graph-original")}
        for m in ("mosaickv_exact", "prefixkv_reimpl")
    }
    configured = study.configurations(records)
    assert len(configured) == 6
    assert sum(r["exact_rotary_dispatch"] for r in configured.values()) == 3
    assert all(r["post_attention_segment"] is segment for r in configured.values())


def test_own_control_storage_cannot_change(monkeypatch):
    monkeypatch.setattr(study, "ORIGINAL_SUMMARY", lambda *a: {})
    audits = {key: dict.fromkeys(study.driver.STORAGE, 100) for key in study.ARMS}
    assert study.summary([], audits)["identical_own_control_initial_final_peak"]
    audits["mosaickv_exact", "joint-kv-rotary"][study.driver.STORAGE[-1]] += 1
    with pytest.raises(ValueError, match="boundary"):
        study.summary([], audits)
