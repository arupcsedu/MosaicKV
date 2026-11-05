import evicastkv_frozen_joint_plan as study
import pytest


def test_common_optimizations_remain_common_and_baseline_frozen_plan_is_a_noop(monkeypatch):
    monkeypatch.setitem(study.COMMON, "segment", True)
    monkeypatch.setitem(study.COMMON, "rotary", True)
    monkeypatch.setitem(study.COMMON, "copy", True)
    records = {
        m: {a: {} for a in ("graph", "graph-lt", "graph-original")}
        for m in ("mosaickv_exact", "prefixkv_reimpl")
    }
    configured = study.configurations(records)
    for (method, arm), record in configured.items():
        assert record["post_attention_segment"] and record["exact_rotary_dispatch"]
        assert record["reuse_source_input_copy"]
        assert record["frozen_joint_plan"] is (arm != "joint-previous")
        assert (record.get("joint_incoming_projection") == "kv") is (method == "mosaickv_exact")


def test_failed_input_copy_is_not_promoted():
    assert not study.copy_policy(dict(complete=False, failures=["error"], rows=[]))
    assert not study.copy_policy(dict(complete=True, failures=[], rows=[{}, {}]))


def test_a_constructor_change_cannot_change_a_storage_boundary(monkeypatch):
    monkeypatch.setattr(study, "ORIGINAL_SUMMARY", lambda *a: {})
    audited = {key: dict.fromkeys(study.driver.STORAGE, 100) for key in study.ARMS}
    assert study.summary([], audited)["own_initial_final_peak_identical"]
    audited["mosaickv_exact", "joint-frozen"][study.driver.STORAGE[-1]] += 1
    with pytest.raises(ValueError, match="boundary"):
        study.summary([], audited)
