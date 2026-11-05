import evicastkv_joint_input_reuse as study
import pytest


def test_common_model_policies_are_not_withheld_from_baselines(monkeypatch):
    monkeypatch.setattr(study, "COMMON_SEGMENT", True)
    monkeypatch.setattr(study, "COMMON_ROTARY", True)
    records = {
        m: {a: dict(reader=a) for a in ("graph", "graph-lt", "graph-original")}
        for m in ("mosaickv_exact", "prefixkv_reimpl")
    }
    configured = study.configurations(records)
    assert len(configured) == 4
    for (method, arm), record in configured.items():
        assert record["reuse_graph_replay_plan"] and record["reuse_patch_dispatch"]
        assert record["post_attention_segment"] and record["exact_rotary_dispatch"]
        assert record["reuse_source_input_copy"] is (arm != "joint-previous")
        assert (record.get("joint_incoming_projection") == "kv") is (method == "mosaickv_exact")


def test_missing_or_failed_common_rotary_cannot_be_promoted():
    assert not study.common_rotary_policy(
        dict(complete=False, failures=["failed control"], rows=[])
    )
    assert not study.common_rotary_policy(dict(complete=True, failures=[], rows=[{}, {}]))


def test_the_copy_has_no_permission_to_change_a_cache_boundary(monkeypatch):
    monkeypatch.setattr(study, "ORIGINAL_SUMMARY", lambda *a: {})
    audits = {key: dict.fromkeys(study.driver.STORAGE, 100) for key in study.ARMS}
    assert study.summary([], audits)["own_initial_final_peak_identical"]
    audits["mosaickv_exact", "joint-input"][study.driver.STORAGE[-1]] += 1
    with pytest.raises(ValueError, match="boundary"):
        study.summary([], audits)


def test_a_baseline_cannot_claim_fake_copy_engagement(monkeypatch):
    measured = dict(
        joint_incoming_projection=dict(source_input_copy=dict(enabled=True, reports=[]))
    )
    monkeypatch.setattr(study, "ORIGINAL_EXECUTE", lambda *a, **k: measured)
    args = (None, None, None, None, None, "prefixkv_reimpl", "graph")
    assert study.execute(*args, dict(reuse_source_input_copy=True)) is measured
    measured["joint_incoming_projection"]["source_input_copy"]["reports"] = [dict(prepared=1)]
    with pytest.raises(RuntimeError, match="no-op"):
        study.execute(*args, dict(reuse_source_input_copy=True))
