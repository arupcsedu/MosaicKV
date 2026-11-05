import evicastkv_post_attention_segment as study
import pytest


def records():
    return {
        m: {a: {"reader": a} for a in ("graph", "graph-lt", "graph-original")}
        for m in ("mosaickv_exact", "prefixkv_reimpl")
    }


def test_all_compared_readers_have_own_common_model_optimization_controls():
    configured = study.configurations(records())
    assert len(configured) == 6
    for (method, arm), record in configured.items():
        assert record["reuse_patch_dispatch"] and record["reuse_graph_replay_plan"]
        assert record["post_attention_segment"] is arm.endswith("-segment")
        assert (record.get("joint_incoming_projection") == "kv") is (method == "mosaickv_exact")
    assert {a for m, a in configured if m == "prefixkv_reimpl"} == {
        "graph-prepared",
        "graph-segment",
        "graph-original-prepared",
        "graph-original-segment",
    }


def test_workspace_is_explicitly_included_in_the_peak_but_not_resident_cache(monkeypatch):
    monkeypatch.setattr(study, "ORIGINAL_SUMMARY", lambda *a: {"formal_latency_acceptance": False})
    audited = {}
    for key, record in study.configurations(records()).items():
        active = record["post_attention_segment"]
        audited[key] = dict(
            zip(study.driver.STORAGE, (100, 200, 300 + 117764 * active), strict=True),
            post_attention_segment=dict(explicit_workspace_bytes=117764),
        )
    assert study.summary([], audited)["exact_common_workspace_charge"]
    audited["mosaickv_exact", "joint-kv-segment"][study.driver.STORAGE[-1]] -= 1
    with pytest.raises(ValueError, match="workspace"):
        study.summary([], audited)


def test_controller_fails_closed_if_the_operator_did_not_engage(monkeypatch):
    state = dict(
        enabled=True,
        calls=4064,
        captures=32,
        replays=4064,
        explicit_workspace_bytes=117764,
        additional_graph_tensor_bytes=0,
        retired=True,
        common_to_every_method=True,
    )
    measured = dict(post_attention_segment=state)
    monkeypatch.setattr(study, "ORIGINAL_EXECUTE", lambda *a, **kw: measured)
    args = (None, None, None, None, None, "mosaickv_exact", "joint-kv-segment")
    assert study.execute(*args, dict(post_attention_segment=True)) is measured
    state["replays"] -= 1
    with pytest.raises(RuntimeError, match="exact common"):
        study.execute(*args, dict(post_attention_segment=True))
