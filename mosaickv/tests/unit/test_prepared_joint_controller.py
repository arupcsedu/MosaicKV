import evicastkv_prepared_joint_replay as study
import pytest


def test_both_baseline_readers_receive_the_same_opt_in_replay_path():
    records = {
        method: {name: {"reader": name} for name in ("graph", "graph-lt", "graph-original")}
        for method in ("mosaickv_exact", "prefixkv_reimpl")
    }
    configured = study.configurations(records)
    assert len(configured) == 7
    assert sum(r["reuse_graph_replay_plan"] for r in configured.values()) == 3
    for (method, arm), record in configured.items():
        assert record["reuse_patch_dispatch"]
        assert record["graph_accounting"] == "event_sum"
        assert record["reuse_graph_replay_plan"] is arm.endswith("-prepared")
        assert (record.get("joint_incoming_projection") == "kv") is arm.startswith("joint-kv")
        if method == "prefixkv_reimpl":
            assert record["reader"] == arm.removesuffix("-prepared")


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("joint", [False, True])
def test_exact_branch_counts_and_scope_are_checked(monkeypatch, prepared, joint):
    measured = dict(
        prompt_graph_reader=dict(
            only_immutable_prompt_captured=not joint,
            prepared_replay_plan=dict(
                enabled=prepared,
                hits=8064 if prepared else 0,
                joint_key_consume_bypasses=4032 if prepared and joint else 0,
                additional_GPU_tensor_bytes=0,
                CPU_only_metadata=True,
            ),
        )
    )
    monkeypatch.setattr(study, "ORIGINAL_EXECUTE", lambda *a, **kw: measured)
    args = (None, None, None, None, None, "mosaickv_exact", "arm")
    record = dict(
        reuse_graph_replay_plan=prepared, joint_incoming_projection="kv" if joint else "none"
    )
    assert study.execute(*args, record) is measured
    measured["prompt_graph_reader"]["prepared_replay_plan"]["hits"] += 1
    with pytest.raises(RuntimeError, match="engagement"):
        study.execute(*args, record)


def test_preparation_cannot_change_any_own_control_storage_boundary(monkeypatch):
    monkeypatch.setattr(study, "ORIGINAL_SUMMARY", lambda *a: {"formal_latency_acceptance": False})
    audited = {key: dict.fromkeys(study.driver.STORAGE, 100) for key in study.ARMS}
    assert study.summary([], audited)["own_control_three_storage_boundaries_identical"]
    audited["prefixkv_reimpl", "graph-prepared"][study.driver.STORAGE[-1]] += 1
    with pytest.raises(ValueError, match="storage"):
        study.summary([], audited)
