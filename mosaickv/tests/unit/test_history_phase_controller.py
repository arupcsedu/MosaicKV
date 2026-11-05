import evicastkv_history_phase_fastpath as control


def test_common_fast_history_is_offered_without_changing_the_prompt_policy(monkeypatch):
    monkeypatch.setattr(control.prior, "REUSE", True)
    monkeypatch.setattr(control.prior.prior, "NORM", True)
    monkeypatch.setattr(
        control.prior.prior.prior,
        "POLICIES",
        dict(segment=True, rotary=True, copy=False, frozen=True, overlap=True),
    )
    records = dict(
        mosaickv_exact={"graph-lt": {}}, prefixkv_reimpl={"graph": {}, "graph-original": {}}
    )
    configs = control.configurations(records)
    assert len(configs) == 6
    for (method, arm), record in configs.items():
        assert record["source_tile_rows"] == 192
        assert record["native_history_reader"] is arm.endswith("-fast-history")
        assert record["input_norm_segment"] and record["reuse_norm_input"]
        assert record["post_attention_segment"] and record["exact_rotary_dispatch"]
        if method == "mosaickv_exact":
            assert record["joint_incoming_projection"] == "kv"


def test_fixed_scope_is_two_consumed_inputs_and_six_same_policy_arms():
    assert control.PREVIOUS.name == "source-192-reader-consumed-20261105-v1"
    assert len(control.ARMS) == 6
    assert len(set(control.ARMS)) == 6
