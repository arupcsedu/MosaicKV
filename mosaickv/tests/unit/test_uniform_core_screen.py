import evicastkv_uniform_core_screen as control


def test_real_configuration_chain_maps_new_arm_to_frozen_parent(monkeypatch):
    monkeypatch.setattr(control.base, "ARMS", control.ARMS)
    monkeypatch.setattr(control.base.base, "ARMS", control.ARMS)
    monkeypatch.setattr(control.base, "overrides", control.overrides)
    records = {
        method: {arm: dict(packet_sha256=f"{method}:{arm}") for arm in arms}
        for method, arms in (
            ("mosaickv_exact", ("graph-lt",)),
            ("prefixkv_reimpl", ("graph", "graph-original")),
        )
    }
    values = control.base.configurations(records)
    assert tuple(values) == control.ARMS
    assert not values["mosaickv_exact", "joint-async-handoff"]["uniform_attention_core"]
    for (method, arm), value in values.items():
        assert value["uniform_attention_core"] is (
            arm == "joint-async-uniform-core" or method != "mosaickv_exact"
        )
        assert value["fused_transport_predicate"] is False
        assert value["native_history_reader"] and value["native_history_graph"]
        if method == "mosaickv_exact":
            assert value["verified_constructor"] == "async_handoff"


def test_operator_proof_is_before_any_inherited_model_request(monkeypatch):
    events = []
    monkeypatch.setattr(control, "pre_request_controls", lambda plan: events.append("words"))

    def execute(*args, **kwargs):
        events.append("request")
        return dict(uniform_attention_core=dict(requested=False))

    monkeypatch.setattr(control, "OLD_EXECUTE", execute)
    control.execute(
        None,
        None,
        None,
        None,
        {},
        "mosaickv_exact",
        "joint-async-handoff",
        dict(uniform_attention_core=False),
    )
    assert events == ["words", "request"]
