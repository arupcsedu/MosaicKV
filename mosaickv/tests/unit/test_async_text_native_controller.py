import evicastkv_async_text_native as control
import pytest


def test_async_is_separate_keeps_common_controls_and_the_original_overlap():
    old = control.OLD_OVERRIDES("mosaickv_exact", "joint-same-buffer")
    new = control.overrides("mosaickv_exact", "joint-async-handoff")
    assert new["verified_constructor"] == "async_handoff"
    assert new["overlap_native_fit"] is True
    assert {k: v for k, v in old.items() if k != "verified_constructor"} == {
        k: v for k, v in new.items() if k != "verified_constructor"
    }


def test_real_configuration_chain_after_launcher_installs_variant_arms(monkeypatch):
    # Reproduce main()'s rebinding without loading weights or submitting jobs.
    # No mocked configuration function: traverse every inherited adapter.
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
    assert control.base.base.ARMS == control.ARMS  # Scoped parent mapping restored.
    for (method, arm), value in values.items():
        assert value["native_history_reader"] is True
        assert value["native_history_graph"] is True
        assert value["overlap_native_fit"] is True
        canonical = "graph-lt" if method == "mosaickv_exact" else arm.removesuffix("-history-graph")
        assert value["packet_sha256"] == records[method][canonical]["packet_sha256"]
    assert (
        values["mosaickv_exact", "joint-async-handoff"]["verified_constructor"] == "async_handoff"
    )
    assert values["mosaickv_exact", "joint-same-buffer"]["verified_constructor"] == "same_buffer"


def test_missing_real_native_callbacks_cannot_be_published(monkeypatch):
    report = dict(
        completed=True,
        deferred_text_ready=True,
        deferred_text=True,
        thread_retired=True,
        native_reference_callbacks=64,
        published_text_layers=32,
        cached_text_layers=2,
        maximum_pending_layers=2,
        engine_count=1,
        outer_stride_word_checks=0,
    )
    value = dict(
        overlapping_native_construction=dict(report=report),
        direct_text_construction=dict(cached_layers=[0, 1]),
    )
    monkeypatch.setattr(control.prior, "execute", lambda *a, **k: value)
    args = (None, None, None, None, None, "mosaickv_exact", "joint-async-handoff", {})
    assert control.execute(*args) is value
    report["native_reference_callbacks"] = 63
    with pytest.raises(RuntimeError, match="shared-reference"):
        control.execute(*args)
