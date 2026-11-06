import evicastkv_verified_constructor as control
import pytest


def test_separate_ablations_preserve_common_controls_and_offer_noop_to_baselines(monkeypatch):
    original = {
        ("mosaickv_exact", "joint-history-graph"): dict(
            overlap_native_fit=True, native_history_graph=True
        ),
        ("prefixkv_reimpl", "graph-history-graph"): dict(
            overlap_native_fit=True, native_history_graph=True
        ),
        ("prefixkv_reimpl", "graph-original-history-graph"): dict(
            overlap_native_fit=True, native_history_graph=True
        ),
    }
    monkeypatch.setattr(control, "OLD_CONFIGURATIONS", lambda records: original)
    values = control.configurations({})
    assert len(values) == 5
    assert values["mosaickv_exact", "joint-same-buffer"]["verified_constructor"] == "same_buffer"
    assert values["mosaickv_exact", "joint-handoff"]["verified_constructor"] == "handoff"
    assert values["mosaickv_exact", "joint-handoff"]["overlap_native_fit"] is False
    assert values["mosaickv_exact", "joint-same-buffer"]["overlap_native_fit"] is True
    assert all(value["native_history_graph"] for value in values.values())
    assert values["prefixkv_reimpl", "graph-history-graph"]["verified_constructor"] == "handoff"


def test_missing_native_word_checks_cannot_be_published_as_a_success(monkeypatch):
    value = dict(
        verified_constructor=dict(
            requested="handoff",
            eligible=True,
            report=dict(components=64, retired=True, native_word_checks=0),
        )
    )
    monkeypatch.setattr(control, "OLD_EXECUTE", lambda *a, **k: value)
    monkeypatch.setattr(control.history, "POLICIES", dict(overlap=True))
    with pytest.raises(RuntimeError, match="complete bounded"):
        control.execute(
            None,
            None,
            None,
            None,
            None,
            "mosaickv_exact",
            "joint-handoff",
            dict(verified_constructor="handoff", overlap_native_fit=False),
        )
    assert control.history.POLICIES["overlap"] is True
