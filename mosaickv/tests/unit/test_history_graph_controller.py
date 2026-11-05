import evicastkv_history_graph_reader as control
import pytest


def test_screen_requires_both_complete_correctness_producers():
    for value in (
        {},
        dict(complete=True, failures=[], rows=[]),
        dict(
            complete=True,
            failures=[],
            rows=[dict(debug_controls_passed=True), dict(debug_controls_passed=False)],
        ),
    ):
        with pytest.raises(ValueError, match="both complete"):
            control.require_debug(value)
    control.require_debug(
        dict(complete=True, failures=[], rows=[dict(debug_controls_passed=True)] * 2)
    )


def test_common_replay_is_offered_to_both_readers_without_changing_prompt_controls(monkeypatch):
    original = {
        ("mosaickv_exact", "joint-192"): dict(native_history_reader=False),
        ("mosaickv_exact", "joint-fast-history"): dict(native_history_reader=True),
        ("prefixkv_reimpl", "graph"): dict(native_history_reader=False),
        ("prefixkv_reimpl", "graph-fast-history"): dict(native_history_reader=True),
        ("prefixkv_reimpl", "graph-original"): dict(native_history_reader=False),
        ("prefixkv_reimpl", "graph-original-fast-history"): dict(native_history_reader=True),
    }
    monkeypatch.setattr(control.prior, "configurations", lambda records: original)
    values = control.configurations({})
    assert len(values) == 6 and len(control.DEBUG_ARMS) == 4
    for (_, arm), record in values.items():
        assert record["native_history_graph"] is arm.endswith("-history-graph")
        assert record["native_history_reader"] is record["native_history_graph"]


def test_replay_metadata_cannot_claim_unused_or_unretired_graphs(monkeypatch):
    graph = dict(
        enabled=True,
        captures=448,
        retired_graphs=448,
        replays=7104,
        maximum_live_graphs=64,
        live_graphs=0,
        retired=True,
        warm_calls=448,
        additional_GPU_tensor_bytes=0,
    )
    value = dict(
        common_native_history=dict(enabled=True, report=dict(immutable_history_graph=graph))
    )
    monkeypatch.setattr(control.prior, "execute", lambda *args, **kwargs: value)
    record = dict(native_history_graph=True)
    assert (
        control.execute(
            None, None, None, None, None, "mosaickv_exact", "joint-history-graph", record
        )
        is value
    )
    graph["additional_GPU_tensor_bytes"] = 8
    with pytest.raises(RuntimeError, match="complete bounded"):
        control.execute(
            None, None, None, None, None, "mosaickv_exact", "joint-history-graph", record
        )
