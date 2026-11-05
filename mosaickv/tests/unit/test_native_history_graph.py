from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.native_history_graph import Entries, Reader


def graph():
    calls = []
    return SimpleNamespace(reset=lambda: calls.append("reset"), calls=calls)


def test_bounded_identity_and_explicit_epoch_retirement():
    table = Entries(1)
    a, b = graph(), graph()
    packet, attention = object(), object()
    key = (id(packet), 0)
    table.add(key, (1, 2), a, packet, attention)
    table.add((id(packet), 1), (1, 3), b, packet, attention)
    assert table.get(key, (1, 2)) is a
    with pytest.raises(RuntimeError, match="bindings"):
        table.get(key, (1, 99))
    with pytest.raises(RuntimeError, match="exhausted"):
        table.add((99, 0), (), graph(), object(), attention)
    assert table.discard({id(packet)}) == 2
    assert a.calls == b.calls == ["reset"]
    assert table.maximum_live == 2 and table.retired == table.captures == 2
    table.close()
    table.close()
    with pytest.raises(RuntimeError, match="retired"):
        table.get(key, (1, 2))


@pytest.mark.parametrize("layers", [True, 0, 33, 32.0])
def test_unsupported_route_counts_fail_before_gpu_access(layers):
    with pytest.raises(ValueError, match="audited"):
        Entries(layers)


def test_tables_are_request_isolated_and_do_not_accept_duplicate_keys():
    a, b = Entries(1), Entries(1)
    packet = object()
    a.add((id(packet), 0), ("A",), graph(), packet, object())
    assert b.get((id(packet), 0), ("A",)) is None
    with pytest.raises(RuntimeError, match="duplicate"):
        a.add((id(packet), 0), ("A",), graph(), packet, object())
    b.close()
    assert len(a.rows) == 1


def test_constructor_targets_never_enter_cuda_capture():
    calls = []
    pool = SimpleNamespace(
        cache=SimpleNamespace(layers=[None]),
        closed=False,
        history_borrowed=False,
        buffer=torch.empty(8),
    )
    state = SimpleNamespace(pool=pool, engine=SimpleNamespace(project=lambda *a: calls.append(a)))
    reader = Reader(state)
    target = torch.empty(8)
    reader.project(object(), object(), 0, target, None)
    assert len(calls) == 1 and reader.direct_calls == 1 and not reader.table.rows


def test_graph_feature_is_explicit_and_requires_native_history():
    from mosaickv.evaluation import native_history_reader as mod

    @mod.managed
    def call(**kwargs):
        return None, None, {}

    for value in (1, None, "true"):
        with pytest.raises(ValueError, match="history graph"):
            call(native_history_reader=True, native_history_graph=value)
    with pytest.raises(ValueError, match="requires"):
        call(native_history_graph=True)
    with pytest.raises(ValueError, match="boolean"):
        mod.State(graph=1)
