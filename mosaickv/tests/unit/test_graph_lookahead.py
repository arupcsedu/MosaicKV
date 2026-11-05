from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.graph_lookahead import Lookahead, attach, use


class FakeCUDA:
    def __init__(self):
        self.operations = []
        self.main = SimpleNamespace(
            wait_event=lambda event: self.operations.append("consumer-wait")
        )

    def current_stream(self):
        return self.main

    def Stream(self, **kwargs):
        return SimpleNamespace(
            wait_event=lambda event: self.operations.append("producer-wait"),
            synchronize=lambda: self.operations.append("producer-sync"),
        )

    def Event(self):
        return SimpleNamespace(record=lambda stream: self.operations.append("record"))

    @contextmanager
    def stream(self, stream):
        yield


def fixture():
    layers = [SimpleNamespace(history_length=1, leased=(), current_component=1) for _ in range(3)]
    owner = torch.zeros(1, 2, 20, 128, dtype=torch.bfloat16)
    layers[0].history_length = 2
    layers[0].leased = (owner,) * 2
    pool = SimpleNamespace(
        cache=SimpleNamespace(layers=layers),
        entries={(id(layer), 0): None for layer in layers},
        buffer=owner,
        lookahead=None,
    )
    cuda = FakeCUDA()
    engine = pool.lookahead = Lookahead(pool, cuda=cuda)

    def reconstruct(layer, component, destination, *, _lookahead_epoch):
        assert engine.authorized(layer, _lookahead_epoch, destination)
        assert component == 0 and destination is owner
        cuda.operations.append("project-existing-K-graph")

    pool.reconstruct = reconstruct
    return pool, layers, engine, cuda


def test_producer_waits_after_PV_and_consumer_waits_before_same_buffer_use():
    pool, layers, engine, cuda = fixture()
    engine.schedule(layers[0])
    assert cuda.operations == ["record", "producer-wait", "project-existing-K-graph", "record"]
    assert engine.pending == (id(layers[1]), 2)
    layers[0].leased = ()
    layers[1].history_length = 2
    layers[1].leased = (pool.buffer[..., :8, :],) * 2
    engine.check_acquire(layers[1])
    assert engine.take(layers[1], layers[1].leased[0])
    assert cuda.operations[-1] == "consumer-wait"
    assert not engine.take(layers[1], layers[1].leased[0])
    engine.close()
    assert engine.record()["hits"] == 1
    assert engine.record()["extra_tensor_workspace_bytes"] == 0
    assert engine.record()["prompt_copies"] == 0


def test_first_capture_and_token_boundary_are_never_prefetched():
    pool, layers, engine, cuda = fixture()
    pool.entries.clear()
    engine.schedule(layers[0])
    layers[-1].leased = (pool.buffer,) * 2
    engine.schedule(layers[-1])
    assert engine.cold_misses == engine.token_boundaries == 1
    assert not cuda.operations and engine.stream is None


@pytest.mark.parametrize("fault", ["foreign", "not_V", "unsynchronized", "duplicate"])
def test_invalid_scheduling_fails_closed(fault):
    _, layers, engine, _ = fixture()
    current = layers[0]
    if fault == "foreign":
        current = SimpleNamespace(leased=(1,), current_component=1)
    elif fault == "not_V":
        current.current_component = 0
    elif fault == "unsynchronized":
        layers[1].history_length = 2
    else:
        engine.last_scheduled = id(current), current.history_length
    with pytest.raises(RuntimeError):
        engine.schedule(current)


def test_cannot_overwrite_or_consume_wrong_request_epoch_or_backing():
    pool, layers, engine, _ = fixture()
    engine.schedule(layers[0])
    with pytest.raises(RuntimeError, match="overwrite"):
        engine.schedule(layers[0])
    with pytest.raises(RuntimeError, match="epoch"):
        engine.check_acquire(layers[2])
    layers[1].history_length = 2
    layers[1].leased = (pool.buffer,) * 2
    with pytest.raises(RuntimeError, match="fixed backing"):
        engine.take(layers[1], pool.buffer.clone())
    with pytest.raises(RuntimeError, match="unconsumed"):
        engine.close()
    assert engine.closed and engine.pending is None


def test_failed_producer_is_synchronized_and_never_published():
    pool, layers, engine, cuda = fixture()

    def fail(*args, **kwargs):
        raise RuntimeError("failed producer")

    pool.reconstruct = fail
    with pytest.raises(RuntimeError, match="failed producer"):
        engine.schedule(layers[0])
    assert cuda.operations[-1] == "producer-sync"
    assert engine.pending is engine.submitting is None
    engine.close()


def test_flag_strict_nonnested_and_no_implicit_non_graph_fallback():
    with pytest.raises(ValueError), use(1):
        pass
    with use(False):
        attach(SimpleNamespace())
        with pytest.raises(ValueError), use(True):
            pass
    with use(True), pytest.raises(ValueError):
        attach(SimpleNamespace())
    with use(False):
        attach(SimpleNamespace())


def test_producer_entry_cannot_be_injected_through_private_keyword():
    from mosaickv.evaluation.prompt_graph_reader import PromptGraphPool

    pool, layers, _, _ = fixture()
    actual = PromptGraphPool(pool.cache)
    actual.buffer = pool.buffer
    with pytest.raises(RuntimeError, match="existing-graph"):
        actual.reconstruct(layers[1], 0, pool.buffer, _lookahead_epoch=2)
