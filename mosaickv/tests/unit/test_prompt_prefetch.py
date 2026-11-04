from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import prompt_prefetch as prefetch
from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


class FakeCUDA:
    def __init__(self):
        self.events = []
        self.main = self.active = self.stream_object("main")

    def stream_object(self, name):
        def wait(event):
            self.events.append((name, "wait", event))

        return SimpleNamespace(wait_event=wait, synchronize=lambda: None, name=name)

    def Stream(self, **_kwargs):
        return self.stream_object("producer")

    def Event(self):
        event = SimpleNamespace(record=lambda stream: self.events.append((stream.name, "record")))
        return event

    def current_stream(self):
        return self.active

    @contextmanager
    def stream(self, stream):
        original, self.active = self.active, stream
        try:
            yield
        finally:
            self.active = original


def cache():
    layers = []
    for offset in (0, 100):
        shape = (1, 2, 8, 128)
        tensor = (torch.arange(2048).reshape(shape) + offset).bfloat16()
        body = PackedBF16(shape, True, (tensor,))
        packet = ExactFormat("kv", shape, (body, body))
        layers.append(
            SimpleNamespace(
                representation=packet,
                prompt_length=8,
                attention=None,
                history_length=1,
                owned_tensors=packet.tensors,
            )
        )
    return SimpleNamespace(layers=layers, shared_tensors=(), observe=lambda *_: None)


def test_prefetch_event_order_exact_bits_charged_storage_and_no_trailing_work():
    base, cuda = cache(), FakeCUDA()
    original = backing_bytes(tuple(t for layer in base.layers for t in layer.owned_tensors()))
    engine = prefetch.Prefetch(base, 3, cuda=cuda)
    size = engine.buffer.numel() * engine.buffer.element_size()
    assert base.initial_bytes == original + size
    target = torch.full((1, 2, 9, 128), -71, dtype=torch.bfloat16)
    assert not engine.take(base.layers[0], target)
    for step in (1, 2):
        for index, layer in enumerate(base.layers):
            layer.history_length = step
            if not (step == 1 and index == 0):
                assert engine.take(layer, target)
                assert torch.equal(target[..., :8, :], layer.representation.pair[0].buffers[0])
                assert (target[..., 8:, :] == -71).all()
            engine.schedule(layer)
    assert engine.hits == engine.launches == 3 and engine.misses == 1
    assert [x[:2] for x in cuda.events[:4]] == [
        ("main", "record"),
        ("producer", "wait"),
        ("producer", "record"),
        ("main", "wait"),
    ]
    engine.close()
    assert engine.buffer is None and not base.shared_tensors


def test_foreign_order_and_aborted_request_cleanup():
    base, cuda = cache(), FakeCUDA()
    engine = prefetch.Prefetch(base, 3, cuda=cuda)
    engine.schedule(base.layers[0])
    with pytest.raises(RuntimeError, match="ordering"):
        engine.take(base.layers[0], torch.empty(1, 2, 9, 128))
    with pytest.raises(RuntimeError, match="overwritten"):
        engine.schedule(base.layers[1])
    engine.close(require_consumed=False)
    assert engine.buffer is None and not base.shared_tensors


def test_private_buffers_and_disabled_policy_no_cuda_or_graph_claim():
    first = prefetch.Prefetch(cache(), 3, cuda=FakeCUDA())
    second = prefetch.Prefetch(cache(), 3, cuda=FakeCUDA())
    assert first.buffer.data_ptr() != second.buffer.data_ptr()
    first.close()
    second.close()
    with prefetch.use(False) as state:
        prefetch.attach(None, 3)
        assert not prefetch.take(None, None)
        prefetch.after_attention(None)
        assert state["engine"] is None
        with pytest.raises(ValueError, match="nonnested"), prefetch.use(False):
            pass
