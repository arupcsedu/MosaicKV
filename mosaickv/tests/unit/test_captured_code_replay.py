
import pytest
import torch

from mosaickv.evaluation import layer_code_delta as runtime
from mosaickv.evaluation.lossless_code_delta_gpu import DeviceCodes
from mosaickv.evaluation.precision_tier import PrecisionFormat, QuantizedBody
from mosaickv.evaluation.replay_descriptor_policy import use


def packets():
    output = []
    for i in range(3):
        body = QuantizedBody((1, 1, 1, 128), 8, 3, 128, None, torch.zeros(1), torch.ones(1))
        roles = []
        for j in range(2):
            values = torch.full((128,), i + j, dtype=torch.uint8)
            # CPU test stand-in payload: real code words are covered by the
            # separate actual CUDA lossless codec control, not these fixtures.
            stream = DeviceCodes(128, values, None if i == 0 else torch.zeros(2, dtype=torch.int32))
            roles.append(runtime.EncodedBody(body, stream))
        template = PrecisionFormat(
            body.shape,
            tuple(roles),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            (torch.empty(0), torch.empty(0)),
        )
        output.append(runtime.DeltaFormat(template))
    return tuple(output)


def test_host_ordering_and_gpu_decode_are_separate_without_new_backing(monkeypatch):
    calls = []

    def decode(previous, stream, destination):
        calls.append((previous.data_ptr(), destination.data_ptr()))
        destination.copy_(stream.payload)
        return destination

    monkeypatch.setattr(runtime, "decode", decode)
    with use(True):
        ordinary = runtime.Replay(packets())
        deferred = runtime.Replay(packets())
    pointers = {r: x.data_ptr() for r, x in deferred.scratch.items()}
    deferred.prepare_graph()
    assert not calls
    for _ in range(3):
        for i in range(3):
            expected = ordinary.view(i)
            count = len(calls)
            actual = deferred.advance_for_graph(i)
            assert len(calls) == count
            deferred.decode_for_graph(i)
            assert all(
                torch.equal(a.codes, b.codes)
                for a, b in zip(actual.pair, expected.pair, strict=True)
            )
            count = len(calls)
            assert deferred.view(i) is actual and len(calls) == count
    assert pointers == {r: x.data_ptr() for r, x in deferred.scratch.items()}


def test_graph_ordering_cannot_skip_layers_or_use_unprepared_descriptors():
    with use(False):
        unprepared = runtime.Replay(packets())
    with pytest.raises(ValueError, match="prepared"):
        unprepared.advance_for_graph(0)
    with use(True):
        prepared = runtime.Replay(packets())
    prepared.prepare_graph()
    with pytest.raises(RuntimeError, match="ordering"):
        prepared.decode_for_graph(1)
    with pytest.raises(RuntimeError, match="ordered"):
        prepared.advance_for_graph(1)


def test_pointers_are_request_private_and_malformed_predecessors_fail():
    with use(True):
        a, b = runtime.Replay(packets()), runtime.Replay(packets())
    a.prepare_graph()
    b.prepare_graph()
    assert all(a.scratch[r].data_ptr() != b.scratch[r].data_ptr() for r in a.scratch)
    broken = list(a.prepared_bindings)
    r, e, d = broken[1][0]
    broken[1] = ((r, e, d[:64]), *broken[1][1:])
    a.prepared_bindings = tuple(broken)
    a.graph_bindings = None
    with pytest.raises(ValueError, match="same-shape"):
        a.prepare_graph()
