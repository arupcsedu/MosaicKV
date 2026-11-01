"""CPU reference substitutes only the codec kernels, never replay semantics."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mosaickv.evaluation import layer_code_delta as runtime
from mosaickv.evaluation.lossless_code_delta import DeltaCodes, decode, encode
from mosaickv.evaluation.lossless_code_delta_gpu import DeviceCodes, validate_bytes
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import PrecisionFormat, QuantizedBody


def cpu_encode(previous, current):
    p = encode(None if previous is None else previous.numpy(), current.numpy())
    return DeviceCodes(
        p.length,
        torch.from_numpy(p.payload),
        None if p.raw else torch.from_numpy(p.offsets.view(np.int32)),
    )


def cpu_decode(previous, packet, destination=None):
    p = DeltaCodes(
        packet.raw,
        packet.length,
        np.empty(0, np.uint32) if packet.raw else packet.offsets.numpy().view(np.uint32),
        packet.payload.numpy(),
    )
    value = torch.from_numpy(decode(None if previous is None else previous.numpy(), p))
    if destination is None:
        return value
    destination.copy_(value)
    return destination


@pytest.fixture
def packets(monkeypatch):
    monkeypatch.setattr(runtime, "encode", cpu_encode)
    monkeypatch.setattr(runtime, "decode", cpu_decode)
    output = []
    for layer in range(8):
        codes = torch.zeros(1024, dtype=torch.uint8)
        codes[layer] = 1
        body = QuantizedBody(
            (1, 1, 8, 128), 8, 3, 128, codes, torch.zeros(1, 1, 8, 1), torch.ones(1, 1, 8, 1)
        )
        output.append(
            PrecisionFormat(
                body.shape,
                (body, replace(body, codes=codes.clone())),
                torch.empty(0, dtype=torch.int32),
                torch.empty(0, dtype=torch.int32),
                tuple(torch.empty(1, 1, 0, 128, dtype=torch.bfloat16) for _ in range(2)),
            )
        )
    return tuple(output)


def test_exact_replay_accounting_and_request_isolation(packets):
    encoded, report = runtime.convert(packets)
    assert report["enabled"]
    a, b = runtime.bind(encoded), runtime.bind(encoded)
    assert a[0].replay is not b[0].replay
    assert all(
        x.data_ptr() != b[0].replay.scratch[role].data_ptr()
        for role, x in a[0].replay.scratch.items()
    )
    actual = backing_bytes(tuple(t for p in a for t in p.tensors()))
    assert actual == report["net_active_bytes"]
    assert actual < report["original_persistent_bytes"]
    for repeat in range(3):
        for original, bound in zip(packets, (a, b, a)[repeat], strict=True):
            view = bound.replay.view(bound.index)
            assert bound.replay.view(bound.index) is view
            for old, new in zip(original.pair, view.pair, strict=True):
                assert torch.equal(old.codes, new.codes)
                assert new.minimum is old.minimum
                assert new.scale is old.scale
            assert view.exact_indices is original.exact_indices
            assert view.exact_values is original.exact_values
    with pytest.raises(ValueError, match="cannot be reused"):
        runtime.bind(a)


def test_out_of_order_and_unbound_use_fail_closed(packets):
    encoded, _ = runtime.convert(packets)
    a = runtime.bind(encoded)
    with pytest.raises(RuntimeError, match="ordered"):
        a[1].replay.view(1)
    with pytest.raises(RuntimeError, match="binding"):
        _ = encoded[0].template.pair[0].codes
    assert all(p.metadata.codes is None for e in encoded for p in e.template.pair)
    a[0].replay.view(0)
    with pytest.raises(RuntimeError, match="ordered"):
        a[2].replay.view(2)


def test_unprofitable_and_length_change_fallback(packets):
    encoded, report = runtime.convert(packets[:1])
    assert not report["enabled"] and encoded[0] is packets[0]
    assert runtime.bind(encoded)[0] is packets[0]
    different = replace(
        packets[1], pair=tuple(replace(b, codes=b.codes[:512].clone()) for b in packets[1].pair)
    )
    _, report = runtime.convert((packets[0], different))
    assert not report["enabled"]
    assert all(r["raw_fallback"] for r in report["records"])


def test_unsupported_packets_and_cpu_kernel_use_rejected(packets):
    with pytest.raises(NotImplementedError):
        runtime.map_bodies(object(), lambda *args: None)
    with pytest.raises(ValueError, match="CUDA"):
        validate_bytes(torch.zeros(128, dtype=torch.uint8))
    with pytest.raises(ValueError, match="complete immutable"):
        runtime.Replay(packets)


def test_bound_component_transport_uses_original_fields(packets, monkeypatch):
    encoded, _ = runtime.convert(packets)
    bound = runtime.bind(encoded)
    seen = []
    monkeypatch.setattr(
        "mosaickv.evaluation.sequential_kv_lease.fill_component",
        lambda p, a, c, d: seen.append((p, a, c, d)),
    )
    target = torch.empty(1)
    bound[0].fill_component("attention", 1, target)
    assert torch.equal(seen[0][0].pair[1].codes, packets[0].pair[1].codes)
    assert seen[0][1:] == ("attention", 1, target)


def test_shared_replay_accounting_matches_live_cache_before_and_after_updates(packets):
    from mosaickv.evaluation.storage_runs import cache_from_formats

    encoded, report = runtime.convert(packets)
    adapter = SimpleNamespace(
        _language_layers=lambda: [
            SimpleNamespace(self_attn=torch.nn.Linear(128, 128, bias=False).bfloat16())
            for _ in packets
        ]
    )
    cache = cache_from_formats(
        adapter, encoded, updates=2, audit=False, sealed=False, sequential_lease=True
    )
    assert cache.initial_bytes == report["net_active_bytes"]
    assert cache.storage_record()["final_active_kv_storage_bytes"] == cache.initial_bytes
    for step in range(2):
        for i, layer in enumerate(cache.layers):
            state = torch.full((1, 1, 1, 128), step, dtype=torch.bfloat16)
            key, value = layer.update(state, state)
            expected = packets[i].materialize()
            assert torch.equal(key[..., :8, :], expected[0])
            layer.materialize_component(1)
            assert torch.equal(value[..., :8, :], expected[1])
            layer.release()
        observed = cache.storage_record()
        assert observed["final_active_kv_storage_bytes"] == backing_bytes(
            tuple(t for layer in cache.layers for t in layer.owned_tensors())
        )
