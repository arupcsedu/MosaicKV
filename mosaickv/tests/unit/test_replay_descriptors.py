from dataclasses import replace

import pytest
import torch
from test_layer_code_delta import make_packets

from mosaickv.evaluation import layer_code_delta as runtime
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.replay_descriptor_policy import current, use


@pytest.fixture
def packets(monkeypatch):
    return make_packets(monkeypatch)


def test_prepared_replay_matches_original_bits_accounting_and_header_identity(packets):
    encoded, report = runtime.convert(packets)
    with use(True) as state:
        cached = runtime.bind(encoded)
    ordinary = runtime.bind(encoded)
    assert state["prepared_formats"] == len(packets) and state["reused_formats"] == 0
    assert (
        backing_bytes(tuple(t for p in cached for t in p.tensors())) == report["net_active_bytes"]
    )
    ids = tuple(id(x) for x in cached[0].replay.prepared_views)
    for _ in range(4):
        for source, a, b in zip(packets, cached, ordinary, strict=True):
            left, right = a.replay.view(a.index), b.replay.view(b.index)
            assert a.replay.view(a.index) is left
            assert id(left) == ids[a.index]
            for original, x, y in zip(source.pair, left.pair, right.pair, strict=True):
                assert torch.equal(x.codes, y.codes) and torch.equal(x.codes, original.codes)
                assert x.minimum is original.minimum and x.scale is original.scale
            assert left.exact_indices is source.exact_indices
    assert state["reused_formats"] == 4 * len(packets)
    assert all(b.metadata.codes is None for p in encoded for b in p.template.pair)


def test_no_eager_decoding_no_repeated_descriptor_construction_and_request_isolation(
    packets, monkeypatch
):
    encoded, _ = runtime.convert(packets)
    decode, mapped = runtime.decode, runtime.map_bodies
    calls = dict(decode=0, maps=0)

    def decoder(*args):
        calls["decode"] += 1
        return decode(*args)

    def mapper(*args):
        calls["maps"] += 1
        return mapped(*args)

    monkeypatch.setattr(runtime, "decode", decoder)
    monkeypatch.setattr(runtime, "map_bodies", mapper)
    with use(True):
        a, b = runtime.bind(encoded), runtime.bind(encoded)
    assert calls["decode"] == 0
    maps = calls["maps"]
    assert all(
        a[0].replay.scratch[r].data_ptr() != t.data_ptr() for r, t in b[0].replay.scratch.items()
    )
    for _ in range(3):
        for index in range(len(packets)):
            for owner in (a, b):
                value = owner[index].replay.view(index)
                assert all(
                    torch.equal(x.codes, y.codes)
                    for x, y in zip(value.pair, packets[index].pair, strict=True)
                )
    assert calls["maps"] == maps and calls["decode"] > 0
    for invalid in (-1, len(packets), True):
        with pytest.raises(ValueError, match="index"):
            a[0].replay.view(invalid)
    with pytest.raises(RuntimeError, match="ordered"):
        a[0].replay.view(2)


def test_varying_lengths_and_raw_restart_preserve_codec_semantics(packets):
    shorter = replace(
        packets[3],
        shape=(1, 1, 4, 128),
        pair=tuple(
            replace(
                b,
                shape=(1, 1, 4, 128),
                codes=b.codes[:512].clone(),
                minimum=b.minimum[:, :, :4],
                scale=b.scale[:, :, :4],
            )
            for b in packets[3].pair
        ),
    )
    sources = (*packets[:3], shorter, *packets[4:])
    encoded, _ = runtime.convert(sources)
    with use(True):
        bound = runtime.bind(encoded)
    if not isinstance(bound[0], runtime.BoundDeltaFormat):
        pytest.fail("fixture must retain a beneficial actual byte chain")
    for _ in range(2):
        for item, source in zip(bound, sources, strict=True):
            actual = item.replay.view(item.index)
            assert all(
                torch.equal(x.codes, y.codes) for x, y in zip(actual.pair, source.pair, strict=True)
            )


def test_policy_restores_on_failure_and_never_enables_implicitly():
    assert current() is None
    with pytest.raises(RuntimeError), use(True) as state:
        assert current() is state
        with use(False):
            assert current() is None
        raise RuntimeError("expected")
    assert current() is None
    for invalid in (1, "yes", None):
        with pytest.raises(ValueError), use(invalid):
            pytest.fail("implicit activation")


def test_foreign_decoder_destination_cannot_populate_prepared_view(packets, monkeypatch):
    encoded, _ = runtime.convert(packets)
    original = runtime.decode
    monkeypatch.setattr(runtime, "decode", lambda *args: original(*args).clone())
    with use(True):
        bound = runtime.bind(encoded)
    bound[0].replay.view(0)
    with pytest.raises(RuntimeError, match="foreign replay"):
        bound[1].replay.view(1)
