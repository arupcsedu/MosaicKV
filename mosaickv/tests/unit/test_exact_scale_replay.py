"""CPU exactness/ownership tests; GPU kernels are verified in a separate gate."""

import gc
import weakref
from dataclasses import replace

import pytest
import torch
from test_layer_code_delta import cpu_decode, cpu_encode, make_packets

from mosaickv.evaluation import exact_scale_replay as headers
from mosaickv.evaluation import layer_code_delta as runtime
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import PrecisionFormat, QuantizedBody
from mosaickv.evaluation.replay_descriptor_policy import use as descriptors


def packets(monkeypatch):
    monkeypatch.setattr(runtime, "encode", cpu_encode)
    monkeypatch.setattr(runtime, "decode", cpu_decode)
    output = []
    for i in range(8):
        codes = torch.zeros(512 * 128, dtype=torch.uint8)
        codes[i] = 1
        scale = ((torch.arange(512) % 7).float() + i + 1).reshape(1, 1, 512, 1)
        body = QuantizedBody(
            (1, 1, 512, 128),
            8,
            3,
            128,
            codes,
            torch.zeros_like(scale, dtype=torch.bfloat16),
            scale,
        )
        output.append(
            PrecisionFormat(
                body.shape,
                (body, replace(body, codes=codes.clone(), scale=scale.clone())),
                torch.empty(0, dtype=torch.int32),
                torch.empty(0, dtype=torch.int32),
                tuple(torch.empty(1, 1, 0, 128, dtype=torch.bfloat16) for _ in range(2)),
            )
        )
    return tuple(output)


@pytest.mark.parametrize("prepared,captured", [(False, False), (True, False), (True, True)])
def test_scale_replay_exact_private_accounted_ordered(monkeypatch, prepared, captured):
    original = packets(monkeypatch)
    encoded, _ = runtime.convert(original)
    compact, report = headers.convert(encoded)
    assert report["enabled"] and report["CPU_residual_bytes"] == 0
    with descriptors(prepared):
        a, b = headers.ScaleReplay(compact), headers.ScaleReplay(compact)
    bound = tuple(runtime.BoundDeltaFormat(p, a, i) for i, p in enumerate(compact))
    actual_bytes = backing_bytes(tuple(t for p in bound for t in p.tensors()))
    assert actual_bytes == report["net_active_bytes"] < report["original_bound_active_bytes"]
    assert len(a.scale_scratch) == 2
    for role in a.scale_scratch:
        assert a.scale_scratch[role].data_ptr() != b.scale_scratch[role].data_ptr()
    with pytest.raises(RuntimeError, match="ordered"):
        a.view(1)
    for _ in range(3):
        for i, packet in enumerate(original):
            view = a.advance_for_graph(i) if captured else a.view(i)
            if captured:
                # Simulate stale workspace before replay of the captured kernel.
                for scratch in a.scale_scratch.values():
                    scratch.fill_(99)
                a.decode_for_graph(i)
            assert a.view(i) is view
            for old, new in zip(packet.pair, view.pair, strict=True):
                assert torch.equal(old.scale.view(torch.int32), new.scale.view(torch.int32))
                assert torch.equal(old.codes, new.codes)
                assert old.minimum is new.minimum
            if prepared:
                assert view is a.prepared_views[i]
    # B was not advanced by A, and its private buffers cannot alias A.
    assert b.last == -1 and b.previous == {}


def test_original_scale_owners_retired_no_views_keep_storage(monkeypatch):
    original = packets(monkeypatch)
    encoded, _ = runtime.convert(original)
    refs = [weakref.ref(b.metadata.scale) for p in encoded for b in p.template.pair]
    compact, report = headers.convert(encoded)
    assert report["enabled"]
    assert all(b.metadata.scale.numel() == 0 for p in compact for b in p.template.pair)
    del original, encoded
    gc.collect()
    assert all(r() is None for r in refs)


def test_complete_transaction_declines_if_shared_workspace_removes_saving(monkeypatch):
    original = packets(monkeypatch)
    encoded, _ = runtime.convert(original)
    compact, report = headers.convert(encoded[:1])
    assert not report["enabled"]
    assert compact[0] is encoded[0]
    small = tuple(
        replace(p, pair=tuple(replace(b, scale=torch.rand_like(b.scale)) for b in p.pair))
        for p in make_packets(monkeypatch)
    )
    small_encoded, _ = runtime.convert(small)
    same, report = headers.convert(small_encoded)
    assert not report["enabled"]
    assert all(a is b for a, b in zip(same, small_encoded, strict=True))


def test_scope_is_default_off_and_restores_hooks(monkeypatch):
    original = packets(monkeypatch)
    old_convert, old_replay = runtime.convert, runtime.Replay
    with headers.use() as reports, descriptors(True):
        encoded, _ = runtime.convert(original)
        bound = runtime.bind(encoded)
        assert reports[0]["enabled"] and isinstance(bound[0].replay, headers.ScaleReplay)
        with pytest.raises(RuntimeError, match="nonnested"), headers.use():
            pass
        assert bound[0].replay.view(0).pair[0].scale.equal(original[0].pair[0].scale)
    assert runtime.convert is old_convert and runtime.Replay is old_replay
    with pytest.raises(ValueError, match="immutable"):
        headers.convert(bound)
    with pytest.raises(ValueError, match="repeated"):
        headers.convert(encoded)
