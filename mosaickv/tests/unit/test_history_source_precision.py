import pytest
import torch

from mosaickv.evaluation.history_source_precision import join, unpack_into
from mosaickv.evaluation.precision_tier import pack, unpack_cpu


def test_triton_dependencies_parse_on_CPU_without_unsupported_language_API():
    pytest.importorskip("triton")
    from mosaickv.evaluation.history_source_precision_kernel import unpack
    from mosaickv.evaluation.packed_prototype_kernel import copy

    assert isinstance(unpack.cache_key, str) and unpack.cache_key
    assert isinstance(copy.cache_key, str) and copy.cache_key


def test_quantized_source_oracle_join_and_workspace_sentinels():
    torch.manual_seed(8)
    first, second = [pack(torch.randn(1, 1, n, 256).bfloat16(), bits=8, axis=3) for n in (16, 32)]
    body = join((first, second))
    expected = torch.cat((unpack_cpu(first), unpack_cpu(second)), 2)
    target = torch.full((128, 256), -99, dtype=torch.bfloat16)
    unpack_into(body, target)
    assert torch.equal(target[:48].flatten(), expected.flatten())
    assert bool((target[48:] == -99).all())
    assert join((first,)) is first
    assert sum(t.numel() * t.element_size() for t in body.tensors()) < expected.numel() * 2
    with pytest.raises(ValueError):
        join(())
    with pytest.raises(ValueError):
        unpack_into(body, target.float())


def test_request_and_rotary_isolation_reject_before_any_projection(monkeypatch):
    import weakref

    from mosaickv.evaluation import native_history_reader as native
    from mosaickv.evaluation.history_source_precision import HistoryFormat

    class State:
        graphs = None
        closed = False

    owner, stranger = State(), State()
    body = pack(torch.randn(1, 1, 16, 256).bfloat16(), bits=8, axis=3)
    cos, sin = torch.ones(1, 16, 128).bfloat16(), torch.zeros(1, 16, 128).bfloat16()
    packet = HistoryFormat(
        body,
        cos,
        sin,
        (1, 32, 16, 128),
        weakref.ref(owner),
        (native.phase_token(cos), native.phase_token(sin)),
    )
    monkeypatch.setattr(native, "current", lambda: stranger)
    with pytest.raises(RuntimeError, match="cross-request"):
        packet.fill_component(None, 0, None)
    monkeypatch.setattr(native, "current", lambda: owner)
    cos.add_(1)
    with pytest.raises(RuntimeError, match="phases changed"):
        packet.fill_component(None, 0, None)


def test_history_transaction_preserves_recent_packet_phases_and_real_storage(monkeypatch):
    from types import SimpleNamespace

    from mosaickv.evaluation import consolidated_history as combined
    from mosaickv.evaluation import native_history_reader as native
    from mosaickv.evaluation.exact_storage import encode
    from mosaickv.evaluation.history_source_precision import HistoryFormat, use
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
    from mosaickv.evaluation.projection_archive import ProjectionArchive

    phases = [
        (torch.ones(1, 16, 128).bfloat16(), torch.zeros(1, 16, 128).bfloat16()) for _ in range(3)
    ]

    def packet(pair):
        source = encode(torch.randn(1, 1, 16, 128).bfloat16())
        ids = torch.arange(16, dtype=torch.int32)
        return native.NativeHistoryFormat(
            "fixture",
            (1, 32, 16, 128),
            archive=ProjectionArchive(source, ids, ids[None].expand(32, -1), 16, *pair),
            output_slots=ids[None].expand(32, -1),
            literal_checkpoints=torch.empty(0, dtype=torch.int16),
        )

    layers = []
    for _ in range(2):
        packets = [packet(p) for p in phases[:2]]
        layers.append(
            SimpleNamespace(
                source_sealed=packets,
                leased=(),
                source_pending=False,
                history_length=32,
                resident_bytes=backing_bytes(tuple(t for f in packets for t in f.private())),
            )
        )
    cache = SimpleNamespace(
        layers=layers,
        history_rotary_chunks=phases[:2],
        shared_tensors=tuple(t for p in phases[:2] for t in p),
        consolidated_phase_versions=[],
        source_consolidation_layers=0,
        source_consolidation_epochs=0,
    )
    cache.shared_bytes = backing_bytes(cache.shared_tensors)

    def observe(extras=()):
        actual = backing_bytes(
            (
                *cache.shared_tensors,
                *(t for layer in layers for f in layer.source_sealed for t in f.private()),
            )
        )
        assert actual == cache.shared_bytes + sum(layer.resident_bytes for layer in layers)

    cache.observe = observe
    original = combined.consolidate

    class State:
        graphs = None

    state = State()
    monkeypatch.setattr(native, "current", lambda: state)
    newest = [layer.source_sealed[-1] for layer in layers]
    with use() as report:
        assert combined.consolidate(cache)
        for layer, recent in zip(layers, newest, strict=True):
            assert isinstance(layer.source_sealed[0], HistoryFormat)
            assert layer.source_sealed[0].shape[-2] == 16
            assert layer.source_sealed[-1] is recent
        # Another complete exact chunk ages the previous recent chunk.
        for layer in layers:
            new = packet(phases[2])
            layer.source_sealed.append(new)
            layer.resident_bytes += backing_bytes(new.private())
            layer.history_length = 48
        cache.history_rotary_chunks.append(phases[2])
        cache.shared_tensors += phases[2]
        cache.shared_bytes = backing_bytes(cache.shared_tensors)
        assert combined.consolidate(cache)
        assert all(layer.source_sealed[0].shape[-2] == 32 for layer in layers)
        assert cache.history_rotary_chunks[0][0].shape[1] == 48
        assert report["conversions"] == 4
        observe()
    assert combined.consolidate is original
