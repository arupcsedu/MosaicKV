from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import native_history_reader as native
from mosaickv.evaluation.exact_storage import decoded, encode
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.mixed_history_projection import MixedHistory, construct, use
from mosaickv.evaluation.projection_archive import ProjectionArchive


def setup(monkeypatch, count=3):
    class Plans:
        def execute(self, source, weight, output, n):
            output[:n].copy_(source[:n] * weight)
            if n >= 32:
                output[0, 0] += 0.125  # force a counted shape-dependent correction

    class State:
        graphs = None
        closed = False
        projections = 0
        engine = SimpleNamespace(
            source=torch.empty(128, 4096, dtype=torch.bfloat16),
            output=torch.empty(128, 4096, dtype=torch.bfloat16),
            plans=Plans(),
        )
        pool = SimpleNamespace(borrow_history=lambda _: nullcontext(None))

    state = State()
    monkeypatch.setattr(native, "current", lambda: state)
    monkeypatch.setattr(
        "mosaickv.evaluation.shared_source_kernel.rotate_native", lambda *args: None
    )
    attention = SimpleNamespace(k_proj=SimpleNamespace(weight=1), v_proj=SimpleNamespace(weight=2))
    formats = []
    cos, sin = torch.ones(1, count * 16, 128).bfloat16(), torch.zeros(1, count * 16, 128).bfloat16()
    for i in range(count):
        source = encode(torch.randn(1, 1, 16, 4096).bfloat16())
        ids = torch.arange(16, dtype=torch.int32)
        c, s = cos[:, i * 16 : (i + 1) * 16], sin[:, i * 16 : (i + 1) * 16]
        formats.append(
            native.NativeHistoryFormat(
                "fixture",
                (1, 32, 16, 128),
                archive=ProjectionArchive(source, ids, ids[None].expand(32, -1), 16, c, s),
                output_slots=ids[None].expand(32, -1),
                literal_checkpoints=torch.empty(0, dtype=torch.int16),
                phase_versions=(native.phase_token(c), native.phase_token(s)),
            )
        )

    def fill(packet, attention, component, target):
        value = decoded(packet.archive.source)[0, 0] * (component + 1)
        target[0].copy_(value.view(16, 32, 128).transpose(0, 1))
        state.projections += 1

    monkeypatch.setattr(native.NativeHistoryFormat, "fill_component", fill)
    return SimpleNamespace(source_sealed=formats, attention=attention), state, cos, sin


def test_one_projection_preserves_unfused_words_with_counted_shape_repairs(monkeypatch):
    layer, state, cos, sin = setup(monkeypatch)
    mixed = construct(layer, cos, sin)
    assert sum(p.indices.numel() for p in mixed.repairs) == 2
    expected = tuple(torch.empty(mixed.shape, dtype=torch.bfloat16) for _ in range(2))
    for i, f in enumerate(layer.source_sealed):
        f.fill(
            layer.attention, tuple(t[..., i * 16 : (i + 1) * 16, :] for t in expected), direct=True
        )
    before = state.projections
    actual = mixed.materialize(layer.attention)
    assert state.projections - before == 2  # K and V, not 2 * three fragments.
    assert all(a.equal(b) for a, b in zip(expected, actual, strict=True))
    for old, stored in zip(layer.source_sealed, mixed.recent, strict=True):
        assert old is stored  # No second persistent source copy.
    sentinel = torch.full((1, 32, 64, 128), -91, dtype=torch.bfloat16)
    from mosaickv.evaluation.sequential_kv_lease import fill_component

    fill_component(mixed, layer.attention, 0, sentinel[..., :48, :])
    assert sentinel[..., :48, :].equal(expected[0])
    assert bool((sentinel[..., 48:, :] == -91).all())


def test_owner_and_phase_mutation_fail_before_projection(monkeypatch):
    layer, state, cos, sin = setup(monkeypatch, count=2)
    mixed = construct(layer, cos, sin)
    before = state.projections
    monkeypatch.setattr(native, "current", lambda: object())
    with pytest.raises(RuntimeError):
        mixed.materialize(layer.attention)
    assert state.projections == before
    monkeypatch.setattr(native, "current", lambda: state)
    cos.add_(1)
    with pytest.raises(RuntimeError, match="phases changed"):
        mixed.materialize(layer.attention)


def test_repeated_aging_preserves_64_exact_sources_and_charges_all_buffers(monkeypatch):
    from mosaickv.evaluation import consolidated_history as combined
    from mosaickv.evaluation.history_source_precision import use as precision

    layer, _state, _, _ = setup(monkeypatch, count=6)
    source_packets = list(layer.source_sealed)
    layer.source_sealed = []
    layer.leased, layer.source_pending, layer.history_length, layer.resident_bytes = (), False, 0, 0
    cache = SimpleNamespace(
        layers=[layer],
        shared_tensors=(),
        shared_bytes=0,
        history_rotary_chunks=[],
        consolidated_phase_versions=[],
        source_consolidation_layers=0,
        source_consolidation_epochs=0,
    )

    def observe(extras=()):
        actual = backing_bytes(
            (*cache.shared_tensors, *(t for f in layer.source_sealed for t in f.private()))
        )
        assert actual == cache.shared_bytes + layer.resident_bytes

    cache.observe = observe
    with precision(feature_group=32, exact_recent_tokens=64), use() as report:
        for index, packet in enumerate(source_packets):
            layer.source_sealed.append(packet)
            layer.history_length += 16
            layer.resident_bytes += backing_bytes(packet.private())
            phases = packet.archive.cos, packet.archive.sin
            cache.history_rotary_chunks.append(phases)
            cache.shared_tensors += phases
            cache.shared_bytes = backing_bytes(cache.shared_tensors)
            combined.consolidate(cache)
            if index:
                mixed = layer.source_sealed[0]
                assert len(layer.source_sealed) == 1 and isinstance(mixed, MixedHistory)
                assert mixed.shape[-2] == layer.history_length
                assert all(
                    a is b
                    for a, b in zip(
                        mixed.recent, source_packets[max(0, index - 3) : index + 1], strict=True
                    )
                )
                assert (mixed.older is None) == (index < 4)
            observe()
        assert report["mixed_constructions"] == 5
        assert report["no_extra_persistent_source_copy"]
