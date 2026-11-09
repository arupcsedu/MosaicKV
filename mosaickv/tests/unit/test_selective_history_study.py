from types import SimpleNamespace

import pytest
import selective_history_study as study
import torch

from mosaickv.evaluation import consolidated_history as combined
from mosaickv.evaluation import native_history_reader as native
from mosaickv.evaluation.exact_storage import encode
from mosaickv.evaluation.history_source_precision import HistoryFormat, use
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.projection_archive import ProjectionArchive


def test_full_horizon_guard_rejects_late_drift_without_changing_thresholds():
    p = dict(predictions=128, guard_tokens=32, maximum_KL=0.02, maximum_logit_difference=1.0)
    reference = torch.zeros(128, 4)
    reference[:, 0] = 10
    candidate = reference.clone()
    candidate[96, 0] += 1.25
    proof = study.fidelity(candidate, reference, p, study.ARMS[1])
    assert proof["first_32_tokens_match"] and proof["all_128_tokens_match"]
    assert proof["maximum_32_logit_difference"] == 0
    assert proof["maximum_128_logit_difference"] == 1.25
    assert not proof["passed"]
    assert study.fidelity(reference, reference, p, study.ARMS[0])["passed"]
    assert not study.fidelity(candidate, reference, p, study.ARMS[0])["passed"]


@pytest.mark.parametrize("recent", [True, 0, 32, 63, 65, 128])
def test_only_preregistered_recent_policies_are_allowed(recent):
    with pytest.raises(ValueError, match="16 or 64"), use(exact_recent_tokens=recent):
        pass


def test_64_recent_tokens_remain_exact_and_all_positions_and_bytes_accounted(monkeypatch):
    class State:
        graphs = None

    owner = State()
    monkeypatch.setattr(native, "current", lambda: owner)
    cache = SimpleNamespace(
        layers=[
            SimpleNamespace(
                source_sealed=[],
                leased=(),
                source_pending=False,
                history_length=0,
                resident_bytes=0,
            )
        ],
        shared_tensors=(),
        shared_bytes=0,
        history_rotary_chunks=[],
        consolidated_phase_versions=[],
        source_consolidation_layers=0,
        source_consolidation_epochs=0,
    )

    def observe(extras=()):
        actual = backing_bytes(
            (
                *cache.shared_tensors,
                *(t for layer in cache.layers for f in layer.source_sealed for t in f.private()),
            )
        )
        assert actual == cache.shared_bytes + sum(item.resident_bytes for item in cache.layers)

    cache.observe = observe
    old_packets = []
    layer = cache.layers[0]
    with use(feature_group=32, exact_recent_tokens=64) as report:
        for index in range(6):
            cos, sin = torch.ones(1, 16, 128).bfloat16(), torch.zeros(1, 16, 128).bfloat16()
            source = encode(torch.randn(1, 1, 16, 128).bfloat16())
            ids = torch.arange(16, dtype=torch.int32)
            packet = native.NativeHistoryFormat(
                "fixture",
                (1, 32, 16, 128),
                archive=ProjectionArchive(source, ids, ids[None].expand(32, -1), 16, cos, sin),
                output_slots=ids[None].expand(32, -1),
                literal_checkpoints=torch.empty(0, dtype=torch.int16),
            )
            old_packets.append(packet)
            layer.source_sealed.append(packet)
            layer.history_length += 16
            layer.resident_bytes += backing_bytes(packet.private())
            cache.history_rotary_chunks.append((cos, sin))
            cache.shared_tensors += (cos, sin)
            cache.shared_bytes = backing_bytes(cache.shared_tensors)
            active = combined.consolidate(cache)
            if index < 4:
                assert not active
                assert all(a is b for a, b in zip(layer.source_sealed, old_packets, strict=True))
            else:
                assert active and isinstance(layer.source_sealed[0], HistoryFormat)
                assert layer.source_sealed[0].shape[-2] == (index - 3) * 16
                assert all(
                    a is b for a, b in zip(layer.source_sealed[-4:], old_packets[-4:], strict=True)
                )
                assert sum(p.shape[-2] for p in layer.source_sealed) == layer.history_length
            observe()
        assert report["latest_sealed_exact_tokens"] == 64
        assert report["conversions"] == 2 and report["epochs"] == 2
        assert report["CPU_residual_bytes"] == 0
