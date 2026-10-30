from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from mosaickv.evaluation.source_history import SourceHistoryCache, build_chunk_direct
from mosaickv.evaluation.storage_runs import cache_from_formats


@pytest.mark.parametrize("policy", ["unknown", "", None, True])
def test_unknown_constructor_is_rejected_before_allocations(policy):
    with pytest.raises(ValueError, match="reference or direct"):
        SourceHistoryCache(audit_workspace=False, constructor=policy)


def test_constructor_cannot_be_silently_enabled_without_source_history():
    with pytest.raises(ValueError, match="explicit source history"):
        cache_from_formats(None, (), updates=128, audit=False, source_history_constructor="direct")


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_direct_builder_keeps_three_final_original_bit_checks(monkeypatch, corrupt, deferred):
    import mosaickv.evaluation.encoded_projection as formats
    import mosaickv.evaluation.encoded_projection_kernel as kernels

    calls = []
    references = tuple(torch.zeros(1, 2, 16, 128, dtype=torch.bfloat16) for _ in range(2))

    @dataclass
    class FakeFormat:
        kind: str
        shape: tuple
        archive: Any
        output_slots: Any
        literal_checkpoints: Any
        patches: tuple = ()

        def materialize(self, attention):
            calls.append(bool(self.patches))
            values = tuple(reference + 1 for reference in references)
            if not corrupt and self.patches:
                for patch, tensor in zip(self.patches, values, strict=True):
                    patch.apply(tensor)
            return values

    monkeypatch.setattr(formats, "EncodedFormat", FakeFormat)
    monkeypatch.setattr(kernels, "checkpoints", lambda body: torch.zeros(4, dtype=torch.int16))
    layer = SimpleNamespace(
        prompt_shape=(1, 2, 1, 128),
        normalized_tail=torch.ones(1, 16, 256, dtype=torch.bfloat16),
        history=references,
        attention=None,
    )
    phase = torch.ones(1, 16, 128, dtype=torch.bfloat16)
    if deferred:
        flags = []
        result = build_chunk_direct(layer, phase, phase, bit_flags=flags)
        assert calls == [False, True, True, True]
        assert len(flags) == 6
        assert bool(torch.stack(flags).all()) == (not corrupt)
    elif corrupt:
        with pytest.raises(RuntimeError, match="original K/V bits"):
            build_chunk_direct(layer, phase, phase)
        assert calls == [False, True]
    else:
        result = build_chunk_direct(layer, phase, phase)
        assert len(result.patches) == 2
        assert calls == [False, True, True, True]
        assert all(
            torch.equal(value, reference)
            for value, reference in zip(result.materialize(None), references, strict=True)
        )
