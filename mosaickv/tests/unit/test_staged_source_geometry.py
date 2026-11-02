from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import shared_source_format, text_source_storage
from mosaickv.evaluation.shared_source_format import SharedSourceFormat
from mosaickv.evaluation.streaming_source_constructor import StreamingDirectSource


@pytest.mark.parametrize("spec", [None, (16, 4), (32, 4), (32, 2)])
def test_only_text_storage_receives_new_geometry_after_original_fit(monkeypatch, spec):
    p = SharedSourceFormat(
        SimpleNamespace(minimum=torch.ones(1), scale=torch.ones(1)),
        (1, 32, 4, 128),
        None,
        (),
        None,
        None,
        None,
        None,
    )
    events = []
    monkeypatch.setattr(shared_source_format, "construct", lambda *args, **kwargs: p)

    def fit(packet, attention):
        assert packet is p and packet.source_row_tile == 16 and packet.source_head_group == 1
        events.append("fit")
        return packet, {}

    def optimize(packet, *args):
        events.append("text")
        expected = (16, 1) if spec is None else spec
        assert (packet.source_row_tile, packet.source_head_group) == expected
        return packet, {}

    monkeypatch.setattr(shared_source_format, "fit_transport", fit)
    monkeypatch.setattr(text_source_storage, "optimize", optimize)
    prepared = SimpleNamespace(
        logical_sequence_length=4, modality_map=(SimpleNamespace(modality="text", start=3, end=4),)
    )
    cache = SimpleNamespace(layers=[None] * 32)
    encoder = StreamingDirectSource(None, prepared, cache, (0.5,) * 32, construction_geometry=spec)
    pair = tuple(torch.zeros((1, 32, 4, 128), dtype=torch.bfloat16) for _ in range(2))
    output = (None, torch.full((1, 32, 4, 4), 0.25, dtype=torch.bfloat16))
    result = encoder(
        0,
        None,
        pair,
        (),
        dict(
            hidden_states=torch.zeros((1, 4, 4096), dtype=torch.bfloat16),
            position_embeddings=tuple(
                torch.ones((1, 4, 128), dtype=torch.bfloat16) for _ in range(2)
            ),
        ),
        output,
    )
    assert events == ["fit", "text"]
    assert (result.source_row_tile, result.source_head_group) == ((16, 4) if spec is None else spec)
    assert result.source is p.source and result.exact is p.exact


def test_invalid_geometry_fails_before_model_access():
    for spec in ((17, 4), (True, 4), (32, 8)):
        with pytest.raises(ValueError, match="construction geometry"):
            StreamingDirectSource(None, None, None, (), construction_geometry=spec)
