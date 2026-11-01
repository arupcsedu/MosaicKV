from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from mosaickv.evaluation.request_precision import online_frozen
from mosaickv.evaluation.streaming_source_constructor import StreamingDirectSource


def test_local_balanced_selection_matches_full_constructor_without_labels():
    torch.manual_seed(277)
    n, h = 8, 32
    probabilities = tuple(torch.softmax(torch.randn(1, h, n, n), -1).bfloat16() for _ in range(2))
    full, _ = online_frozen(
        SimpleNamespace(attention_weights=probabilities),
        (0.8, 0.2),
        0.5,
        method="mosaickv_exact",
        allocation_mode="balanced",
    )
    expected = full["uniform_0.5_control"]["anchor"]["positions"]
    prepared = SimpleNamespace(
        logical_sequence_length=n,
        modality_map=(SimpleNamespace(modality="text", start=6, end=8),),
    )
    cache = SimpleNamespace(layers=[None, None], extra_tensors=())
    encoder = StreamingDirectSource(None, prepared, cache, (0.8, 0.2))
    phase = (torch.ones(1, n, 128).bfloat16(), torch.zeros(1, n, 128).bfloat16())
    captured = []

    def construct(x, original, memberships, *args, **kwargs):
        captured.append(memberships)
        return object()

    with (
        patch("mosaickv.evaluation.shared_source_format.construct", construct),
        patch("mosaickv.evaluation.shared_source_format.fit_transport", lambda p, a: (p, {})),
        patch("mosaickv.evaluation.text_source_storage.optimize", lambda p, *a: (p, {})),
        patch("mosaickv.evaluation.source_retile.retile", lambda p, g: p),
    ):
        for i in range(2):
            x = torch.randn(1, n, h * 128).bfloat16()
            pair = tuple(torch.randn(1, h, n, 128).bfloat16() for _ in range(2))
            encoder(
                i,
                None,
                pair,
                (),
                dict(hidden_states=x, position_embeddings=phase),
                (None, probabilities[i]),
            )
    assert tuple(captured) == expected
    assert tuple(encoder.memberships) == expected
    assert cache.extra_tensors
    encoder.finish()
    assert not cache.extra_tensors and encoder.finished
    with pytest.raises(ValueError):
        encoder(2, None, pair, (), {}, None)


def test_odd_capacity_layouts_fail_before_construction():
    with pytest.raises(NotImplementedError):
        StreamingDirectSource(
            None,
            SimpleNamespace(
                logical_sequence_length=7,
                modality_map=(SimpleNamespace(modality="text", start=6, end=7),),
            ),
            None,
            (0.5,),
        )
