import pytest
import torch

from mosaickv.evaluation.complete_transport_predicate import _predicate, fused, oracle, use
from mosaickv.evaluation.shared_source_format import transport_bad_mask


def test_triton_parse_without_gpu():
    assert isinstance(_predicate.cache_key, str) and _predicate.cache_key


@pytest.mark.parametrize("shift", [0, 1, 128, 8191])
def test_literal_bf16_rounding_formula_matches_pinned_aten_on_all_words(shift):
    x = torch.arange(65536, dtype=torch.int32).short().view(torch.bfloat16)
    y = x.roll(shift)
    # The GPU implementation must emulate these separate BF16 tensor operators.
    error = (y.float() - x.float()).bfloat16().abs()
    allowed = ((0.01 * x.float()).bfloat16().abs().float() + 2**-6).bfloat16()
    close = (x == y) | (torch.isfinite(error) & (error <= allowed))
    assert torch.equal(close, torch.isclose(y, x, rtol=0.01, atol=2**-6))
    bad, finite = oracle(x, y)
    assert torch.equal(bad, transport_bad_mask(x, y))
    assert torch.equal(finite, torch.isfinite(y).all())


def test_scope_restores_frozen_default_and_rejects_unsupported_cpu():
    from mosaickv.evaluation import shared_source_format as source

    original = source.transport_corrections, source.assert_transport_close
    with use() as report:
        assert source.transport_corrections is not original[0]
        with pytest.raises(NotImplementedError):
            fused(torch.ones(4).bfloat16(), torch.ones(4).bfloat16())
    assert (source.transport_corrections, source.assert_transport_close) == original
    assert not report["correction_calls"] and not report["verification_calls"]


def test_componentwise_explicit_chunk_uses_registered_complete_span(monkeypatch):
    from mosaickv.evaluation import complete_transport_predicate as module
    from mosaickv.evaluation import shared_source_format as source
    from mosaickv.evaluation.deferred_transport_checks import use as validation

    # CPU numerical oracle substitutes the CUDA kernel, not span-policy logic.
    monkeypatch.setattr(module, "fused", lambda x, y, **_: oracle(x, y))
    x = torch.ones(1, 1, 8, 128, dtype=torch.bfloat16)
    y = x.clone()
    y.flatten()[-1] = 2
    empty = torch.empty(0, dtype=torch.int32)
    with use(), validation(True, coalesce_nonzero=True, full_predicate_span=True):
        patch = source.transport_corrections((x,), (y,), empty, chunk_scalars=17)[0]
    assert patch.indices.tolist() == [1023]
    assert patch.values.tolist() == [1]
    with (
        use(),
        validation(True, coalesce_nonzero=True, full_predicate_span=False),
        pytest.raises(ValueError, match="full-span"),
    ):
        source.transport_corrections((x,), (y,), empty, chunk_scalars=17)
