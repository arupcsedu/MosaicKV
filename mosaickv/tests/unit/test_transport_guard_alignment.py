"""CPU synthetic numerical regression, never benchmark evidence."""

import pytest
import torch

from mosaickv.evaluation.shared_source_format import (
    assert_transport_close,
    transport_bad_mask,
    transport_corrections,
)


@pytest.mark.parametrize("chunk", [None, 1, 127, 524288])
def test_asymmetric_bf16_boundary_is_repaired_without_changing_guard(chunk):
    reference = torch.full((1, 1, 1, 128), 0.375, dtype=torch.bfloat16)
    actual = reference.clone()
    actual.flatten()[0] = 0.39453125
    assert torch.isclose(reference.float(), actual.float(), rtol=0.01, atol=2**-6).all()
    with pytest.raises(AssertionError):
        assert_transport_close(actual, reference)
    patch = transport_corrections(
        (reference,), (actual,), torch.tensor([], dtype=torch.int32), chunk_scalars=chunk
    )[0]
    assert patch.indices.tolist() == [0]
    patch.apply(actual)
    assert_transport_close(actual, reference)
    assert torch.equal(actual, reference)


def test_new_repairs_never_hide_protected_rows():
    reference = torch.full((1, 1, 1, 128), 0.375, dtype=torch.bfloat16)
    actual = reference.clone()
    actual.flatten()[0] = 0.39453125
    with pytest.raises(RuntimeError, match="protected"):
        transport_corrections((reference,), (actual,), torch.tensor([0], dtype=torch.int32))


def test_every_previously_valid_packet_keeps_its_original_mask():
    torch.manual_seed(42)
    for scale in (0, 0.001, 0.01, 0.1, 1):
        reference = torch.randn(4096).bfloat16()
        actual = (reference.float() + torch.randn(4096) * scale).bfloat16()
        legacy = ~torch.isclose(reference.float(), actual.float(), rtol=0.01, atol=2**-6)
        old_repaired = torch.where(legacy, reference, actual)
        fixed = transport_bad_mask(reference, actual)
        assert not (legacy & ~fixed).any()
        new_repaired = torch.where(fixed, reference, actual)
        assert_transport_close(new_repaired, reference)
        if torch.isclose(old_repaired, reference, rtol=0.01, atol=2**-6).all():
            assert torch.equal(fixed, legacy)
            assert torch.equal(old_repaired, new_repaired)
