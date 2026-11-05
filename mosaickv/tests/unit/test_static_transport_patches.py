import pytest
import torch

from mosaickv.evaluation.exact_storage import SparseBitPatch
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.precision_tier import pack
from mosaickv.evaluation.shared_source_format import (
    SharedSourceFormat,
    apply_transport_patch,
    transport_corrections,
)
from mosaickv.evaluation.static_transport_patches import build, finalize, use


def packet(patches):
    n, heads = 5, 2
    body = torch.zeros(1, 1, 1, 128, dtype=torch.bfloat16)
    return SharedSourceFormat(
        source=pack(torch.zeros(1, 1, n, 256, dtype=torch.bfloat16), bits=3, axis=2),
        shape=(1, heads, n, 128),
        row_positions=torch.tensor([2], dtype=torch.int32),
        exact=(PackedBF16(tuple(body.shape), True, (body,)),) * 2,
        cos=torch.ones(1, n, 128, dtype=torch.bfloat16),
        sin=torch.zeros(1, n, 128, dtype=torch.bfloat16),
        union=torch.arange(n, dtype=torch.int32),
        slots=torch.arange(n, dtype=torch.int16)[None].expand(heads, n),
        transport_patches=patches,
    )


@pytest.mark.parametrize("count", [0, 3, 8])
def test_static_corrections_decode_and_canonical_storage_match_original(count):
    original = torch.zeros(1, 2, 5, 128, dtype=torch.bfloat16)
    expected = original.clone()
    expected.view(-1)[:count] = 2.0
    protected = torch.tensor([2], dtype=torch.int32)
    reference = transport_corrections((expected,), (original,), protected)[0]
    with use(True, capacity=8) as state:
        temporary = transport_corrections((expected,), (original,), protected)[0]
        actual = torch.full((1, 2, 7, 128), -71, dtype=torch.bfloat16)
        actual[..., :5, :].copy_(original)
        apply_transport_patch(temporary, actual, prompt_length=5)
        assert torch.equal(actual[..., :5, :].view(torch.int16), expected.view(torch.int16))
        assert bool((actual[..., 5:, :] == -71).all())
        committed = finalize((packet((temporary,)),), state)[0].transport_patches[0]
    assert type(committed) is SparseBitPatch
    assert torch.equal(committed.indices, reference.indices)
    assert torch.equal(committed.values.view(torch.int16), reference.values.view(torch.int16))
    assert state.report()["canonical_patch_bytes"] == count * 6
    assert state.report()["count_barrier_CPU_bytes"] == 8
    assert state.closed and not state.patches


def test_capacity_overflow_rejects_without_truncation_or_decode():
    expected = torch.zeros(1, 2, 5, 128, dtype=torch.bfloat16)
    mask = torch.zeros_like(expected, dtype=torch.bool)
    mask.reshape(-1)[:9] = True
    with pytest.raises(RuntimeError, match="overflow"), use(True, capacity=8) as state:
        temporary = build(expected, mask)
        finalize((packet((temporary,)),), state)
    assert not state.closed and not state.patches


def test_protected_mismatch_cannot_be_hidden_by_static_patch():
    expected = torch.zeros(1, 2, 5, 128, dtype=torch.bfloat16)
    actual = expected.clone()
    actual[0, 0, 2, 0] = 2.0
    with pytest.raises(RuntimeError, match="protected"), use(True, capacity=8):
        transport_corrections((expected,), (actual,), torch.tensor([2]))


def test_nested_source_wrappers_keep_positions_headers_and_canonical_patches():
    from mosaickv.evaluation.cached_text_reader import CachedTextFormat

    expected = torch.zeros(1, 2, 5, 128, dtype=torch.bfloat16)
    with use(True, capacity=8) as state:
        temporary = build(expected, torch.zeros_like(expected, dtype=torch.bool))
        source = packet((temporary,))
        wrapper = CachedTextFormat(source, torch.empty(0, dtype=torch.int32), ())
        result = finalize((wrapper,), state)[0]
    assert result.flat_tokens is wrapper.flat_tokens
    assert result.base.source is source.source
    assert result.base.row_positions is source.row_positions
    assert type(result.base.transport_patches[0]) is SparseBitPatch


def test_discarded_or_foreign_temporary_cannot_escape_finalization():
    expected = torch.zeros(1, 2, 5, 128, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="every temporary"), use(True, capacity=8) as state:
        build(expected, torch.zeros_like(expected, dtype=torch.bool))
        finalize((packet(()),), state)


def test_scope_is_strict_even_when_disabled_and_requires_commit():
    with use(False), pytest.raises(ValueError), use(True):
        pass
    with pytest.raises(RuntimeError, match="commit before decode"), use(True):
        pass
    with use(False) as state:
        item = packet(())
        assert finalize((item,), state)[0] is item


def test_dense_source_phase_original_values_are_never_mutated():
    source = torch.zeros(1, 2, 5, 128, dtype=torch.bfloat16)
    original = source.clone()
    with use(True, capacity=8) as state:
        mask = torch.zeros_like(source, dtype=torch.bool)
        mask.reshape(-1)[0] = True
        temporary = build(source, mask)
        temporary.values[0] = 3.0
        result = finalize((packet((temporary,)),), state)[0]
    assert torch.equal(source, original)
    assert result.transport_patches[0].values[0] == 3.0


def test_temporary_owners_must_retire_not_merely_disappear_from_canonical_packet():
    import gc

    expected = torch.zeros(1, 2, 5, 128, dtype=torch.bfloat16)
    with use(True, capacity=8) as state:
        temporary = build(expected, expected != 0)
        provisional = packet((temporary,))
        canonical = finalize((provisional,), state)
        with pytest.raises(RuntimeError, match="owners must retire"):
            state.check_retired()
        del temporary, provisional
        gc.collect()
        state.check_retired()
        assert type(canonical[0].transport_patches[0]) is SparseBitPatch


def test_original_finite_protected_and_close_flags_all_reach_private_commit():
    from mosaickv.evaluation.deferred_transport_checks import use as validation
    from mosaickv.evaluation.shared_source_format import assert_transport_close

    expected = torch.zeros(1, 2, 5, 128, dtype=torch.bfloat16)
    expected.reshape(-1)[0] = 2.0
    actual = torch.zeros_like(expected)
    with (
        use(True, capacity=8) as state,
        validation(True, coalesce_nonzero=True, full_predicate_span=True) as checks,
    ):
        temporary = transport_corrections((expected,), (actual,), torch.tensor([2]))[0]
        apply_transport_patch(temporary, actual, prompt_length=5)
        assert_transport_close(actual, expected)
        finalize((packet((temporary,)),), state)
    assert checks.closed
    assert checks.report()["counts"] == {"finite": 1, "protected": 1, "close": 1}
