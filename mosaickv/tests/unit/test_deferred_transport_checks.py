import pytest
import torch

from mosaickv.evaluation.deferred_transport_checks import enqueue, use
from mosaickv.evaluation.shared_source_format import assert_transport_close, transport_corrections


def test_same_sparse_patch_words_and_bounded_flags_not_source_tensors():
    generator = torch.Generator().manual_seed(99)
    reference = torch.randn(1, 2, 17, 128, generator=generator).bfloat16()
    actual = reference.clone()
    actual.flatten()[::127] += 0.25
    protected = torch.empty(0, dtype=torch.int32)
    original = transport_corrections((reference,), (actual,), protected, chunk_scalars=128)[0]
    with use(True) as state:
        candidate = transport_corrections((reference,), (actual,), protected, chunk_scalars=128)[0]
        assert_transport_close(reference, reference.clone(), chunk_scalars=128)
        assert len(state.flags) == 3 and all(f.numel() == 1 for f in state.flags)
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(candidate.tensors, original.tensors, strict=True)
    )
    assert state.report()["commit_flushes"] == 1 and not state.flags
    assert state.peak_bytes == 13


@pytest.mark.parametrize(
    "kind,error", [("finite", ValueError), ("protected", RuntimeError), ("close", AssertionError)]
)
def test_failed_predicate_aborts_before_commit_and_resets_state(kind, error):
    committed = False
    with pytest.raises(error):
        with use(True):
            enqueue(torch.tensor(False), kind)
        committed = True
    assert not committed
    assert not enqueue(torch.tensor(True), "close")
    with use(True):
        assert enqueue(torch.tensor(True), "close")


def test_real_failures_preserve_predicates_and_error_categories():
    x = torch.ones(1, 1, 8, 128, dtype=torch.bfloat16)
    for mode in ("finite", "protected", "close"):
        y = x.clone()
        y.flatten()[0] = float("nan") if mode == "finite" else 2.0
        error = {"finite": ValueError, "protected": RuntimeError, "close": AssertionError}[mode]
        with pytest.raises(error), use(True):
            if mode == "close":
                assert_transport_close(y, x)
            else:
                transport_corrections((x,), (y,), torch.tensor([0], dtype=torch.int32))


def test_empty_disabled_nested_and_capacity_controls():
    with use(True) as state, pytest.raises(ValueError, match="nonnested"), use(True):
        pass
    assert state.report()["commit_flushes"] == 0
    with use(False):
        assert not enqueue(torch.tensor(True), "finite")
    with use(True):
        for _ in range(256):
            enqueue(torch.tensor(True), "close")
        with pytest.raises(RuntimeError, match="exhausted"):
            enqueue(torch.tensor(True), "close")
