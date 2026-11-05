"""CPU arithmetic/provenance fixtures, not GPU performance measurements."""

from dataclasses import replace

import pytest
import torch
from test_lt_source_format import fixture

from mosaickv.evaluation import selective_text_construction as selective


@pytest.mark.parametrize(
    "positions,n,expected",
    [
        ((0, 1, 577, 599), 600, ((0, 16), (576, 24))),
        ((0, 1, 577, 603), 604, ((0, 16), (576, 28))),
        ((15, 16), 600, ((0, 32),)),
        ((599,), 600, ((592, 8),)),
    ],
)
def test_windows_retain_original_mma_row_alignment(positions, n, expected):
    windows = selective.aligned_windows(positions, n, 16)
    assert windows == expected
    assert all(start % 16 == 0 for start, _ in windows)
    assert all(any(start <= p < start + take for start, take in windows) for p in positions)


@pytest.mark.parametrize(
    "positions,n,tile",
    [((), 600, 16), ((-1,), 600, 16), ((600,), 600, 16), ((1.5,), 600, 16), ((1,), 600, 8)],
)
def test_bad_windows_fail_before_projection(positions, n, tile):
    with pytest.raises(ValueError):
        selective.aligned_windows(positions, n, tile)


@pytest.mark.parametrize("text", [(0,), (8,), tuple(range(9))])
def test_stripping_existing_protected_values_matches_full_materialization(text):
    from mosaickv.evaluation.text_source_storage import strip_text_overlays

    attention, base = fixture()
    positions = (tuple(range(9)),) * 2
    reference = base.materialize(attention)
    old = strip_text_overlays(base, reference, positions, frozenset(text))
    cpu = replace(base, row_positions_cpu=base.row_positions.cpu().clone())
    with selective.use(True) as state:
        new = selective.strip_exact(cpu, frozenset(text))
        assert state["cpu_metadata_hits"] == 1
    assert all(
        torch.equal(x.view(torch.uint8), y.view(torch.uint8))
        for x, y in zip(old.tensors(), new.tensors(), strict=True)
    )
    assert torch.equal(new.row_positions.cpu(), new.row_positions_cpu)


def test_partial_pair_never_reads_unwritten_visual_values(monkeypatch):
    from test_full_prompt_source import example

    from mosaickv.evaluation.direct_text_construction import build

    attention, base = example(29)
    base = replace(base, protected_patches_disjoint=True)
    h, n, _d = base.shape[1:]
    protected = set(base.row_positions.tolist())
    p = next(p for p in range(n) if any(head * n + p not in protected for head in range(h)))
    pair = base.materialize(attention)
    positions = (tuple(range(n)),) * h
    old = build(base, pair, positions, (p,), ("resident", "resident"), attention=attention)
    poisoned = tuple(torch.full_like(t, float("nan")) for t in pair)
    for a, b in zip(poisoned, pair, strict=True):
        a[:, :, p, :] = b[:, :, p, :]
    monkeypatch.setattr(selective, "text_pair", lambda *a: poisoned)
    with selective.use(True):
        new = build(
            replace(base, row_positions_cpu=base.row_positions.cpu().clone()),
            pair,
            positions,
            (p,),
            ("resident", "resident"),
            attention=attention,
        )
    assert all(
        torch.equal(x.view(torch.uint8), y.view(torch.uint8))
        for x, y in zip(old.tensors(), new.tensors(), strict=True)
    )


def test_context_is_private_exception_safe_and_rejects_invalid_values():
    assert not selective.enabled()
    with pytest.raises(RuntimeError), selective.use(True):
        assert selective.enabled()
        with pytest.raises(ValueError), selective.use(True):
            pass
        raise RuntimeError("abort")
    assert not selective.enabled()
    with pytest.raises(ValueError), selective.use(1):
        pass


def test_cpu_index_metadata_is_accounted_separately_and_never_contains_KV_values():
    from mosaickv.evaluation.cached_text_reader import CachedTextFormat

    _, base = fixture()
    assert selective.cpu_metadata_bytes((base,)) == 0
    cpu = base.row_positions.cpu().clone()
    value = replace(base, row_positions_cpu=cpu)
    wrapped = CachedTextFormat(value, torch.empty(0, dtype=torch.int32), ())
    assert selective.cpu_metadata_bytes((value, wrapped)) == cpu.untyped_storage().nbytes()
    assert all(a is b for a, b in zip(base.tensors(), value.tensors(), strict=True))
    with pytest.raises(ValueError):
        selective.source_rows(replace(base, row_positions_cpu=cpu.flip(0)))


def test_cpu_index_creation_preserves_all_original_cache_tensor_words():
    from test_full_prompt_source import example

    _attention, old = example(29)
    assert old.row_positions_cpu is None
    with selective.use(True):
        _, new = example(29)
    assert new.row_positions_cpu.dtype == torch.int32
    assert torch.equal(new.row_positions_cpu, new.row_positions.cpu())
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip(old.tensors(), new.tensors(), strict=True)
    )


def test_selective_source_cannot_bypass_original_identity_position_rule():
    from mosaickv.evaluation.direct_text_construction import build

    attention, base = fixture()
    with selective.use(True), pytest.raises(ValueError, match="identity positions"):
        build(
            base,
            base.materialize(attention),
            (tuple(reversed(range(9))),) * 2,
            (0,),
            ("resident", "resident"),
            attention=attention,
        )


@pytest.mark.parametrize("value", [True, 1])
def test_missing_frozen_text_policy_or_nonboolean_fails_before_model(value):
    from mosaickv.evaluation.common_streaming_timing import request

    with pytest.raises(ValueError, match="selective text"):
        request(None, None, None, "mosaickv_exact", selective_text_rows=value)
