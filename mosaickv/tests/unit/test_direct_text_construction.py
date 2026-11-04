from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import direct_text_construction as direct
from mosaickv.evaluation.fused_lossless_attention import bits_equal
from mosaickv.evaluation.head_local_precision import localize
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.precision_tier import construct


def example():
    torch.manual_seed(924)
    pair = tuple(torch.randn(1, 2, 17, 128).bfloat16() for _ in range(2))
    positions, text = (tuple(range(17)),) * 2, (1, 4, 16)
    packet = construct(pair, bits=3, protected=text, encoded_exact=True)
    packet = localize(packet, pair, positions, set(text), policy="head3")
    return packet, pair, positions, text


@pytest.mark.parametrize("modes", list(product(("encoded", "resident"), repeat=2)))
def test_direct_borrowing_preserves_parent_bits_memberships_and_history(modes):
    source, pair, positions, text = example()
    candidate = direct.build(source, pair, positions, text, modes)
    assert candidate.flat_tokens.tolist() == [1, 4, 16, 18, 21, 33]
    for original, value in zip(source.materialize(None), candidate.materialize(None), strict=True):
        assert bits_equal(original, value)
    targets = tuple(torch.full((1, 2, 22, 128), -71, dtype=torch.bfloat16) for _ in range(2))
    candidate.fill(None, targets, direct=True)
    assert all((t[..., 17:, :] == -71).all() for t in targets)


def test_unprotected_rows_cannot_be_borrowed_as_if_parent_exact():
    source, pair, positions, text = example()
    source = replace(source, flat_tokens=torch.empty(0, dtype=torch.int32), protected=())
    with pytest.raises(ValueError, match="already be parent-exact"):
        direct.build(source, pair, positions, text, ("resident", "resident"))


def test_scope_no_labels_count_exact_bytes_and_rollback(monkeypatch):
    source, pair, positions, text = example()
    built = direct.build(source, pair, positions, text, ("encoded", "resident"))
    size = backing_bytes(built.private())
    policy = dict(
        modes=[["encoded", "resident"]],
        maximum_added_bytes=0,
        base_private_bytes=size,
        expected_private_bytes=size,
    )
    with direct.use(policy) as state:
        value, _ = direct.optimize(source, {}, positions, text, None, reference_pair=pair)
        assert state.finish((value,))["bypassed_text_fits"] == 1
        with pytest.raises(RuntimeError, match="nonnested"), direct.use(policy):
            pass
        with pytest.raises(RuntimeError, match="more construction"):
            direct.optimize(source, {}, positions, text, None, reference_pair=pair)
    monkeypatch.setattr(
        "mosaickv.evaluation.text_source_storage.optimize", lambda *args: ("original", args)
    )
    result = direct.optimize(source, {}, positions, text, None, reference_pair=pair)
    assert result[0] == "original" and len(result[1]) == 5
    with pytest.raises(ValueError, match="labels forbidden"):
        direct.Construction(dict(policy, answer="hidden"))
    with direct.use(dict(policy, expected_private_bytes=size - 1)) as state:
        value, _ = direct.optimize(source, {}, positions, text, None, reference_pair=pair)
        with pytest.raises(RuntimeError, match="storage differs"):
            state.finish((value,))
    with direct.use(policy) as state, pytest.raises(RuntimeError, match="exactly one"):
        state.finish(())


def test_shared_source_keeps_exact_values_and_retiles_without_reencoding():
    from mosaickv.evaluation.shared_source_format import construct as source_construct
    from mosaickv.evaluation.source_retile import retile

    torch.manual_seed(42)
    x = torch.randn(1, 17, 256).bfloat16()
    attention = SimpleNamespace(
        k_proj=torch.nn.Linear(256, 256, bias=False).bfloat16(),
        v_proj=torch.nn.Linear(256, 256, bias=False).bfloat16(),
    )
    with torch.inference_mode():
        pair = tuple(
            p(x).reshape(1, 17, 2, 128).transpose(1, 2)
            for p in (attention.k_proj, attention.v_proj)
        )
    positions, text = (tuple(range(17)),) * 2, (0, 1, 16)
    identity = (
        torch.arange(17, dtype=torch.int32),
        torch.arange(17, dtype=torch.int16)[None].expand(2, 17),
    )
    phases = torch.ones(1, 17, 128).bfloat16(), torch.zeros(1, 17, 128).bfloat16()
    source = source_construct(x, pair, positions, text, phases, identity)
    value = direct.build(source, pair, positions, text, ("encoded", "resident"))
    moved = retile(value, 2)
    assert moved.base.source is value.base.source and moved.base.source_head_group == 2
    for old, new in zip(source.materialize(attention), value.materialize(attention), strict=True):
        assert bits_equal(old, new)
