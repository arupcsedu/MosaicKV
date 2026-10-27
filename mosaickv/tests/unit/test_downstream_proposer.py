"""Exact-capacity directional proposer and unchanged descriptive-screen fixtures."""

import pytest
import torch
from test_draft_head_substitution import synthetic

from mosaickv.evaluation.downstream_head_diagnostic import downstream_screen
from mosaickv.selection.draft_head_substitution import draft_substitution_pool
from mosaickv.selection.head_substitution import pack_substitution


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_downstream_directional_pairs_match_exhaustive_and_preserve_mandatory_storage():
    full, anchor, evidence = synthetic()
    rng = torch.Generator().manual_seed(91)
    gradients = tuple(torch.randn(row["target"].shape, generator=rng) for row in evidence)
    pool, trace = draft_substitution_pool(
        full, anchor, evidence, limit=16, output_gradients=gradients
    )
    expected = []
    for i, source in enumerate(full.layers):
        for h in range(2):

            def score(positions, i=i, h=h, source=source):
                data = evidence[i]
                p = data["prompt_probabilities"][h, :, positions]
                num = p @ source.value[0, h, positions] + data["generated_numerator"][h]
                den = p.sum(-1) + data["generated_mass"][h]
                return float((num / den[:, None] * gradients[i][h]).mean())

            before = score([0, 2, 5])
            after, d, a = min(
                (score(sorted(({0, 2, 5} - {d}) | {a})), d, a) for d in (0, 2) for a in (1, 3, 4)
            )
            if after < before:
                expected.append((before - after, i, h, d, a))
    expected.sort(key=lambda x: (-x[0], *x[1:]))
    assert pool and [(p.layer, p.head, p.removed_position, p.added_position) for p in pool] == [
        (i, h, d, a) for _, i, h, d, a in expected
    ]
    for proposal, record, (gain, *_rest) in zip(pool, trace, expected, strict=True):
        assert record["predicted_gain"] == pytest.approx(gain, rel=1e-5, abs=1e-6)
        assert record["ranking_objective"] == "frozen_downstream_directional_KL"
        assert proposal.removed_position != 5
        packed = pack_substitution(full, anchor, proposal)
        assert sum(
            x.key.numel() * x.key.element_size() + x.value.numel() * x.value.element_size()
            for x in packed.snapshot.layers
        ) == sum(
            x.key.numel() * x.key.element_size() + x.value.numel() * x.value.element_size()
            for x in anchor.snapshot.layers
        )
    assert (pool, trace) == draft_substitution_pool(
        full, anchor, evidence, limit=16, output_gradients=gradients
    )
    opposite, _ = draft_substitution_pool(
        full, anchor, evidence, limit=16, output_gradients=tuple(-g for g in gradients)
    )
    assert opposite != pool


def test_zero_or_bad_sensitivity_cannot_create_a_promoted_pool():
    full, anchor, evidence = synthetic()
    zero = tuple(torch.zeros_like(row["target"]) for row in evidence)
    assert draft_substitution_pool(full, anchor, evidence, limit=16, output_gradients=zero) == (
        (),
        [],
    )
    with pytest.raises(ValueError, match="cover every"):
        draft_substitution_pool(full, anchor, evidence, limit=16, output_gradients=zero[:-1])
    with pytest.raises(ValueError, match="nonfinite"):
        draft_substitution_pool(
            full, anchor, evidence, limit=16, output_gradients=tuple(x + float("nan") for x in zero)
        )
    with pytest.raises(TypeError):
        draft_substitution_pool(
            full, anchor, evidence, limit=16, benchmark_references=["forbidden"]
        )


def test_screen_requires_directional_gain_beyond_reconstruction_and_shared_prefix():
    rows = [
        dict(
            prefix=1,
            predicted_gain=i + 1,
            reconstruction_gain=16 - i,
            actual_KL_gain=(i + 1) * 1e-4,
            anchor_KL=0.1,
        )
        for i in range(16)
    ]
    assert downstream_screen(rows)["passed"]
    assert not downstream_screen([dict(r, predicted_gain=-r["predicted_gain"]) for r in rows])[
        "passed"
    ]
    assert not downstream_screen([dict(r, reconstruction_gain=r["predicted_gain"]) for r in rows])[
        "passed"
    ]
    assert not downstream_screen([dict(rows[0], prefix=4)])["passed"]
