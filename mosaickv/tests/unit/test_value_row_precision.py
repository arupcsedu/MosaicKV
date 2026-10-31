import pytest
import torch

from mosaickv.evaluation.head_local_precision import localize, refit_corrections
from mosaickv.evaluation.precision_corrections import fund_corrections
from mosaickv.evaluation.precision_tier import construct
from mosaickv.evaluation.value_row_precision import cpu_row_packet, upgrade_rows


def fixture(fraction=0.005):
    generator = torch.Generator().manual_seed(43)
    pair = tuple(torch.randn(1, 3, 141, 128, generator=generator).bfloat16() for _ in range(2))
    heads = tuple(tuple(range(h, h + 141)) for h in range(3))
    text = {40, 72, 99}
    union = tuple(sorted({j for head in heads for j, p in enumerate(head) if p in text}))
    fmt, _ = fund_corrections(
        construct(pair, bits=3, protected=union, encoded_exact=True), pair, fraction=fraction
    )
    packet, _ = refit_corrections(localize(fmt, pair, heads, text, policy="head3k2v_refit"), pair)
    demand = torch.rand(1, 3, 16, 143, generator=generator)
    return pair, heads, packet, demand


def test_value_rows_spend_no_more_bytes_and_cover_more_values_than_scalar_patches():
    pair, heads, packet, demand = fixture()
    revised, audit = upgrade_rows(packet, pair, demand, heads)
    again, _ = upgrade_rows(packet, pair, demand, heads)
    assert revised.component_bytes <= packet.component_bytes
    assert audit["actual_row_upgrade_bytes"] <= audit["old_scalar_exception_bytes"]
    assert audit["scalar_values_covered"] > audit["previous_scalar_values"]
    assert audit["next_prefix_bytes"] > audit["old_scalar_exception_bytes"]
    assert torch.equal(revised.positions, again.positions)
    actual = revised.materialize()
    original = packet.materialize()
    assert torch.equal(actual[0].view(torch.int16), original[0].view(torch.int16))
    flat = actual[1].reshape(1, 1, 3 * 141, 128)
    reference = pair[1].reshape_as(flat)
    assert torch.equal(
        flat[..., revised.positions.long(), :].view(torch.int16),
        reference[..., revised.positions.long(), :].view(torch.int16),
    )
    oracle = cpu_row_packet(revised).materialize()
    assert all(
        torch.equal(a.view(torch.int16), b.view(torch.int16))
        for a, b in zip(actual, oracle, strict=True)
    )


def test_zero_budget_and_strided_history_are_safe():
    pair, heads, packet, demand = fixture(fraction=0)
    revised, audit = upgrade_rows(packet, pair, demand, heads)
    assert revised.values is None and audit["exact_value_rows"] == 0
    targets = tuple(torch.full((1, 3, 151, 128), -7, dtype=torch.bfloat16) for _ in range(2))
    revised.fill(None, targets, direct=True)
    assert all(bool((t[..., 141:, :] == -7).all()) for t in targets)


def test_negative_or_missing_query_evidence_is_rejected():
    pair, heads, packet, demand = fixture()
    demand[0, 0, 0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        upgrade_rows(packet, pair, demand, heads)


def test_all_true_filter_is_identical_and_query_demand_remains_individual():
    pair, heads, packet, demand = fixture()
    unfiltered, _ = upgrade_rows(packet, pair, demand, heads)
    filtered, audit = upgrade_rows(
        packet, pair, demand, heads, query_mask=torch.ones(16, dtype=torch.bool)
    )
    assert torch.equal(unfiltered.positions, filtered.positions)
    assert audit["prompt_queries_considered"] == 16
    # Deliberately opposing visual/text queries; the filter, not averaging,
    # decides which original value row receives the funded exact upgrade.
    demand.zero_()
    demand[..., :8, 45] = 1
    demand[..., 8:, 88] = 1
    text, _ = upgrade_rows(packet, pair, demand, heads, query_mask=torch.arange(16) >= 8)
    visual, _ = upgrade_rows(packet, pair, demand, heads, query_mask=torch.arange(16) < 8)
    assert set((text.positions.long() % 141).tolist()) == {88, 87, 86}
    assert set((visual.positions.long() % 141).tolist()) == {45, 44, 43}


@pytest.mark.parametrize(
    "mask", [torch.zeros(16, dtype=torch.bool), torch.ones(15, dtype=torch.bool), torch.ones(16)]
)
def test_invalid_query_masks_fail_closed(mask):
    pair, heads, packet, demand = fixture()
    with pytest.raises(ValueError, match="nonempty boolean query mask"):
        upgrade_rows(packet, pair, demand, heads, query_mask=mask)
