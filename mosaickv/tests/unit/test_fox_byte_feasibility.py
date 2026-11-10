"""CPU-only exact byte arithmetic and conservative ownership checks."""

from itertools import product
from random import Random

import evicastkv_review_byte_feasibility as study
import pytest


def test_reachability_matches_exhaustive_small_instances():
    rng = Random(917)
    for _ in range(30):
        costs = [[rng.randrange(1, 30) for _ in range(3)] for _ in range(5)]
        targets = range(1, 170)
        reference = {
            11 + sum(costs[layer][i] for layer, i in enumerate(choices))
            for choices in product(range(3), repeat=5)
        }
        actual = study.reachability(costs, 11, targets)
        assert set(actual) == reference
        for size, witness in actual.items():
            assert 11 + sum(row[i] for row, i in zip(costs, witness, strict=True)) == size


def test_gcd_one_does_not_fill_gaps():
    assert study.reachability([[10, 12], [10, 13]], 0, [20, 21, 22, 23, 24, 25]) == {
        20: [0, 0],
        22: [1, 0],
        23: [0, 1],
        25: [1, 1],
    }


def test_bounded_search_rejects_massive_span():
    with pytest.raises(ValueError, match="resource bound"):
        study.reachability([[1, 1 + study.MAX_SPAN + 1]], 0, [1])


def packet(shared_paths=None):
    def owner(size, paths):
        return dict(
            bytes=size,
            paths=paths,
            category="test",
            device="cuda:0",
            dtype="torch.uint8",
            observed_view_shape=[size],
        )

    return {
        "components": [
            owner(10, ["layer[0].codes"]),
            owner(20, ["layer[1].codes"]),
            owner(3, shared_paths or ["layer[0].cos", "layer[1].cos"]),
        ],
        "representation_storage_bytes": 33,
        "category_bytes": {"test": 33},
        "includes_codes_scales_indices_rotary_exact_overlays_and_numerical_corrections": True,
        "CPU_residual_bytes": 0,
        "generated_history": "not_yet_allocated_reported_separately_after_decode",
    }


def test_shared_owner_counted_once_and_unknown_coupling_rejected():
    local, shared = study.inventory(packet(), 2)
    assert local == [10, 20]
    assert shared["cos"][0] == 3
    with pytest.raises(ValueError, match="unmodeled"):
        study.inventory(packet(["layer[0].cos", "layer[1].sin"]), 2)
    with pytest.raises(ValueError, match="multiply charged"):
        study.inventory(packet(["layer[0].codes", "layer[1].cos"]), 2)


def test_mixing_refuses_changed_shared_owner_layout():
    records = {key: packet() for key in study.CHOICES}
    records[study.CHOICES[1]]["components"][2]["observed_view_shape"] = [1, 3]
    with pytest.raises(ValueError, match="shared owner layout"):
        study.arm_grid({"physical_allocation": {"candidate_inventories": records}}, 2, [33])


def test_corrupted_component_sum_rejected():
    altered = packet()
    altered["representation_storage_bytes"] += 1
    with pytest.raises(ValueError, match="does not reconcile"):
        study.inventory(altered, 2)


def test_changed_parent_record_rejected(tmp_path):
    path = tmp_path / "record.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="immutable evidence"):
        study.read_bound(path, "0" * 64)
