"""CPU semantics/operation-count controls; no CUDA timing or quality claim."""

import hashlib
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from mosaickv.evaluation import bounded_ragged_raw as raw
from mosaickv.evaluation.bounded_component_attention import ComponentWorkspace
from mosaickv.evaluation.request_precision import ranked_positions


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class OperationCounts(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("mode", ("random", "ties", "zeros"))
@pytest.mark.parametrize("layer", (0, 17, 18, 19, 31))
def test_120_frozen_shape_rankings_match_old_individual_heads(seed, mode, layer):
    torch.manual_seed(seed)
    if mode == "random":
        scores = torch.randn(32, 600)
    elif mode == "ties":
        scores = torch.randint(0, 4, (32, 600)).float()
    else:
        scores = torch.zeros(32, 600)
    before = scores.clone()
    expected = tuple(
        ranked_positions(scores[head : head + 1], 300 + (layer * 32 + head < 600), (0, 599))[0]
        for head in range(32)
    )
    actual = raw._grouped_cpu_positions(scores, 300, 600, layer_index=layer, mandatory=(0, 599))
    assert actual == expected
    assert torch.equal(scores, before)
    assert all({0, 599}.issubset(row) for row in actual)


@pytest.mark.parametrize(
    "heads,length,layers,lower,remainder",
    [(1, 7, 3, 3, 0), (3, 7, 2, 3, 1), (3, 7, 2, 3, 5), (4, 13, 3, 6, 7), (2, 7, 2, 6, 3)],
)
def test_small_and_boundary_geometries_preserve_exact_order(
    heads, length, layers, lower, remainder
):
    scores = torch.arange(length).float().repeat(heads, 1)
    for layer in range(layers):
        actual = raw._grouped_cpu_positions(
            scores, lower, remainder, layer_index=layer, mandatory=(0,)
        )
        expected = tuple(
            ranked_positions(scores[h : h + 1], lower + int(layer * heads + h < remainder), (0,))[0]
            for h in range(heads)
        )
        assert actual == expected


@pytest.mark.parametrize("layer,expected_calls", [(0, 1), (18, 2), (31, 1)])
def test_at_most_two_ranking_calls_and_fewer_real_operations(monkeypatch, layer, expected_calls):
    scores = torch.ones(32, 600)
    ranking = Mock(wraps=ranked_positions)
    monkeypatch.setattr(raw, "ranked_positions", ranking)
    grouped = OperationCounts()
    with grouped:
        raw._grouped_cpu_positions(scores, 300, 600, layer_index=layer, mandatory=(0, 599))
    assert ranking.call_count == expected_calls
    assert sum(grouped.counts.values()) <= 32
    individual = OperationCounts()
    with individual:
        tuple(
            ranked_positions(scores[h : h + 1], 300 + int(layer * 32 + h < 600), (0, 599))[0]
            for h in range(32)
        )
    assert sum(grouped.counts.values()) < sum(individual.counts.values())


@pytest.mark.parametrize("mode", ("random", "ties", "zeros"))
def test_complete_streamed_raw_layer_words_bytes_maps_and_hashes_match_old_builder(mode):
    torch.manual_seed(31)
    layers, heads, length = 3, 4, 13
    original = tuple(
        tuple(torch.randn(1, heads, length, 128).bfloat16() for _ in range(2))
        for _ in range(layers)
    )
    scores = (
        torch.randn(heads, length)
        if mode == "random"
        else torch.randint(0, 3, (heads, length)).float()
        if mode == "ties"
        else torch.zeros(heads, length)
    )
    adapter = SimpleNamespace(_language_layers=lambda: tuple(range(layers)))
    target = (layers * heads * 6 + 7) * 512
    old, _ = raw.build_raw_packets(
        adapter, original, {i: scores for i in range(layers)}, target, (0, 12)
    )
    workspace = ComponentWorkspace(device="cpu", heads=heads)
    for layer in range(layers):
        observed = []
        packet, report = raw.build_raw_packet_layer(
            original[layer],
            scores,
            (0, 12),
            layer_index=layer,
            total_layers=layers,
            target_bytes=target,
            workspace=workspace,
            workspace_observer=lambda tensors, observed=observed: observed.append(len(tensors)),
        )
        reference = old[layer]
        assert packet.component_bytes == reference.component_bytes == report["actual_raw_KV_bytes"]
        assert packet.prompt_lengths == reference.prompt_lengths
        assert all(
            torch.equal(a, b)
            for a, b in zip(packet.original_positions, reference.original_positions, strict=True)
        )
        assert all(
            torch.equal(a.view(torch.int16), b.view(torch.int16))
            for a, b in zip(packet.tensors(), reference.tensors(), strict=True)
        )
        assert report["persistent_padding_bytes"] == 0
        assert report["allocation_layout"] == "joint_KV_per_head_2ND_exact_cover"
        assert report["original_positions"] == [p.tolist() for p in reference.original_positions]
        assert (
            report["position_map_sha256"]
            == hashlib.sha256(
                b"".join(p.numpy().tobytes() for p in reference.original_positions)
            ).hexdigest()
        )
        assert report["CPU_ranking_groups"] in (1, 2)
        assert len(observed) == 1 + 2 * heads


def packet_pair(count=3):
    owner = torch.zeros(2, count, 128, dtype=torch.bfloat16)
    return owner[0], owner[1]


@pytest.mark.parametrize(
    "case",
    (
        "negative",
        "upper",
        "duplicate",
        "descending",
        "float",
        "bool",
        "shape",
        "strided",
        "meta",
        "not_tensor",
    ),
)
def test_exact_cpu_metadata_guards_remain_fail_closed(case):
    positions = torch.tensor([0, 1, 4], dtype=torch.int64)
    if case == "negative":
        positions = torch.tensor([-1, 1, 4])
    elif case == "upper":
        positions = torch.tensor([0, 1, 5])
    elif case == "duplicate":
        positions = torch.tensor([0, 1, 1])
    elif case == "descending":
        positions = torch.tensor([0, 4, 1])
    elif case == "float":
        positions = positions.float()
    elif case == "bool":
        positions = positions.bool()
    elif case == "shape":
        positions = positions[:, None]
    elif case == "strided":
        positions = torch.tensor([0, 99, 1, 99, 4, 99])[::2]
    elif case == "meta":
        positions = torch.empty(3, dtype=torch.int64, device="meta")
    elif case == "not_tensor":
        positions = [0, 1, 4]
    with pytest.raises(ValueError, match="sorted CPU positions"):
        raw.RaggedRawPacket((packet_pair(),), (positions,), original_length=5, prompt_capacity=3)


def test_valid_cpu_metadata_checks_create_no_transient_boolean_tensors():
    pairs = tuple(packet_pair() for _ in range(4))
    positions = tuple(torch.tensor([0, 1, 4]) for _ in range(4))
    operations = OperationCounts()
    with operations:
        packet = raw.RaggedRawPacket(pairs, positions, original_length=5, prompt_capacity=3)
    assert packet.original_positions == positions
    assert not any(
        "any" in name or "lt." in name or "ge." in name or "le." in name
        for name in operations.counts
    )


@pytest.mark.parametrize("change", ({"lower": 0}, {"remainder": -1}, {"layer_index": -1}))
def test_bad_group_capacity_metadata_fails_closed(change):
    kwargs = dict(lower=2, remainder=1, layer_index=0, mandatory=(0,))
    kwargs.update(change)
    with pytest.raises(ValueError, match="balanced-capacity"):
        raw._grouped_cpu_positions(torch.ones(2, 7), **kwargs)


def test_nonfinite_scores_still_fail_before_ranked_selection():
    scores = torch.ones(2, 7)
    scores[1, 4] = float("nan")
    with pytest.raises(ValueError, match="invalid scores"):
        raw._grouped_cpu_positions(scores, 3, 1, layer_index=0, mandatory=(0,))
