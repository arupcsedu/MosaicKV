"""Opt-in layer-owner correctness; CPU-only, not CUDA/quality evidence."""

import hashlib

import pytest
import torch

from mosaickv.evaluation.bounded_component_attention import ComponentWorkspace
from mosaickv.evaluation.bounded_ragged_raw import (
    PackedRaggedRawPacket,
    RaggedRawPacket,
    build_raw_packet_layer,
    packed_selftest,
)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def packet_parts(counts=(3, 2)):
    owner = torch.arange(2 * sum(counts) * 128, dtype=torch.int16).view(torch.bfloat16)
    pairs, maps, cursor = [], [], 0
    for count in counts:
        elements = count * 128
        pairs.append(
            (
                owner[cursor : cursor + elements].view(count, 128),
                owner[cursor + elements : cursor + 2 * elements].view(count, 128),
            )
        )
        maps.append(torch.tensor((0, *range(2, count + 1)), dtype=torch.int64))
        cursor += 2 * elements
    return owner, tuple(pairs), tuple(maps)


@pytest.mark.parametrize("mode", ("random", "ties", "zeros"))
@pytest.mark.parametrize(
    "heads,length,layers,lower,remainder",
    ((1, 7, 3, 3, 0), (3, 7, 2, 3, 1), (3, 7, 2, 3, 5), (4, 13, 3, 6, 7), (32, 600, 32, 300, 600)),
)
def test_packed_matches_per_head_words_positions_bytes_and_remainder(
    mode, heads, length, layers, lower, remainder
):
    torch.manual_seed(37)
    # Include arbitrary BF16 encodings: signed zeros/NaN payloads are still
    # copied bitwise, not numerically approximated.
    original = tuple(
        torch.randint(-32768, 32767, (1, heads, length, 128), dtype=torch.int16).view(
            torch.bfloat16
        )
        for _ in range(2)
    )
    before = tuple(t.view(torch.int16).clone() for t in original)
    scores = (
        torch.randn(heads, length)
        if mode == "random"
        else torch.randint(0, 3, (heads, length)).float()
        if mode == "ties"
        else torch.zeros(heads, length)
    )
    target = (layers * heads * lower + remainder) * 512
    workspace = ComponentWorkspace(device="cpu", heads=heads)
    for layer in sorted({0, layers // 2, layers - 1}):
        kwargs = dict(
            layer_index=layer,
            total_layers=layers,
            target_bytes=target,
            workspace=workspace,
        )
        legacy, old_report = build_raw_packet_layer(original, scores, (0, length - 1), **kwargs)
        observations = []
        packed, report = build_raw_packet_layer(
            original,
            scores,
            (0, length - 1),
            packed_layer=True,
            workspace_observer=lambda tensors, observations=observations: observations.append(
                tuple(tensors)
            ),
            **kwargs,
        )
        assert type(legacy) is RaggedRawPacket
        assert type(packed) is PackedRaggedRawPacket
        assert len(packed.tensors()) == 1
        assert packed.tensors()[0] is packed.owner
        assert packed.component_bytes == legacy.component_bytes == report["actual_raw_KV_bytes"]
        assert packed.owner.untyped_storage().nbytes() == packed.owner.numel() * 2
        assert packed.prompt_lengths == legacy.prompt_lengths
        assert report["prompt_lengths"] == old_report["prompt_lengths"]
        assert report["position_map_sha256"] == old_report["position_map_sha256"]
        assert report["persistent_padding_bytes"] == 0
        assert report["constructor_KV_gather_temporary_allocations"] == 0
        assert report["allocation_layout"] == "joint_KV_layer_ragged_head_exact_cover"
        assert (
            report["kind"]
            == "original_BF16_streaming_layer_packed_exact_selection_transport_control"
        )
        assert len(observations) == 1 + 2 * heads
        assert all(any(t is packed.owner for t in row) for row in observations)
        cursor = 0
        for h, (new_pair, old_pair, positions) in enumerate(
            zip(packed.pair_by_head, legacy.pair_by_head, packed.original_positions, strict=True)
        ):
            count = lower + (layer * heads + h < remainder)
            assert packed.prompt_lengths[h] == count
            assert torch.equal(positions, legacy.original_positions[h])
            assert {0, length - 1}.issubset(positions.tolist())
            for component, (new, old) in enumerate(zip(new_pair, old_pair, strict=True)):
                assert new.storage_offset() == cursor + component * count * 128
                assert new.untyped_storage()._cdata == packed.owner.untyped_storage()._cdata
                assert torch.equal(new.view(torch.int16), old.view(torch.int16))
            cursor += 2 * count * 128
        assert cursor == packed.owner.numel()
        assert (
            report["position_map_sha256"]
            == hashlib.sha256(
                b"".join(p.numpy().tobytes() for p in packed.original_positions)
            ).hexdigest()
        )
        for component in (0, 1):
            actual = torch.empty(packed.shape, dtype=torch.bfloat16)
            expected = torch.empty(legacy.shape, dtype=torch.bfloat16)
            packed.fill_component(None, component, actual)
            legacy.fill_component(None, component, expected)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        for start, stop in ((0, 1), (max(0, heads - 2), heads)):
            actual = torch.empty((1, stop - start, packed.shape[2], 128), dtype=torch.bfloat16)
            expected = torch.empty_like(actual)
            packed.fill_group(None, 1, start, stop, actual)
            legacy.fill_group(None, 1, start, stop, expected)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert all(
        torch.equal(t.view(torch.int16), old) for t, old in zip(original, before, strict=True)
    )


def test_legacy_guard_still_rejects_shared_cross_head_storage():
    owner, pairs, positions = packet_parts()
    with pytest.raises(ValueError, match="exact-cover"):
        RaggedRawPacket(pairs, positions, original_length=7, prompt_capacity=3)
    packet = PackedRaggedRawPacket(owner, pairs, positions, original_length=7, prompt_capacity=3)
    assert packet.component_bytes == 5 * 512


@pytest.mark.parametrize(
    "case", ("tail", "offset", "overlap", "swapped", "foreign", "strided", "dtype", "nonflat")
)
def test_malformed_owner_or_component_layout_fails_closed(case):
    owner, pairs, positions = packet_parts((3, 3))
    if case == "tail":
        owner = torch.cat((owner, torch.zeros(128, dtype=torch.bfloat16)))
        pairs = tuple(
            tuple(
                owner[t.storage_offset() : t.storage_offset() + t.numel()].view(t.shape)
                for t in pair
            )
            for pair in pairs
        )
    elif case == "offset":
        larger = torch.empty(owner.numel() + 1, dtype=torch.bfloat16)
        owner = larger[1:]
    elif case == "overlap":
        pairs = (pairs[0], pairs[0])
    elif case == "swapped":
        pairs = (tuple(reversed(pairs[0])), pairs[1])
    elif case == "foreign":
        pairs = (pairs[0], (pairs[1][0].clone(), pairs[1][1]))
    elif case == "strided":
        pairs = ((pairs[0][0].t().contiguous().t(), pairs[0][1]), pairs[1])
    elif case == "dtype":
        owner = owner.float()
    elif case == "nonflat":
        owner = owner.view(2, -1)
    with pytest.raises(ValueError):
        PackedRaggedRawPacket(owner, pairs, positions, original_length=7, prompt_capacity=3)


@pytest.mark.parametrize(
    "case", ("negative", "upper", "duplicate", "descending", "float", "missing", "strided")
)
def test_position_guards_and_mandatory_preservation(case):
    owner, pairs, positions = packet_parts((3, 3))
    bad = positions[0]
    if case == "negative":
        bad = torch.tensor([-1, 2, 3])
    elif case == "upper":
        bad = torch.tensor([0, 2, 7])
    elif case == "duplicate":
        bad = torch.tensor([0, 2, 2])
    elif case == "descending":
        bad = torch.tensor([0, 3, 2])
    elif case == "float":
        bad = bad.float()
    elif case == "missing":
        bad = torch.tensor([1, 2, 3])
    elif case == "strided":
        bad = torch.tensor([0, 99, 2, 99, 3, 99])[::2]
    with pytest.raises(ValueError):
        PackedRaggedRawPacket(
            owner, pairs, (bad, positions[1]), original_length=7, prompt_capacity=3, mandatory=(0,)
        )


@pytest.mark.parametrize("mandatory", ((2, 0), (0, 0), (-1,), (7,), (True,), [0]))
def test_invalid_mandatory_metadata(mandatory):
    owner, pairs, positions = packet_parts()
    with pytest.raises(ValueError, match="mandatory"):
        PackedRaggedRawPacket(
            owner, pairs, positions, original_length=7, prompt_capacity=3, mandatory=mandatory
        )


def test_owner_workspace_alias_and_independently_wrapped_alias_rejected():
    owner, pairs, positions = packet_parts((3, 3))
    packet = PackedRaggedRawPacket(owner, pairs, positions, original_length=7, prompt_capacity=3)
    direct = owner[: 2 * 3 * 128].view(packet.shape)
    # A separately wrapped CPU storage has a distinct _cdata but the same
    # address interval. Checking storage identity alone would miss this alias.
    wrapped = torch.frombuffer(memoryview(owner.view(torch.int16).numpy()), dtype=torch.bfloat16)
    assert wrapped.untyped_storage()._cdata != owner.untyped_storage()._cdata
    indirect = wrapped[: 2 * 3 * 128].view(packet.shape)
    for target in (direct, indirect):
        with pytest.raises(ValueError, match="must not alias"):
            packet.fill_component(None, 0, target)


@pytest.mark.parametrize("value", (1, 0, None, "true"))
def test_builder_layout_opt_in_requires_boolean(value):
    original = tuple(torch.zeros(1, 2, 7, 128, dtype=torch.bfloat16) for _ in range(2))
    with pytest.raises(ValueError, match="explicit Boolean"):
        build_raw_packet_layer(
            original,
            torch.ones(2, 7),
            (0,),
            layer_index=0,
            total_layers=2,
            target_bytes=4 * 3 * 512,
            workspace=ComponentWorkspace(device="cpu", heads=2),
            packed_layer=value,
        )


@pytest.mark.parametrize("target", (511, 512, 4 * 8 * 512, 0, True))
def test_infeasible_global_budget_still_rejected(target):
    original = tuple(torch.zeros(1, 2, 7, 128, dtype=torch.bfloat16) for _ in range(2))
    with pytest.raises(ValueError):
        build_raw_packet_layer(
            original,
            torch.ones(2, 7),
            (0, 6),
            layer_index=0,
            total_layers=2,
            target_bytes=target,
            workspace=ComponentWorkspace(device="cpu", heads=2),
            packed_layer=True,
        )


def test_packed_operator_selftest_is_cpu_executable_without_weights():
    report = packed_selftest(device="cpu")
    assert report["failures"] == []
    assert report["measured_allocator_or_quality_claim"] is False
    assert len(report["cases"]) == 3
    for case in report["cases"]:
        assert case["word_parity"] is True
        assert case["packed_allocation_owners"] == 1
        assert case["legacy_allocation_owners"] == 32
        assert case["persistent_padding_bytes"] == 0
        assert case["constructor_observations"] == 65
