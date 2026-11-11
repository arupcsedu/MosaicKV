"""Random/tiny-HF CPU transport fixtures; no pretrained or paper evidence."""

import hashlib
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

from mosaickv.adapters.huggingface.base import HuggingFaceMultimodalAdapter
from mosaickv.adapters.huggingface.types import (
    AdapterCapabilities,
    CachedKeyState,
    PreparedInputs,
    QueryVectorState,
)
from mosaickv.evaluation.bounded_ragged_raw import (
    RaggedRawPacket,
    build_raw_packets,
    native_same_selected_state,
    run_native_same_selected,
)
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(*, layers=2, heads=3, length=7):
    torch.manual_seed(25)
    pairs = tuple(
        tuple(torch.randn(1, heads, length, 128).bfloat16() for _ in range(2))
        for _ in range(layers)
    )
    scores = {i: torch.ones(1, heads, 2, length) for i in range(layers)}
    return SimpleNamespace(_language_layers=lambda: tuple(range(layers))), pairs, scores


def test_exact_512_byte_lattice_balances_independent_head_lengths_without_storage_padding():
    adapter, pairs, scores = fixture()
    packets, report = build_raw_packets(adapter, pairs, scores, 20 * 512, (0, 6))
    assert [p.prompt_lengths for p in packets] == [(4, 4, 3), (3, 3, 3)]
    assert {p.shape for p in packets} == {(1, 3, 4, 128)}
    assert sum(p.component_bytes for p in packets) == report["actual_raw_KV_bytes"] == 20 * 512
    assert report["persistent_padding_bytes"] == 0
    assert report["workspace_prompt_capacity"] == 4
    assert report["CPU_position_metadata_bytes"] == 20 * 8
    for layer, packet in enumerate(packets):
        for head, logical in enumerate(packet.original_positions):
            assert {0, 6}.issubset(logical.tolist())
            assert logical.device.type == "cpu"
            assert logical.tolist() == sorted(set(logical.tolist()))
            for component in (0, 1):
                assert torch.equal(
                    packet.pair_by_head[head][component].view(torch.int16),
                    pairs[layer][component][0, head, logical].view(torch.int16),
                )


def test_all_tied_scores_use_source_order_and_do_not_mutate_input_evidence():
    adapter, pairs, scores = fixture(layers=1, heads=2)
    before = scores[0].clone()
    packets, report = build_raw_packets(adapter, pairs, scores, 7 * 512, (6,))
    assert packets[0].original_positions[0].tolist() == [0, 1, 2, 6]
    assert packets[0].original_positions[1].tolist() == [0, 1, 6]
    assert torch.equal(scores[0], before)
    assert not report["benchmark_labels_or_reference_answers_used"]
    assert not report["FullKV_quality_preservation_claim"]


def test_fill_copies_actual_rows_and_zeros_only_the_shared_workspace_holes():
    adapter, pairs, scores = fixture(layers=1, heads=2)
    packets, _ = build_raw_packets(adapter, pairs, scores, 7 * 512, (6,))
    packet = packets[0]
    original_bytes = packet.component_bytes
    before = tuple(t.clone() for t in packet.tensors())
    workspace = torch.full(packet.shape, 99.0, dtype=torch.bfloat16)
    packet.fill_component(None, 0, workspace)
    for head, count in enumerate(packet.prompt_lengths):
        assert torch.equal(workspace[0, head, :count], packet.pair_by_head[head][0])
        assert torch.count_nonzero(workspace[0, head, count:]) == 0
    assert packet.component_bytes == original_bytes
    assert all(torch.equal(a, b) for a, b in zip(before, packet.tensors(), strict=True))
    group = torch.empty(1, 1, packet.shape[-2], 128, dtype=torch.bfloat16)
    packet.fill_group(None, 1, 1, 2, group)
    assert torch.equal(group[0, 0, :3], packet.pair_by_head[1][1])
    assert torch.count_nonzero(group[..., 3:, :]) == 0


@pytest.mark.parametrize(
    "target,mandatory", [(1, (0,)), (513, (0,)), (5 * 512, (0, 6)), (43 * 512, (0,)), (0, ())]
)
def test_non_lattice_over_budget_or_mandatory_infeasible_targets_fail_closed(target, mandatory):
    adapter, pairs, scores = fixture()
    with pytest.raises(ValueError, match="target"):
        build_raw_packets(adapter, pairs, scores, target, mandatory)


def test_malformed_mandatory_or_nonfinite_scores_fail_closed():
    adapter, pairs, scores = fixture()
    with pytest.raises(ValueError, match="sorted unique"):
        build_raw_packets(adapter, pairs, scores, 20 * 512, (6, 0))
    scores[0][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        build_raw_packets(adapter, pairs, scores, 20 * 512, (0,))


def test_padded_or_aliasing_persistent_views_are_not_valid_packets():
    data = torch.zeros(4, 128, dtype=torch.bfloat16)
    other = data.clone()
    logical = torch.tensor([0, 1, 2])
    with pytest.raises(ValueError, match="exact-size backing"):
        RaggedRawPacket(((data[:3], other[:3]),), (logical,), original_length=4, prompt_capacity=3)
    with pytest.raises(ValueError, match=r"not padded views|must not alias|disjoint exact-cover"):
        RaggedRawPacket(((data, data),), (torch.arange(4),), original_length=4, prompt_capacity=4)


def test_joint_heads_cover_one_exact_owner_and_remove_odd_component_rounding():
    adapter, pairs, scores = fixture(layers=1, heads=2)
    packets, report = build_raw_packets(adapter, pairs, scores, 7 * 512, (6,))
    packet = packets[0]
    assert packet.prompt_lengths == (4, 3)
    assert report["allocation_layout"] == "joint_KV_per_head_2ND_exact_cover"
    assert report["raw_KV_allocation_owner_count"] == 2
    assert report["constructor_KV_gather_temporary_allocations"] == 0
    assert packet.allocation_layout == report["allocation_layout"]
    for key, value in packet.pair_by_head:
        owner = key.untyped_storage()
        assert owner._cdata == value.untyped_storage()._cdata
        assert key.storage_offset() == 0
        assert value.storage_offset() == key.numel()
        assert key.is_contiguous() and value.is_contiguous()
        assert owner.nbytes() == 2 * key.numel() * key.element_size()
        assert owner.nbytes() % 512 == 0
    assert packet.component_bytes == 7 * 512


def test_joint_packing_has_identical_semantic_word_sha_and_positions_to_legacy_gathers():
    adapter, pairs, scores = fixture()
    packets, report = build_raw_packets(adapter, pairs, scores, 20 * 512, (0, 6))
    actual, expected = [], []
    for layer, packet in enumerate(packets):
        legacy_pairs = []
        for head, positions in enumerate(packet.original_positions):
            legacy = tuple(t[0, head].index_select(0, positions) for t in pairs[layer])
            legacy_pairs.append(legacy)
            actual.extend(packet.pair_by_head[head])
            expected.extend(legacy)
        legacy_packet = RaggedRawPacket(
            legacy_pairs, packet.original_positions, original_length=7, prompt_capacity=4
        )
        assert legacy_packet.allocation_layout == "separate_K_and_V_legacy_exact_cover"
        assert legacy_packet.component_bytes == packet.component_bytes

    def words_sha(tensors):
        return hashlib.sha256(
            b"".join(t.view(torch.int16).numpy().tobytes() for t in tensors)
        ).hexdigest()

    assert words_sha(actual) == words_sha(expected)
    assert (
        report["position_map_sha256"]
        == hashlib.sha256(
            b"".join(p.numpy().tobytes() for packet in packets for p in packet.original_positions)
        ).hexdigest()
    )


@pytest.mark.parametrize("case", ("overlap", "reversed", "unused_tail", "cross_head"))
def test_joint_exact_cover_rejects_overlap_reversal_tail_and_cross_head_sharing(case):
    owner = torch.zeros(2, 3, 128, dtype=torch.bfloat16)
    pairs = ((owner[0], owner[1]),)
    maps = (torch.arange(3),)
    if case == "overlap":
        pairs = ((owner[0], owner[0]),)
    elif case == "reversed":
        pairs = ((owner[1], owner[0]),)
    elif case == "unused_tail":
        owner = torch.zeros(2, 4, 128, dtype=torch.bfloat16)
        pairs = ((owner[0, :3], owner[1, :3]),)
    else:
        pairs, maps = pairs * 2, maps * 2
    with pytest.raises(ValueError, match=r"disjoint exact-cover|must not alias"):
        RaggedRawPacket(pairs, maps, original_length=7, prompt_capacity=3)


def test_joint_and_separate_exact_cover_heads_can_coexist_without_cross_head_aliases():
    owner = torch.zeros(2, 3, 128, dtype=torch.bfloat16)
    separate = tuple(torch.ones(3, 128, dtype=torch.bfloat16) for _ in range(2))
    packet = RaggedRawPacket(
        ((owner[0], owner[1]), separate),
        (torch.arange(3),) * 2,
        original_length=7,
        prompt_capacity=3,
    )
    assert packet.allocation_layout == "mixed_exact_cover_heads"
    assert packet.component_bytes == 6 * 512


class TinyAdapter(HuggingFaceMultimodalAdapter):
    capabilities = AdapterCapabilities(
        model_family="tiny_cpu_fixture",
        architectures=("LlamaForCausalLM",),
        attention_implementations=("eager",),
        image=False,
        multi_image=False,
        video=False,
        cache_classes=("DynamicCache",),
        cache_sequence_dimension=2,
        cached_key_state=CachedKeyState.POST_ROPE,
        query_vector_state=QueryVectorState.Q_PROJ_PRE_ROPE,
        supports_prototype_merge=False,
        supports_residual_repair=False,
    )

    def _language_layers(self):
        return self.model.model.layers

    def _image_token_id(self):
        return None

    def _processor_kwargs(self, prompt, images, videos):
        raise NotImplementedError("CPU fixture uses explicit token inputs")


def tiny_prefill():
    torch.manual_seed(25)
    model = (
        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=31,
                hidden_size=256,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                head_dim=128,
                max_position_embeddings=2048,
                attn_implementation="eager",
            )
        )
        .bfloat16()
        .eval()
    )
    adapter = TinyAdapter(model, None)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    prepared = PreparedInputs(dict(input_ids=tokens, attention_mask=torch.ones_like(tokens)), (), 7)
    prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=False)
    pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
    scores = {i: torch.ones(1, 2, 2, 7) for i in range(2)}
    return adapter, prefill, pairs, scores


def test_native_reference_uses_real_per_head_masks_and_original_next_rope_position():
    adapter, prefill, pairs, scores = tiny_prefill()
    packets, _ = build_raw_packets(adapter, pairs, scores, 9 * 512, (0,))
    originals = tuple(tuple(t.clone() for t in pair) for pair in pairs)
    state = native_same_selected_state(adapter, prefill, packets)
    masks = state.model_state["mosaickv_validity_masks"]
    assert [mask.sum(-1).tolist() for mask in masks] == [[3, 2], [2, 2]]
    assert state.active_cache_length == 3
    assert state.next_decode_position == state.logical_sequence_length == 7
    native = DynamicCache()
    for layer, packet in enumerate(packets):
        copied = tuple(torch.empty(packet.shape, dtype=torch.bfloat16) for _ in range(2))
        for component, target in enumerate(copied):
            packet.fill_component(None, component, target)
        native.update(*copied, layer)
    token = prefill.next_token_id
    # Independent direct model call with explicit per-layer masks supplied by
    # the same public mask override that the adapter uses for packed caches.
    from mosaickv.adapters.huggingface.base import LayerAttentionMaskOverride

    layer_masks = []
    for validity in masks:
        mask = torch.zeros(1, 2, 1, 4, dtype=torch.bfloat16)
        mask[..., :3].masked_fill_(~validity[None, :, None], torch.finfo(torch.bfloat16).min)
        layer_masks.append(mask)
    override = LayerAttentionMaskOverride(
        tuple(layer.self_attn for layer in adapter._language_layers())
    )
    with torch.inference_mode(), override.activate(tuple(layer_masks)):
        expected = adapter.model(
            input_ids=token,
            past_key_values=native,
            cache_position=torch.tensor([7]),
            position_ids=torch.tensor([[7]]),
            use_cache=True,
            logits_to_keep=1,
        ).logits[:, -1]
    actual = adapter.decode_one_token(token, state, capture_queries=False)
    assert torch.equal(expected, actual.logits)
    assert state.next_decode_position == 8
    assert prefill.state.next_decode_position == 7
    assert all(
        torch.equal(a, b)
        for x, y in zip(originals, pairs, strict=True)
        for a, b in zip(x, y, strict=True)
    )


def test_retention_one_native_transport_matches_ordinary_original_cache_decode():
    from mosaickv.evaluation.exact_selected_control import (
        build_raw_formats,
    )
    from mosaickv.evaluation.exact_selected_control import (
        run_native_same_selected as old_native,
    )

    adapter, prefill, pairs, scores = tiny_prefill()
    full_bytes = backing_bytes(tuple(t for pair in pairs for t in pair))
    packets, _ = build_raw_packets(adapter, pairs, scores, full_bytes, (0,))
    positions = tuple(tuple(tuple(range(7)) for _ in range(2)) for _ in range(2))
    ordinary = build_raw_formats(adapter, pairs, positions, mandatory=(0,))
    expected, _ = old_native(adapter, prefill, ordinary, tokens=128)
    actual, report = run_native_same_selected(adapter, prefill, packets, tokens=128)
    assert torch.equal(expected, actual)
    assert actual.shape == (128, 31)
    assert report["native_verifier_padding_excluded_from_candidate_measurements"] is True
    assert prefill.state.next_decode_position == 7


def test_joint_and_legacy_ragged_owners_have_128_prediction_native_transport_parity():
    adapter, prefill, pairs, scores = tiny_prefill()
    packets, _ = build_raw_packets(adapter, pairs, scores, 9 * 512, (0,))
    legacy = tuple(
        RaggedRawPacket(
            tuple(tuple(t.clone() for t in pair) for pair in packet.pair_by_head),
            tuple(p.clone() for p in packet.original_positions),
            original_length=packet.original_length,
            prompt_capacity=packet.shape[-2],
        )
        for packet in packets
    )
    forced = (1,) * 128
    expected, old_report = run_native_same_selected(adapter, prefill, legacy, forced=forced)
    actual, report = run_native_same_selected(adapter, prefill, packets, forced=forced)
    assert torch.equal(expected, actual)
    assert actual.shape == (128, 31)
    assert (
        report["original_next_decode_position"]
        == old_report["original_next_decode_position"]
        == 134
    )
    assert sum(p.component_bytes for p in legacy) == sum(p.component_bytes for p in packets)
