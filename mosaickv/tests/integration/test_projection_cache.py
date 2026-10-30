"""Model-level isolated source cache and audited tensor lifetimes; CPU only."""

import weakref

import pytest
import torch
from test_hf_tiny_models import _llava, _runtime_request

from mosaickv.adapters.huggingface.types import DecodeState
from mosaickv.evaluation.delimiter_control import source_sha
from mosaickv.evaluation.lossless_lifetime import standard_greedy_reference
from mosaickv.evaluation.lossless_prompt_cache import IsolatedStorageTransport, backing_bytes
from mosaickv.evaluation.projection_archive import ProjectionCapture
from mosaickv.evaluation.projection_cache import (
    ScratchAudit,
    construct_projection_cache,
    identity,
    runtime_controls,
    source_layer_choices,
)


@pytest.mark.parametrize("policy", ["source_all", "hybrid", "kv_only", "bounded_sources"])
def test_archive_runtime_matches_original_exact_greedy_and_audits_shared_backing(policy):
    torch.manual_seed(34)
    adapter = _llava(kv_heads=2)
    adapter.model.to(torch.bfloat16)
    with ProjectionCapture(adapter) as capture:
        prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request(policy).messages))
    capture.validate()
    pairs, _ = adapter._legacy_layers(prefill.state.past_key_values)
    before = source_sha(pairs)
    positions = (((0, 3), (0, 3)),) * 2
    reference = standard_greedy_reference(adapter, prefill, positions)
    cache = construct_projection_cache(adapter, prefill, capture, positions, policy=policy)
    state = DecodeState(
        cache,
        torch.ones(1, 2, dtype=torch.long),
        2,
        4,
        4,
        prefill.state.modality_map,
        dict(
            prefill.state.model_state, mosaickv_packed_all_valid=True, mosaickv_cache_all_valid=True
        ),
    )
    values, token = [prefill.logits.detach().reshape(-1).float()], prefill.next_token_id
    with IsolatedStorageTransport(adapter, cache):
        for _ in range(31):
            output = adapter.decode_one_token(token, state, capture_queries=False)
            values.append(output.logits.detach().reshape(-1).float())
            token = output.next_token_id
    assert torch.equal(torch.stack(values), reference)
    assert source_sha(pairs) == before
    record = cache.storage_record()
    assert record["outstanding_layer_leases"] == 0
    assert record["history_updates_per_layer"] == [31, 31]
    assert record["final_active_kv_storage_bytes"] == backing_bytes(
        tuple(t for layer in cache.layers for t in layer.owned_tensors())
    )
    if policy == "source_all":
        assert record["reconstruction_operator_observations"] > 100
        assert cache.shared_bytes > 0
        assert record["peak_active_kv_storage_bytes"] > record["final_active_kv_storage_bytes"]


def test_scratch_mode_deduplicates_views_excludes_weights_and_releases_all_temporaries():
    base = torch.ones(128, dtype=torch.bfloat16)
    weights = torch.ones(128, dtype=torch.bfloat16)
    observed = []
    mode = ScratchAudit(
        excluded={identity(base), identity(weights)},
        observe=lambda extras: observed.append(backing_bytes(extras)),
    )
    with mode:
        alias = weights.view(-1)
        first = base.clone()
        second = first + base
    refs = weakref.ref(first), weakref.ref(second)
    assert max(observed) == 512
    assert identity(alias) != identity(first)
    del first, second
    assert all(ref() is None for ref in refs)


def test_bounded_source_choices_charge_shared_activation_and_break_ties_deterministically():
    assert source_layer_choices([1, 1, 4, 8], [10, 10, 9, 9], 5, 2) == (0, 1)
    assert source_layer_choices([1, 1, 4, 8], [10, 10, 9, 9], 50, 2) == ()
    assert source_layer_choices([1, 1], [10, 10], 0, 0) == ()
    with pytest.raises(ValueError, match="integer costs"):
        source_layer_choices([True], [10], 0, 1)


def test_source_choice_matches_tiny_exhaustive_cost_solver():
    from itertools import combinations

    source = [15, 5, 17, 6, 30]
    kv = [20, 20, 20, 20, 20]
    fee = 8
    selected = source_layer_choices(source, kv, fee, 3)

    def cost(ids):
        return sum(source[i] if i in ids else kv[i] for i in range(5)) + (fee if ids else 0)

    assert cost(selected) == min(
        cost(ids) for count in range(4) for ids in combinations(range(5), count)
    )


def test_full_retention_source_transport_matches_unmodified_cache_for_32_tokens():
    adapter = _llava(kv_heads=2)
    adapter.model.to(torch.bfloat16)
    with ProjectionCapture(adapter) as capture:
        prefill = adapter.prefill(
            adapter.prepare_inputs(_runtime_request("identity-source").messages)
        )
    result = runtime_controls(
        adapter, prefill, capture, (((0, 1, 2, 3), (0, 1, 2, 3)),) * 2, policies=("source_all",)
    )
    assert result["source_all"]["bitwise_logits_equal_reference"]
    assert result["source_all"]["source_archived_layers"] == 2


def test_mask_extent_does_not_assume_the_first_layer_is_longest():
    adapter = _llava(kv_heads=2)
    adapter.model.to(torch.bfloat16)
    with ProjectionCapture(adapter) as capture:
        prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request("mask-source").messages))
    cache = construct_projection_cache(
        adapter, prefill, capture, (((0,), (0,)), ((0, 1, 3), (0, 1, 3))), policy="source_all"
    )
    assert cache.get_seq_length(0) == 1
    assert cache.get_seq_length(1) == 3
    assert cache.get_seq_length() == 3
    assert cache.get_mask_sizes(torch.tensor([4]), 0) == (4, 0)
