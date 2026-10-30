"""Tiny random-weight CPU LLaVA: same backing, positions and mask transport."""

from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
from test_hf_tiny_models import _llava, _runtime_request

import mosaickv.adapters.huggingface.bounded_history_cache as bounded
import mosaickv.backends.hf_runtime as runtime_module
from mosaickv.adapters.huggingface.incremental_history_cache import install_incremental_history
from mosaickv.adapters.huggingface.layer_local_cache import gather_layer_local_snapshot
from mosaickv.backends.hf_allocation_guard import packed_positions_sha
from mosaickv.config import IncrementalHistoryConfig
from mosaickv.evaluation.delimiter_control import source_sha
from mosaickv.evaluation.donor_isolation import DonorMaskControl
from mosaickv.measurements.cache_lifecycle import CacheLifecycle


def _fixture(adapter):
    prefill = adapter.prefill(adapter.prepare_inputs(_runtime_request("donor").messages))
    source, _ = adapter._legacy_layers(prefill.state.past_key_values)
    heads = source[0][0].shape[1]
    positions = tuple(tuple((0, 3) for _ in range(heads)) for _ in source)
    snapshot = gather_layer_local_snapshot(adapter, prefill.state.past_key_values, positions)
    records = tuple(
        dict(layer=layer, kv_head=h, slot=s, logical_position=p, tier="exact")
        for layer in range(len(source))
        for h in range(heads)
        for s, p in enumerate((0, 3))
    )
    packed = runtime_module.PackedRuntimeCache(snapshot, (), 2, True, (), (), records)
    cost = 2 * source[0][0].shape[-1] * source[0][0].element_size()
    tx = dict(
        mandatory_positions=[3],
        delimiter_positions=[2, 3],
        changed_routes=len(source) * heads,
        total_replaced_bytes=len(source) * heads * cost,
        events=[
            dict(
                layer=layer,
                kv_head=h,
                removed_positions=[0],
                added_positions=[2],
                position_bytes=cost,
                replaced_bytes=cost,
            )
            for layer in range(len(source))
            for h in range(heads)
        ],
        before_positions_sha256=packed_positions_sha(packed),
        source_KV_sha256=source_sha(source),
    )
    return prefill, packed, tx


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_32_prefix_zero_mask_parity_donor_zero_mass_storage_and_cleanup(kv_heads):
    torch.manual_seed(7)
    adapter = _llava(kv_heads=kv_heads)
    original = (
        runtime_module._state_from_packed,
        adapter.decode_one_token,
        bounded.bounded_history_masks,
    )
    outcomes = {}
    for mode in ("unobserved", "zero_mask", "donor_only"):
        prefill, packed, tx = _fixture(adapter)
        control = (
            None
            if mode == "unobserved"
            else DonorMaskControl(adapter, runtime_module, tx, mode=mode)
        )
        with control if control is not None else nullcontext():
            state = runtime_module._state_from_packed(adapter, prefill.state, packed)
            install_incremental_history(
                adapter,
                state,
                generated_capacity=32,
                sink_tokens=4,
                cap_bytes=65536,
                config=IncrementalHistoryConfig(
                    chunk_tokens=16,
                    recent_tokens=4,
                    policy="sink_recent",
                    fixed_prompt_reservation=True,
                ),
            )
            before = source_sha(
                tuple((x.keys.clone(), x.values.clone()) for x in state.past_key_values.layers)
            )
            logits, storage = [], []
            for step in range(31):
                output = adapter.decode_one_token(
                    torch.tensor([[3 + step % 5]]),
                    state,
                    capture_queries=False,
                    capture_attentions=True,
                )
                logits.append(output.logits.clone())
                storage.append(
                    CacheLifecycle.count(
                        tuple((x.keys, x.values) for x in state.past_key_values.layers)
                    )[1]
                )
            prompt = tuple(
                (x.keys[..., :2, :], x.values[..., :2, :]) for x in state.past_key_values.layers
            )
            assert source_sha(prompt) == before
            if control is not None:
                control.validate_complete()
                assert control.record()["physical_KV_bytes_saved"] == 0
                if mode == "donor_only":
                    assert all(
                        r["maximum_donor_attention_probability"] == 0 for r in control.records
                    )
                else:
                    assert any(
                        r["maximum_donor_attention_probability"] > 0 for r in control.records
                    )
            outcomes[mode] = torch.stack(logits), storage
        assert original == (
            runtime_module._state_from_packed,
            adapter.decode_one_token,
            bounded.bounded_history_masks,
        )
        assert adapter._packed_mask_dispatcher.active_masks is None
    assert torch.equal(outcomes["unobserved"][0], outcomes["zero_mask"][0])
    assert not torch.equal(outcomes["donor_only"][0], outcomes["zero_mask"][0])
    assert outcomes["unobserved"][1] == outcomes["zero_mask"][1] == outcomes["donor_only"][1]


def test_saved_source_mismatch_fails_and_cleans_hooks():
    adapter = _llava()
    prefill, packed, tx = _fixture(adapter)
    tx = deepcopy(tx)
    tx["source_KV_sha256"] = "wrong"
    original = (
        runtime_module._state_from_packed,
        adapter.decode_one_token,
        bounded.bounded_history_masks,
    )
    with (
        pytest.raises(ValueError, match="original source"),
        DonorMaskControl(adapter, runtime_module, tx, mode="donor_only"),
    ):
        runtime_module._state_from_packed(adapter, prefill.state, packed)
    assert original == (
        runtime_module._state_from_packed,
        adapter.decode_one_token,
        bounded.bounded_history_masks,
    )
