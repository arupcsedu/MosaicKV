"""Random-weight CPU transport fixtures, never pretrained quality evidence."""

from dataclasses import replace

import pytest
import torch
from test_hf_tiny_models import _llava

from mosaickv.backends.hf_attention_bias import bias_cache_storage, install_attention_bias_cache
from mosaickv.backends.hf_runtime import _clone_decode_state
from mosaickv.prototypes.attention_compaction import CompactedHead


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def heads(adapter, state, positions=None):
    result = []
    for keys, values in adapter._legacy_layers(state.past_key_values)[0]:
        p = tuple(range(keys.shape[-2])) if positions is None else positions
        result.append(
            tuple(
                CompactedHead(
                    keys[0, h, list(p)].clone(),
                    values[0, h, list(p)].clone(),
                    torch.zeros(len(p)),
                    p,
                    False,
                    {},
                )
                for h in range(keys.shape[1])
            )
        )
    return tuple(result)


def install(adapter, prefill, rows, **kwargs):
    return install_attention_bias_cache(
        adapter,
        prefill.state,
        rows,
        required_logical_positions=(0, 3),
        generated_capacity=32,
        sink_tokens=4,
        cap_bytes=2**20,
        **kwargs,
    )


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_zero_bias_retention_one_matches_ordinary_fullkv_for_32_tokens(kv_heads):
    torch.manual_seed(73)
    adapter = _llava(kv_heads=kv_heads)
    prefill = adapter.prefill(
        adapter.prepare_inputs([dict(role="user", content="tiny")]), capture_queries=False
    )
    rows = heads(adapter, prefill.state)
    original = _clone_decode_state(adapter, prefill.state)
    transported = install(adapter, prefill, rows, retention_one=True)
    token = prefill.next_token_id
    initial = bias_cache_storage(adapter, transported)
    weights = [p.detach().clone() for p in adapter.model.parameters()]
    source = [
        (k.clone(), v.clone()) for k, v in adapter._legacy_layers(prefill.state.past_key_values)[0]
    ]
    for step in range(32):
        full = adapter.decode_one_token(token, original, capture_queries=False)
        out = adapter.decode_one_token(token, transported, capture_queries=False)
        assert torch.equal(full.logits, out.logits)
        assert torch.equal(full.next_token_id, out.next_token_id)
        assert out.state.next_decode_position == 5 + step
        assert out.state.logical_sequence_length == 5 + step
        assert (
            bias_cache_storage(adapter, transported)["active_kv_backing_bytes"]
            == initial["active_kv_backing_bytes"]
        )
        token = full.next_token_id
    assert prefill.state.next_decode_position == 4
    assert all(
        torch.equal(k, a) and torch.equal(v, b)
        for (k, v), (a, b) in zip(
            source, adapter._legacy_layers(prefill.state.past_key_values)[0], strict=True
        )
    )
    assert all(torch.equal(a, b) for a, b in zip(weights, adapter.model.parameters(), strict=True))


def test_nonzero_bias_reaches_eager_softmax_and_stays_out_of_generated_ring():
    torch.manual_seed(73)
    adapter = _llava()
    prefill = adapter.prefill(
        adapter.prepare_inputs([dict(role="user", content="tiny")]), capture_queries=False
    )
    original = heads(adapter, prefill.state)
    rows = tuple(
        tuple(replace(row, log_bias=torch.tensor([0.0, 1.0, 0.0, 0.0])) for row in layer)
        for layer in original
    )
    state = install(adapter, prefill, rows)
    baseline = install(adapter, prefill, original)
    a = adapter.decode_one_token(
        prefill.next_token_id, baseline, capture_queries=False, capture_attentions=True
    )
    b = adapter.decode_one_token(
        prefill.next_token_id, state, capture_queries=False, capture_attentions=True
    )
    # First-layer Q/K are identical, so odds change by exp(1) for slot 1.
    odds_before = a.attention_weights[0][0, :, 0, 1] / a.attention_weights[0][0, :, 0, 0]
    odds_after = b.attention_weights[0][0, :, 0, 1] / b.attention_weights[0][0, :, 0, 0]
    torch.testing.assert_close(odds_after / odds_before, torch.full_like(odds_before, torch.e))
    assert not torch.equal(a.logits, b.logits)
    mask = state.model_state["mosaickv_bounded_history_masks"][0]
    assert (mask[..., 4:] == 0).all()
    assert state.model_state["attention_compaction_representation"].endswith("mass_bias_v1")


def test_transport_rejects_changed_keys_mandatory_values_false_identity_and_unsupported_kernel():
    adapter = _llava()
    prefill = adapter.prefill(
        adapter.prepare_inputs([dict(role="user", content="tiny")]), capture_queries=False
    )
    rows = heads(adapter, prefill.state)
    for field, changed, match in (
        ("keys", rows[0][0].keys + 1, "post-RoPE"),
        ("values", rows[0][0].values + 1, "mandatory"),
        ("log_bias", torch.ones(4), "mandatory"),
    ):
        bad = ((replace(rows[0][0], **{field: changed}),), rows[1])
        with pytest.raises(ValueError, match=match):
            install(adapter, prefill, bad)
    subset = heads(adapter, prefill.state, positions=(0, 3))
    with pytest.raises(ValueError, match="identity"):
        install(adapter, prefill, subset, retention_one=True)
    with pytest.raises(RuntimeError, match="eager"):
        install(_llava(attention="sdpa"), prefill, rows)


def test_separate_requests_have_no_aliases_and_ring_preserves_original_positions():
    adapter = _llava()
    prefill = adapter.prefill(
        adapter.prepare_inputs([dict(role="user", content="tiny")]), capture_queries=False
    )
    rows = heads(adapter, prefill.state, positions=(0, 3))
    states = [
        install_attention_bias_cache(
            adapter,
            prefill.state,
            rows,
            required_logical_positions=(0, 3),
            generated_capacity=2,
            sink_tokens=0,
            cap_bytes=2**20,
        )
        for _ in range(2)
    ]
    for _ in range(5):
        adapter.decode_one_token(prefill.next_token_id, states[0], capture_queries=False)
    assert states[0].next_decode_position == 9 and states[1].next_decode_position == 4
    assert sorted(states[0].past_key_values.layers[0].generated_logical_positions) == [7, 8]
    assert states[1].past_key_values.layers[0].generated_logical_positions == ()
    for key in ("attention_compaction_log_bias", "mosaickv_bounded_history_masks"):
        assert states[0].model_state[key][0].data_ptr() != states[1].model_state[key][0].data_ptr()


def test_complete_random_weight_diagnostic_has_real_fits_unchanged_guard_and_aba_controls():
    from mosaickv.evaluation.attention_compaction_diagnostic import diagnose_bias_transport
    from mosaickv.prototypes.attention_compaction import AttentionCompactionPolicy

    torch.manual_seed(73)
    adapter = _llava()
    prepared = adapter.prepare_inputs([dict(role="user", content="tiny")])
    events = []
    record, artifacts = diagnose_bias_transport(
        adapter,
        prepared,
        prompt_positions=3,
        cap_bytes=2**20,
        policy=AttentionCompactionPolicy(True),
        event=events.append,
    )
    assert record["engineering_passed"] and record["zero_bias_retention_one_passed"]
    assert all(row["bitwise_equal"] for row in record["retention_one_parity"])
    assert record["original_guard"] == dict(
        horizon=32, checkpoints=[1, 4, 16, 32], absolute_tolerance=1e-7, relative_tolerance=1e-6
    )
    assert all(control["passed"] for control in record["controls"].values())
    assert record["constructed_heads"] == 2
    assert record["fullkv_prefills"] == 1 and not record["second_prefill_performed"]
    assert record["source_weights_unchanged"] and record["source_cache_unchanged"]
    assert not record["fresh_cohort_authorized"] and not record["benchmark_labels_accessed"]
    assert all(row["total_persistent_backing_bytes"] <= 2**20 for row in record["storage"])
    assert len(artifacts["logits"]["fullkv"]) == 32
    assert len(artifacts["logits"]["null_repack_after_candidate"]) == 32
    assert set(record["phase_timing"]) == {
        e["phase"] for e in events if e["event"] == "phase_complete"
    }
    import yaml
    from attention_compaction_hf_gate import PROTOCOL, review_result

    reviewed = review_result(record, artifacts["logits"], yaml.safe_load(PROTOCOL.read_text()))
    assert reviewed["engineering_passed"]
    assert reviewed["diagnostic_passed"] == record["diagnostic_passed"]


def test_failed_identity_emits_complete_parity_and_stops_before_fitting(monkeypatch):
    from mosaickv.evaluation.attention_compaction_diagnostic import diagnose_bias_transport
    from mosaickv.prototypes.attention_compaction import AttentionCompactionPolicy

    adapter = _llava()
    prepared = adapter.prepare_inputs([dict(role="user", content="tiny")])
    original = adapter.decode_one_token

    def perturbed(token, state, **kwargs):
        output = original(token, state, **kwargs)
        if state.model_state.get("attention_compaction_representation"):
            return replace(output, logits=output.logits + 0.1)
        return output

    monkeypatch.setattr(adapter, "decode_one_token", perturbed)
    events = []
    with pytest.raises(RuntimeError, match="parity failed"):
        diagnose_bias_transport(
            adapter,
            prepared,
            prompt_positions=3,
            cap_bytes=2**20,
            policy=AttentionCompactionPolicy(True),
            event=events.append,
        )
    rows = next(row["steps"] for row in events if row["event"] == "retention_one_parity")
    assert len(rows) == 32 and not all(row["bitwise_equal"] for row in rows)
    assert not any(row.get("phase") == "cpu_value_and_mass_construction" for row in events)
