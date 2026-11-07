from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation.offloaded_fullkv_draft import DraftLayer, run, signatures, snapshot


def test_cpu_exact_prompt_and_single_update_are_isolated_and_accounted():
    torch.manual_seed(81)
    pair = tuple(torch.randn(1, 2, 7, 4).bfloat16() for _ in range(2))
    copied = snapshot(pair)
    assert all(a.data_ptr() != b.data_ptr() for a, b in zip(pair, copied, strict=True))
    seen = []
    layer = DraftLayer(copied, device=torch.device("cpu"), observe=lambda t: seen.append(t))
    token = tuple(torch.randn(1, 2, 1, 4).bfloat16() for _ in range(2))
    materialized = layer.update(*token)
    for source, new, actual in zip(pair, token, materialized, strict=True):
        assert torch.equal(actual[..., :7, :], source)
        assert torch.equal(actual[..., 7:, :], new)
    assert layer.get_seq_length() == 8 and seen
    assert layer.resident_bytes == sum(t.numel() * t.element_size() for t in token)
    layer.release()
    assert layer.lease_bytes == 0 and not layer.leased
    with pytest.raises(ValueError, match="maximum"):
        layer.update(*token)


def test_exact_draft_uses_existing_prefill_no_labels_and_one_request_local_cache(monkeypatch):
    import mosaickv.evaluation.offloaded_fullkv_draft as module

    class Scope:
        def __init__(self, adapter, cache):
            self.cache = cache

        def __enter__(self):
            return self

        def __exit__(self, *args):
            for layer in self.cache.layers:
                layer.release()

    monkeypatch.setattr(module, "IsolatedStorageTransport", Scope)
    pair = tuple(torch.zeros(1, 2, 7, 4).bfloat16() for _ in range(2))
    calls = []

    def decode(token, state, *, capture_queries):
        calls.append((token, state))
        for layer in state.past_key_values.layers:
            layer.update(torch.ones(1, 2, 1, 4).bfloat16(), torch.ones(1, 2, 1, 4).bfloat16())
        return SimpleNamespace(
            logits=torch.tensor([[0.1, 0.2, 0.3]]),
            state=SimpleNamespace(logical_sequence_length=8, next_decode_position=8),
        )

    adapter = SimpleNamespace(
        _language_layers=lambda: (0, 1),
        device=torch.device("cpu"),
        model=torch.nn.Linear(4, 4).eval(),
        decode_one_token=decode,
    )
    prefill = SimpleNamespace(
        logits=torch.tensor([[0.3, 0.2, 0.1]]),
        next_token_id=torch.tensor([[0]]),
        state=SimpleNamespace(
            logical_sequence_length=7, next_decode_position=7, modality_map=(), model_state={}
        ),
    )
    result, evidence = run(adapter, prefill, (snapshot(pair), snapshot(pair)))
    assert len(calls) == 1 and result.shape == (2, 3)
    assert evidence["layer_updates"] == [1, 1]
    assert evidence["outstanding_layer_leases"] == 0
    assert evidence["temporary_cpu_original_kv_bytes"] == 2 * 2 * 1 * 2 * 7 * 4 * 2
    assert evidence["second_prefill_performed"] is False
    assert prefill.state.model_state == {}


def test_inference_snapshots_are_checked_without_unsupported_version_counters():
    with torch.inference_mode():
        pair = tuple(torch.ones(1, 2, 3, 4).bfloat16() for _ in range(2))
        before = signatures((pair,))
        pair[0].zero_()
        assert before != signatures((pair,))


def test_global_draft_requires_an_active_explicit_forecast_policy():
    from storage_six_ablation import component_policies

    from mosaickv.evaluation.storage_six_overlay import OverlayPolicy

    with pytest.raises(ValueError, match="active"):
        OverlayPolicy(readout_target="fullkv_draft")
    policies = component_policies(global_draft_study=True)
    assert policies["readout_joint"]["readout_target"] == "fullkv_draft"
    assert policies["no_forecast"]["readout_target"] == "conditional_head"
    assert policies["no_forecast"]["readout_selection"] == "none"
