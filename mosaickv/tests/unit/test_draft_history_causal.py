from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from mosaickv.evaluation.draft_history_causal import HistoryReference, HistoryRepairTransport
from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.storage_runs import cache_from_formats


def fixture():
    cfg = LlamaConfig(hidden_size=256, num_attention_heads=2, num_key_value_heads=2, head_dim=128)
    blocks = [SimpleNamespace(self_attn=LlamaAttention(cfg, i).bfloat16().eval()) for i in range(2)]

    class Adapter:
        def _language_layers(self):
            return blocks

        def decode_one_token(self, token, state, **kwargs):
            for layer in state.past_key_values.layers:
                k = torch.full((1, 2, 1, 128), -1.0, dtype=torch.bfloat16)
                layer.update(k, k * 2)
                layer.release()
            state.next_decode_position += 1
            return SimpleNamespace(state=state)

    adapter = Adapter()
    prompt = torch.ones(1, 2, 3, 128).bfloat16()
    packet = ExactFormat(
        "kv",
        tuple(prompt.shape),
        pair=tuple(PackedBF16(tuple(prompt.shape), True, (prompt.clone(),)) for _ in range(2)),
    )
    second = ExactFormat(
        "kv",
        tuple(prompt.shape),
        pair=tuple(PackedBF16(tuple(prompt.shape), True, (prompt.clone(),)) for _ in range(2)),
    )
    cache = cache_from_formats(adapter, (packet, second), updates=4, audit=False, sealed=False)
    reference = HistoryReference(
        torch.arange(1, 4)
        .bfloat16()[:, None, None, None, None, None]
        .expand(3, 2, 2, 2, 1, 128)
        .contiguous(),
        torch.tensor([7, 8, 9]),
        9,
    )
    return adapter, cache, reference


@pytest.mark.parametrize("enabled", [False, True])
def test_private_repair_prefix_stop_state_restoration_and_exact_accounting(enabled):
    adapter, cache, reference = fixture()
    original = adapter.decode_one_token.__func__
    initial = cache.initial_bytes
    state = SimpleNamespace(past_key_values=cache, next_decode_position=9)
    with HistoryRepairTransport(adapter, cache, reference, enabled=enabled) as transport:
        for token in (7, 99, 9):
            adapter.decode_one_token(torch.tensor([[token]]), state)
    assert adapter.decode_one_token.__func__ is original
    assert "decode_one_token" not in adapter.__dict__
    assert all("update" not in layer.__dict__ for layer in cache.layers)
    for layer in cache.layers:
        assert torch.all(layer.history[0][..., 0, :] == (1 if enabled else -1))
        assert torch.all(layer.history[0][..., 1:3, :] == -1)
    record = transport.record()
    assert record["repaired_token_steps"] == int(enabled)
    assert record["first_input_prefix_mismatch"] == 1
    assert cache.initial_bytes == initial + transport.stage.numel() * 2
    assert record["cpu_residual_storage_bytes"] == reference.kv.numel() * 2 + 3 * 8
    cache.storage_record()


def test_foreign_request_fails_closed_and_reference_requires_real_positions():
    adapter, cache, reference = fixture()
    with (
        pytest.raises(RuntimeError, match="foreign"),
        HistoryRepairTransport(adapter, cache, reference, enabled=True),
    ):
        adapter.decode_one_token(torch.tensor([[7]]), SimpleNamespace(past_key_values=object()))
    assert "decode_one_token" not in adapter.__dict__
    with pytest.raises(ValueError):
        HistoryReference(reference.kv, reference.tokens, 0)
