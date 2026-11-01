import pytest
import torch

from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.streaming_prefill import RetiringPrefillCache, lossless_kv_encoder


def pair():
    return tuple(torch.randn(1, 2, 16, 128).bfloat16() for _ in range(2))


def test_native_layers_retire_in_order_and_logical_lengths_do_not_reset():
    cache = RetiringPrefillCache(3, audit=True)
    attention = torch.nn.Linear(2, 2)
    for i in range(3):
        values = pair()
        returned = cache.update(*values, i, dict(cache_position=torch.arange(16)))
        assert returned[0] is values[0]
        assert cache.get_seq_length(i) == 16
        cache.seal(i, attention, lossless_kv_encoder)
        assert cache.layers[i].keys is None and cache.layers[i].values is None
        assert cache.get_seq_length(i) == 16
        decoded = cache.layers[i].packet.materialize(attention)
        assert all(torch.equal(a, b) for a, b in zip(values, decoded, strict=True))
    assert len(cache.formats()) == 3
    record = cache.record()
    assert record["original_native_layers_remaining"] == 0
    assert record["observed_prefill_cache_peak_bytes"] >= record["final_prefill_cache_bytes"]
    assert not record["whole_request_gpu_peak_certified"]
    with pytest.raises(ValueError):
        cache.update(*pair(), 0)


def test_failure_never_publishes_or_drops_native_cache():
    cache = RetiringPrefillCache(1, audit=False)
    values = pair()
    cache.update(*values, 0)

    def fail(*args):
        raise RuntimeError("encoder rejected")

    with pytest.raises(RuntimeError, match="encoder rejected"):
        cache.seal(0, torch.nn.Linear(2, 2), fail)
    assert cache.next_layer == 0 and cache.layers[0].keys is values[0]
    with pytest.raises(RuntimeError):
        cache.formats()
    assert cache.record(require_complete=False)["logical_lengths"] == [16]


def test_aliasing_and_out_of_order_updates_rejected():
    cache = RetiringPrefillCache(2, audit=False)
    with pytest.raises(ValueError):
        cache.update(*pair(), 1)
    values = pair()
    cache.update(*values, 0)

    def aliases(*args):
        return ExactFormat(
            "kv",
            tuple(values[0].shape),
            pair=tuple(PackedBF16(tuple(t.shape), True, (t,)) for t in values),
        )

    with pytest.raises(RuntimeError, match="aliases"):
        cache.seal(0, torch.nn.Linear(2, 2), aliases)


def test_original_position_offsets_not_silently_renumbered():
    cache = RetiringPrefillCache(1, audit=False)
    with pytest.raises(ValueError, match="original contiguous"):
        cache.update(*pair(), 0, dict(cache_position=torch.arange(3, 19)))
    assert cache.get_seq_length() == 0


def test_no_decode_crop_beam_or_partial_export():
    cache = RetiringPrefillCache(1, audit=False)
    with pytest.raises(RuntimeError):
        cache.formats()
    with pytest.raises(NotImplementedError):
        cache.crop(2)
    with pytest.raises(NotImplementedError):
        cache.reorder_cache(torch.tensor([0]))


@pytest.mark.parametrize("count", [True, 0, -1, 1.0])
def test_invalid_layer_count_rejected(count):
    with pytest.raises(ValueError):
        RetiringPrefillCache(count)


def test_real_hf_cpu_prefill_hook_preserves_logits_and_retires_cache():
    from types import SimpleNamespace

    from transformers import LlamaConfig, LlamaForCausalLM

    from mosaickv.evaluation.streaming_prefill import SealAfterAttention

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=32,
        attn_implementation="eager",
    )
    torch.manual_seed(127)
    model = LlamaForCausalLM(config).bfloat16().eval()
    adapter = SimpleNamespace(_language_layers=lambda: model.model.layers)
    tokens = torch.arange(8)[None]
    cache = RetiringPrefillCache(2, audit=True)
    with torch.inference_mode():
        reference = model(tokens, use_cache=True).logits
        with SealAfterAttention(adapter, cache, lossless_kv_encoder):
            result = model(tokens, use_cache=True, past_key_values=cache)
    assert result.past_key_values is cache
    assert torch.equal(reference, result.logits)
    assert cache.get_seq_length() == 8 and len(cache.formats()) == 2
    assert all(x.keys is None and x.values is None for x in cache.layers)
    assert cache.record()["original_KV_tensor_objects_alive"] == 0
    for other in (tokens.flip(1), tokens):
        fresh = RetiringPrefillCache(2, audit=True)
        with torch.inference_mode(), SealAfterAttention(adapter, fresh, lossless_kv_encoder):
            replay = model(other, use_cache=True, past_key_values=fresh).logits
        assert all(not block.self_attn._forward_hooks for block in model.model.layers)
    assert torch.equal(reference, replay)
