from types import SimpleNamespace

import pytest
import torch

from mosaickv.evaluation import joint_value_transaction as joint
from mosaickv.evaluation.exact_storage import ExactFormat
from mosaickv.evaluation.lossless_bf16 import PackedBF16
from mosaickv.evaluation.lossless_prompt_cache import backing_bytes
from mosaickv.evaluation.sequential_source_history import SequentialSourceHistoryLayer


def layer():
    values = tuple(
        torch.arange(5 * 2 * 128, dtype=torch.bfloat16).reshape(1, 2, 5, 128) for _ in range(2)
    )
    packet = ExactFormat(
        "kv",
        tuple(values[0].shape),
        pair=tuple(PackedBF16(tuple(x.shape), True, (x,)) for x in values),
    )
    return SequentialSourceHistoryLayer(
        packet,
        torch.nn.Linear(256, 256, bias=False),
        observe=lambda extras=(): None,
        direct=True,
        updates=128,
        audit=False,
    )


@torch.inference_mode()
def test_split_transaction_matches_original_history_and_physical_storage_exactly():
    original, split = layer(), layer()
    keys, values = [], []
    for step in range(12):
        key = torch.full((1, 2, 1, 128), step + 1, dtype=torch.bfloat16)
        value = torch.full_like(key, -step - 1)
        keys.append(key)
        values.append(value)
        original.source_pending = split.source_pending = True
        reference = original.update(key, value)[0].clone()
        actual = joint.begin_key(split, key)[0]
        assert torch.equal(actual.view(torch.int16), reference.view(torch.int16))
        assert split.source_pending and split._joint_pending_value == step
        # Fill the pending raw V slot with NaN: a correct K-only stage must not read it.
        split.history[1][..., step : step + 1, :].fill_(torch.nan)
        assert torch.isfinite(actual).all()
        joint.commit_value(split, value)
        assert not split.source_pending and split._joint_pending_value is None
        original.materialize_component(1)
        split.materialize_component(1)
        assert torch.equal(split.leased[0].view(torch.int16), original.leased[0].view(torch.int16))
        assert split.history_length == original.history_length == step + 1
        assert split.resident_bytes == original.resident_bytes
        assert split.lease_bytes == original.lease_bytes
        assert backing_bytes(split.owned_tensors()) == backing_bytes(original.owned_tensors())
        split.release()
        original.release()


@torch.inference_mode()
def test_no_value_commit_without_a_real_key_slot_or_more_than_once():
    state = layer()
    value = torch.zeros(1, 2, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="same-slot"):
        joint.commit_value(state, value)
    state.source_pending = True
    joint.begin_key(state, value)
    joint.commit_value(state, value)
    with pytest.raises(ValueError, match="same-slot"):
        joint.commit_value(state, value)
    state.release()


@pytest.mark.parametrize("corruption", ["no_source", "bad_dtype", "pending", "full_tail"])
@torch.inference_mode()
def test_invalid_begin_cannot_advance_history(corruption):
    state, key = layer(), torch.zeros(1, 2, 1, 128, dtype=torch.bfloat16)
    state.source_pending = True
    if corruption == "no_source":
        state.source_pending = False
    elif corruption == "bad_dtype":
        key = key.float()
    elif corruption == "pending":
        state._joint_pending_value = 0
    else:
        state.tail_used = 16
    with pytest.raises(ValueError):
        joint.begin_key(state, key)
    assert state.history_length == 0 and not state.leased


def test_each_layer_value_must_commit_before_the_next_layer():
    ledger = joint.ValueLedger(2)
    for update in range(4):
        for index in range(2):
            ledger.enter(index, update, ready=update > 0)
            if update:
                ledger.produced()
                ledger.consume(index, update + 1)
            with pytest.raises(RuntimeError, match="previous actual value"):
                ledger.enter((index + 1) % 2, update, ready=update > 0)
            ledger.value_ready(index, update + 1)
            with pytest.raises(RuntimeError, match="exactly one"):
                ledger.value_ready(index, update + 1)
    result = ledger.finish(4)
    assert result["key_calls"] == 8 and result["folded_keys"] == 6
    assert result["value_commits"] == 8 and result["fake_raw_value_tensors"] == 0
    assert not result["all_value_projections_original"]


def test_wrong_epoch_value_or_missing_value_prevents_success():
    ledger = joint.ValueLedger(1)
    ledger.enter(0, 0, ready=False)
    with pytest.raises(RuntimeError, match="exactly one"):
        ledger.value_ready(0, 2)
    with pytest.raises(RuntimeError, match="value transaction"):
        ledger.finish(1)


def test_commit_hook_ignores_all_unmodified_readers_and_keys():
    joint.after_prompt(SimpleNamespace(), 1)
    joint.after_prompt(
        SimpleNamespace(
            prompt_graph_pool=SimpleNamespace(joint_projection=SimpleNamespace(mode="keys"))
        ),
        1,
    )
    joint.after_prompt(
        SimpleNamespace(
            prompt_graph_pool=SimpleNamespace(joint_projection=SimpleNamespace(mode="kv"))
        ),
        0,
    )


@torch.inference_mode()
def test_pinned_full_width_attention_matches_reference_without_calling_value_projection(
    monkeypatch,
):
    from transformers import LlamaConfig
    from transformers.cache_utils import DynamicCache
    from transformers.models.llama.modeling_llama import LlamaAttention

    from mosaickv.evaluation.sequential_kv_lease import SequentialAttentionTransport, fill_component

    config = LlamaConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        attention_dropout=0.0,
        attention_bias=False,
        num_hidden_layers=1,
    )
    config._attn_implementation = "eager"
    with torch.inference_mode(False):
        attention = LlamaAttention(config, layer_idx=0).bfloat16().eval()
    versions = [(p.data_ptr(), p._version) for p in attention.parameters()]
    prompt = tuple(torch.randn(1, 32, 5, 128).bfloat16() for _ in range(2))
    hidden = torch.randn(1, 1, 4096).bfloat16()
    phase = (torch.ones(1, 1, 128).bfloat16(), torch.zeros(1, 1, 128).bfloat16())
    reference_cache = DynamicCache()
    reference_cache.update(*prompt, 0)
    reference_output, reference_weights = attention(
        hidden, phase, attention_mask=None, past_key_value=reference_cache
    )
    raw_value = attention.v_proj(hidden).view(1, 1, 32, 128).transpose(1, 2)
    packet = ExactFormat(
        "kv",
        tuple(prompt[0].shape),
        pair=tuple(PackedBF16(tuple(t.shape), True, (t,)) for t in prompt),
    )
    state = SequentialSourceHistoryLayer(
        packet, attention, observe=lambda extras=(): None, direct=True, updates=128, audit=False
    )
    state.source_pending = True
    cache = SimpleNamespace(layers=[state])
    engine = SimpleNamespace(tail=0, output=raw_value.reshape(1, 4096))
    pool = SimpleNamespace(lt_engine=engine, lookahead=None)
    owner = SimpleNamespace(
        cache=cache, pool=pool, mode="kv", ledger=joint.ValueLedger(1), indices={id(state): 0}
    )
    pool.joint_projection = owner
    pool.reconstruct = lambda layer, comp, dest: fill_component(packet, attention, comp, dest)
    state.prompt_graph_pool = pool
    owner.ledger.enter(0, 0, ready=False)

    def forbid(*args, **kwargs):
        raise RuntimeError("separate value projection was called")

    monkeypatch.setattr(attention.v_proj, "forward", forbid)
    adapter = SimpleNamespace(_language_layers=lambda: [SimpleNamespace(self_attn=attention)])
    with SequentialAttentionTransport(adapter, cache):
        actual_output, actual_weights = joint.attention_forward(owner, state)(
            hidden, phase, past_key_value=cache
        )
    assert torch.equal(actual_output.view(torch.int16), reference_output.view(torch.int16))
    assert torch.equal(actual_weights.view(torch.int16), reference_weights.view(torch.int16))
    assert torch.equal(state.history[1][..., :1, :].view(torch.int16), raw_value.view(torch.int16))
    assert not state.source_pending and state._joint_pending_value is None
    assert owner.ledger.finish(1)["value_commits"] == 1
    assert versions == [(p.data_ptr(), p._version) for p in attention.parameters()]
    state.release()
