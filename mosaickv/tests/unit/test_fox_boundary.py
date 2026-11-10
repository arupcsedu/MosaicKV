"""Synthetic Boundary-B continuation and endpoint-accounting regressions."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from mosaickv.adapters.huggingface import PreparedInputs
from mosaickv.cache_state import Modality, ModalitySpan
from mosaickv.evaluation.fox_boundary import run_boundary_b


@dataclass(frozen=True)
class ImmutableState:
    next_decode_position: int
    past_key_values: tuple


class SyntheticAdapter:
    """Both adapter state conventions must obey the same explicit protocol."""

    def __init__(self, *, immutable):
        self.immutable = immutable
        self.capabilities = SimpleNamespace(model_family="llava_1_5")
        self.model = object()
        self.processor = object()
        self.positions = []
        self.states = []

    def prepare_inputs(self, _messages):
        return PreparedInputs(
            dict(input_ids=torch.tensor([[1, 2, 3, 4]]), attention_mask=torch.ones(1, 4)),
            (ModalitySpan(0, 4, Modality.TEXT),),
            4,
        )

    def new_state(self, length):
        cache = ((torch.zeros(1, 2, length, 3, dtype=torch.float16),) * 2,)
        cls = ImmutableState if self.immutable else SimpleNamespace
        return cls(next_decode_position=length, past_key_values=cache)

    def prefill(self, prepared, **capture_flags):
        assert capture_flags == dict(capture_queries=False, capture_attentions=False)
        self.prefix_state = self.new_state(prepared.logical_sequence_length)
        return SimpleNamespace(state=self.prefix_state)

    def decode_one_token(self, token, state, **capture_flags):
        assert capture_flags == dict(capture_queries=False, capture_attentions=False)
        assert tuple(token.shape) == (1, 1)
        assert state.next_decode_position == 3 + len(self.positions)
        self.positions.append(state.next_decode_position)
        self.states.append(state)
        updated = self.new_state(state.next_decode_position + 1)
        if not self.immutable:
            state.next_decode_position = updated.next_decode_position
            state.past_key_values = updated.past_key_values
            updated = state
        return SimpleNamespace(
            state=updated,
            next_token_id=torch.tensor([[len(self.positions)]]),
            logits=torch.zeros(1, 80),
        )

    @staticmethod
    def _legacy_layers(cache):
        return cache, "synthetic_tuple"


@pytest.mark.parametrize("immutable", [False, True])
def test_boundary_continuation_uses_returned_state_and_final_storage(
    tmp_path, monkeypatch, immutable
):
    # Import the runtime before replacing generation helpers: its own imports
    # must retain the real functions for tests executed later in this process.
    import mosaickv.backends.hf_runtime as runtime
    import mosaickv.generation as generation

    assert callable(runtime.HuggingFaceMosaicKVModel.generate)
    scored_ids = []

    def decode_quality_output(_processor, ids, _terminal_ids):
        scored_ids.extend(ids)
        return SimpleNamespace(text="synthetic fixture only")

    monkeypatch.setattr(generation, "decode_quality_output", decode_quality_output)
    monkeypatch.setattr(generation, "terminal_token_ids", lambda *_: ())
    adapter = SyntheticAdapter(immutable=immutable)
    config = SimpleNamespace(
        method=SimpleNamespace(is_full_cache=True),
        forecasting=SimpleNamespace(prompt_window=16),
    )
    result = run_boundary_b(adapter, config, (), object(), tmp_path)
    assert adapter.positions == list(range(3, 67))
    assert scored_ids == list(range(1, 65))
    assert result["held_token_forward_count"] == 1
    assert result["answer_token_forwards"] == 63
    assert result["prefix_logical_length"] == 3
    assert result["original_logical_length"] == 4
    # K and V alias in this synthetic cache: physical storage is counted once.
    per_position_bytes = 1 * 2 * 3 * 2
    assert result["initial_kv_bytes"] == 4 * per_position_bytes
    assert result["final_kv_bytes"] == 67 * per_position_bytes
    assert result["final_kv_bytes"] - result["initial_kv_bytes"] == 63 * per_position_bytes
    assert result["peak_kv_bytes"] == result["final_kv_bytes"]
    assert result["measurement_type"] == "boundary_diagnostic_not_timing"
    assert result["directly_byte_matched_to_A"] is False
    if immutable:
        assert adapter.prefix_state.next_decode_position == 3
        assert len({id(state) for state in adapter.states}) == 64
    else:
        assert len({id(state) for state in adapter.states}) == 1
