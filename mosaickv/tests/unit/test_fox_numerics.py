from pathlib import Path

import pytest


def test_independent_real_hf_oracle_never_calls_production_mask_builder(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    from verify_hf_adapters import tiny_adapters

    from mosaickv.backends.hf_runtime import _clone_decode_state
    from mosaickv.evaluation.fox_numerics import (
        independent_state,
        validate_independent_layout,
    )
    from mosaickv.evaluation.messages import build_multimodal_messages

    _, adapter = tiny_adapters()[0]
    prefill = adapter.prefill(
        adapter.prepare_inputs(build_multimodal_messages("tiny prompt")),
        capture_queries=False,
        capture_attentions=False,
    )
    positions = {
        (layer, head): (0, 2, 3)
        for layer, (key, _value) in enumerate(
            adapter._legacy_layers(prefill.state.past_key_values)[0]
        )
        for head in range(key.shape[1])
    }
    state, _masks = independent_state(adapter, prefill.state, positions)
    candidate = _clone_decode_state(adapter, state)
    # All compact slots are valid: native eager decode needs no special mask.
    tokens = [int(prefill.next_token_id.item())]
    token = prefill.next_token_id
    for _ in range(17):
        out = adapter.decode_one_token(
            token, state, capture_queries=False, capture_attentions=False
        )
        token = out.next_token_id
        tokens.append(int(token.item()))

    def forbidden(*args, **kwargs):
        raise AssertionError("independent references must not call production masks")

    monkeypatch.setattr(adapter, "_packed_decode_masks", forbidden)
    result = validate_independent_layout(adapter, prefill.state, candidate, positions, tokens)
    assert result["independent_compact_passed"]
    assert result["unpruned_control_passed"]
    assert result["original_masked_normwise_passed"]
    assert result["forced_steps"] == 16
    assert result["source_entries_bitwise"]


def test_independent_source_gather_rejects_duplicate_positions(monkeypatch):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    from verify_hf_adapters import tiny_adapters

    from mosaickv.evaluation.fox_numerics import independent_state
    from mosaickv.evaluation.messages import build_multimodal_messages

    _, adapter = tiny_adapters()[0]
    output = adapter.prefill(
        adapter.prepare_inputs(build_multimodal_messages("tiny prompt")),
        capture_queries=False,
        capture_attentions=False,
    )
    positions = {
        (layer, head): (0, 0)
        for layer, (key, _v) in enumerate(adapter._legacy_layers(output.state.past_key_values)[0])
        for head in range(key.shape[1])
    }
    with pytest.raises(ValueError, match="invalid independent"):
        independent_state(adapter, output.state, positions)
