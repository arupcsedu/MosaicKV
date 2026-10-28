"""Exact baseline labels do not authorize transformed or corrupted entries."""

from types import SimpleNamespace

import pytest
from evicastkv_worker import layout_capture
from verify_hf_adapters import tiny_adapters

import mosaickv.backends.hf_runtime as runtime
from mosaickv.evaluation.fox_numerics import independent_state
from mosaickv.evaluation.messages import build_multimodal_messages


@pytest.mark.parametrize("tier", ["exact", "vl_cache_reimpl_exact"])
@pytest.mark.parametrize("corrupt", [False, True])
def test_exact_baseline_label_still_requires_bitwise_source_kv(monkeypatch, tier, corrupt):
    _, adapter = tiny_adapters()[0]
    output = adapter.prefill(
        adapter.prepare_inputs(build_multimodal_messages("a tiny test prompt")),
        capture_queries=False,
        capture_attentions=False,
    )
    layers = adapter._legacy_layers(output.state.past_key_values)[0]
    positions = {
        (layer, head): (0, 2) for layer, (k, _) in enumerate(layers) for head in range(k.shape[1])
    }
    candidate, _ = independent_state(adapter, output.state, positions)
    if corrupt:
        adapter._legacy_layers(candidate.past_key_values)[0][0][0][0, 0, 0, 0] += 1
    slots = tuple(
        dict(tier=tier, layer=layer, kv_head=head, slot=slot, logical_position=logical)
        for (layer, head), ids in positions.items()
        for slot, logical in enumerate(ids)
    )
    monkeypatch.setattr(runtime, "_state_from_packed", lambda *args: candidate)
    with layout_capture(adapter) as capture:
        if corrupt:
            with pytest.raises(AssertionError, match="retained KV changed"):
                runtime._state_from_packed(
                    adapter, output.state, SimpleNamespace(slot_records=slots)
                )
        else:
            runtime._state_from_packed(adapter, output.state, SimpleNamespace(slot_records=slots))
            assert capture["retained_entries"] == capture["padded_entries"]


@pytest.mark.parametrize("tier", ["prototype", "residual", "lookm_reimpl_merge", "anything_exact"])
def test_unknown_or_transformed_tiers_remain_rejected(monkeypatch, tier):
    _, adapter = tiny_adapters()[0]
    output = adapter.prefill(
        adapter.prepare_inputs(build_multimodal_messages("a tiny test prompt")),
        capture_queries=False,
        capture_attentions=False,
    )
    monkeypatch.setattr(runtime, "_state_from_packed", lambda *args: output.state)
    with layout_capture(adapter), pytest.raises(AssertionError, match="original exact KV"):
        runtime._state_from_packed(
            adapter, output.state, SimpleNamespace(slot_records=(dict(tier=tier),))
        )


def test_retry_reuses_only_successes_and_only_known_diagnostic_failures():
    from evicastkv_opportunity_study import arms, retry_parent_rows

    rows = [
        dict(
            sample_id=sid,
            method=method,
            budget_bytes=None if cap is None else cap * 2**20,
            variant=variant,
            status="failed" if method == "vl_cache_reimpl" else "completed",
            error="AssertionError: campaign must retain original exact KV"
            if method == "vl_cache_reimpl"
            else None,
        )
        for sid in ("consumed-a", "consumed-b")
        for method, cap, variant in arms()
    ]
    valid, failed = retry_parent_rows(rows)
    assert len(valid) == 22 and len(failed) == 4
    failed[0]["error"] = "different numerical error"
    with pytest.raises(ValueError):
        retry_parent_rows(rows)
