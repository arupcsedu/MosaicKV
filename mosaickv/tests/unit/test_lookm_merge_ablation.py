from types import SimpleNamespace

import pytest
import torch

from mosaickv.adapters.huggingface.types import CachedKeyState, CacheLayerSnapshot, CacheSnapshot
from mosaickv.evaluation.lookm_merge_ablation import ARMS, comparison, decode, make_payload


def fixture():
    source = tuple(
        (torch.arange(48).reshape(1, 2, 6, 4).bfloat16() + i, torch.ones(1, 2, 6, 4).bfloat16())
        for i in range(2)
    )
    positions = (((0, 5), (1, 5)), ((0, 2, 5), (1, 3, 5)))
    template = CacheSnapshot(
        tuple(CacheLayerSnapshot(k, v, 2) for k, v in source),
        tuple,
        "tuple",
        6,
        CachedKeyState.POST_ROPE,
    )
    selected = tuple(
        tuple(torch.stack([t[0, h, list(p)] for h, p in enumerate(heads)])[None] for t in pair)
        for pair, heads in zip(source, positions, strict=True)
    )
    merged = tuple((k + 2, v + 3) for k, v in selected)
    adapter = SimpleNamespace(
        _legacy_layers=lambda _: (source, "tuple"),
        borrow_past_key_values=lambda _: template,
    )
    prefill = SimpleNamespace(
        state=SimpleNamespace(past_key_values=object(), logical_sequence_length=6)
    )
    return adapter, prefill, source, positions, selected, merged


@pytest.mark.parametrize("arm", ARMS)
def test_only_registered_key_or_value_changes_with_identical_bytes_and_membership(arm):
    adapter, prefill, source, positions, selected, merged = fixture()
    originals = tuple(tuple(t.clone() for t in pair) for pair in source)
    payload = make_payload(adapter, prefill, merged, positions, arm)
    assert payload.head_positions == positions
    snapshot = payload.snapshot
    assert snapshot.active_sequence_length == 3 and snapshot.independent_layer_lengths
    assert snapshot.cached_key_state == CachedKeyState.POST_ROPE
    for layer, pair, m in zip(snapshot.layers, selected, merged, strict=True):
        assert torch.equal(
            layer.key, m[0] if arm in ("key_merge_only", "key_value_merge") else pair[0]
        )
        assert torch.equal(
            layer.value, m[1] if arm in ("value_merge_only", "key_value_merge") else pair[1]
        )
        assert layer.key.data_ptr() != m[0].data_ptr()
        assert layer.value.data_ptr() != m[1].data_ptr()
    assert (
        sum(
            t.numel() * t.element_size()
            for layer in snapshot.layers
            for t in (layer.key, layer.value)
        )
        == 160
    )
    snapshot.layers[0].key.zero_()
    assert all(
        torch.equal(a, b)
        for aa, bb in zip(source, originals, strict=True)
        for a, b in zip(aa, bb, strict=True)
    )


@pytest.mark.parametrize(
    "bad",
    [
        (((0, 4), (1, 5)), ((0, 2, 5), (1, 3, 5))),
        (((5, 0), (1, 5)), ((0, 2, 5), (1, 3, 5))),
        (((0, 5), (5,)), ((0, 2, 5), (1, 3, 5))),
    ],
)
def test_missing_mandatory_unsorted_or_ragged_rejected(bad):
    adapter, prefill, _, _, _, merged = fixture()
    with pytest.raises(ValueError):
        make_payload(adapter, prefill, merged, bad, "selection_only")


def test_wrong_arm_dtype_and_capacity_rejected():
    adapter, prefill, _, positions, _, merged = fixture()
    with pytest.raises(ValueError, match="unknown"):
        make_payload(adapter, prefill, merged, positions, "official")
    for changed in ((merged[0][0].float(), merged[0][1]), (merged[0][0][:, :, :1], merged[0][1])):
        with pytest.raises(ValueError, match="capacity"):
            make_payload(adapter, prefill, (changed, merged[1]), positions, "selection_only")


def test_fixed_prefix_alignment_uses_all_forced_tokens_and_isolates_state(monkeypatch):
    from mosaickv.backends import hf_runtime

    adapter, prefill, source, positions, _, merged = fixture()
    prefill.logits, prefill.next_token_id = torch.tensor([[0.0, 3.0, 1.0]]), torch.tensor([1])
    prefill.state.next_decode_position = 6
    seen = []
    states = []

    def inject(a, original, packed):
        assert original is prefill.state
        state = SimpleNamespace(
            logical_sequence_length=6,
            next_decode_position=6,
            model_state={},
            past_key_values=tuple((x.key, x.value) for x in packed.snapshot.layers),
        )
        states.append(state)
        return state

    def step(token, state, **kwargs):
        seen.append(int(token.item()))
        state.next_decode_position += 1
        return SimpleNamespace(
            logits=torch.tensor([[2.0, 1.0, 0.0]]), next_token_id=torch.tensor([0])
        )

    adapter.decode_one_token = step
    adapter._legacy_layers = lambda cache: (cache, "tuple")
    prefill.state.past_key_values = source
    monkeypatch.setattr(hf_runtime, "_state_from_packed", inject)
    first = make_payload(adapter, prefill, merged, positions, "selection_only")
    logits, record = decode(adapter, prefill, first, predictions=4, forced_prefix=(2, 1, 2))
    assert seen == [2, 1, 2] and torch.equal(logits[0], prefill.logits[0])
    assert first.snapshot is None and record["final_next_decode_position"] == 9
    assert prefill.state.next_decode_position == 6
    assert record["initial_active_kv_storage_bytes"] == 160
    assert record["fixed_FullKV_prefix_evaluation_only"]
    with pytest.raises(ValueError, match="already consumed"):
        decode(adapter, prefill, first, predictions=4)
    decode(
        adapter,
        prefill,
        make_payload(adapter, prefill, merged, positions, "selection_only"),
        predictions=4,
    )
    assert seen[3:] == [1, 0, 0] and states[0] is not states[1]


@pytest.mark.parametrize("prefix", [(0,), (0, 1, -1), (0, 1, True)])
def test_invalid_forced_prefix_fails_before_payload_consumed(prefix):
    adapter, prefill, _, positions, _, merged = fixture()
    payload = make_payload(adapter, prefill, merged, positions, "selection_only")
    with pytest.raises(ValueError, match="forced token"):
        decode(adapter, prefill, payload, predictions=4, forced_prefix=prefix)
    assert payload.snapshot is not None


def test_comparison_preserves_registered_guard_and_reports_per_prefix_errors():
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert comparison(a, a)["registered_guard_pass"]
    b = a.clone()
    b[1] = torch.tensor([4.0, 0.0])
    result = comparison(a, b)
    assert result["first_divergent_prediction"] == 1
    assert result["token_agreement"] == 0.5 and not result["registered_guard_pass"]
    assert result["per_prefix_KL_FP64"][0] == 0
    assert result["per_prefix_KL_FP64"][1] > 0
    with pytest.raises(ValueError, match="shapes"):
        comparison(a, b[:1])


@pytest.mark.parametrize("arm", ARMS)
def test_full_membership_identity_when_no_evicted_cache_exists(arm):
    adapter, prefill, source, _, _, _ = fixture()
    positions = tuple(tuple(tuple(range(6)) for _ in range(2)) for _ in source)
    payload = make_payload(adapter, prefill, source, positions, arm)
    for layer, pair in zip(payload.snapshot.layers, source, strict=True):
        assert torch.equal(layer.key, pair[0]) and torch.equal(layer.value, pair[1])
        assert layer.key.data_ptr() != pair[0].data_ptr()


def test_analyzer_records_missing_producers_without_inventing_results(tmp_path, monkeypatch):
    import evicastkv_lookm_selection_merge as driver

    monkeypatch.setattr(
        driver.previous.common,
        "inputs",
        lambda *args: dict(jobs=[dict(index=0), dict(index=1)], source_sha="fixture"),
    )
    driver.analyze(tmp_path, "fixture_plan")
    result = driver.previous.read(tmp_path / "analysis/decision.json")
    assert not result["complete"] and len(result["failures"]) == 2
    assert result["summaries"] == []
    assert result["fresh_cohort_and_larger_campaign_blocked"]
