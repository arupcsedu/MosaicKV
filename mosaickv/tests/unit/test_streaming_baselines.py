import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mosaickv.adapters.huggingface.types import CachedKeyState, CacheLayerSnapshot, CacheSnapshot
from mosaickv.cache_state import ModalitySpan
from mosaickv.config import CacheConfig, ModelConfig, synthetic_smoke_config
from mosaickv.evaluation.exact_storage import ExactFormat, encode
from mosaickv.evaluation.request_precision import online_frozen
from mosaickv.evaluation.storage_baselines import baseline_payload
from mosaickv.evaluation.streaming_baselines import (
    SelectedEncoding,
    SelectedRetiringCache,
    StreamingBaseline,
    encode_layer,
    prefix_capacities,
    source_capacities,
    vl_capacities,
)
from mosaickv.evaluation.streaming_prefill import lossless_kv_encoder
from mosaickv.types import BudgetUnit, Precision


def fixture(tmp_path, *, length=48):
    torch.manual_seed(913)
    count, n, heads, dim = 6, length, 2, 128
    pairs = tuple(
        tuple(torch.randn(1, heads, n, dim).bfloat16() for _ in range(2)) for _ in range(count)
    )
    probabilities = tuple(
        torch.softmax(torch.randn(1, heads, n, n).tril(), -1).bfloat16() for _ in range(count)
    )
    profile = (0.7, 0.2, 0.6, 0.5, 0.3, 0.8)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    prepared = SimpleNamespace(
        logical_sequence_length=n,
        model_inputs={"input_ids": torch.arange(n)[None]},
        modality_map=(
            ModalitySpan(0, 3, "text"),
            ModalitySpan(3, n - 13, "image"),
            ModalitySpan(n - 13, n, "text"),
        ),
    )
    snapshot = CacheSnapshot(
        tuple(CacheLayerSnapshot(*p, 2) for p in pairs),
        tuple,
        "legacy",
        n,
        CachedKeyState.POST_ROPE,
    )
    adapter = SimpleNamespace(
        _legacy_layers=lambda cache: (cache, None), borrow_past_key_values=lambda cache: snapshot
    )
    prefill = SimpleNamespace(
        attention_weights=probabilities,
        state=SimpleNamespace(
            past_key_values=pairs, logical_sequence_length=n, next_decode_position=n
        ),
    )
    cfg = replace(
        synthetic_smoke_config(),
        model=ModelConfig(
            "llava-hf/llava-1.5-7b-hf", "b234b804b114d9e37bb655e11cbbb5f5e971b7a9", Precision.BF16
        ),
        cache=CacheConfig(100, BudgetUnit.BYTES, 1.0, 1),
    )
    return profile, path, pairs, probabilities, prepared, adapter, prefill, cfg


@pytest.mark.parametrize(
    "method", ["prompt_attention_topk", "prefixkv_reimpl", "vl_cache_reimpl", "lookm_reimpl"]
)
def test_local_allocation_and_actual_merges_match_frozen_global_implementation(tmp_path, method):
    from mosaickv.baselines.vl_cache import post_vision_attention_statistics

    profile, path, pairs, probabilities, prepared, adapter, pref, cfg = fixture(tmp_path)
    frozen, _ = online_frozen(
        pref,
        profile,
        0.5,
        method=method,
        allocation_mode="balanced" if method == "prompt_attention_topk" else "profile",
    )
    memberships, transformed, proof = baseline_payload(
        adapter, prepared, pref, cfg, method, frozen, str(path), borrow_source=True
    )
    expected = (
        memberships if memberships is not None else proof["selected_source_positions_by_layer"]
    )
    encoder = StreamingBaseline(prepared, SimpleNamespace(extra_tensors=()), profile, method)
    if method == "vl_cache_reimpl":
        encoder.statistics = [
            post_vision_attention_statistics((p,), kv_heads_by_layer=(2,), post_vision_start=35)
            for p in probabilities
        ]
        encoder.capacities = vl_capacities(encoder.statistics, 48, encoder.target)
    for i, pair in enumerate(pairs):
        heads, selected = encoder._heads_pair(i, pair, probabilities[i])
        assert heads == tuple(map(tuple, expected[i]))
        if transformed is not None:
            assert all(torch.equal(x, y) for x, y in zip(selected, transformed[i], strict=True))
    assert sum(encoder.capacities) == sum(source_capacities(profile, 48))
    if method == "prefixkv_reimpl":
        assert encoder.capacities == tuple(len(heads[0]) for heads in expected)


def test_explicit_selected_mapping_preserves_original_positions_and_dense_storage():
    cache = SelectedRetiringCache(1, audit=True)
    pair = tuple(torch.randn(1, 2, 16, 128).bfloat16() for _ in range(2))
    cache.update(*pair, 0)
    heads = ((0, 4, 15), (2, 8, 15))
    packet = ExactFormat(
        "kv",
        (1, 2, 3, 128),
        pair=tuple(
            encode(torch.stack([t[0, h, list(row)] for h, row in enumerate(heads)])[None])
            for t in pair
        ),
    )
    cache.seal(0, torch.nn.Linear(2, 2), lambda *a: SelectedEncoding(packet, heads))
    assert cache.formats()[0].shape == (1, 2, 3, 128)
    assert cache.get_seq_length() == 16
    assert cache.memberships[0] == heads
    assert cache.layers[0].keys is None


@pytest.mark.parametrize(
    "heads",
    [((0, 4, 14), (2, 8, 15)), ((0, 15, 4), (2, 8, 15)), ((0, 0, 15), (2, 8, 15)), ((0, 4, 15),)],
)
def test_invalid_selected_membership_never_retires_native_cache(heads):
    cache = SelectedRetiringCache(1, audit=False)
    pair = tuple(torch.randn(1, 2, 16, 128).bfloat16() for _ in range(2))
    cache.update(*pair, 0)
    packet = SimpleNamespace(shape=(1, 2, 3, 128))
    with pytest.raises(ValueError, match="mapping"):
        cache.seal(0, torch.nn.Linear(2, 2), lambda *a: SelectedEncoding(packet, heads))
    assert cache.next_layer == 0 and cache.layers[0].keys is pair[0] and not cache.memberships


def test_ordinary_encoder_cannot_silently_drop_physical_slots():
    cache = SelectedRetiringCache(1, audit=False)
    pair = tuple(torch.randn(1, 2, 16, 128).bfloat16() for _ in range(2))
    cache.update(*pair, 0)
    with pytest.raises(RuntimeError, match="physical prompt geometry"):
        cache.seal(0, torch.nn.Linear(2, 2), lambda *a: SimpleNamespace(shape=(1, 2, 3, 128)))


def test_lossless_global_planning_archive_finalizes_one_layer_without_native_alias(tmp_path):
    cache = SelectedRetiringCache(1, audit=True)
    pair = tuple(torch.randn(1, 2, 16, 128).bfloat16() for _ in range(2))
    cache.update(*pair, 0)
    attention = torch.nn.Linear(2, 2)
    cache.seal(0, attention, lossless_kv_encoder)
    original_archive = cache.formats()[0]
    heads = ((0, 4, 15), (2, 8, 15))

    def build(original):
        assert all(torch.equal(x, y) for x, y in zip(original, pair, strict=True))
        packet = ExactFormat(
            "kv",
            (1, 2, 3, 128),
            pair=tuple(
                encode(torch.stack([t[0, h, list(row)] for h, row in enumerate(heads)])[None])
                for t in original
            ),
        )
        return SelectedEncoding(packet, heads)

    cache.finalize_lossless_layer(0, attention, build)
    assert cache.formats()[0] is not original_archive
    assert cache.get_seq_length() == 16
    assert (
        cache.record()["observed_prefill_cache_peak_bytes"]
        > cache.record()["final_prefill_cache_bytes"]
    )
    with pytest.raises(RuntimeError):
        cache.finalize_lossless_layer(0, attention, build)


def test_unsupported_method_or_geometry_fails_before_model_access(tmp_path):
    profile, _, _, _, prepared, *_ = fixture(tmp_path)
    with pytest.raises(ValueError, match="four baseline"):
        StreamingBaseline(prepared, None, profile, "invented")
    with pytest.raises(NotImplementedError):
        source_capacities(profile, 47)
    assert sum(prefix_capacities(profile, 48, 100)) == 100


@pytest.mark.parametrize(
    "method", ["prompt_attention_topk", "prefixkv_reimpl", "vl_cache_reimpl", "lookm_reimpl"]
)
def test_original_global_bundle_and_local_encoder_payloads_are_identical(tmp_path, method):
    from mosaickv.baselines.vl_cache import post_vision_attention_statistics
    from mosaickv.evaluation.head_local_precision import bundle
    from mosaickv.evaluation.text_source_storage import optimize

    # The 48-token fixture protects almost all selected slots. This longer
    # fixture leaves nonzero boundary exceptions and catches a missing refit.
    profile, path, pairs, probabilities, prepared, adapter, pref, cfg = fixture(
        tmp_path, length=128
    )
    modules = [
        SimpleNamespace(
            head_dim=128,
            q_proj=torch.nn.Linear(256, 256, bias=False).bfloat16(),
            k_proj=torch.nn.Linear(256, 256, bias=False).bfloat16(),
            v_proj=torch.nn.Linear(256, 256, bias=False).bfloat16(),
        )
        for _ in pairs
    ]
    adapter._language_layers = lambda: tuple(SimpleNamespace(self_attn=a) for a in modules)
    hidden = tuple(torch.randn(1, 128, 256).bfloat16() for _ in pairs)
    phase = (torch.ones(1, 128, 128).bfloat16(), torch.zeros(1, 128, 128).bfloat16())
    capture = SimpleNamespace(
        records={
            i: dict(
                normalized=x[:, -16:].clone(),
                cos=phase[0][:, -16:].clone(),
                sin=phase[1][:, -16:].clone(),
            )
            for i, x in enumerate(hidden)
        }
    )
    encoder = StreamingBaseline(prepared, SimpleNamespace(extra_tensors=()), profile, method)
    if method == "vl_cache_reimpl":
        encoder.statistics = [
            post_vision_attention_statistics((p,), kv_heads_by_layer=(2,), post_vision_start=115)
            for p in probabilities
        ]
        encoder.capacities = vl_capacities(encoder.statistics, 128, encoder.target)
    text = encoder.text_positions
    with torch.inference_mode():
        old, proof = bundle(
            adapter,
            prepared,
            pref,
            capture,
            cfg,
            method,
            str(path),
            0.5,
            policies=("head3_boundary8",),
            allocation_mode="balanced" if method == "prompt_attention_topk" else "profile",
        )
        for i, (module, original) in enumerate(zip(modules, pairs, strict=True)):
            record = dict(
                normalized=hidden[i][:, text].clone(),
                cos=phase[0][:, text].clone(),
                sin=phase[1][:, text].clone(),
            )
            parent, _ = optimize(
                old["head3_boundary8"][i], record, proof["source_memberships"][i], text, module
            )
            heads, pair = encoder._heads_pair(i, original, probabilities[i])
            new = encode_layer(
                i,
                len(pairs),
                module,
                pair,
                heads,
                text,
                capture.records[i],
                probabilities[i],
                record,
            )
            assert len(parent.tensors()) == len(new.tensors())
            assert all(
                torch.equal(a, b) for a, b in zip(parent.tensors(), new.tensors(), strict=True)
            )


def test_temporary_vl_summaries_do_not_retain_full_attention_or_hidden_backings(
    tmp_path, monkeypatch
):
    import mosaickv.evaluation.streaming_baselines as module

    profile, _, _, _, prepared, *_ = fixture(tmp_path)
    encoder = StreamingBaseline(
        prepared, SimpleNamespace(extra_tensors=()), profile, "vl_cache_reimpl"
    )
    x = torch.randn(1, 48, 4096).bfloat16()
    probabilities = torch.randn(1, 32, 48, 48).softmax(-1).bfloat16()
    phase = (torch.ones(1, 48, 128).bfloat16(), torch.zeros(1, 48, 128).bfloat16())
    original = tuple(torch.randn(1, 32, 48, 128).bfloat16() for _ in range(2))
    monkeypatch.setattr(
        "mosaickv.evaluation.streaming_prefill.lossless_kv_encoder", lambda *a: "archive"
    )
    packet = encoder(
        0,
        None,
        original,
        (),
        dict(hidden_states=x, position_embeddings=phase),
        (None, probabilities),
    )
    assert packet == "archive"
    record, window, _ = encoder.pending[0]
    assert record["normalized"].shape == (1, 16, 4096)
    assert record["normalized"].untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    assert window.untyped_storage().data_ptr() != probabilities.untyped_storage().data_ptr()
    assert encoder.cache.extra_tensors
    assert module.source_capacities(profile, 48) == source_capacities(profile, 48)
