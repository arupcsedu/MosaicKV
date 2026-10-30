"""Random-weight CPU baseline plumbing; synthetic modality metadata only."""

from dataclasses import replace

import pytest
import torch
from test_hf_tiny_models import _baseline_runtime_config, _llava, _runtime_request

from mosaickv.cache_state import Modality, ModalitySpan
from mosaickv.evaluation.storage_baselines import baseline_payload, ordinary_pairs_reference
from mosaickv.types import MosaicKVMethod


@pytest.mark.parametrize("method", ["prefixkv_reimpl", "vl_cache_reimpl", "lookm_reimpl"])
def test_published_adaptations_spend_exact_source_slots_and_keep_merged_values(tmp_path, method):
    torch.manual_seed(14)
    adapter = _llava(kv_heads=2)
    adapter.model.to(torch.bfloat16)
    prepared = adapter.prepare_inputs(_runtime_request(method).messages)
    prepared.modality_map = (
        ModalitySpan(0, 1, Modality.TEXT),
        ModalitySpan(1, 3, Modality.IMAGE, image_index=0),
        ModalitySpan(3, 4, Modality.TEXT),
    )
    prefill = adapter.prefill(prepared, capture_queries=False, capture_attentions=True)
    cfg = replace(
        _baseline_runtime_config(MosaicKVMethod.PROMPT_ATTENTION_TOPK, 0.5),
        generation=replace(
            _baseline_runtime_config(MosaicKVMethod.PROMPT_ATTENTION_TOPK, 0.5).generation,
            max_new_tokens=32,
        ),
    )
    profile = tmp_path / "profile.json"
    profile.write_text("[0.5,0.5]")
    positions = (((0, 3), (1, 3)),) * 2
    frozen = {
        r: {"anchor": {"positions": positions}}
        for r in ("window_topk_control", "uniform_0.5_control")
    }
    membership, pairs, proof = baseline_payload(
        adapter, prepared, prefill, cfg, method, frozen, str(profile)
    )
    assert proof["actual_source_slot_bytes"] == proof["source_slot_byte_cap"]
    assert proof["append_only"]
    assert not proof["paper_native_reproduction"]
    if method == "lookm_reimpl":
        assert membership is None and pairs is not None
        assert proof["merged_KV_preserved"]
        assert not proof["source_regeneration_eligible"]
        before = tuple((k.clone(), v.clone()) for k, v in pairs)
        reference = ordinary_pairs_reference(adapter, prefill, pairs, tokens=16)
        assert reference.shape == (16, 64)
        for (k, v), (a, b) in zip(pairs, before, strict=True):
            assert torch.equal(k, a) and torch.equal(v, b)
    else:
        assert pairs is None
        assert all(3 in head for layer in membership for head in layer)
