import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mosaickv.adapters.huggingface.types import CachedKeyState, CacheLayerSnapshot, CacheSnapshot
from mosaickv.cache_state import Modality, ModalitySpan
from mosaickv.config import synthetic_smoke_config
from mosaickv.evaluation.storage_baselines import baseline_payload


@pytest.mark.parametrize("method", ["prefixkv_reimpl", "vl_cache_reimpl", "lookm_reimpl"])
def test_readonly_borrow_preserves_positions_merged_values_and_source(method, tmp_path):
    g = torch.Generator().manual_seed(7)
    source = tuple(
        tuple(torch.randn(1, 2, 20, 4, generator=g).bfloat16() for _ in range(2)) for _ in range(2)
    )
    originals = tuple(tuple(t.clone() for t in pair) for pair in source)
    calls = []

    def snapshot(copy):
        calls.append("clone" if copy else "borrow")
        return CacheSnapshot(
            tuple(
                CacheLayerSnapshot(k.clone() if copy else k, v.clone() if copy else v, 2)
                for k, v in source
            ),
            tuple,
            "legacy",
            20,
            CachedKeyState.POST_ROPE,
        )

    adapter = SimpleNamespace(
        _legacy_layers=lambda cache: (source, "legacy"),
        extract_past_key_values=lambda cache: snapshot(True),
        borrow_past_key_values=lambda cache: snapshot(False),
    )
    p = (
        torch.rand(1, 2, 20, 20, generator=g)
        .masked_fill(torch.ones(20, 20).triu(1).bool(), -1000)
        .softmax(-1)
    )
    prefill = SimpleNamespace(
        state=SimpleNamespace(
            past_key_values="owned", logical_sequence_length=20, next_decode_position=20
        ),
        attention_weights=(p, p),
    )
    prepared = SimpleNamespace(
        modality_map=(
            ModalitySpan(0, 1, Modality.TEXT),
            ModalitySpan(1, 10, Modality.IMAGE, image_index=0),
            ModalitySpan(10, 20, Modality.TEXT),
        ),
        model_inputs={"input_ids": torch.arange(20)[None]},
    )
    cfg = synthetic_smoke_config()
    cfg = replace(cfg, cache=replace(cfg.cache, block_size=1, physical_budget_block_size=1))
    frozen = {"uniform_0.5_control": {"anchor": {"positions": ((tuple(range(12)),) * 2,) * 2}}}
    path = tmp_path / "profile.json"
    path.write_text(json.dumps([0.5, 0.5]))
    cloned = baseline_payload(adapter, prepared, prefill, cfg, method, frozen, str(path))
    borrowed = baseline_payload(
        adapter, prepared, prefill, cfg, method, frozen, str(path), borrow_source=True
    )
    assert calls == ["clone", "borrow"]
    assert cloned[0] == borrowed[0]
    if cloned[1] is not None:
        assert all(
            torch.equal(a, b)
            for old, new in zip(cloned[1], borrowed[1], strict=True)
            for a, b in zip(old, new, strict=True)
        )
    assert borrowed[2]["read_only_source_views_no_redundant_snapshot_copy"]
    assert all(
        torch.equal(a, b)
        for old, new in zip(originals, source, strict=True)
        for a, b in zip(old, new, strict=True)
    )
