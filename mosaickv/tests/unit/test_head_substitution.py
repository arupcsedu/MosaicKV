"""Synthetic CPU invariants, never pretrained answer-quality evidence."""

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from mosaickv.adapters.huggingface.types import CachedKeyState, CacheLayerSnapshot, CacheSnapshot
from mosaickv.backends.hf_runtime import PackedRuntimeCache
from mosaickv.cache_state import FullKVState
from mosaickv.config import DraftAllocationGuardConfig
from mosaickv.selection.draft_shortlist import draft_shortlist, validate_shortlist_trace
from mosaickv.selection.head_substitution import pack_substitution, substitution_pool


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture():
    layers = tuple(
        (torch.arange(48).reshape(1, 2, 6, 4).float() + i * 100, torch.ones(1, 2, 6, 4) * i)
        for i in range(2)
    )
    logical = (10, 20, 30, 40, 50, 60)
    full = FullKVState.from_tensors(
        layers,
        block_size=1,
        logical_positions=logical,
        mandatory_logical_positions=(60,),
        original_logical_sequence_length=61,
        next_decode_position=61,
    )
    snapshot = CacheSnapshot(
        tuple(
            CacheLayerSnapshot(k[:, :, [0, 2, 5]].clone(), v[:, :, [0, 2, 5]].clone(), 2)
            for k, v in layers
        ),
        tuple,
        "tuple",
        3,
        CachedKeyState.POST_ROPE,
    )
    records = tuple(
        dict(layer=i, kv_head=h, slot=s, logical_position=logical[p], tier="exact")
        for i in range(2)
        for h in range(2)
        for s, p in enumerate((0, 2, 5))
    )
    anchor = PackedRuntimeCache(snapshot, (), 3, True, (), (), records)
    attention = {
        (i, h): np.array([0.1, 0.3, 0.05, 0.15, 0.2, 0.2]) for i in range(2) for h in range(2)
    }
    return full, anchor, attention


def test_one_head_only_exact_values_positions_capacity_mandatory_and_alias_isolation():
    full, anchor, attention = fixture()
    cap = full.active_bytes // 2
    pool = substitution_pool(full, attention, anchor, prompt_bytes=cap, limit=4)
    assert pool == substitution_pool(full, attention, anchor, prompt_bytes=cap, limit=4)
    assert len(pool) == 4
    source = [(layer.key.clone(), layer.value.clone()) for layer in anchor.snapshot.layers]
    p = pool[0]
    changed = pack_substitution(full, anchor, p)
    for i, layer in enumerate(changed.snapshot.layers):
        assert layer.key.shape == anchor.snapshot.layers[i].key.shape
        for h in range(2):
            if (i, h) == (p.layer, p.head):
                assert not torch.equal(layer.key[:, h], source[i][0][:, h])
            else:
                assert torch.equal(layer.key[:, h], source[i][0][:, h])
        if i != p.layer:
            assert layer.key is anchor.snapshot.layers[i].key
    assert all(
        torch.equal(k, layer.key) and torch.equal(v, layer.value)
        for (k, v), layer in zip(source, anchor.snapshot.layers, strict=True)
    )
    rows = sorted(
        (r for r in changed.slot_records if (r["layer"], r["kv_head"]) == (p.layer, p.head)),
        key=lambda r: r["slot"],
    )
    assert 60 in [r["logical_position"] for r in rows]
    assert [r["logical_position"] for r in rows] == sorted(r["logical_position"] for r in rows)
    for row in rows:
        assert torch.equal(
            changed.snapshot.layers[p.layer].key[0, p.head, row["slot"]],
            full.layers[p.layer].key[0, p.head, row["physical_position"]],
        )
    with pytest.raises(ValueError, match="mandatory"):
        pack_substitution(full, anchor, replace(p, removed_position=5))
    with pytest.raises(ValueError, match="metadata"):
        pack_substitution(full, anchor, replace(p, replaced_bytes=p.replaced_bytes + 1))


def row(step, kl=0.05):
    return dict(
        step=step,
        anchor=dict(logit_kl=0.1, maximum_logit_difference=0.5, greedy_token_agreement=True),
        candidate=dict(logit_kl=kl, maximum_logit_difference=0.4, greedy_token_agreement=True),
    )


def test_actual_logit_shortlist_can_choose_last_pool_item_and_reject_late_failure():
    calls, closed = [], []

    def probe(name):
        try:
            for step in range(1, 33):
                calls.append((name, step))
                yield row(
                    step,
                    0.2
                    if name == "best_early" and step == 16
                    else 0.01
                    if name == "best_early"
                    else 0.03
                    if name == "last"
                    else 0.08,
                )
        finally:
            closed.append(name)

    result = draft_shortlist(
        tuple((name, name) for name in ("first", "best_early", "last")),
        probe,
        DraftAllocationGuardConfig(),
    )
    assert result["shortlist"] == ["best_early", "last"]
    assert result["winner"] == "last" and result["meaningful"]
    assert max(step for name, step in calls if name == "first") == 4
    assert max(step for name, step in calls if name == "best_early") == 16
    assert max(step for name, step in calls if name == "last") == 32
    assert sorted(closed) == ["best_early", "first", "last"]
    validate_shortlist_trace(result)
    bad = copy.deepcopy(result)
    bad["shortlist"] = ["first", "last"]
    with pytest.raises(ValueError, match="actual draft"):
        validate_shortlist_trace(bad)
    bad = copy.deepcopy(result)
    bad["events"][-1]["steps"].pop()
    with pytest.raises(ValueError, match="entire registered horizon"):
        validate_shortlist_trace(bad)


def test_ties_are_not_meaningful_and_rejections_fail_fast():
    def tie(_):
        for step in range(1, 33):
            r = row(step)
            r["candidate"] = r["anchor"].copy()
            yield r

    result = draft_shortlist((("tie", None),), tie, DraftAllocationGuardConfig())
    assert result["accepted"] and not result["meaningful"]

    def rejected(_):
        for step in range(1, 33):
            yield row(step, 0.2)

    result = draft_shortlist((("bad", None),), rejected, DraftAllocationGuardConfig())
    assert not result["accepted"] and result["candidate_updates"] == 1
    validate_shortlist_trace(result)


@pytest.mark.parametrize("count", [0, 17])
def test_bounded_pool(count):
    with pytest.raises(ValueError, match="bounded"):
        draft_shortlist(
            tuple((str(i), None) for i in range(count)),
            lambda _: iter(()),
            DraftAllocationGuardConfig(),
        )


def test_diagnostic_launcher_redacts_annotations_and_keeps_scope_disabled():
    import importlib.util
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[2] / "scripts/head_substitution_v7.py"
    spec = importlib.util.spec_from_file_location("head_v7_tooling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sample = dict(
        sample_id="synthetic",
        prompt="question",
        references=["forbidden"],
        teacher_forcing_target="forbidden",
        score=1.0,
        media=[dict(kind="image", path="/scratch/fake.jpg", sha256="a" * 64, answer="forbidden")],
    )
    redacted = module.redact(sample)
    assert set(redacted) == {"sample_id", "prompt", "media"}
    assert set(redacted["media"][0]) == {"kind", "path", "sha256"}
    protocol = yaml.safe_load(module.PROTOCOL.read_text())
    assert protocol["samples_per_task"] == 1 and protocol["workers"] == 2
    assert not protocol["fresh_cohort_authorized"] and not protocol["calibration_authorized"]
    assert not protocol["paper_campaign_authorized"]
    assert protocol["diagnostic"]["guard"]["horizon"] == 32
    assert protocol["diagnostic"]["guard"]["absolute_tolerance"] == 1e-7
    assert protocol["diagnostic"]["guard"]["relative_tolerance"] == 1e-6
