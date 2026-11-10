"""Synthetic-only source-control scope, protection, ownership and replay tests."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def module():
    path = Path(__file__).resolve().parents[2] / "scripts/evicastkv_review_source_controls.py"
    spec = importlib.util.spec_from_file_location("fox_source_controls_test", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


@pytest.fixture
def driver():
    return module()


@pytest.mark.parametrize("n", (1, 4, 8, 12, 17, 64, 129))
def test_optional_protection_preserves_every_mandatory_position(driver, n):
    scores = torch.rand(3, n)
    text = tuple(range(0, n, 5))
    masks, info = driver.protection_masks(scores, text, fraction=0.1)
    mandatory = masks["mandatory"]
    assert mandatory[:, list(text)].all()
    assert mandatory[:, : min(4, n)].all()
    assert mandatory[:, max(0, n - 8) :].all()
    for name in ("attention", "random"):
        assert masks[name][mandatory].all()
        assert ((masks[name] & ~mandatory).sum(-1) == info["optional_count_per_head"]).all()


def test_random_is_seeded_and_equal_cardinality_not_best_seed_selected(driver):
    scores = torch.arange(200).reshape(2, 100).float()
    first, _ = driver.protection_masks(scores, (0, 99), fraction=0.12, seed=13)
    second, _ = driver.protection_masks(scores, (0, 99), fraction=0.12, seed=13)
    third, _ = driver.protection_masks(scores, (0, 99), fraction=0.12, seed=14)
    assert all(torch.equal(first[k], second[k]) for k in first)
    assert not torch.equal(first["random"], third["random"])
    assert torch.equal(first["random"].sum(-1), first["attention"].sum(-1))


def test_attention_ties_use_original_position_not_random_order(driver):
    masks, info = driver.protection_masks(torch.ones(2, 80), (), fraction=0.1)
    optional = masks["attention"] & ~masks["mandatory"]
    assert info["optional_count_per_head"] == 8
    assert optional[0].nonzero().flatten().tolist() == list(range(4, 12))


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -0.1, 1.1))
def test_reject_invalid_fraction(driver, bad):
    with pytest.raises(ValueError, match="fraction"):
        driver.protection_masks(torch.ones(2, 20), (), fraction=bad)


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), -1.0))
def test_reject_nonfinite_or_negative_attention(driver, bad):
    scores = torch.ones(2, 20)
    scores[0, 5] = bad
    with pytest.raises(ValueError, match="finite"):
        driver.protection_masks(scores, ())


def test_direct3bit_is_local_and_exact_overlay_is_per_head(driver):
    torch.manual_seed(0)
    pair = tuple(torch.randn(1, 3, 24, 128).bfloat16() for _ in range(2))
    scores = torch.rand(3, 24)
    masks, _ = driver.protection_masks(scores, (7, 10), fraction=0.2)
    mask = masks["attention"]
    fmt = driver.direct_projected(pair, mask)
    assert [body.bits for body in fmt.base.pair] == [3, 3]
    assert [body.axis for body in fmt.base.pair] == [2, 3]
    for actual, expected in zip(fmt.materialize(), pair, strict=True):
        assert actual.shape == expected.shape
        assert torch.equal(actual[0][mask].view(torch.int16), expected[0][mask].view(torch.int16))
        assert torch.isfinite(actual).all()
    assert all(t.device.type == "cpu" for t in fmt.tensors())
    from mosaickv.evaluation.lossless_prompt_cache import backing_bytes

    assert fmt.component_bytes == backing_bytes(fmt.tensors())
    assert fmt.component_bytes > sum(b.codes.numel() for b in fmt.base.pair)


def test_direct_does_not_mutate_original_pair(driver):
    pair = tuple(torch.randn(1, 2, 16, 128).bfloat16() for _ in range(2))
    before = [t.clone() for t in pair]
    masks, _ = driver.protection_masks(torch.ones(2, 16), ())
    driver.direct_projected(pair, masks["attention"]).materialize()
    assert all(torch.equal(a, b) for a, b in zip(pair, before, strict=True))


def test_replay_accepts_only_identical_representation_and_retired_owners(driver):
    storage = dict(
        graph_private_live_bytes=0, active_workspace_bytes=0, graph_private_accounting_verified=True
    )
    off = dict(
        arm="source3_attention_optional",
        representation_sha256="abc",
        replay_executed=False,
        storage=storage,
    )
    on = dict(
        arm="source3_attention_optional_replay_on",
        representation_sha256="abc",
        replay_executed=True,
        storage=storage,
    )
    assert driver.replay_check(off, on)["actual_replay_engaged"]
    for key, value in (("representation_sha256", "changed"), ("replay_executed", False)):
        bad = copy.deepcopy(on)
        bad[key] = value
        with pytest.raises(ValueError):
            driver.replay_check(off, bad)
    for key, value in (
        ("graph_private_live_bytes", 1),
        ("active_workspace_bytes", 1),
        ("graph_private_accounting_verified", False),
    ):
        bad = copy.deepcopy(on)
        bad["storage"][key] = value
        with pytest.raises(ValueError, match="owner"):
            driver.replay_check(off, bad)


def test_inputs_are_consumed_and_labels_are_rejected(driver, tmp_path):
    paths = [tmp_path / f"{i}.json" for i in range(2)]
    for i, path in enumerate(paths):
        path.write_text(json.dumps(dict(task="x", sample_id=str(i), prompt="hello", media=[])))
    parent = dict(jobs=[dict(inputs=[str(path)]) for path in paths])
    assert len(driver.consumed_inputs(parent)) == 2
    raw = json.loads(paths[0].read_text())
    raw["answer"] = "forbidden"
    paths[0].write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="prohibit"):
        driver.consumed_inputs(parent)


def test_representation_hash_detects_dtype_shape_and_payload(driver):
    data = torch.arange(8).bfloat16()
    packet = SimpleNamespace(kind="toy", shape=(1, 1, 1, 8), tensors=lambda: (data,))
    old = driver._packet_hash((packet,))
    data[0] = 10
    assert driver._packet_hash((packet,)) != old
    packet.kind = "changed"
    assert driver._packet_hash((packet,)) != old


def test_policy_explicitly_keeps_historical_source_and_exact_separate(driver):
    assert driver.MODEL == ("llava-hf/llava-1.5-7b-hf", "b234b804b114d9e37bb655e11cbbb5f5e971b7a9")
    assert "not_frozen" in driver.POLICY["id"]
    assert driver.POLICY["guard"] == dict(
        tokens=32, token_agreement=1.0, maximum_KL_FP32=0.02, logit_difference=1.0
    )
    assert len(driver.ARMS) == 7
    assert all("KIVI" not in arm for arm in driver.ARMS)


def test_inventory_never_calls_unverified_ports_official(driver):
    rows = driver.baseline_inventory()
    assert {r["method"] for r in rows} == {"prefixkv_reimpl", "lookm_reimpl", "vl_cache_reimpl"}
    assert not any(r["official_fidelity_for_this_new_diagnostic"] for r in rows)
    assert all(r["evidence_sha256"] for r in rows)
