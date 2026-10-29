"""Synthetic layer schedules, hash compatibility and sideband capture controls."""

import json
from dataclasses import replace

import evicastkv_layerwise_prior as controller
import pytest
import torch
from test_joint_lifetime_controller import parent_fixture
from test_multiscale_prompt_capture import AttentionFixture

from mosaickv.adapters.huggingface.base import AttentionCapture
from mosaickv.config import (
    ConfigurationError,
    IncrementalHistoryConfig,
    RunConfig,
    canonical_config,
    config_sha256,
)
from mosaickv.evaluation.fox_campaign import sha
from mosaickv.measurements.selection_fingerprints import selection_fingerprints


def test_four_fixed_distinct_balanced_schedules_depend_only_on_profile():
    profile = (0.3, 0.1, 0.4, 0.2)
    schedules = controller.layer_schedules(profile)
    assert tuple(schedules) == controller.POLICIES
    assert len(set(schedules.values())) == 4
    assert schedules["profile_high_global"] == (0.5, 1, 0.5, 1)
    for weights in schedules.values():
        assert weights.count(1) == weights.count(0.5) == 2
    assert controller.layer_schedules(profile) == schedules


@pytest.mark.parametrize("value", [None, "0.5", [True], [float("nan")], [-0.1], [1.1]])
def test_mapping_rejects_malformed_layer_weights(value):
    with pytest.raises(ConfigurationError, match="layer_weights"):
        IncrementalHistoryConfig.from_mapping(
            dict(full_prompt_prior_weight=0.5, full_prompt_prior_layer_weights=value)
        )


def test_default_hashes_unchanged_and_vectors_roundtrip(tmp_path):
    parent, _ = parent_fixture(tmp_path)
    cfg = RunConfig.from_mapping(
        controller.common.read(parent / "plan.json")["jobs"][1]["arms"][0]["config"]
    )
    before = canonical_config(cfg)
    assert "full_prompt_prior_layer_weights" not in before["bounded_decode"]["incremental_history"]
    assert config_sha256(cfg) == config_sha256(RunConfig.from_mapping(before))
    mixed = replace(
        cfg,
        bounded_decode=replace(
            cfg.bounded_decode,
            incremental_history=replace(
                cfg.bounded_decode.incremental_history,
                full_prompt_prior_weight=0.5,
                full_prompt_prior_head_pooling="shared_heads",
                full_prompt_prior_layer_weights=(0.5, 1),
            ),
        ),
    )
    assert config_sha256(mixed) == config_sha256(RunConfig.from_mapping(canonical_config(mixed)))
    assert config_sha256(mixed) != config_sha256(cfg)


def test_different_weights_apply_only_to_private_evidence():
    modules = [AttentionFixture(), AttentionFixture()]
    raw = torch.arange(1, 65, dtype=torch.float32).reshape(1, 4, 4, 4)
    raw = raw / raw.sum(-1, keepdim=True)
    original = raw.clone()
    hidden = torch.ones(1, 4)
    with AttentionCapture(
        modules,
        query_window=2,
        full_prompt_prior_weight=0.5,
        full_prompt_prior_head_pooling="shared_heads",
        full_prompt_prior_layer_weights=(0.5, 1),
    ) as c:
        for module in modules:
            output = module(hidden, raw)
            assert output[0] is hidden and output[1] is raw
    prior = raw.mean(1, keepdim=True).float().mean(-2, keepdim=True)
    for actual, weight in zip(c.result(), (0.5, 1), strict=True):
        assert torch.equal(
            actual, ((1 - weight) * raw[..., -2:, :] + weight * prior).expand_as(actual)
        )
    assert torch.equal(raw, original)
    assert not torch.equal(*c.result())


@pytest.mark.parametrize("weights", [(0.5,), (True, 1), (0.5, float("nan")), (-1, 1)])
def test_capture_fails_closed_on_incomplete_or_invalid_layer_vector(weights):
    with pytest.raises(ValueError, match="layer prior"):
        AttentionCapture(
            [AttentionFixture(), AttentionFixture()],
            query_window=2,
            full_prompt_prior_weight=0.5,
            full_prompt_prior_layer_weights=weights,
        )


def test_position_fingerprints_localize_substitution_without_storing_kv():
    records = [
        dict(layer=layer, kv_head=head, slot=slot, logical_position=slot, tier="exact")
        for layer in range(2)
        for head in range(2)
        for slot in range(3)
    ]
    before = selection_fingerprints(records)
    records[-1] = dict(records[-1], logical_position=5)
    after = selection_fingerprints(records)
    assert before[:-1] == after[:-1]
    assert before[-1]["sha256"] != after[-1]["sha256"]
    assert before[-1]["positions"] == after[-1]["positions"] == 3
    assert all(set(r) == {"layer", "kv_head", "positions", "sha256"} for r in after)


def test_no_synthetic_prototype_fingerprint_allowed():
    with pytest.raises(ValueError, match="exact"):
        selection_fingerprints(
            [dict(layer=0, kv_head=0, slot=0, logical_position=1, tier="prototype")]
        )


@pytest.mark.parametrize("passing_parent", [False, True])
def test_controller_is_bounded_and_rejects_promotion_of_passing_parent(
    tmp_path, monkeypatch, passing_parent
):
    parent, profile = parent_fixture(tmp_path)
    profile.write_text("[0.3,0.1,0.4,0.2]")
    old = controller.common.read(parent / "plan.json")
    jobs = []
    for template in old["jobs"]:
        if template.get("mandatory_policy") != "method":
            continue
        cfg = RunConfig.from_mapping(template["arms"][0]["config"])
        weights = (0.5, 1.0) if cfg.method.is_mosaickv else (0.0,)
        for weight in weights:
            h = cfg.bounded_decode.incremental_history
            candidate = replace(
                cfg,
                cache=replace(cfg.cache, budget_value=289 * 2**20),
                bounded_decode=replace(
                    cfg.bounded_decode,
                    generated_capacity=256,
                    incremental_history=replace(
                        h,
                        layer_profile_sha256=sha(profile),
                        full_prompt_prior_weight=weight,
                        full_prompt_prior_head_pooling="shared_heads" if weight else "per_head",
                    ),
                ),
            )
            label = f"fixture-{candidate.method.value}-{weight}"
            job = dict(
                template,
                index=len(jobs),
                diagnostic_history_capacity=256,
                diagnostic_full_prompt_prior_weight=weight,
                arms=[
                    dict(
                        label=label,
                        config=canonical_config(candidate),
                        config_sha=config_sha256(candidate),
                    )
                ],
            )
            row = parent / f"workers/{job['index']}/rows/{label}.json"
            row.parent.mkdir(parents=True, exist_ok=True)
            row.write_text(json.dumps(dict(status="completed", traces={}, memory_audit_traces={})))
            jobs.append(job)
    (parent / "plan.json").write_text(
        json.dumps(dict(old, bound_files={str(profile): sha(profile)}, jobs=jobs))
    )
    (parent / "analysis/GATE.json").write_text(
        json.dumps(
            dict(
                status="complete_consumed_diagnostic",
                failures=[],
                point_eligible_configurations=["passing"] if passing_parent else [],
            )
        )
    )
    monkeypatch.setattr(controller, "clean_source", lambda *_: "fixture-clean-sha")
    monkeypatch.setattr(controller, "audit_saved", lambda *_: dict(label_blind=True))
    output = tmp_path / "successor"
    if passing_parent:
        with pytest.raises(ValueError, match="negative"):
            controller.prepare(output, parent, sha(parent / "plan.json"))
        assert not output.exists()
        return
    digest = controller.prepare(output, parent, sha(parent / "plan.json"))
    assert digest == sha(output / "plan.json")
    plan = controller.common.read(output / "plan.json")
    assert len(plan["jobs"]) == 14
    assert not plan["fresh_evaluation_authorized"] and not plan["paper_campaign_authorized"]
    assert plan["capture_selection_fingerprints"]
    for job in plan["jobs"]:
        cfg = RunConfig.from_mapping(job["arms"][0]["config"])
        assert cfg.cache.budget_value == 289 * 2**20
        assert cfg.bounded_decode.generated_capacity == 256
        assert config_sha256(cfg) == job["arms"][0]["config_sha"]
        if job["diagnostic_layer_policy"] in controller.POLICIES:
            assert len(cfg.bounded_decode.incremental_history.full_prompt_prior_layer_weights) == 4
            assert job["equivalence_control_row"] is None
        else:
            assert not cfg.bounded_decode.incremental_history.full_prompt_prior_layer_weights
            assert job["equivalence_control_row"]
