"""Synthetic scope/control/config checks; no real samples or scheduler calls."""

import json
from dataclasses import replace

import evicastkv_multiscale_prompt as module
import pytest
from test_joint_lifetime_controller import parent_fixture

from mosaickv.config import RunConfig, canonical_config, config_sha256
from mosaickv.evaluation.fox_campaign import sha


@pytest.mark.parametrize("prior_pass", [False, True])
def test_shared_prior_plan_binds_current_controls_and_preserves_legacy_hashes(
    tmp_path, monkeypatch, prior_pass
):
    parent, _profile = parent_fixture(tmp_path)
    old = module.common.read(parent / "plan.json")
    source = [j for j in old["jobs"] if j.get("mandatory_policy") == "method"]
    jobs = []
    for j in source:
        for history in (128, 256, 512):
            cfg = RunConfig.from_mapping(j["arms"][0]["config"])
            cfg = replace(
                cfg, bounded_decode=replace(cfg.bounded_decode, generated_capacity=history)
            )
            label = f"fixture-{cfg.method.value}-{history}"
            job = dict(
                j,
                index=len(jobs),
                diagnostic_history_capacity=history,
                arms=[
                    dict(label=label, config=canonical_config(cfg), config_sha=config_sha256(cfg))
                ],
            )
            row = parent / f"workers/{job['index']}/rows/{label}.json"
            row.parent.mkdir(parents=True, exist_ok=True)
            row.write_text(json.dumps(dict(status="completed", traces={}, memory_audit_traces={})))
            jobs.append(job)
    (parent / "plan.json").write_text(json.dumps(dict(old, jobs=jobs)))
    (parent / "analysis/GATE.json").write_text(
        json.dumps(
            dict(
                status="complete_consumed_diagnostic",
                failures=[],
                point_eligible_history_capacities=[],
            )
        )
    )
    digest = sha(parent / "plan.json")
    equiv = tmp_path / "equiv"
    (equiv / "analysis").mkdir(parents=True)
    (equiv / "analysis/GATE.json").write_text(
        json.dumps(dict(status="passed_equivalence_and_allocator_reduction"))
    )
    (equiv / "plan.json").write_text(
        json.dumps(dict(bound_files={str(parent / "plan.json"): digest}))
    )
    prior = tmp_path / "prior"
    (prior / "analysis").mkdir(parents=True)
    (prior / "plan.json").write_text("{}")
    (prior / "analysis/GATE.json").write_text(
        json.dumps(
            dict(
                status="complete_consumed_diagnostic",
                failures=[],
                point_eligible_configurations=["fixture"] if prior_pass else [],
            )
        )
    )
    monkeypatch.setattr(module, "clean_source", lambda *_: "fixture-source")
    out = tmp_path / "out"
    if prior_pass:
        with pytest.raises(ValueError, match="negative"):
            module.prepare(out, parent, digest, equiv, "shared_heads", prior)
        assert not out.exists()
        return
    module.prepare(out, parent, digest, equiv, "shared_heads", prior)
    plan = module.common.read(out / "plan.json")
    assert len(plan["jobs"]) == 24
    assert plan["contemporaneous_window_topk_controls"]
    zero = [j for j in plan["jobs"] if j["diagnostic_full_prompt_prior_weight"] == 0]
    assert len(zero) == 6 and all(
        j["arms"][0]["config"]["method"] == "prompt_attention_topk" for j in zero
    )
    for j in plan["jobs"]:
        arm = j["arms"][0]
        cfg = RunConfig.from_mapping(arm["config"])
        assert config_sha256(cfg) == arm["config_sha"]
        h = cfg.bounded_decode.incremental_history
        if h.full_prompt_prior_weight == 0:
            assert (
                "full_prompt_prior_head_pooling"
                not in arm["config"]["bounded_decode"]["incremental_history"]
            )
        else:
            assert h.full_prompt_prior_head_pooling == "shared_heads"
    assert not plan["fresh_evaluation_authorized"] and not plan["paper_campaign_authorized"]
