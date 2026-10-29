"""Synthetic orchestration tests: no real data access or scheduler calls."""

import json
import math
from dataclasses import replace

import evicastkv_joint_lifetime as module
import pytest

from mosaickv.config import (
    BoundedDecodeConfig,
    CacheConfig,
    DatasetConfig,
    ExecutionConfig,
    ForecastingConfig,
    GenerationConfig,
    IncrementalHistoryConfig,
    ModelConfig,
    PrototypeConfig,
    RepairConfig,
    ResidualConfig,
    RunConfig,
    canonical_config,
    config_sha256,
)
from mosaickv.evaluation.fox_campaign import sha
from mosaickv.types import (
    Backend,
    BudgetUnit,
    ForecastMode,
    MosaicKVMethod,
    Precision,
    RepairPolicy,
)


def parent_fixture(tmp_path):
    parent = tmp_path / "parent"
    (parent / "analysis").mkdir(parents=True)
    (parent / "analysis/GATE.json").write_text(
        json.dumps(
            dict(status="complete_consumed_diagnostic", failures=[], point_eligible_policies=[])
        )
    )
    profile = tmp_path / "profile.json"
    profile.write_text("[0.1,0.8]")
    base = RunConfig(
        model=ModelConfig("llava-hf/llava-1.5-7b-hf", "a" * 40, Precision.BF16),
        dataset=DatasetConfig("fixture-task", "b" * 40, "development"),
        generation=GenerationConfig(max_new_tokens=1024),
        execution=ExecutionConfig(backend=Backend.HUGGINGFACE, attention_implementation="eager"),
        cache=CacheConfig(
            block_size=1,
            physical_budget_block_size=1,
            budget_unit=BudgetUnit.BYTES,
            budget_value=208 * 2**20,
            retention_ratio=math.nextafter(1.0, 0.0),
        ),
        forecasting=ForecastingConfig(mode=ForecastMode.PROMPT_WINDOW, draft_steps=0),
        method=MosaicKVMethod.MOSAICKV_EXACT,
        prototypes=PrototypeConfig(enabled=False),
        residual=ResidualConfig(enabled=False),
        repair=RepairConfig(enabled=False, policy=RepairPolicy.NONE),
        bounded_decode=BoundedDecodeConfig(
            128,
            4,
            incremental_history=IncrementalHistoryConfig(
                policy="sink_recent",
                fixed_prompt_reservation=True,
                layer_profile_path=str(profile),
                layer_profile_sha256=sha(profile),
            ),
        ),
    )
    jobs = []
    for task in ("llava_description", "mmvet_prefixkv"):
        index = len(jobs)
        path = parent / f"workers/{index}/rows/prefixkv_reimpl-native-control.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(traces={}, memory_audit_traces={})))
        jobs.append(
            dict(
                index=index,
                task=task,
                sample_id=f"consumed-{task}",
                measurement_role="remeasured_native_control",
                arms=[dict(config={})],
            )
        )
        for policy in ("method", "question_and_boundaries"):
            for method in (MosaicKVMethod.PROMPT_ATTENTION_TOPK, MosaicKVMethod.MOSAICKV_EXACT):
                cfg = replace(
                    base,
                    method=method,
                    forecasting=replace(base.forecasting, enabled=method.is_mosaickv),
                    graph=replace(base.graph, enabled=method.is_mosaickv),
                    selection=replace(base.selection, enabled=method.is_mosaickv),
                )
                jobs.append(
                    dict(
                        index=len(jobs),
                        task=task,
                        sample_id=f"consumed-{task}",
                        mandatory_policy=policy,
                        arms=[
                            dict(
                                label=f"{method.value}-{policy}-v3",
                                config=canonical_config(cfg),
                                config_sha=config_sha256(cfg),
                            )
                        ],
                    )
                )
    (parent / "plan.json").write_text(
        json.dumps(
            dict(
                source_sha="fixture-parent-sha", bound_files={str(profile): sha(profile)}, jobs=jobs
            )
        )
    )
    return parent, profile


def test_joint_plan_preserves_consumed_scope_and_reuses_bound_controls(tmp_path, monkeypatch):
    parent, profile = parent_fixture(tmp_path)
    monkeypatch.setattr(module, "clean_source", lambda *_: "fixture-source-sha")
    out = tmp_path / "out"
    digest = module.prepare(out, parent, sha(parent / "plan.json"))
    assert digest == sha(out / "plan.json")
    plan = module.common.read(out / "plan.json")
    assert len(plan["jobs"]) == 8
    assert not plan["fresh_evaluation_authorized"] and not plan["paper_campaign_authorized"]
    assert plan["native_control_producer_sha"] == "fixture-parent-sha"
    assert plan["bound_files"][str(profile)] == sha(profile)
    for job in plan["jobs"]:
        assert job["sample_id"] == f"consumed-{job['task']}"
        assert (
            sha(module.Path(job["native_control_row"]))
            == plan["bound_files"][job["native_control_row"]]
        )
        arm = job["arms"][0]
        cfg = RunConfig.from_mapping(arm["config"])
        assert config_sha256(cfg) == arm["config_sha"]
        assert cfg.bounded_decode.incremental_history.history_allocation == "profile_weighted"
        assert job["retention_one_config"]["method"] == "mosaickv_exact"


@pytest.mark.parametrize(
    "reason", ["prior_pass", "prior_failure", "changed_profile", "changed_plan"]
)
def test_joint_plan_fails_closed(tmp_path, monkeypatch, reason):
    parent, profile = parent_fixture(tmp_path)
    monkeypatch.setattr(module, "clean_source", lambda *_: "fixture-source-sha")
    digest = sha(parent / "plan.json")
    if reason in {"prior_pass", "prior_failure"}:
        gate = module.common.read(parent / "analysis/GATE.json")
        gate["point_eligible_policies" if reason == "prior_pass" else "failures"] = ["fixture"]
        (parent / "analysis/GATE.json").write_text(json.dumps(gate))
    elif reason == "changed_profile":
        profile.write_text("[0.2,0.8]")
    else:
        digest = "wrong"
    with pytest.raises(ValueError):
        module.prepare(tmp_path / "out", parent, digest)
    assert not (tmp_path / "out").exists()
