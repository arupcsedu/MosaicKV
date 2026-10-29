"""Synthetic orchestration fixtures never submit jobs or load model weights."""

import json
import math
from dataclasses import replace

import evicastkv_profile_lifetime as module
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


@pytest.mark.parametrize(
    "state,eligible", [("failed", []), ("complete_consumed_diagnostic", ["prior"])]
)
def test_controller_never_submits_on_prior_failure_or_advance(
    tmp_path, monkeypatch, state, eligible
):
    parent = tmp_path / "parent"
    (parent / "analysis").mkdir(parents=True)
    (parent / "plan.json").write_text("{}")
    (parent / "analysis/GATE.json").write_text(
        json.dumps(dict(status=state, failures=[], consumed_followup_eligible_labels=eligible))
    )
    monkeypatch.setattr(module, "clean_source", lambda *_: "fixture")
    monkeypatch.setattr(
        module.v2, "submit", lambda *_args, **_kwargs: pytest.fail("unexpected submission")
    )
    out = tmp_path / "out"
    module.conditional_submit(out, parent, sha(parent / "plan.json"))
    assert module.common.read(out / "DECISION.json")["gpu_submitted"] is False


def test_controller_binds_only_consumed_inputs_and_profile_hash(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    (parent / "analysis").mkdir(parents=True)
    (parent / "analysis/GATE.json").write_text(
        json.dumps(
            dict(
                status="complete_consumed_diagnostic",
                failures=[],
                consumed_followup_eligible_labels=[],
            )
        )
    )
    profile = tmp_path / "profile.json"
    profile.write_text("[0.1,0.8]")
    config = RunConfig(
        model=ModelConfig(
            id="llava-hf/llava-1.5-7b-hf", revision="a" * 40, precision=Precision.BF16
        ),
        dataset=DatasetConfig("fixture", "b" * 40, "development"),
        generation=GenerationConfig(max_new_tokens=1024),
        method=MosaicKVMethod.MOSAICKV_EXACT,
        execution=ExecutionConfig(backend=Backend.HUGGINGFACE, attention_implementation="eager"),
        cache=CacheConfig(
            block_size=1,
            physical_budget_block_size=1,
            budget_unit=BudgetUnit.BYTES,
            budget_value=208 * 2**20,
            retention_ratio=math.nextafter(1.0, 0.0),
        ),
        forecasting=ForecastingConfig(mode=ForecastMode.PROMPT_WINDOW, draft_steps=0),
        prototypes=PrototypeConfig(enabled=False),
        residual=ResidualConfig(enabled=False),
        repair=RepairConfig(enabled=False, policy=RepairPolicy.NONE),
        bounded_decode=BoundedDecodeConfig(
            128,
            4,
            incremental_history=IncrementalHistoryConfig(
                fixed_prompt_reservation=True, borrow_prefill_snapshot=True, policy="sink_recent"
            ),
        ),
    )
    jobs = []
    for i, task in enumerate(("llava_description", "mmvet_prefixkv")):
        w = parent / f"workers/{i}/rows"
        w.mkdir(parents=True)
        (w / "prefixkv_reimpl-native-control.json").write_text(
            json.dumps(dict(memory_audit_traces={}))
        )
        (w.parent / "fullkv_reference.json").write_text("{}")
        arms = [
            dict(
                label="prefixkv_reimpl-native-control",
                config=dict(prefixkv=dict(profile_path=str(profile))),
            )
        ]
        for method in (MosaicKVMethod.PROMPT_ATTENTION_TOPK, MosaicKVMethod.MOSAICKV_EXACT):
            cfg = replace(
                config,
                method=method,
                forecasting=replace(config.forecasting, enabled=method.is_mosaickv),
                graph=replace(config.graph, enabled=method.is_mosaickv),
                selection=replace(config.selection, enabled=method.is_mosaickv),
            )
            arms.append(
                dict(label=f"{method.value}-sink_recent-208mib-v2", config=canonical_config(cfg))
            )
        jobs.append(
            dict(
                index=i,
                task=task,
                ledger=f"consumed-ledger-{i}",
                sample_id=f"consumed-{i}",
                arms=arms,
            )
        )
    plan = dict(bound_files={}, jobs=jobs)
    (parent / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(module, "clean_source", lambda *_: "fixture")
    calls = []
    monkeypatch.setattr(module.v2, "submit", lambda *_args, **_kwargs: calls.append(True))
    out = tmp_path / "out"
    module.conditional_submit(out, parent, sha(parent / "plan.json"))
    result = module.common.read(out / "plan.json")
    assert len(calls) == 1 and len(result["jobs"]) == 10
    assert not result["fresh_evaluation_authorized"] and not result["paper_campaign_authorized"]
    assert result["bound_files"][str(profile)] == sha(profile)
    for job in result["jobs"]:
        assert job["sample_id"] in {"consumed-0", "consumed-1"}
        arm = job["arms"][0]
        if job.get("measurement_role") == "remeasured_native_control":
            assert job["retention_one_config"]
            continue
        cfg = RunConfig.from_mapping(arm["config"])
        assert config_sha256(cfg) == arm["config_sha"]
        assert cfg.bounded_decode.incremental_history.layer_profile_sha256 == sha(profile)


def test_peak_gate_uses_shared_transaction_boundary_not_legacy_samples(tmp_path, monkeypatch):
    jobs = []
    for task in ("llava_description", "mmvet_prefixkv"):
        prior = tmp_path / f"prior-{task}.json"
        native = dict(
            answer="synthetic_fixture",
            nll_sum=10.0,
            evaluated_tokens=10,
            teacher_forced_ppl=3.1,
            rouge_l_f1=0.5,
            metrics={k: v for k, v in zip(module.common.STORAGE, (200, 300, 1), strict=True)},
            audited_decode_transaction_peak_bytes=350,
        )
        prior.write_text(json.dumps(native))
        for policy, method in [("native_control", "prefixkv_reimpl")] + [
            (p, m)
            for p in ("method", "question_and_boundaries")
            for m in ("prompt_attention_topk", "mosaickv_exact")
        ]:
            index = len(jobs)
            label = f"{method}-{policy}"
            job = dict(
                index=index,
                task=task,
                mandatory_policy=policy,
                native_control_row=str(prior),
                arms=[dict(label=label)],
            )
            if method == "prefixkv_reimpl":
                job["measurement_role"] = "remeasured_native_control"
            jobs.append(job)
            worker = tmp_path / f"workers/{index}"
            (worker / "rows").mkdir(parents=True)
            (worker / "result.json").write_text(json.dumps(dict(failures=[])))
            row = dict(
                native,
                task=task,
                sample_id=f"fixture-{task}",
                label=label,
                method=method,
                status="completed",
                traces={},
                memory_audit_traces={},
                audited_request_owned_peak_bytes=500,
                pytorch_request_peak_allocated_bytes=1000,
                memory_observer_token_parity=True,
            )
            row["metrics"] = dict(row["metrics"], end_to_end_time=1.0)
            if method != "prefixkv_reimpl":
                row.update(
                    teacher_forced_ppl=3.0,
                    rouge_l_f1=0.6,
                    audited_decode_transaction_peak_bytes=180,
                )
                # Deliberately contradictory legacy peak: it must not decide
                # the common transaction-peak comparison.
                row["metrics"].update(
                    dict(zip(module.common.STORAGE, (100, 150, 400), strict=True))
                )
            (worker / f"rows/{label}.json").write_text(json.dumps(row))
    plan = dict(source_sha="fixture", jobs=jobs)
    monkeypatch.setattr(module.common, "inputs", lambda *_: plan)
    module.analyze(tmp_path, "fixture")
    gate = module.common.read(tmp_path / "analysis/GATE.json")
    assert gate["status"] == "complete_consumed_diagnostic"
    assert gate["peak_comparison_field"] == "audited_decode_transaction_peak_bytes"
    assert all(c["smaller_initial_final_peak_than_native"] for c in gate["checks"])
    assert not gate["superiority_established"] and not gate["paper_campaign_authorized"]
