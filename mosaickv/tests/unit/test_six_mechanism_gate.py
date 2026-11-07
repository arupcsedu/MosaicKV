from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from mosaickv.adapters.huggingface import AdapterCapabilities, CachedKeyState, QueryVectorState
from mosaickv.backends.hf_runtime import build_compression_plan
from mosaickv.cache_state import FullKVState, Modality, ModalitySpan
from mosaickv.config import RunConfig, canonical_config
from mosaickv.evaluation.six_mechanism_gate import ARMS, all_pass, build_config, validate_evidence
from mosaickv.prototypes.types import TierConstructionMode
from mosaickv.repair import RepairCacheState, repair_decode_step


def protocol():
    return json.loads(
        (
            Path(__file__).parents[2] / "configs/experiments/six_mechanism_feasibility_v1.json"
        ).read_text()
    )


@pytest.mark.parametrize("arm", ARMS)
def test_all_registered_configurations_are_strict_and_do_not_use_storage_variant(arm):
    cfg = build_config(protocol(), arm)
    assert RunConfig.from_mapping(canonical_config(cfg)) == cfg
    assert cfg.output_preserving is None and cfg.bounded_decode is None
    assert cfg.execution.attention_implementation == "eager"
    assert cfg.model.id == "llava-hf/llava-1.5-7b-hf"
    assert cfg.generation.max_new_tokens == 16
    assert cfg.cache.block_size == 1
    assert cfg.repair.max_blocks_per_step == 1
    if arm not in {"full_kv", "retention_one"}:
        assert cfg.cache.budget_value == 150 * 2**20


def test_unknown_arm_fails_closed():
    with pytest.raises(ValueError, match="unregistered"):
        build_config(protocol(), "silent_exact_fallback")


def orchestration():
    file = Path(__file__).parents[2] / "scripts/six_mechanism_feasibility.py"
    spec = importlib.util.spec_from_file_location("six_gate_script", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gpu_launch_is_held_one_gpu_twelve_cpus_thirty_minutes(tmp_path, monkeypatch):
    script = orchestration()
    (tmp_path / "plan.json").write_text("{}")
    captured = []
    monkeypatch.setattr(
        script.subprocess, "check_output", lambda args, **kwargs: captured.append(args) or "12345\n"
    )
    assert script.submit(tmp_path, "worker", array=2) == "12345"
    assert "--gres=gpu:a100:1" in captured[0]
    assert "--cpus-per-task=12" in captured[0]
    assert "--array=0-1%2" in captured[0] and "--hold" in captured[0]
    assert "--time=00:30:00" in captured[0]
    assert "--reservation=bi_fox_dgx" in captured[0]


def test_failed_producer_is_analyzed_and_cannot_launch_component_study(tmp_path, monkeypatch):
    script = orchestration()
    monkeypatch.setattr(
        script,
        "checked",
        lambda root, expected: dict(
            protocol=protocol(), classification="synthetic_failure_fixture", source_sha="a" * 40
        ),
    )
    monkeypatch.setattr(script, "launch", lambda *args: pytest.fail("blocked study was submitted"))
    source = tmp_path / "canary/workers/0/result.json"
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(dict(all_checks_passed=False, error="controlled failure")))
    script.analyze(tmp_path, "b" * 64, "canary")
    decision = json.loads((tmp_path / "canary_analysis/decision.json").read_text())
    assert not decision["all_gates_passed"]
    assert decision["expected_rows"] == 12
    assert not decision["larger_campaign_authorized"]


def test_cpu_analysis_uses_afterany_not_success_dependency(tmp_path, monkeypatch):
    script = orchestration()
    (tmp_path / "plan.json").write_text("{}")
    captured = []
    monkeypatch.setattr(
        script.subprocess, "check_output", lambda args, **kwargs: captured.append(args) or "12346\n"
    )
    script.submit(tmp_path, "analyze", after="12345")
    assert "--dependency=afterany:12345" in captured[0]
    assert not any("--gres" in value for value in captured[0])


def fixture_evidence():
    # Synthetic validation fixture only; never an experimental measurement.
    trace = dict(
        status="completed",
        tier_mode="three_tier",
        effective_method="mosaickv_full",
        forecast_statistics=dict(
            provenance=dict(
                mode="hybrid",
                completed_draft_steps=2,
                reused_original_prefill=True,
                draft_cache_isolated=True,
            )
        ),
        utility_diagnostics=dict(
            weights=dict(lambda_v=-0.25), value_contribution=dict(maximum=0.3)
        ),
        tier_storage_accounting=dict(
            exact_active_bytes=32, prototype_active_bytes=8, residual_cpu_bytes=64
        ),
        residual_configuration=dict(storage_dtype="lossless", all_payloads_pinned=True),
        request_KV_ownership=dict(includes_registered_inactive_source_and_transient_KV=True),
        graph_edges=[{}],
        prototypes=[{}],
        selected_blocks=[{}],
        repair_events=[
            dict(
                triggered=True,
                restored_bytes=8,
                restored_block_ids=[1],
                re_decode_count=1,
                transfer_was_asynchronous=True,
            )
        ],
        timing_breakdown={
            key: 0.1
            for key in (
                "forecast",
                "graph",
                "utility_and_selection",
                "tier_construction",
                "cache_packing",
                "decode",
                "total",
            )
        },
    )
    ops = dict(
        prefill=1,
        forecast=1,
        graph=1,
        cross_modal_edges=1,
        utility=1,
        planner=1,
        selector=1,
        prototype_members=1,
        repair=1,
        decode_checks=[True],
        rope_position_checks=[True],
        plan_invariants=True,
        packed_storage_checks=True,
        repair_physical_cap_checks=True,
        promoted_kv_exact_checks=[True],
    )
    return trace, ops


def test_complete_executed_fixture_passes():
    trace, ops = fixture_evidence()
    assert all_pass(validate_evidence(trace, ops, controlled=True))


@pytest.mark.parametrize(
    "operation",
    [
        "forecast",
        "graph",
        "cross_modal_edges",
        "utility",
        "planner",
        "selector",
        "prototype_members",
        "repair",
    ],
)
def test_enabled_flag_without_executed_mechanism_is_rejected(operation):
    trace, ops = fixture_evidence()
    ops[operation] = 0
    assert not all_pass(validate_evidence(trace, ops))


@pytest.mark.parametrize(
    "mode",
    [
        "full_cache",
        "exact_only_prototype_merge_unsafe",
        "exact_only_incompatible_assignments",
        "retention_one",
    ],
)
def test_fallback_and_identity_do_not_prove_compressed_three_tier_execution(mode):
    trace, ops = fixture_evidence()
    trace["tier_mode"] = mode
    assert not all_pass(validate_evidence(trace, ops))


@pytest.mark.parametrize(
    "label", ["full_kv", "mosaickv_exact", "mosaickv_full__fallback", "unknown"]
)
def test_arbitrary_effective_labels_fail_even_when_flags_say_three_tier(label):
    trace, ops = fixture_evidence()
    trace["effective_method"] = label
    assert not all_pass(validate_evidence(trace, ops))


@pytest.mark.parametrize(
    "field,value",
    [
        ("re_decode_count", 0),
        ("re_decode_count", 2),
        ("restored_bytes", 0),
        ("transfer_was_asynchronous", False),
    ],
)
def test_repair_must_transfer_and_recompute_exactly_once(field, value):
    trace, ops = fixture_evidence()
    trace["repair_events"][0][field] = value
    assert not all_pass(validate_evidence(trace, ops, controlled=True))


def test_nan_timings_missing_storage_and_violated_positions_fail():
    trace, ops = fixture_evidence()
    bad = copy.deepcopy(trace)
    bad["timing_breakdown"]["graph"] = float("nan")
    assert not all_pass(validate_evidence(bad, ops))
    ops["decode_checks"] = [True, False]
    assert not all_pass(validate_evidence(trace, ops))
    assert not all_pass({}) and not all_pass({"a": 1})


def test_routed_bounded_prototype_plan_and_actual_cpu_promotion():
    # Real component execution on synthetic CPU tensors; not checkpoint evidence.
    rng = np.random.default_rng(17)
    layers = tuple(
        (
            torch.tensor(rng.normal(size=(1, 2, 32, 8)), dtype=torch.float32),
            torch.tensor(rng.normal(size=(1, 2, 32, 8)), dtype=torch.float32),
        )
        for _ in range(2)
    )
    full = FullKVState.from_tensors(
        layers,
        block_size=1,
        mandatory_logical_positions=(0, 31),
        cached_key_state=CachedKeyState.NOT_APPLICABLE,
        modality_spans=(
            ModalitySpan(0, 4, Modality.TEXT),
            ModalitySpan(4, 28, Modality.IMAGE, image_index=0),
            ModalitySpan(28, 32, Modality.TEXT),
        ),
    )
    cfg = build_config(protocol(), "controlled_repair")
    cfg = replace(
        cfg,
        cache=replace(cfg.cache, budget_value=full.active_bytes // 2),
        prototypes=replace(cfg.prototypes, group_size=8),
        residual=replace(cfg.residual, require_pinned_memory=False),
    )
    caps = AdapterCapabilities(
        "fixture",
        ("Fixture",),
        ("eager",),
        True,
        False,
        False,
        ("tuple",),
        -2,
        CachedKeyState.NOT_APPLICABLE,
        QueryVectorState.Q_PROJ_PRE_ROPE,
        True,
        True,
    )
    scores = {i: (block.physical_cache_indices[0] + 1) / 528 for i, block in enumerate(full.blocks)}
    attention = tuple(torch.ones((1, 2, 4, 32)) / 32 for _ in layers)
    plan = build_compression_plan(
        full,
        None,
        scores,
        caps,
        cfg,
        position_limits={(layer, head): 16 for layer in range(2) for head in range(2)},
        prompt_attentions=attention,
        prompt_attention_by_node=scores,
        graph_head_uncertainty={
            (layer, head): 0.9 if (layer, head) == (0, 0) else 0.1
            for layer in range(2)
            for head in range(2)
        },
    )
    assert plan.construction.mode is TierConstructionMode.THREE_TIER
    assert len(plan.construction.prototypes) >= 2
    assert plan.construction.state.statistics.active_kv_bytes <= cfg.cache.budget_value
    repair = RepairCacheState.from_construction(full, plan.construction)
    calls = []
    result = repair_decode_step(
        repair,
        cfg.repair,
        step_index=1,
        provisional_logits=torch.tensor([[1.0, 2.0, 3.0]]),
        prototype_attention_mass={
            p.prototype_id: 0.5 / len(plan.construction.prototypes)
            for p in plan.construction.prototypes
        },
        re_decode=lambda state: calls.append(state) or torch.tensor([[1.0, 3.0, 2.0]]),
    )
    assert result.event.triggered and len(calls) == 1
    assert result.event.active_cost_after <= cfg.cache.budget_value
