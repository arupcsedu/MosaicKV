"""Finite scheduling and route-count changes do not manufacture quality evidence."""

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from evicastkv_opportunity_followup import (
    REGISTRATION,
    next_action,
    prepare_derived,
    r16_template,
    submit_derived,
    validate_registration,
)
from test_hf_runtime import _capabilities
from test_opportunity_demand import source

from mosaickv.config import EvidencePolicyConfig, RunConfig
from mosaickv.evaluation.fox_campaign import source_config
from mosaickv.selection.opportunity_demand import normalized_queries


def gate_fixture(positive=False):
    return dict(
        status="passed_execution_development_only",
        actual_rows=832,
        quality_decisions={
            v: dict(required_contrasts=4, positive_development_signal=positive)
            for v in ("evidence-routing-v1", "evidence-coverage-v1", "evidence-both-v1")
        },
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("budgets_mib", [10, 40]),
        ("development_samples", 128),
        ("minimum_query_mass_fraction", 0.9),
        ("confirmation_authorized", True),
        ("no_optional_enlargement", False),
        ("maximum_additional_gpu_requests", 9999),
    ],
)
def test_scope_is_not_enlarged_or_floor_weakened(key, value):
    registration = yaml.safe_load(REGISTRATION.read_text())
    validate_registration(registration)
    registration[key] = value
    with pytest.raises(ValueError):
        validate_registration(registration)


def test_template_changes_only_two_consistent_participation_fields():
    base = source_config("coco")
    original = copy.deepcopy(base)
    revised = r16_template(base)
    assert base == original
    for name in ("graph", "prototypes"):
        assert revised[name]["uncertain_head_fraction"] == 1 / 16
        revised[name]["uncertain_head_fraction"] = 1 / 256
    assert revised == original


def test_wrong_reference_fraction_is_not_silently_treated_as_frozen():
    base = source_config("coco")
    base["graph"]["uncertain_head_fraction"] = 1 / 4
    with pytest.raises(ValueError, match="registered"):
        r16_template(base)


def test_positive_development_stops_instead_of_launching_another_sweep():
    assert next_action(gate_fixture(True)).startswith("stop_development_signal")


def test_negative_development_only_authorizes_the_bounded_numerical_canary():
    assert next_action(gate_fixture()) == "run_registered_r16_numerical_canary"


@pytest.mark.parametrize(
    "status,rows", [("failed_execution", 832), ("passed_execution_development_only", 831)]
)
def test_incomplete_first_study_never_triggers_an_algorithm_retry(status, rows):
    gate = gate_fixture()
    gate.update(status=status, actual_rows=rows)
    assert next_action(gate).startswith("blocked_")


@pytest.mark.parametrize("malformation", ["missing", "wrong_count", "boolean_string"])
def test_malformed_quality_decisions_fail_closed(malformation):
    gate = gate_fixture()
    if malformation == "missing":
        gate["quality_decisions"].pop("evidence-routing-v1")
    elif malformation == "wrong_count":
        gate["quality_decisions"]["evidence-routing-v1"]["required_contrasts"] = 3
    else:
        gate["quality_decisions"]["evidence-routing-v1"]["positive_development_signal"] = "false"
    with pytest.raises(ValueError):
        next_action(gate)


@pytest.mark.parametrize("variant", ["routing_only", "coverage_only", "both"])
def test_multiple_routed_heads_keep_exact_capacity_queries_and_mandatory_tokens(variant):
    import mosaickv.backends.hf_runtime as runtime

    state = source(heads=32, length=12)
    base = RunConfig.from_mapping(r16_template(source_config("coco")))
    config = replace(
        base,
        cache=replace(base.cache, budget_value=32 * 6 * 32),
        evidence_policy=EvidencePolicyConfig(variant=variant),
    )
    rng = np.random.default_rng(22)
    a = normalized_queries(rng.random((32 * 4, 12))).reshape(1, 32, 4, 12)
    prefill = SimpleNamespace(attention_weights=(a,))
    token_attention = runtime._prompt_token_attention_by_head(state, prefill, prompt_window=16)
    scores = runtime._block_attention_values_from_token_attention(state, token_attention)
    plan = runtime.build_compression_plan(
        state,
        None,
        scores,
        _capabilities(safe=False),
        config,
        position_limits={(0, head): 6 for head in range(32)},
        prompt_attention_by_node=scores,
        prompt_attentions=prefill.attention_weights,
        graph_head_uncertainty=runtime._attention_uncertainty_by_head(token_attention),
    )
    assert len(plan.graph_head_routes) == 2
    assert plan.construction.state.statistics.active_kv_bytes == config.cache.budget_value
    for head in range(32):
        entries = [b for b in plan.construction.state.exact.blocks if b.kv_head == head]
        assert len(entries) == 6 and any(b.mandatory for b in entries)
    if variant != "routing_only":
        checks = plan.evidence_policy_statistics["query_guard"]["routes"]
        assert len(checks) == 2
        assert all(c["final_minimum_query_mass_ratio"] >= 0.95 for c in checks)


def derived_fixture(tmp_path):
    first = tmp_path / "first"
    first.mkdir()
    template = first / "template.json"
    template.write_text(json.dumps(source_config("coco")))
    ledger = first / "ledger.json"
    ledger.write_text(json.dumps(dict(samples=[dict(sample_id=str(i)) for i in range(64)])))
    lock = dict(
        templates={"coco": dict(path=str(template))},
        cohorts={
            "coco": {"development": dict(path=str(ledger), sample_ids=[str(i) for i in range(64)])}
        },
        historical_bindings={},
        source_root=str(tmp_path / "runtime"),
        source_sha="fixture",
    )
    (first / "protocol_lock.json").write_text(json.dumps(lock))
    root = tmp_path / "controller"
    root.mkdir()
    plan = dict(
        registration=dict(
            first_root=str(first),
            canary_wall_minutes=30,
            resources=dict(maximum_a100s=6, array_concurrency=6, wall_minutes=90),
        ),
        bindings={},
        controller_sha="fixture",
    )
    (root / "plan.json").write_text(json.dumps(plan))
    return root, plan


def test_derived_canary_uses_only_two_consumed_ids_and_both_shared_caps(tmp_path):
    root, plan = derived_fixture(tmp_path)
    target, lock = prepare_derived(root, plan, "evidence_canary")
    cohort = lock["cohorts"]["coco"]["consumed_canary"]
    assert cohort["sample_ids"] == ["0", "1"]
    jobs = json.loads((target / "jobs/evidence_canary.json").read_text())["jobs"]
    assert len(jobs) == 2 and all(len(j["arms"]) == 13 for j in jobs)
    assert lock["routing_followup"]["comparator_not_the_original_frozen_r256"]
    assert lock["resources"]["wall_minutes"] == 30
    assert lock["resources"]["array_concurrency"] == 2


def test_failed_canary_never_creates_a_development_execution_root(tmp_path):
    root, plan = derived_fixture(tmp_path)
    gate = root / "r16-canary/analysis/evidence_canary/GATE_evidence_canary.json"
    gate.parent.mkdir(parents=True)
    gate.write_text(json.dumps(dict(status="failed")))
    with pytest.raises(ValueError, match="canary did not pass"):
        prepare_derived(root, plan, "evidence_development")
    assert not (root / "r16-development").exists()


def test_gpu_submission_is_durable_before_cpu_callback_and_honors_wall_limit(tmp_path, monkeypatch):
    import evicastkv_opportunity_followup as module

    target = tmp_path / "execution"
    (target / "jobs").mkdir(parents=True)
    (target / "jobs/evidence_development.json").write_text(json.dumps(dict(jobs=[{}, {}])))
    calls = []

    def scheduler(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            return ""
        if len(calls) == 2:
            return "100\n"
        durable = json.loads((target / "jobs/gpu_submission.json").read_text())
        assert durable["array"] == "100"
        return "101\n"

    monkeypatch.setattr(module.subprocess, "check_output", scheduler)
    monkeypatch.setenv("USER", "fixture")
    lock = dict(
        source_root=str(tmp_path / "runtime"),
        resources=dict(
            maximum_a100s=6,
            array_concurrency=6,
            wall_minutes=90,
        ),
    )
    assert submit_derived(target, lock, "evidence_development") == "101"
    assert "--time=01:30:00" in calls[1]
    assert "--array=0-1%6" in calls[1]
    assert "--dependency=afterany:100" in calls[2]


def test_no_gpu_submission_when_aggregate_six_gpu_cap_is_already_reached(tmp_path, monkeypatch):
    import evicastkv_opportunity_followup as module

    calls = []

    def scheduler(command, **kwargs):
        calls.append(command)
        return "fixture|gpu:a100:6\n"

    monkeypatch.setattr(module.subprocess, "check_output", scheduler)
    monkeypatch.setenv("USER", "fixture")
    lock = dict(source_root=str(tmp_path), resources=dict(maximum_a100s=6, array_concurrency=6))
    with pytest.raises(ValueError, match="admission"):
        submit_derived(tmp_path, lock, "evidence_development")
    assert len(calls) == 1 and calls[0][0] == "squeue"
