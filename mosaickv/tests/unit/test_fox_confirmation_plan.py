"""Synthetic CPU metadata tests; never confirmation or empirical evidence."""

import builtins
import hashlib
import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.fox_confirmation_candidate import CONFIG_SHA256, CONTROLS, PROOF_FIELDS
from mosaickv.evaluation.fox_confirmation_plan import build_confirmation_plan
from mosaickv.evaluation.paper_readiness_v63 import AUTHORIZATION_KEYS


def capsule(path, record):
    sha = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return dict(path=path, sha256=sha, record=record)


@pytest.fixture
def inputs():
    protocol = json.loads(
        (
            Path(__file__).parents[2]
            / "configs/experiments/evicastkv_fox/review_completion_v63.json"
        ).read_text()
    )
    source = "a" * 40
    review = capsule("synthetic/review.json", dict(source_sha=source))
    proofs = {
        kind: dict(path=f"synthetic/{kind}.json", sha256=str(index) * 64)
        for index, kind in enumerate(
            ("correctness", "byte_matching", "metric", "baseline_fidelity"), 1
        )
    }
    candidate = dict(
        source_sha=source,
        qualified=True,
        parameters_frozen=True,
        candidate_config_file_sha256=CONFIG_SHA256,
        qualification_proofs=proofs,
        selected_config=dict(review={key: review[key] for key in ("path", "sha256")}),
        unresolved=["history_media_cohort_and_confirmation_admission_required"],
        authorization=dict.fromkeys(AUTHORIZATION_KEYS, False),
    )
    binding = dict(
        route=dict(dataset_id=CONTROLS["dataset_id"]),
        processor_bound=True,
        binding_sha256="b" * 64,
        unresolved=[],
        authorization=dict.fromkeys(AUTHORIZATION_KEYS, False),
    )
    history_capsules = {}
    for kind in ("history", "media"):
        history_capsules[kind] = capsule(
            f"synthetic/{kind}.json",
            dict(
                schema="evicastkv_original_complete_qualified_proof_v36",
                kind=kind,
                complete=True,
                failures=[],
                source_sha=source,
                candidate_config_sha256=CONFIG_SHA256,
                controls=deepcopy(CONTROLS),
                evidence_files=[dict(path=f"synthetic/closed-{kind}.json", sha256="d" * 64)],
                **dict.fromkeys(PROOF_FIELDS[kind], True),
            ),
        )
    ids = [f"synthetic-{index}" for index in range(256)]
    evidence = dict(
        schema="fox_confirmation_evidence_v1",
        reviewed=True,
        source_sha=source,
        candidate_config_file_sha256=CONFIG_SHA256,
        input_binding_sha256="b" * 64,
        gate_proofs=dict(
            source_fidelity=proofs["correctness"]["sha256"],
            constructed_byte_matching=proofs["byte_matching"]["sha256"],
            scorer=proofs["metric"]["sha256"],
            baseline_fidelity=proofs["baseline_fidelity"]["sha256"],
            **{
                k: {field: history_capsules[k][field] for field in ("path", "sha256")}
                for k in ("history", "media")
            },
        ),
        proof_capsules=history_capsules,
        review_capsule=review,
        history=dict(
            unbound=0,
            partial=0,
            semantic_exposure_unresolved=0,
            identity_coverage_complete=True,
            media_source_group_coverage_complete=True,
        ),
        cohort=dict(
            ledger_path="synthetic/confirmation-ledger.json",
            ledger_sha256="e" * 64,
            sample_ids=ids,
            timing_sample_ids=ids[:32],
        ),
        power=dict(independently_registered=True, sample_size=256, registration_sha256="f" * 64),
        measurement=dict(
            runtime_lock_sha256="9" * 64,
            boundary="A",
            initial_final_peak_storage_required=True,
            append_only=True,
            generated_predictions=64,
            cache_appends=63,
            warmups=5,
            repetitions=3,
        ),
    )
    common = dict(
        source_sha=source,
        candidate_config_file_sha256=CONFIG_SHA256,
        input_binding_sha256="b" * 64,
        ledger_sha256=evidence["cohort"]["ledger_sha256"],
    )
    statistics = dict(
        sample_size=256,
        family_size=6,
        familywise_alpha=0.05,
        bootstrap_replicates=10000,
        bootstrap_seed=20271014,
    )
    analysis = capsule(
        "synthetic/power-analysis.json",
        dict(
            schema="fox_confirmation_power_analysis_v1",
            **common,
            **statistics,
            complete=True,
            evidence_scope="source_development_only",
            no_confirmation_outcomes_used=True,
            procedure="corpus_CIDEr_cluster_resampling",
            assumptions="Synthetic fixture only: registered cluster-resampling alternative.",
            assumed_effect_cider=2.0,
            target_power=0.8,
            estimated_power_lower_bound=0.85,
            evidence_files=[
                dict(path="synthetic/closed-development-analysis.json", sha256="7" * 64)
            ],
        ),
    )
    registration = capsule(
        "synthetic/power-registration.json",
        dict(
            schema="fox_confirmation_power_registration_v1",
            **common,
            **statistics,
            reviewed=True,
            independently_registered=True,
            outcome_blind=True,
            no_confirmation_outcomes_used=True,
            target_power=0.8,
            analysis_binding={k: analysis[k] for k in ("path", "sha256")},
        ),
    )
    evidence["cohort"]["input_media_source_group_inventory_sha256"] = "6" * 64
    disjoint = capsule(
        "synthetic/cohort-disjointness.json",
        dict(
            schema="fox_confirmation_cohort_disjointness_v1",
            **common,
            reviewed=True,
            complete=True,
            outcome_blind=True,
            no_confirmation_outcomes_used=True,
            sample_count=256,
            checked_sample_count=256,
            sample_ids_sha256=digest(ids),
            input_media_source_group_inventory_sha256="6" * 64,
            history_proof=evidence["gate_proofs"]["history"],
            media_proof=evidence["gate_proofs"]["media"],
            identity_intersections=0,
            media_intersections=0,
            source_group_intersections=0,
        ),
    )
    evidence.update(
        power_registration_capsule=registration,
        power_analysis_capsule=analysis,
        cohort_disjointness_capsule=disjoint,
    )
    evidence["power"]["registration_sha256"] = registration["sha256"]
    evidence["cohort"]["disjointness_report_sha256"] = disjoint["sha256"]
    return dict(
        protocol=protocol,
        candidate_binding=candidate,
        input_binding=binding,
        source_sha=source,
        source_clean=True,
        confirmation_evidence=evidence,
    )


def assert_disabled(result):
    assert not any(result["authorization"].values())
    assert result["execution_enabled"] is False
    assert result["fresh_confirmation_authorized"] is False
    assert result["GPU_execution_authorized"] is False
    assert result["materialization_authorized"] is False
    assert result["automatic_stage_progression"] is False
    assert result["scheduler_commands"] == []
    assert result["actual_requests_executed"] == 0
    assert result["measured_GPU_jobs"] == 0
    assert result["completed_proofs_replayed"] == 0


def test_missing_evidence_stays_blocked_with_actionable_dependencies(inputs):
    inputs["confirmation_evidence"] = None
    inputs["candidate_binding"].update(qualified=False, parameters_frozen=False)
    inputs["input_binding"].update(processor_bound=False)
    result = build_confirmation_plan(**inputs)
    assert result["status"] == "BLOCKED_DEPENDENCIES"
    assert "candidate_parameters_not_frozen" in result["blockers"]
    assert "candidate_controls_not_qualified" in result["blockers"]
    assert "processor_preprocessing_binding_unresolved" in result["blockers"]
    assert "historical_identity_and_media_closure_not_qualified" in result["blockers"]
    assert result["reviewed_runtime_lock_sha256"] is None
    assert_disabled(result)


def test_synthetic_complete_metadata_still_never_authorizes_confirmation(inputs):
    before = deepcopy(inputs)
    result = build_confirmation_plan(**inputs)
    assert inputs == before
    assert result["status"] == "METADATA_READY_EXECUTION_DISABLED"
    assert result["prerequisite_metadata_validated"] is True
    assert result["blockers"] == []
    assert result["reviewed_runtime_lock_sha256"] == "9" * 64
    assert result["planned_unique_inputs"] == 256
    assert result["planned_main_requests"] == 1792
    assert len(result["prospective_arms"]) == 7
    assert result["prospective_arms"][0]["role"] == "quality_reference"
    assert all(
        arm["initial_budget_bytes"] in (20971520, 41943040)
        for arm in result["prospective_arms"][1:]
    )
    assert result["independent_quality_confirmation_established"] is False
    assert result["latency_superiority_established"] is False
    assert result["six_component_benefit_established"] is False
    assert_disabled(result)


def test_pure_planning_never_opens_data_imports_models_or_schedules(inputs, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("planner crossed an execution or file boundary")

    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "datasets"}:
            forbidden()
        return original_import(name, *args, **kwargs)

    for owner, attribute in (
        (Path, "open"),
        (builtins, "open"),
        (os, "open"),
        (subprocess, "run"),
        (subprocess, "check_output"),
        (subprocess, "Popen"),
    ):
        monkeypatch.setattr(owner, attribute, forbidden)
    monkeypatch.setattr(builtins, "__import__", checked_import)
    assert build_confirmation_plan(**inputs)["status"] == "METADATA_READY_EXECUTION_DISABLED"


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("statistics", "registered_primary_quality_family_size", 2),
        ("statistics", "bootstrap_replicates", 9999),
        ("statistics", "adjusted_latency_ratio_upper_bound_maximum", 1.05),
        ("statistics", "corpus_CIDEr_recomputed_each_draw_with_unique_resample_IDs", False),
        ("proposed_resources", "maximum_aggregate_A100s", 7),
        ("proposed_resources", "maximum_RAM_GiB", 64),
        ("proposed_resources", "CPUs_per_shard", 16),
        ("proposed_resources", "GPUs_per_shard", 2),
        ("proposed_resources", "shard_time_limit", "03:00:00"),
        ("memory_and_pairing", "continuation_appends", 64),
        ("memory_and_pairing", "maximum_predictions", 32),
        ("variant_contract", "draft_generation_enabled", True),
        ("variant_contract", "generated_history", "bounded"),
        ("variant_contract", "prototypes_enabled", True),
        ("timing", "warmups", 0),
    ],
)
def test_registered_limits_and_statistical_contract_cannot_drift(inputs, section, key, value):
    inputs["protocol"][section][key] = value
    with pytest.raises(ValueError):
        build_confirmation_plan(**inputs)


def test_any_new_permission_is_rejected(inputs):
    inputs["protocol"]["authorization"]["GPU_execution_authorized"] = True
    with pytest.raises(ValueError, match="explicitly false"):
        build_confirmation_plan(**inputs)


@pytest.mark.parametrize("key,value", [("source_clean", False), ("source_clean", 1)])
def test_clean_source_is_literal_and_required_even_with_complete_metadata(inputs, key, value):
    inputs[key] = value
    result = build_confirmation_plan(**inputs)
    assert "immutable_clean_source_required" in result["blockers"]
    assert_disabled(result)


def test_old_source_proofs_never_transfer_automatically(inputs):
    inputs["candidate_binding"]["source_sha"] = "c" * 40
    result = build_confirmation_plan(**inputs)
    assert "candidate_source_differs_no_automatic_proof_transfer" in result["blockers"]


@pytest.mark.parametrize(
    "field",
    [
        "source_fidelity",
        "constructed_byte_matching",
        "scorer",
        "baseline_fidelity",
        "history",
        "media",
    ],
)
def test_all_six_proof_bindings_are_required(inputs, field):
    del inputs["confirmation_evidence"]["gate_proofs"][field]
    with pytest.raises(ValueError, match="six prerequisite proofs"):
        build_confirmation_plan(**inputs)


def test_history_status_assertion_cannot_replace_qualified_proof(inputs):
    inputs["confirmation_evidence"]["proof_capsules"]["history"]["record"] = dict(status="passed")
    with pytest.raises(ValueError, match="qualified"):
        build_confirmation_plan(**inputs)


@pytest.mark.parametrize("value", [None, True, "unknown", "g" * 64, "9" * 63])
def test_reviewed_runtime_lock_requires_explicit_immutable_hash(inputs, value):
    inputs["confirmation_evidence"]["measurement"]["runtime_lock_sha256"] = value
    with pytest.raises(ValueError, match="reviewed runtime-lock hash required"):
        build_confirmation_plan(**inputs)


def test_missing_runtime_lock_hash_cannot_authorize_late_runtime_resolution(inputs):
    del inputs["confirmation_evidence"]["measurement"]["runtime_lock_sha256"]
    with pytest.raises(ValueError, match="reviewed runtime-lock hash required"):
        build_confirmation_plan(**inputs)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("history", "unbound", 67),
        ("history", "partial", 47),
        ("history", "semantic_exposure_unresolved", 135),
        ("cohort", "sample_ids", ["duplicate"] * 256),
        ("cohort", "timing_sample_ids", [f"outside-{i}" for i in range(32)]),
        ("cohort", "ledger_path", "../outside.json"),
        ("cohort", "ledger_path", "/home/private.json"),
        ("power", "independently_registered", False),
        ("power", "sample_size", 512),
        ("measurement", "cache_appends", 64),
    ],
)
def test_closure_identity_power_and_measurement_must_be_bound(inputs, section, field, value):
    inputs["confirmation_evidence"][section][field] = value
    with pytest.raises(ValueError):
        build_confirmation_plan(**inputs)


@pytest.mark.parametrize(
    "field", ["power_registration_capsule", "power_analysis_capsule", "cohort_disjointness_capsule"]
)
def test_bare_hash_cannot_replace_explicit_registration_or_report_capsule(inputs, field):
    del inputs["confirmation_evidence"][field]
    with pytest.raises(ValueError, match="capsule required"):
        build_confirmation_plan(**inputs)


def test_nonexistent_arbitrary_power_registration_hash_is_rejected(inputs):
    inputs["confirmation_evidence"]["power"]["registration_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="capsule hash differs"):
        build_confirmation_plan(**inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha", "c" * 40),
        ("candidate_config_file_sha256", "c" * 64),
        ("input_binding_sha256", "c" * 64),
        ("ledger_sha256", "c" * 64),
        ("sample_size", 512),
        ("family_size", 2),
        ("familywise_alpha", 0.1),
        ("bootstrap_replicates", 9999),
        ("bootstrap_seed", 1),
        ("no_confirmation_outcomes_used", False),
    ],
)
def test_power_registration_must_match_frozen_controls(inputs, field, value):
    inputs["confirmation_evidence"]["power_registration_capsule"]["record"][field] = value
    with pytest.raises(ValueError):
        build_confirmation_plan(**inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("estimated_power_lower_bound", 0.79),
        ("estimated_power_lower_bound", float("nan")),
        ("estimated_power_lower_bound", float("inf")),
        ("estimated_power_lower_bound", True),
        ("assumed_effect_cider", 0.0),
        ("target_power", 0.1),
        ("assumptions", " "),
        ("evidence_files", []),
        ("procedure", "normal_approximation_of_mean_shard_CIDEr"),
        ("evidence_scope", "confirmation_outputs"),
        ("complete", 1),
    ],
)
def test_actual_development_power_analysis_is_required_without_inventing_a_normal_model(
    inputs, field, value
):
    inputs["confirmation_evidence"]["power_analysis_capsule"]["record"][field] = value
    with pytest.raises(ValueError):
        build_confirmation_plan(**inputs)


def test_power_analysis_receipt_must_match_registration_binding(inputs):
    inputs["confirmation_evidence"]["power_registration_capsule"]["record"]["analysis_binding"] = {
        "path": "synthetic/other-analysis.json",
        "sha256": "d" * 64,
    }
    with pytest.raises(ValueError, match="linked analysis"):
        build_confirmation_plan(**inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("identity_intersections", 1),
        ("media_intersections", 1),
        ("source_group_intersections", 1),
        ("identity_intersections", False),
        ("checked_sample_count", 255),
        ("complete", False),
        ("source_sha", "c" * 40),
        ("ledger_sha256", "c" * 64),
        ("input_media_source_group_inventory_sha256", "c" * 64),
    ],
)
def test_reused_consumed_or_partially_checked_cohort_cannot_pass_disjointness(inputs, field, value):
    inputs["confirmation_evidence"]["cohort_disjointness_capsule"]["record"][field] = value
    with pytest.raises(ValueError):
        build_confirmation_plan(**inputs)


def test_cohort_order_is_bound_to_disjointness_report(inputs):
    ids = inputs["confirmation_evidence"]["cohort"]["sample_ids"]
    ids[0], ids[1] = ids[1], ids[0]
    with pytest.raises(ValueError, match="sample_ids_sha256"):
        build_confirmation_plan(**inputs)


def test_source_group_inventory_must_be_bound_independently_in_cohort(inputs):
    del inputs["confirmation_evidence"]["cohort"]["input_media_source_group_inventory_sha256"]
    with pytest.raises(ValueError, match="independently bound"):
        build_confirmation_plan(**inputs)


def test_disjointness_proof_hash_and_history_media_links_must_agree(inputs):
    report = inputs["confirmation_evidence"]["cohort_disjointness_capsule"]["record"]
    report["history_proof"] = dict(path="synthetic/other-history.json", sha256="a" * 64)
    with pytest.raises(ValueError, match="history_proof"):
        build_confirmation_plan(**inputs)
