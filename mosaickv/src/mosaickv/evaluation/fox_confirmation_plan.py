"""Pure metadata planning for the disabled COCO confirmation route.

This module performs no I/O, model import, dataset access or scheduling. Caller
records must be read by the bounded hash-binding loader; dictionary validation
is not a substitute for original scientific evidence or execution authority.
"""

from __future__ import annotations

import math
from pathlib import PurePosixPath

from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.paper_readiness_v63 import AUTHORIZATION_KEYS, disabled_authorization

PROOF_KINDS = {
    "source_fidelity": "correctness",
    "constructed_byte_matching": "byte_matching",
    "scorer": "metric",
    "baseline_fidelity": "baseline_fidelity",
}


def _hash(value, length=64):
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def _exact(record, expected, label):
    if not isinstance(record, dict):
        raise ValueError(f"{label}: object required")
    for key, value in expected.items():
        actual = record.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"{label}: registered {key} differs")


def _registered_protocol(protocol):
    if not isinstance(protocol, dict):
        raise ValueError("protocol object required")
    disabled_authorization(protocol.get("authorization"))
    _exact(protocol, {"enabled": False, "execution_enabled": False}, "protocol")
    _exact(
        protocol.get("execution"),
        {"auto_start": False, "requires_separate_explicit_enablement_after_all_gates": True},
        "execution",
    )
    _exact(
        protocol.get("statistics"),
        {
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20271014,
            "registered_primary_quality_family_size": 6,
            "registered_six_contrast_family_unchanged": True,
            "COCO_primary_contrasts": 2,
            "MileBench_registered_but_NOT_RUN_development_only_contrasts": 4,
            "ordinary_interval_coverage": 0.95,
            "familywise_alpha": 0.05,
            "adjusted_latency_ratio_upper_bound_maximum": 1.0,
            "latency_ratio_1_05_is_acceptance": False,
            "corpus_CIDEr_recomputed_each_draw_with_unique_resample_IDs": True,
            "resampling_unit": "paired_input_derived_reusable_media_source_group_cluster",
            "mean_shard_CIDEr_is_corpus_CIDEr": False,
            "ratio_recomputed_each_bootstrap_draw": True,
            "confirmation_outcome_dependent_family_change_or_optional_stopping": False,
        },
        "statistics",
    )
    _exact(protocol.get("cohort"), {"proposed_unique_COCO_inputs": 256}, "cohort")
    _exact(
        protocol.get("matrix"),
        {
            "workload": "COCO_caption_only",
            "proposed_caps_MiB": [20, 40],
            "proposed_caps_bytes": [20971520, 41943040],
            "compressed_methods": ["prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact"],
            "planned_main_requests": 1792,
            "actual_requests_executed": 0,
        },
        "matrix",
    )
    _exact(
        protocol["matrix"].get("MileBench"),
        {"status": "NOT_RUN_DEVELOPMENT_ONLY", "confirmation_requests_planned": 0},
        "MileBench",
    )
    _exact(
        protocol["matrix"].get("quality_reference"),
        {
            "method": "full_kv",
            "matched_byte_competitor": False,
            "compressed_latency_ranking_member": False,
        },
        "FullKV reference",
    )
    _exact(
        protocol.get("recovered_reference_controls", {}).get("generation"),
        {"do_sample": False, "max_new_tokens": 64, "temperature": 0.0, "top_p": 1.0},
        "generation",
    )
    _exact(
        protocol.get("variant_contract"),
        {
            "generated_history": "append_only",
            "draft_generation_enabled": False,
            "draft_steps": 0,
            "prototypes_enabled": False,
            "residual_repair_enabled": False,
            "new_backend_enabled": False,
            "new_batching_scheduler_enabled": False,
            "unchanged_retained_KV_entries_required": True,
            "original_logical_positions_and_masks_required": True,
            "full_six_mechanism_benefit_claim": False,
        },
        "variant",
    )
    _exact(
        protocol.get("memory_and_pairing"),
        {
            "maximum_predictions": 64,
            "continuation_appends": 63,
            "initial_final_and_peak_active_unique_backing_bytes_required": True,
            "per_input_initial_physical_byte_equality_required": True,
            "padding_preallocation_metadata_and_transient_owned_storage_charged": True,
            "no_FullKV_fallback_or_nominal_token_ratio_substitution": True,
        },
        "memory",
    )
    _exact(
        protocol.get("timing"),
        {
            "warmups": 5,
            "input_count": 32,
            "repeats_per_primary_method_budget": 3,
            "counterbalanced_method_order_required": True,
            "complete_requests_recomputed_unprofiled": True,
            "input_only_selection_and_same_GPU_within_each_pair_required": True,
            "summarize_repeats_per_input_before_bootstrap": True,
            "FullKV_is_compressed_latency_comparator": False,
        },
        "timing",
    )
    resources = protocol.get("proposed_resources", {})
    _exact(resources, {"GPUs_per_shard": 1, "CPUs_per_shard": 12}, "resources")
    for key, maximum in (
        ("maximum_aggregate_A100s", 6),
        ("maximum_RAM_GiB", 32),
        ("maximum_analysis_workers", 8),
    ):
        value = resources.get(key)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"resources: invalid {key}")
    time_limit = resources.get("shard_time_limit")
    if (
        not isinstance(time_limit, str)
        or len(time_limit) != 8
        or len(time_limit.split(":")) != 3
        or any(len(part) != 2 or not part.isdigit() for part in time_limit.split(":"))
    ):
        raise ValueError("resources: bounded wall time required")
    hours, minutes, seconds = (int(part) for part in time_limit.split(":"))
    if minutes >= 60 or seconds >= 60 or not 0 < hours * 3600 + minutes * 60 + seconds <= 7200:
        raise ValueError("resources: bounded wall time required")
    if resources.get("GPU_execution_authorized") is not False:
        raise ValueError("resource ceiling is not execution authorization")


def _sample_ids(values, count, label):
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(not isinstance(value, str) or not value.strip() for value in values)
        or len(set(values)) != count
    ):
        raise ValueError(f"{label}: exactly {count} unique nonempty identities required")
    return set(values)


def _metadata_binding(value, label):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label}: metadata path/hash binding required")
    path = value["path"]
    if (
        not isinstance(path, str)
        or not path
        or "\x00" in path
        or ".." in PurePosixPath(path).parts
        or PurePosixPath(path).suffix != ".json"
        or any(term in path.lower() for term in (".codex", "credential", "secret", ".ssh"))
        or not _hash(value["sha256"])
        or value["sha256"] == "0" * 64
    ):
        raise ValueError(f"{label}: safe metadata path and immutable hash required")
    return dict(value)


def _capsule(evidence, key, schema, common):
    """Validate a caller-read capsule without opening any linked raw evidence."""
    value = evidence.get(key)
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "record"}:
        raise ValueError(f"{key}: explicit hash-bound metadata capsule required")
    bound = _metadata_binding({name: value[name] for name in ("path", "sha256")}, key)
    _exact(value["record"], dict(schema=schema, **common), key)
    return value["record"], bound


def _registration_capsules(evidence, candidate, inputs, source_sha, cohort, proofs):
    """Require actual power and cohort reports, not unverifiable PASS hashes.

    This validates saved receipt semantics, NOT the analysis itself. In
    particular no additive/normal power model is invented for corpus CIDEr.
    """
    common = dict(
        source_sha=source_sha,
        candidate_config_file_sha256=candidate["candidate_config_file_sha256"],
        input_binding_sha256=inputs["binding_sha256"],
        ledger_sha256=cohort["ledger_sha256"],
    )
    statistics = dict(
        sample_size=256,
        family_size=6,
        familywise_alpha=0.05,
        bootstrap_replicates=10000,
        bootstrap_seed=20271014,
    )
    registration, registered_binding = _capsule(
        evidence,
        "power_registration_capsule",
        "fox_confirmation_power_registration_v1",
        dict(
            common,
            **statistics,
            reviewed=True,
            independently_registered=True,
            outcome_blind=True,
            no_confirmation_outcomes_used=True,
        ),
    )
    if evidence["power"].get("registration_sha256") != registered_binding["sha256"]:
        raise ValueError("power registration: capsule hash differs from registered hash")
    analysis, analysis_binding = _capsule(
        evidence,
        "power_analysis_capsule",
        "fox_confirmation_power_analysis_v1",
        dict(
            common,
            **statistics,
            complete=True,
            evidence_scope="source_development_only",
            no_confirmation_outcomes_used=True,
            procedure="corpus_CIDEr_cluster_resampling",
        ),
    )
    if registration.get("analysis_binding") != analysis_binding:
        raise ValueError("power registration: linked analysis capsule differs")
    target, lower, effect = (
        registration.get("target_power"),
        analysis.get("estimated_power_lower_bound"),
        analysis.get("assumed_effect_cider"),
    )
    if (
        any(type(v) not in (int, float) or not math.isfinite(v) for v in (target, lower, effect))
        or not 0 < target < 1
        or not target <= lower <= 1
        or effect <= 0
        or type(analysis.get("target_power")) is not type(target)
        or analysis["target_power"] != target
        or not isinstance(analysis.get("assumptions"), str)
        or not analysis["assumptions"].strip()
    ):
        raise ValueError(
            "power analysis: finite preregistered assumptions and adequate bound required"
        )
    files = analysis.get("evidence_files")
    if not isinstance(files, list) or not files:
        raise ValueError("power analysis: actual development evidence bindings required")
    paths = []
    for item in files:
        # Bind references only; the reusable proof's raw evidence is not replayed.
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("power analysis: development evidence path/hash required")
        bound = _metadata_binding(item, "power analysis development evidence")
        paths.append(bound["path"])
    if len(paths) != len(set(paths)):
        raise ValueError("power analysis: duplicate development evidence path")
    inventory = cohort.get("input_media_source_group_inventory_sha256")
    if not _hash(inventory) or inventory == "0" * 64:
        raise ValueError("cohort: independently bound input media/source-group inventory required")
    _, disjoint_binding = _capsule(
        evidence,
        "cohort_disjointness_capsule",
        "fox_confirmation_cohort_disjointness_v1",
        dict(
            common,
            reviewed=True,
            complete=True,
            outcome_blind=True,
            no_confirmation_outcomes_used=True,
            sample_count=256,
            checked_sample_count=256,
            sample_ids_sha256=digest(cohort["sample_ids"]),
            input_media_source_group_inventory_sha256=inventory,
            history_proof=proofs["history"],
            media_proof=proofs["media"],
            identity_intersections=0,
            media_intersections=0,
            source_group_intersections=0,
        ),
    )
    if cohort.get("disjointness_report_sha256") != disjoint_binding["sha256"]:
        raise ValueError("cohort: disjointness report hash differs")


def _evidence(evidence, candidate, inputs, source_sha):
    _exact(
        evidence,
        {"schema": "fox_confirmation_evidence_v1", "reviewed": True, "source_sha": source_sha},
        "confirmation evidence",
    )
    for key, expected in (
        ("candidate_config_file_sha256", candidate.get("candidate_config_file_sha256")),
        ("input_binding_sha256", inputs.get("binding_sha256")),
    ):
        if not _hash(expected) or evidence.get(key) != expected:
            raise ValueError(f"confirmation evidence: mismatched {key}")
    proofs = evidence.get("gate_proofs")
    if not isinstance(proofs, dict) or set(proofs) != {*PROOF_KINDS, "history", "media"}:
        raise ValueError("confirmation evidence: exact six prerequisite proofs required")
    candidate_proofs = candidate.get("qualification_proofs", {})
    if not isinstance(candidate_proofs, dict):
        raise ValueError("confirmation evidence: qualified candidate proof mapping required")
    for gate, kind in PROOF_KINDS.items():
        item = candidate_proofs.get(kind)
        expected = item.get("sha256") if isinstance(item, dict) else None
        if not _hash(expected) or proofs[gate] != expected:
            raise ValueError(f"confirmation evidence: candidate proof differs: {gate}")
    from mosaickv.evaluation.fox_confirmation_candidate import validate_qualified_proof

    capsules = evidence.get("proof_capsules", {})
    review = evidence.get("review_capsule")
    if not isinstance(review, dict) or set(review) != {"path", "sha256", "record"}:
        raise ValueError("confirmation evidence: hash-bound original review required")
    bound_review = candidate.get("selected_config", {}).get("review")
    if bound_review != {key: review[key] for key in ("path", "sha256")}:
        raise ValueError("confirmation evidence: reviewed candidate binding differs")
    for kind in ("history", "media"):
        qualified = validate_qualified_proof(
            kind, capsules.get(kind), review["record"], expected_source_sha=source_sha
        )
        if proofs[kind] != qualified:
            raise ValueError(f"confirmation evidence: {kind} proof binding differs")
    _exact(
        evidence.get("history"),
        {
            "unbound": 0,
            "partial": 0,
            "semantic_exposure_unresolved": 0,
            "identity_coverage_complete": True,
            "media_source_group_coverage_complete": True,
        },
        "history closure",
    )
    cohort = evidence.get("cohort", {})
    path = cohort.get("ledger_path")
    if (
        not isinstance(path, str)
        or not path
        or PurePosixPath(path).is_absolute()
        or any(part in (".", "..") for part in PurePosixPath(path).parts)
        or any(term in path.lower() for term in (".codex", "credential", "secret", ".ssh"))
        or not _hash(cohort.get("ledger_sha256"))
    ):
        raise ValueError("cohort: bounded-loader relative ledger path and hash required")
    all_ids = _sample_ids(cohort.get("sample_ids"), 256, "cohort")
    timing_ids = _sample_ids(cohort.get("timing_sample_ids"), 32, "timing cohort")
    if not timing_ids <= all_ids:
        raise ValueError("timing cohort: identities must belong to confirmation cohort")
    power = evidence.get("power", {})
    _exact(power, {"independently_registered": True, "sample_size": 256}, "power registration")
    if not _hash(power.get("registration_sha256")):
        raise ValueError("power registration: hash-bound independent registration required")
    _registration_capsules(evidence, candidate, inputs, source_sha, cohort, proofs)
    _exact(
        evidence.get("measurement"),
        {
            "boundary": "A",
            "initial_final_peak_storage_required": True,
            "append_only": True,
            "generated_predictions": 64,
            "cache_appends": 63,
            "warmups": 5,
            "repetitions": 3,
        },
        "measurement contract",
    )
    runtime_lock_sha256 = evidence["measurement"].get("runtime_lock_sha256")
    if not _hash(runtime_lock_sha256):
        raise ValueError("measurement contract: reviewed runtime-lock hash required")
    return runtime_lock_sha256


def build_confirmation_plan(
    protocol,
    candidate_binding,
    input_binding,
    *,
    confirmation_evidence=None,
    source_sha,
    source_clean=False,
    readiness_capsule=None,
):
    """Validate prospective metadata without granting any execution permission."""
    _registered_protocol(protocol)
    if not _hash(source_sha, 40):
        raise ValueError("valid current source SHA required")
    if not isinstance(candidate_binding, dict) or not isinstance(input_binding, dict):
        raise ValueError("candidate and input bindings must be objects")
    disabled_authorization(candidate_binding.get("authorization"))
    disabled_authorization(input_binding.get("authorization"))
    blockers = []
    if source_clean is not True:
        blockers.append("immutable_clean_source_required")
    if candidate_binding.get("source_sha") != source_sha:
        blockers.append("candidate_source_differs_no_automatic_proof_transfer")
    if candidate_binding.get("parameters_frozen") is not True:
        blockers.append("candidate_parameters_not_frozen")
    if candidate_binding.get("qualified") is not True:
        blockers.append("candidate_controls_not_qualified")
    if input_binding.get("processor_bound") is not True:
        blockers.append("processor_preprocessing_binding_unresolved")
    if not isinstance(input_binding.get("route"), dict):
        blockers.append("dataset_route_unresolved")
    for label, value in (("candidate", candidate_binding), ("input", input_binding)):
        unresolved = value.get("unresolved", [])
        if not isinstance(unresolved, list) or any(
            not isinstance(item, str) for item in unresolved
        ):
            raise ValueError(f"{label}: unresolved dependency list required")
        blockers.extend(
            f"{label}:{item}"
            for item in unresolved
            if not (
                confirmation_evidence is not None
                and label == "candidate"
                and item == "history_media_cohort_and_confirmation_admission_required"
            )
        )
    runtime_lock_sha256 = None
    if confirmation_evidence is None:
        blockers.extend(
            (
                "reviewed_confirmation_evidence_absent",
                "historical_identity_and_media_closure_not_qualified",
                "independent_power_and_256_identity_cohort_not_bound",
            )
        )
    else:
        runtime_lock_sha256 = _evidence(
            confirmation_evidence, candidate_binding, input_binding, source_sha
        )
    if readiness_capsule is not None and not isinstance(readiness_capsule, dict):
        raise ValueError("reused readiness capsule must be an object")
    # Historical readiness is retained as context, not replayed or treated as
    # proof for a new source, and eventual quality outcomes are not prerequisites.
    arms = [dict(method="full_kv", initial_budget_bytes=None, role="quality_reference")]
    arms.extend(
        dict(method=method, initial_budget_bytes=cap, role="compressed_comparison")
        for cap in (20971520, 41943040)
        for method in ("prompt_attention_topk", "vl_cache_reimpl", "mosaickv_exact")
    )
    return dict(
        schema="fox_confirmation_dependency_plan_v1",
        status="BLOCKED_DEPENDENCIES" if blockers else "METADATA_READY_EXECUTION_DISABLED",
        classification="prospective_metadata_plan_not_scientific_measurements",
        source_sha=source_sha,
        reviewed_runtime_lock_sha256=runtime_lock_sha256,
        blockers=list(dict.fromkeys(blockers)),
        prerequisite_metadata_validated=not blockers,
        prospective_arms=arms,
        planned_unique_inputs=256,
        planned_main_requests=1792,
        actual_requests_executed=0,
        measured_GPU_jobs=0,
        execution_enabled=False,
        authorization=dict.fromkeys(AUTHORIZATION_KEYS, False),
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        GPU_execution_authorized=False,
        scheduler_commands=[],
        automatic_stage_progression=False,
        completed_proofs_replayed=0,
        independent_quality_confirmation_established=False,
        latency_superiority_established=False,
        six_component_benefit_established=False,
    )
