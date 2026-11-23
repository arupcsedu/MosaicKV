"""Pure candidate binding for disabled Fox COCO confirmation preparation.

Callers supply already hash-bound metadata capsules read with CapsuleReader.
This module opens no paths, follows no raw-evidence links, selects no inputs,
and never creates a freeze or an execution permission. V36 permits the
preregistered original default without a positive development p-value.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from pathlib import PurePosixPath

from mosaickv.evaluation.paper_readiness_v63 import AUTHORIZATION_KEYS

CONFIG_SHA256 = "389b80ff6529df2072c2a1839469f4be3687be0ec45e38bc124cd834273004cd"
CANONICAL_CONFIG_SHA256 = "52b2cc02506a631e1263c5e9e2ce378a86325c783eeee78d0199b0c9a8148727"
FREEZE_SCHEMA = "evicastkv_original_complete_freeze_v36"
REVIEW_SCHEMA = "evicastkv_original_complete_prospective_review_v36"
AUTH_SCHEMA = "evicastkv_original_complete_delegated_review_authorization_v36"
CORE_PROOFS = ("correctness", "metric", "byte_matching", "baseline_fidelity")
CONTROLS = dict(
    model_id="llava-hf/llava-v1.6-mistral-7b-hf",
    model_revision="2424fdd47412fccc66d91719126b420e9fbd7065",
    precision="fp16",
    attention_backend="eager",
    backend="huggingface",
    dataset_id="lmms-lab/COCO-Caption",
    dataset_revision="c3213b54e173ae86517d481de17f31439d6f8c72",
    split="val",
    method="mosaickv_exact",
    variant="complete",
    initial_budgets_mib=[20, 40],
    block_size=1,
    maximum_predictions=64,
    continuation_steps=63,
    generated_history="append_only",
)
# Mirrored static V36 contract; the regression suite checks exact equality to
# that existing implementation without importing its script in production.
SCIENTIFIC_CONTRACT = dict(
    candidate="original_complete_not_a_selected_opportunity_successor",
    controls=CONTROLS,
    quality_metric="pinned_lmms_eval_PTBTokenizer_COCO_corpus_CIDEr_x100",
    quality_superiority="unrounded_simultaneous_lower_bound_strictly_greater_than_zero",
    primary_family_size=6,
    local_baseline_secondary_family_size=6,
    bootstrap_replicates=10000,
    bootstrap_seed=20271014,
    resampling="paired_input_derived_shared_media_clusters_corpus_reducer_each_draw",
    initial_budgets_mib=[20, 40],
    per_sample_physical_byte_matching=True,
    final_peak_matching_requires_matched_forward_schedule=True,
    retention_one_max_absolute_logit_error=1e-4,
    independent_compact_max_absolute_logit_error=1e-4,
    masked_unpruned_normwise_FP16_bound=8 * 2**-10,
    reference="independently_gathered_same_selected_source_KV_not_unrestricted_FullKV",
    quality_samples=256,
    no_optional_enlargement=True,
    timing_inputs=32,
    warmups=5,
    timing_repeats=3,
    latency_adjusted_upper_candidate_baseline_ratio_maximum=1.0,
    all_results_including_negative_confirmation_retained=True,
    no_registered_minimum_memory_reduction=True,
)
PROOF_FIELDS = {
    "correctness": (
        "retention_one_1e4_pass",
        "independent_compact_same_selected_source_1e4_pass",
        "unpruned_padding_control_normwise_pass",
        "original_KV_positions_mandatory_masks_preserved",
    ),
    "metric": (
        "pinned_official_CIDEr_fixed_fixture_pass",
        "actual_cohort_sparse_official_reducer_equivalence_pass",
        "corpus_reducer_recomputed_each_bootstrap_draw",
    ),
    "byte_matching": (
        "per_sample_initial_physical_byte_match_pass",
        "per_sample_final_physical_byte_match_pass",
        "per_sample_peak_physical_byte_match_pass",
        "padding_preallocation_unique_backing_charged",
        "matched_64_predictions_63_appends",
    ),
    "baseline_fidelity": (
        "vl_cache_reimpl_qualified_as_local_reimplementation",
        "registered_formula_and_native_metric_validation_pass",
    ),
    "history": (
        "global_historical_identity_coverage_complete",
        "all_registered_frontiers_exhausted",
        "all_ambiguities_resolved_with_evidence",
    ),
    "media": (
        "global_historical_media_coverage_complete",
        "reusable_media_and_source_group_inventory_complete",
        "all_ambiguities_resolved_with_evidence",
    ),
}


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _hash(value, size=64):
    if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{size}}}", value) is None:
        raise ValueError("explicit immutable hash required")
    return value


def _binding(value):
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "record"}:
        raise ValueError("already hash-bound metadata capsule required")
    path = value["path"]
    if not isinstance(path, str) or not path or ".." in PurePosixPath(path).parts:
        raise ValueError("explicit safe capsule path required")
    if not isinstance(value["record"], dict):
        raise ValueError("metadata capsule object required")
    return {"path": path, "sha256": _hash(value["sha256"])}


def validate_delegation(capsule, review_record, frozen_record):
    """Validate the explicitly loaded V36 delegation, never follow its path.

    The caller has already bound the record's bytes with CapsuleReader. This
    check neither infers human approval nor grants experiment authorization.
    """
    if not isinstance(review_record, dict) or not isinstance(frozen_record, dict):
        raise ValueError("review and freeze objects required for delegation binding")
    binding = _binding(capsule)
    record = capsule["record"]
    text = record.get("user_instruction_text")
    if (
        record.get("schema") != AUTH_SCHEMA
        or record.get("instruction_origin") != "user_current_turn"
        or not isinstance(text, str)
        or not text.strip()
        or record.get("user_instruction_sha256") != hashlib.sha256(text.encode()).hexdigest()
        or record.get("delegated_original_default_review_authorized") is not True
        or record.get("review_actor") != "codex_agent"
        or record.get("reviewed_by_user") is not False
        or record.get("candidate_config_sha256") != CONFIG_SHA256
        or record.get("scientific_gates_unchanged") is not True
        or record.get("failed_successors_remain_ineligible") is not True
        or record.get("confirmation_ids_known_at_review") is not False
    ):
        raise ValueError("exact hash-bound delegated review required; no invented human approval")
    if (
        review_record.get("authorization_binding") != binding
        or review_record.get("authorization_instruction_sha256")
        != record["user_instruction_sha256"]
        or frozen_record.get("authorization_instruction_sha256")
        != record["user_instruction_sha256"]
    ):
        raise ValueError("loaded delegation differs from the reviewed freeze binding")
    return binding


def validate_qualified_proof(kind, capsule, review_record, *, expected_source_sha):
    """Validate V36 metadata semantics; do NOT reopen its evidence-file links."""
    if kind not in PROOF_FIELDS:
        raise ValueError("unregistered proof kind")
    source = _hash(expected_source_sha, 40)
    if not isinstance(review_record, dict) or review_record.get("source_sha") != source:
        raise ValueError("proof and reviewed source must agree")
    binding = _binding(capsule)
    record = capsule["record"]
    if (
        record.get("schema") != "evicastkv_original_complete_qualified_proof_v36"
        or record.get("kind") != kind
        or record.get("complete") is not True
        or record.get("failures") != []
        or record.get("source_sha") != source
        or record.get("candidate_config_sha256") != CONFIG_SHA256
        or _digest(record.get("controls")) != _digest(CONTROLS)
        or any(record.get(field) is not True for field in PROOF_FIELDS[kind])
    ):
        raise ValueError(f"qualified source/config-bound {kind} metadata required")
    evidence = record.get("evidence_files")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("qualified proof must identify actual completed evidence")
    paths = []
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("qualified evidence path/hash binding required")
        _binding(dict(item, record={}))
        paths.append(item["path"])
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate qualified evidence path")
    if kind == "correctness" and any(
        record.get(key) != expected
        for key, expected in {
            "reference_scope": "equivalent_selected_original_source_not_unrestricted_FullKV",
            "model_scope": "FP16_GQA_COCO_exact_selection",
            "retention_one_max_absolute_logit_error": 1e-4,
            "independent_compact_max_absolute_logit_error": 1e-4,
            "masked_unpruned_normwise_FP16_bound": 8 * 2**-10,
        }.items()
    ):
        raise ValueError("applicable FP16/GQA exact-selection correctness required")
    if kind == "byte_matching" and record.get("compared_methods") != [
        "prompt_attention_topk",
        "vl_cache_reimpl",
        "mosaickv_exact",
    ]:
        raise ValueError("all three compressed methods require actual byte proofs")
    return binding


def resolve_candidate_dependency(
    protocol,
    config,
    *,
    config_file_sha256,
    frozen=None,
    review=None,
    delegation=None,
    proof_capsules=None,
    expected_source_sha=None,
):
    """Bind the original policy without turning source inspection into a freeze."""
    if not isinstance(protocol, dict) or not isinstance(config, dict):
        raise ValueError("protocol and candidate configuration objects required")
    if config_file_sha256 != CONFIG_SHA256 or _digest(config) != CANONICAL_CONFIG_SHA256:
        raise ValueError("only the exact registered original configuration is supported")
    reference = protocol.get("source_references", {}).get("configs", {}).get("mosaickv_exact", {})
    if reference.get("file_sha256") != CONFIG_SHA256:
        raise ValueError("protocol must bind the registered original configuration")
    base_source = _hash(protocol.get("source_references", {}).get("base_source_sha"), 40)
    policy = {
        "routing": {
            "mode": "normalized_pooled_prompt_attention_entropy",
            "fraction": config["graph"]["uncertain_head_fraction"],
            "threshold": config["graph"]["uncertain_head_threshold"],
            "rounding": (
                "max(1,ceil(eligible_route_count*fraction)); threshold; best-route fallback"
            ),
            "tie_break": "layer_then_KV_head",
        },
        "coefficients": {
            "utility": deepcopy(config["utility"]),
            "lambda_g": config["selection"]["lambda_g"],
            "lambda_m": config["selection"]["lambda_m"],
        },
        "forecast": deepcopy(config["forecasting"]),
        "graph": deepcopy(config["graph"]),
        "selection": deepcopy(config["selection"]),
        "capacity": deepcopy(config["cache"]),
        "controls": deepcopy(CONTROLS),
    }
    result = dict(
        schema="evicastkv_confirmation_candidate_dependency_v1",
        status="reference_bound_not_qualified",
        qualified=False,
        parameters_frozen=False,
        candidate_config_file_sha256=CONFIG_SHA256,
        source_sha=base_source,
        canonical_policy_sha256=_digest(policy),
        resolved_policy=policy,
        qualification_proofs={},
        selected_config=None,
        unresolved=["qualified_original_default_freeze_required"],
        authorization=dict.fromkeys(AUTHORIZATION_KEYS, False),
        fresh_confirmation_authorized=False,
        GPU_execution_authorized=False,
    )
    if frozen is None:
        return result
    if not isinstance(frozen, dict) or frozen.get("schema") != FREEZE_SCHEMA:
        raise ValueError("only the V36 reviewed original-default freeze is supported")
    if review is None or proof_capsules is None:
        result["unresolved"] = ["review_and_four_qualified_proof_capsules_required"]
        return result
    if delegation is None:
        result["unresolved"] = ["reviewed_delegation_capsule_required"]
        return result
    review_binding = _binding(review)
    reviewed = review["record"]
    source = _hash(frozen.get("source_sha"), 40)
    if expected_source_sha is not None and source != _hash(expected_source_sha, 40):
        raise ValueError("current-source qualification or reviewed source bridge required")
    for record, schema in ((frozen, FREEZE_SCHEMA), (reviewed, REVIEW_SCHEMA)):
        if (
            record.get("schema") != schema
            or record.get("source_sha") != source
            or record.get("candidate_config_sha256") != CONFIG_SHA256
            or _digest(record.get("controls")) != _digest(CONTROLS)
            or _digest(record.get("scientific_contract")) != _digest(SCIENTIFIC_CONTRACT)
            or record.get("review_actor") != "codex_agent"
            or record.get("reviewed_by_user") is not False
            or record.get("review_mode")
            != "explicitly_delegated_prospective_original_default_review"
            or record.get("positive_development_selection_claimed") is not False
        ):
            raise ValueError("immutable reviewed original-default contract differs")
    if (
        frozen.get("frozen") is not True
        or frozen.get("candidate_parameters_frozen") is not True
        or frozen.get("review_sha256") != review_binding["sha256"]
        or frozen.get("never_selected_from_successor_win") is not True
        or frozen.get("confirmation_sample_ids_not_selected") is not True
        or frozen.get("independent_confirmation_may_be_negative") is not True
        or frozen.get("history_media_coverage_is_confirmation_not_parameter_selection_gate")
        is not True
        or reviewed.get("confirmation_ids_known_at_review") is not False
        or reviewed.get("selection_basis")
        != "original_registered_complete_default_not_favorable_successor_outcomes"
    ):
        raise ValueError("prospective original-default freeze binding required")
    instruction_sha = _hash(reviewed.get("authorization_instruction_sha256"))
    if frozen.get("authorization_instruction_sha256") != instruction_sha:
        raise ValueError("freeze and delegated review instruction binding differ")
    authority = reviewed.get("authorization_binding")
    if not isinstance(authority, dict) or set(authority) != {"path", "sha256"}:
        raise ValueError("original delegated-review authorization binding required")
    _binding(dict(authority, record={}))
    delegation_binding = validate_delegation(delegation, reviewed, frozen)
    for record in (frozen, reviewed):
        for name in (
            "materialization_authorized",
            "fresh_confirmation_authorized",
            "paper_claim_allowed",
        ):
            if record.get(name) is not False:
                raise ValueError("candidate metadata must not authorize execution")
    if not isinstance(proof_capsules, dict) or set(proof_capsules) != set(CORE_PROOFS):
        raise ValueError("exact four original-default engineering proof capsules required")
    paths = frozen.get("proof_paths_by_kind")
    hashes = frozen.get("qualifying_proof_bindings")
    if (
        not isinstance(paths, dict)
        or set(paths) != set(CORE_PROOFS)
        or not isinstance(hashes, dict)
    ):
        raise ValueError("frozen proof path/hash inventory required")
    bindings = {}
    for kind in CORE_PROOFS:
        bound = validate_qualified_proof(
            kind, proof_capsules[kind], reviewed, expected_source_sha=source
        )
        if paths[kind] != bound["path"] or hashes.get(bound["path"]) != bound["sha256"]:
            raise ValueError("frozen qualified proof binding changed")
        bindings[kind] = bound
    if len({v["path"] for v in bindings.values()}) != 4 or set(hashes) != {
        v["path"] for v in bindings.values()
    }:
        raise ValueError("exact four distinct frozen proof paths required")
    result.update(
        status="parameters_frozen_confirmation_blocked",
        qualified=True,
        parameters_frozen=True,
        qualification_proofs=deepcopy(bindings),
        source_sha=source,
        selected_config={
            "schema": FREEZE_SCHEMA,
            "source_sha": source,
            "candidate_config_sha256": CONFIG_SHA256,
            "review": review_binding,
            "delegation": delegation_binding,
            "proofs": bindings,
            "scientific_contract": deepcopy(SCIENTIFIC_CONTRACT),
        },
        unresolved=["history_media_cohort_and_confirmation_admission_required"],
    )
    return result
