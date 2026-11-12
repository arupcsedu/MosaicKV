#!/usr/bin/env python3
"""Prospective delegated review of the ORIGINAL complete COCO policy, not a win.

Prompt 06 permits a reviewed trade-off/default configuration; its requirement
is not the later, stricter opportunity-successor selection rule. This separate
route preserves both failed successor gates. It never selects a new variant,
reads benchmark outputs, chooses IDs, launches jobs, or waives scientific gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import evicastkv_coco_confirmation_admission_v35 as prior

from mosaickv.evaluation.fox_campaign import clean_source, sha, write_once

SOURCE = Path(__file__).resolve().parents[2]
AUTH_SCHEMA = "evicastkv_original_complete_delegated_review_authorization_v36"
REVIEW_SCHEMA = "evicastkv_original_complete_prospective_review_v36"
FREEZE_SCHEMA = "evicastkv_original_complete_freeze_v36"
PROOFS = (
    "correctness",
    "metric",
    "byte_matching",
    "baseline_fidelity",
    "history",
    "media",
)
CANDIDATE_PROOFS = PROOFS[:4]
SCIENTIFIC_CONTRACT = dict(
    candidate="original_complete_not_a_selected_opportunity_successor",
    controls=prior.CONTROLS,
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


def validate_authorization(record):
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
        or record.get("candidate_config_sha256") != prior.CANDIDATE_SHA
        or record.get("scientific_gates_unchanged") is not True
        or record.get("failed_successors_remain_ineligible") is not True
        or record.get("confirmation_ids_known_at_review") is not False
    ):
        raise ValueError("exact hash-bound delegated review required; no invented human approval")


def original_evidence(reader):
    config = reader.read(prior.CANDIDATE, prior.CANDIDATE_SHA)
    if config != reader.read(prior.REVIEW_TEMPLATE, prior.CANDIDATE_SHA):
        raise ValueError("original reviewed candidate content changed")
    if prior.config_controls(config) != prior.CONTROLS:
        raise ValueError("original COCO policy/model changed")
    negatives = []
    for path, expected in prior.SELECTION_GATES:
        gate = reader.read(path, expected)
        decisions = gate.get("quality_decisions", {})
        if (
            gate.get("status") != "passed_execution_development_only"
            or not decisions
            or any(
                value.get("positive_development_signal") is not False
                or value.get("required_contrasts") != 4
                for value in decisions.values()
            )
        ):
            raise ValueError("failed registered successor selection evidence changed")
        negatives.append(dict(path=str(path), sha256=expected, quality_decisions=decisions))
    for relative in (
        "00_MASTER.md",
        "06_MECHANISM_ABLATIONS.md",
        "07_FRESH_CONFIRMATION.md",
        "09_STATISTICS_AND_EXPORT.md",
    ):
        path = SOURCE / "mosaickv/evicastkv_experiment_prompts" / relative
        reader.bindings[str(path.resolve(strict=True))] = dict(
            actual_sha256=sha(path), expected_sha256=None, hash_verified=True
        )
    return negatives


def preregister(root, authorization, authorization_sha256):
    revision = clean_source(SOURCE)
    root = Path(root)
    if root.exists():
        raise FileExistsError("a new immutable review root is required")
    if re.fullmatch(r"[0-9a-f]{64}", authorization_sha256) is None:
        raise ValueError("explicit authorization-ledger SHA256 required")
    reader = prior.MetadataReader()
    authority = reader.read(authorization, authorization_sha256)
    validate_authorization(authority)
    negatives = original_evidence(reader)
    review = dict(
        schema=REVIEW_SCHEMA,
        source_sha=revision,
        candidate_config_sha256=prior.CANDIDATE_SHA,
        controls=prior.CONTROLS,
        scientific_contract=SCIENTIFIC_CONTRACT,
        review_actor="codex_agent",
        reviewed_by_user=False,
        review_mode="explicitly_delegated_prospective_original_default_review",
        procedural_change="delegated_actor_disclosed_not_fabricated_human_review",
        selection_basis="original_registered_complete_default_not_favorable_successor_outcomes",
        original_development_outcomes_already_exist=True,
        positive_development_selection_claimed=False,
        successor_variants_rejected_not_revived=negatives,
        authorization_binding=dict(
            path=str(Path(authorization).resolve(strict=True)), sha256=authorization_sha256
        ),
        authorization_instruction_sha256=authority["user_instruction_sha256"],
        evidence_bindings=reader.bindings,
        confirmation_ids_known_at_review=False,
        new_sample_ids_accessed=False,
        materialization_authorized=False,
        fresh_confirmation_authorized=False,
        paper_claim_allowed=False,
        jobs_submitted=0,
    )
    write_once(root / "original_complete_review.json", review)
    return dict(status="preregistered_original_default_not_yet_frozen", jobs_submitted=0)


def load_review(root):
    root = Path(root).resolve(strict=True)
    reader = prior.MetadataReader()
    review = reader.read(root / "original_complete_review.json")
    clean_source(SOURCE, review["source_sha"])
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("controls") != prior.CONTROLS
        or review.get("candidate_config_sha256") != prior.CANDIDATE_SHA
        or review.get("scientific_contract") != SCIENTIFIC_CONTRACT
        or review.get("review_actor") != "codex_agent"
        or review.get("reviewed_by_user") is not False
        or review.get("positive_development_selection_claimed") is not False
        or review.get("confirmation_ids_known_at_review") is not False
    ):
        raise ValueError("original default review or scientific contract changed")
    for path, binding in review["evidence_bindings"].items():
        if sha(Path(path)) != binding["actual_sha256"]:
            raise ValueError("preregistered evidence binding changed")
    validate_authorization(reader.linked(review, "authorization_binding"))
    original_evidence(reader)
    return review, reader


def qualifying_proof(kind, record, review, reader):
    """New derived proof metadata must bind actual evidence, not a bare PASS."""
    if (
        record.get("schema") != "evicastkv_original_complete_qualified_proof_v36"
        or record.get("kind") != kind
        or record.get("complete") is not True
        or record.get("failures") != []
        or record.get("source_sha") != review["source_sha"]
        or record.get("candidate_config_sha256") != prior.CANDIDATE_SHA
        or record.get("controls") != prior.CONTROLS
        or not isinstance(record.get("evidence_files"), list)
        or not record["evidence_files"]
    ):
        raise ValueError(f"complete current-source/model/config-bound {kind} proof required")
    for binding in record["evidence_files"]:
        if (
            not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or not isinstance(binding["path"], str)
            or re.fullmatch(r"[0-9a-f]{64}", str(binding["sha256"])) is None
        ):
            raise ValueError("actual evidence path and SHA256 binding required")
        path = Path(binding["path"]).resolve(strict=True)
        actual = sha(path)
        reader.bindings[str(path)] = dict(
            actual_sha256=actual,
            expected_sha256=binding["sha256"],
            hash_verified=actual == binding["sha256"],
        )
        if actual != binding["sha256"]:
            raise ValueError("qualified proof actual evidence binding changed")
    required = {
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
    }[kind]
    if any(record.get(name) is not True for name in required):
        raise ValueError(f"unchanged {kind} scientific prerequisite has not passed")
    if kind == "byte_matching" and record.get("compared_methods") != [
        "prompt_attention_topk",
        "vl_cache_reimpl",
        "mosaickv_exact",
    ]:
        raise ValueError("byte proof must cover all three registered compressed methods")
    if kind == "correctness" and (
        record.get("reference_scope")
        != "equivalent_selected_original_source_not_unrestricted_FullKV"
        or record.get("model_scope") != "FP16_GQA_COCO_exact_selection"
        or record.get("retention_one_max_absolute_logit_error") != 1e-4
        or record.get("independent_compact_max_absolute_logit_error") != 1e-4
        or record.get("masked_unpruned_normwise_FP16_bound") != 8 * 2**-10
    ):
        raise ValueError("matching compressed GQA proof required; no FullKV/BF16 source transfer")
    return True


def _decide(root, proofs, *, kinds, decision_name):
    root = Path(root)
    if (root / decision_name).exists():
        raise FileExistsError("finalization is write-once; retain blocked decisions")
    review, reader = load_review(root)
    proofs = dict(proofs)
    selected_path = root / "selected_config.json"
    existing_freeze = None
    if selected_path.exists():
        existing_freeze = reader.read(selected_path)
        if (
            existing_freeze.get("schema") != FREEZE_SCHEMA
            or existing_freeze.get("candidate_parameters_frozen") is not True
            or existing_freeze.get("source_sha") != review["source_sha"]
            or existing_freeze.get("candidate_config_sha256") != prior.CANDIDATE_SHA
            or existing_freeze.get("controls") != prior.CONTROLS
            or existing_freeze.get("scientific_contract") != SCIENTIFIC_CONTRACT
            or existing_freeze.get("review_sha256") != sha(root / "original_complete_review.json")
        ):
            raise ValueError("immutable existing candidate freeze changed")
        for kind in CANDIDATE_PROOFS:
            frozen_path = existing_freeze["proof_paths_by_kind"][kind]
            if proofs.get(kind) is not None and Path(proofs[kind]).resolve() != Path(frozen_path):
                raise ValueError(
                    "engineering proof replacement after parameter freeze is forbidden"
                )
            proofs[kind] = frozen_path
        for path, expected in existing_freeze["qualifying_proof_bindings"].items():
            if sha(Path(path)) != expected:
                raise ValueError("frozen engineering evidence changed")
    results, failures = {}, []
    for kind in kinds:
        try:
            if proofs.get(kind) is None:
                raise ValueError(f"missing qualifying {kind} proof")
            record = reader.read(proofs[kind])
            results[kind] = qualifying_proof(kind, record, review, reader)
        except (OSError, ValueError, KeyError, TypeError) as error:
            results[kind] = False
            failures.append(dict(prerequisite=kind, error=str(error)))
    decision = dict(
        status="blocked",
        source_sha=review["source_sha"],
        candidate_config_sha256=prior.CANDIDATE_SHA,
        review_sha256=sha(root / "original_complete_review.json"),
        prerequisites=results,
        failures=failures,
        evidence_bindings=reader.bindings,
        positive_development_selection_claimed=False,
        successor_gates_unchanged=True,
        cohort_disjointness_not_yet_checked=True,
        independent_quality_not_yet_tested=True,
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        paper_claim_allowed=False,
        jobs_submitted=0,
    )
    candidate_ready = all(results.get(kind) is True for kind in CANDIDATE_PROOFS)
    if candidate_ready and existing_freeze is None:
        write_once(
            selected_path,
            dict(
                schema=FREEZE_SCHEMA,
                frozen=True,
                candidate_parameters_frozen=True,
                source_sha=review["source_sha"],
                candidate_config_sha256=prior.CANDIDATE_SHA,
                controls=prior.CONTROLS,
                scientific_contract=SCIENTIFIC_CONTRACT,
                review_actor="codex_agent",
                reviewed_by_user=False,
                review_mode=review["review_mode"],
                review_sha256=decision["review_sha256"],
                authorization_instruction_sha256=review["authorization_instruction_sha256"],
                qualifying_proof_bindings={
                    str(Path(proofs[kind]).resolve()): sha(Path(proofs[kind]))
                    for kind in CANDIDATE_PROOFS
                },
                proof_paths_by_kind={
                    kind: str(Path(proofs[kind]).resolve()) for kind in CANDIDATE_PROOFS
                },
                evidence_bindings=reader.bindings,
                positive_development_selection_claimed=False,
                never_selected_from_successor_win=True,
                confirmation_sample_ids_not_selected=True,
                history_media_coverage_is_confirmation_not_parameter_selection_gate=True,
                independent_confirmation_may_be_negative=True,
                materialization_authorized=False,
                fresh_confirmation_authorized=False,
                paper_claim_allowed=False,
                jobs_submitted=0,
            ),
        )
    decision["candidate_parameters_frozen"] = candidate_ready
    if candidate_ready:
        decision["selected_config_sha256"] = sha(selected_path)
        if kinds == CANDIDATE_PROOFS:
            decision["status"] = "frozen_original_default_not_a_quality_win"
        elif all(results.values()):
            decision["status"] = "prerequisites_passed_cohort_disjointness_still_required"
    write_once(root / decision_name, decision)
    return decision


def freeze_candidate(root, proofs):
    """Freeze fixed parameters before any fresh cohort is selected or audited."""
    return _decide(
        root, proofs, kinds=CANDIDATE_PROOFS, decision_name="candidate_freeze_decision.json"
    )


def finalize(root, proofs):
    """Check all engineering/coverage gates without authorizing confirmation."""
    return _decide(root, proofs, kinds=PROOFS, decision_name="freeze_decision.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preregister", "freeze-candidate", "finalize"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256")
    for kind in PROOFS:
        parser.add_argument(f"--{kind.replace('_', '-')}-proof", type=Path)
    args = parser.parse_args()
    if args.mode == "preregister":
        if args.authorization is None or args.authorization_sha256 is None:
            parser.error("preregister requires exact authorization path and SHA256")
        result = preregister(args.root, args.authorization, args.authorization_sha256)
    else:
        function = freeze_candidate if args.mode == "freeze-candidate" else finalize
        result = function(args.root, {kind: getattr(args, f"{kind}_proof") for kind in PROOFS})
    print(json.dumps(dict(status=result["status"], jobs_submitted=0), sort_keys=True))


if __name__ == "__main__":
    main()
