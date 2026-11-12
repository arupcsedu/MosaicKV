"""Synthetic delegated-review fixtures, not a real candidate freeze or GPU result."""

import hashlib
import json
from copy import deepcopy

import evicastkv_original_complete_admission_v36 as admission
import pytest


def write(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return dict(path=str(path), sha256=admission.sha(path))


def authorization():
    text = "Synthetic delegation fixture, never an assertion of an actual user instruction."
    return dict(
        schema=admission.AUTH_SCHEMA,
        instruction_origin="user_current_turn",
        user_instruction_text=text,
        user_instruction_sha256=hashlib.sha256(text.encode()).hexdigest(),
        delegated_original_default_review_authorized=True,
        review_actor="codex_agent",
        reviewed_by_user=False,
        candidate_config_sha256=admission.prior.CANDIDATE_SHA,
        scientific_gates_unchanged=True,
        failed_successors_remain_ineligible=True,
        confirmation_ids_known_at_review=False,
    )


@pytest.fixture
def reviewed(monkeypatch, tmp_path):
    monkeypatch.setattr(admission, "clean_source", lambda *_args: "a" * 40)
    monkeypatch.setattr(admission, "original_evidence", lambda _reader: [])
    path = tmp_path / "authorization.json"
    write(path, authorization())
    root = tmp_path / "review"
    admission.preregister(root, path, admission.sha(path))
    return root


def proofs(root):
    review = json.loads((root / "original_complete_review.json").read_text())
    evidence = write(root / "synthetic-engineering-gate.json", dict(synthetic=True))
    fields = {
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
    records, paths = {}, {}
    for kind in admission.PROOFS:
        record = dict(
            schema="evicastkv_original_complete_qualified_proof_v36",
            kind=kind,
            complete=True,
            failures=[],
            source_sha=review["source_sha"],
            candidate_config_sha256=admission.prior.CANDIDATE_SHA,
            controls=deepcopy(admission.prior.CONTROLS),
            evidence_files=[evidence],
            **dict.fromkeys(fields[kind], True),
        )
        if kind == "correctness":
            record.update(
                reference_scope="equivalent_selected_original_source_not_unrestricted_FullKV",
                model_scope="FP16_GQA_COCO_exact_selection",
                retention_one_max_absolute_logit_error=1e-4,
                independent_compact_max_absolute_logit_error=1e-4,
                masked_unpruned_normwise_FP16_bound=8 * 2**-10,
            )
        if kind == "byte_matching":
            record["compared_methods"] = [
                "prompt_attention_topk",
                "vl_cache_reimpl",
                "mosaickv_exact",
            ]
        path = root / f"synthetic-{kind}-proof.json"
        write(path, record)
        records[kind], paths[kind] = record, path
    return records, paths


def test_review_is_preregistered_without_fabricating_user_or_quality_approval(reviewed):
    record = json.loads((reviewed / "original_complete_review.json").read_text())
    assert record["review_actor"] == "codex_agent"
    assert record["reviewed_by_user"] is False
    assert record["positive_development_selection_claimed"] is False
    assert record["confirmation_ids_known_at_review"] is False
    assert not (reviewed / "selected_config.json").exists()
    assert record["jobs_submitted"] == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("reviewed_by_user", True),
        ("review_actor", "human"),
        ("scientific_gates_unchanged", False),
        ("failed_successors_remain_ineligible", False),
        ("confirmation_ids_known_at_review", True),
        ("user_instruction_sha256", "f" * 64),
        ("candidate_config_sha256", "c" * 64),
    ],
)
def test_no_forged_approval_changed_candidate_or_weakened_gates(key, value):
    record = authorization()
    record[key] = value
    with pytest.raises(ValueError, match="delegated review"):
        admission.validate_authorization(record)


def test_absent_proofs_leave_default_unfrozen_without_reviving_successors(reviewed):
    decision = admission.finalize(reviewed, {})
    assert decision["status"] == "blocked"
    assert len(decision["failures"]) == 6
    assert not (reviewed / "selected_config.json").exists()
    assert decision["successor_gates_unchanged"] is True
    assert decision["jobs_submitted"] == 0
    with pytest.raises(FileExistsError):
        admission.finalize(reviewed, {})


def test_all_synthetic_qualified_proofs_freeze_default_not_superiority_or_execution(reviewed):
    _records, paths = proofs(reviewed)
    decision = admission.finalize(reviewed, paths)
    assert decision["status"] == "prerequisites_passed_cohort_disjointness_still_required"
    frozen = json.loads((reviewed / "selected_config.json").read_text())
    assert frozen["schema"] == admission.FREEZE_SCHEMA
    assert frozen["controls"] == admission.prior.CONTROLS
    assert frozen["positive_development_selection_claimed"] is False
    assert frozen["never_selected_from_successor_win"] is True
    assert frozen["independent_confirmation_may_be_negative"] is True
    assert frozen["reviewed_by_user"] is False
    for key in (
        "fresh_confirmation_authorized",
        "materialization_authorized",
        "paper_claim_allowed",
    ):
        assert frozen[key] is False
    assert frozen["jobs_submitted"] == 0


@pytest.mark.parametrize("kind", admission.PROOFS)
def test_engineering_failure_blocks_freeze_coverage_failure_blocks_confirmation(reviewed, kind):
    records, paths = proofs(reviewed)
    records[kind]["complete"] = False
    write(paths[kind], records[kind])
    result = admission.finalize(reviewed, paths)
    assert result["prerequisites"][kind] is False
    assert (reviewed / "selected_config.json").exists() is (kind not in admission.CANDIDATE_PROOFS)


def test_parameters_freeze_before_freshness_while_confirmation_stays_blocked(reviewed):
    _records, paths = proofs(reviewed)
    core = {kind: paths[kind] for kind in admission.CANDIDATE_PROOFS}
    frozen = admission.freeze_candidate(reviewed, core)
    assert frozen["candidate_parameters_frozen"] is True
    assert frozen["status"] == "frozen_original_default_not_a_quality_win"
    assert set(frozen["prerequisites"]) == set(admission.CANDIDATE_PROOFS)
    result = admission.finalize(reviewed, {})
    assert result["candidate_parameters_frozen"] is True
    assert result["status"] == "blocked"
    assert {row["prerequisite"] for row in result["failures"]} == {"history", "media"}
    assert result["fresh_confirmation_authorized"] is False


def test_later_coverage_checks_reuse_frozen_core_without_selecting_new_parameters(reviewed):
    _records, paths = proofs(reviewed)
    admission.freeze_candidate(reviewed, paths)
    frozen_sha = admission.sha(reviewed / "selected_config.json")
    result = admission.finalize(reviewed, {kind: paths[kind] for kind in ("history", "media")})
    assert result["status"] == "prerequisites_passed_cohort_disjointness_still_required"
    assert admission.sha(reviewed / "selected_config.json") == frozen_sha
    assert result["cohort_disjointness_not_yet_checked"] is True
    assert result["fresh_confirmation_authorized"] is False


def test_changed_engineering_artifact_after_parameter_freeze_is_rejected(reviewed):
    records, paths = proofs(reviewed)
    admission.freeze_candidate(reviewed, paths)
    records["correctness"]["extra"] = "changed"
    write(paths["correctness"], records["correctness"])
    with pytest.raises(ValueError, match="frozen engineering evidence changed"):
        admission.finalize(reviewed, paths)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha", "d" * 40),
        ("model_scope", "FP16_GQA_COCO_FullKV_evaluation_only_boundary_control"),
        ("model_scope", "BF16_MHA_LLaVA_1_5_source_reader"),
        ("reference_scope", "unrestricted_FullKV"),
        ("retention_one_max_absolute_logit_error", 0.001),
        ("independent_compact_max_absolute_logit_error", 0.001),
        ("masked_unpruned_normwise_FP16_bound", 0.1),
    ],
)
def test_no_model_source_reference_or_tolerance_transfer(reviewed, field, value):
    records, paths = proofs(reviewed)
    records["correctness"][field] = value
    write(paths["correctness"], records["correctness"])
    result = admission.finalize(reviewed, paths)
    assert result["prerequisites"]["correctness"] is False
    assert not (reviewed / "selected_config.json").exists()


def test_proof_must_bind_real_unchanged_evidence(reviewed):
    _records, paths = proofs(reviewed)
    write(reviewed / "synthetic-engineering-gate.json", dict(synthetic="modified"))
    result = admission.finalize(reviewed, paths)
    assert not any(result["prerequisites"].values())
    assert not (reviewed / "selected_config.json").exists()


def test_scientific_contract_uses_original_not_successor_family_and_thresholds():
    contract = admission.SCIENTIFIC_CONTRACT
    assert contract["primary_family_size"] == 6
    assert contract["bootstrap_replicates"] == 10000
    assert contract["quality_samples"] == 256
    assert contract["warmups"] == 5 and contract["timing_repeats"] == 3
    assert contract["retention_one_max_absolute_logit_error"] == 1e-4
    assert contract["latency_adjusted_upper_candidate_baseline_ratio_maximum"] == 1.0
    assert contract["no_optional_enlargement"] is True


@pytest.mark.parametrize("revived", [False, True])
def test_original_review_preserves_failed_successor_records(monkeypatch, tmp_path, revived):
    config = dict(
        model=dict(
            id=admission.prior.CONTROLS["model_id"],
            revision=admission.prior.CONTROLS["model_revision"],
            precision="fp16",
        ),
        dataset=dict(
            id=admission.prior.CONTROLS["dataset_id"],
            revision=admission.prior.CONTROLS["dataset_revision"],
            split="val",
        ),
        execution=dict(attention_implementation="eager", backend="huggingface"),
        method="mosaickv_exact",
        cache=dict(block_size=1),
        generation=dict(max_new_tokens=64),
    )
    candidate_path = tmp_path / "candidate.json"
    write(candidate_path, config)
    monkeypatch.setattr(admission.prior, "CANDIDATE", candidate_path)
    monkeypatch.setattr(admission.prior, "REVIEW_TEMPLATE", candidate_path)
    monkeypatch.setattr(admission.prior, "CANDIDATE_SHA", admission.sha(candidate_path))
    gate_path = tmp_path / "failed-successor.json"
    write(
        gate_path,
        dict(
            status="passed_execution_development_only",
            quality_decisions={
                "evidence-routing-v1": dict(
                    required_contrasts=4, positive_development_signal=revived
                )
            },
        ),
    )
    monkeypatch.setattr(
        admission.prior, "SELECTION_GATES", ((gate_path, admission.sha(gate_path)),)
    )
    reader = admission.prior.MetadataReader()
    if revived:
        with pytest.raises(ValueError, match="successor selection evidence changed"):
            admission.original_evidence(reader)
    else:
        negatives = admission.original_evidence(reader)
        assert (
            negatives[0]["quality_decisions"]["evidence-routing-v1"]["positive_development_signal"]
            is False
        )
