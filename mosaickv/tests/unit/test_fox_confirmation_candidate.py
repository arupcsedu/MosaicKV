"""CPU-only synthetic contracts; no scientific freeze, model or data execution."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from mosaickv.evaluation import fox_confirmation_candidate as candidate

ROOT = Path(__file__).parents[2]


def capsule(path, record):
    raw = json.dumps(record, sort_keys=True, allow_nan=False).encode()
    return dict(path=path, sha256=hashlib.sha256(raw).hexdigest(), record=record)


@pytest.fixture
def inputs():
    protocol = json.loads(
        (ROOT / "configs/experiments/evicastkv_fox/review_completion_v63.json").read_text()
    )
    config = json.loads(
        (ROOT / "configs/experiments/evicastkv_fox/coco/mosaickv_exact.json").read_text()
    )
    source = "a" * 40
    instruction = "Synthetic unit-fixture delegation; not a real execution authorization."
    delegation = capsule(
        "/synthetic/authorization.json",
        dict(
            schema=candidate.AUTH_SCHEMA,
            instruction_origin="user_current_turn",
            user_instruction_text=instruction,
            user_instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
            delegated_original_default_review_authorized=True,
            review_actor="codex_agent",
            reviewed_by_user=False,
            candidate_config_sha256=candidate.CONFIG_SHA256,
            scientific_gates_unchanged=True,
            failed_successors_remain_ineligible=True,
            confirmation_ids_known_at_review=False,
        ),
    )
    review = dict(
        schema=candidate.REVIEW_SCHEMA,
        source_sha=source,
        candidate_config_sha256=candidate.CONFIG_SHA256,
        controls=deepcopy(candidate.CONTROLS),
        scientific_contract=deepcopy(candidate.SCIENTIFIC_CONTRACT),
        review_actor="codex_agent",
        reviewed_by_user=False,
        review_mode="explicitly_delegated_prospective_original_default_review",
        positive_development_selection_claimed=False,
        confirmation_ids_known_at_review=False,
        selection_basis="original_registered_complete_default_not_favorable_successor_outcomes",
        authorization_instruction_sha256=delegation["record"]["user_instruction_sha256"],
        authorization_binding={k: delegation[k] for k in ("path", "sha256")},
        materialization_authorized=False,
        fresh_confirmation_authorized=False,
        paper_claim_allowed=False,
    )
    review = capsule("/synthetic/review.json", review)
    proofs = {}
    for kind in candidate.PROOF_FIELDS:
        record = dict(
            schema="evicastkv_original_complete_qualified_proof_v36",
            kind=kind,
            complete=True,
            failures=[],
            source_sha=source,
            candidate_config_sha256=candidate.CONFIG_SHA256,
            controls=deepcopy(candidate.CONTROLS),
            evidence_files=[dict(path=f"/synthetic/closed-{kind}.json", sha256="d" * 64)],
            **dict.fromkeys(candidate.PROOF_FIELDS[kind], True),
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
        proofs[kind] = capsule(f"/synthetic/proof-{kind}.json", record)
    frozen = deepcopy(review["record"])
    frozen.update(
        schema=candidate.FREEZE_SCHEMA,
        frozen=True,
        candidate_parameters_frozen=True,
        review_sha256=review["sha256"],
        never_selected_from_successor_win=True,
        confirmation_sample_ids_not_selected=True,
        independent_confirmation_may_be_negative=True,
        history_media_coverage_is_confirmation_not_parameter_selection_gate=True,
        proof_paths_by_kind={k: proofs[k]["path"] for k in candidate.CORE_PROOFS},
        qualifying_proof_bindings={
            proofs[k]["path"]: proofs[k]["sha256"] for k in candidate.CORE_PROOFS
        },
    )
    return dict(
        protocol=protocol,
        config=config,
        config_file_sha256=candidate.CONFIG_SHA256,
        frozen=frozen,
        review=review,
        delegation=delegation,
        proof_capsules={k: proofs[k] for k in candidate.CORE_PROOFS},
        expected_source_sha=source,
    ), proofs


def test_static_contract_exactly_matches_existing_v36_default():
    import evicastkv_original_complete_admission_v36 as original

    assert candidate.CONTROLS == original.prior.CONTROLS
    assert candidate.SCIENTIFIC_CONTRACT == original.SCIENTIFIC_CONTRACT
    assert candidate.FREEZE_SCHEMA == original.FREEZE_SCHEMA
    assert candidate.CORE_PROOFS == original.CANDIDATE_PROOFS
    assert candidate.AUTH_SCHEMA == original.AUTH_SCHEMA


def test_reference_binding_recovers_policy_but_is_not_selection_or_qualification(inputs):
    args, _ = inputs
    result = candidate.resolve_candidate_dependency(
        args["protocol"], args["config"], config_file_sha256=candidate.CONFIG_SHA256
    )
    assert result["status"] == "reference_bound_not_qualified"
    assert result["selected_config"] is None
    assert result["qualified"] is False and result["parameters_frozen"] is False
    assert result["qualification_proofs"] == {}
    assert result["resolved_policy"]["routing"]["fraction"] == 1 / 256
    assert result["resolved_policy"]["forecast"]["prompt_window"] == 16
    assert result["resolved_policy"]["coefficients"]["utility"]["lambda_q"] == 1.0
    assert not any(result["authorization"].values())


def test_synthetic_qualified_default_needs_no_positive_development_pvalue(inputs, monkeypatch):
    args, _ = inputs
    before = deepcopy(args)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pure adapter must not open any files")

    monkeypatch.setattr(Path, "open", forbidden)
    result = candidate.resolve_candidate_dependency(**args)
    assert args == before
    assert result["status"] == "parameters_frozen_confirmation_blocked"
    assert result["qualified"] is True and result["parameters_frozen"] is True
    assert set(result["qualification_proofs"]) == set(candidate.CORE_PROOFS)
    assert result["fresh_confirmation_authorized"] is False
    assert result["GPU_execution_authorized"] is False
    assert not any(result["authorization"].values())
    assert result["unresolved"] == ["history_media_cohort_and_confirmation_admission_required"]


@pytest.mark.parametrize("missing", ("review", "proof_capsules", "delegation"))
def test_missing_capsules_never_qualify(inputs, missing):
    args, _ = inputs
    args[missing] = None
    result = candidate.resolve_candidate_dependency(**args)
    assert result["qualified"] is False and result["selected_config"] is None


@pytest.mark.parametrize("kind", candidate.CORE_PROOFS)
def test_each_engineering_proof_is_required_and_source_scoped(inputs, kind):
    args, _ = inputs
    args["proof_capsules"][kind]["record"]["source_sha"] = "b" * 40
    with pytest.raises(ValueError, match="source/config-bound"):
        candidate.resolve_candidate_dependency(**args)


@pytest.mark.parametrize("kind", ("history", "media"))
def test_history_media_remain_separate_confirmation_gates(inputs, kind):
    args, proofs = inputs
    result = candidate.validate_qualified_proof(
        kind, proofs[kind], args["review"]["record"], expected_source_sha="a" * 40
    )
    assert result == dict(path=proofs[kind]["path"], sha256=proofs[kind]["sha256"])
    proofs[kind]["record"][candidate.PROOF_FIELDS[kind][0]] = False
    with pytest.raises(ValueError, match="source/config-bound"):
        candidate.validate_qualified_proof(
            kind, proofs[kind], args["review"]["record"], expected_source_sha="a" * 40
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "config",
        "file_sha",
        "freeze_schema",
        "proof_hash",
        "review_hash",
        "source",
        "policy",
        "authorization",
        "bare_pass",
    ),
)
def test_changed_or_bare_pass_metadata_fails_closed(inputs, mutation):
    args, _ = inputs
    if mutation == "config":
        args["config"]["utility"]["lambda_q"] = 2.0
    elif mutation == "file_sha":
        args["config_file_sha256"] = "f" * 64
    elif mutation == "freeze_schema":
        args["frozen"]["schema"] = "evicastkv_coco_confirmation_freeze_v35"
    elif mutation == "proof_hash":
        args["proof_capsules"]["correctness"]["sha256"] = "f" * 64
    elif mutation == "review_hash":
        args["review"]["sha256"] = "f" * 64
    elif mutation == "source":
        args["expected_source_sha"] = "f" * 40
    elif mutation == "policy":
        args["frozen"]["scientific_contract"][
            "latency_adjusted_upper_candidate_baseline_ratio_maximum"
        ] = 1.05
    elif mutation == "authorization":
        args["frozen"]["fresh_confirmation_authorized"] = True
    elif mutation == "bare_pass":
        args["proof_capsules"]["correctness"]["record"] = dict(status="passed")
    with pytest.raises(ValueError):
        candidate.resolve_candidate_dependency(**args)


@pytest.mark.parametrize("kind", ("correctness", "metric", "byte_matching", "baseline_fidelity"))
def test_negative_completed_proof_cannot_be_promoted(inputs, kind):
    args, _ = inputs
    args["proof_capsules"][kind]["record"]["complete"] = False
    with pytest.raises(ValueError):
        candidate.resolve_candidate_dependency(**args)


def test_fullkv_or_wrong_dtype_geometry_cannot_replace_compressed_exact_proof(inputs):
    args, _ = inputs
    args["proof_capsules"]["correctness"]["record"]["model_scope"] = (
        "BF16_MHA_LLaVA_1_5_source_reader"
    )
    with pytest.raises(ValueError, match="FP16/GQA"):
        candidate.resolve_candidate_dependency(**args)


def test_proof_links_are_not_opened_or_treated_as_new_attestations(inputs):
    args, _ = inputs
    args["proof_capsules"]["metric"]["record"]["evidence_files"] = []
    with pytest.raises(ValueError, match="actual completed evidence"):
        candidate.resolve_candidate_dependency(**args)


def test_missing_delegation_record_is_explicitly_unqualified(inputs):
    args, _ = inputs
    args["delegation"] = None
    result = candidate.resolve_candidate_dependency(**args)
    assert result["qualified"] is False
    assert result["unresolved"] == ["reviewed_delegation_capsule_required"]
    assert result["selected_config"] is None
    assert not any(result["authorization"].values())


def test_loaded_delegation_matches_exact_existing_v36_contract(inputs, monkeypatch):
    import evicastkv_original_complete_admission_v36 as original

    args, _ = inputs

    def forbidden(*_args, **_kwargs):
        pytest.fail("pure delegation validation opened a file or scheduled a process")

    monkeypatch.setattr(Path, "open", forbidden)
    original.validate_authorization(args["delegation"]["record"])
    result = candidate.validate_delegation(
        args["delegation"], args["review"]["record"], args["frozen"]
    )
    assert result == {k: args["delegation"][k] for k in ("path", "sha256")}
    selected = candidate.resolve_candidate_dependency(**args)["selected_config"]
    assert selected["delegation"] == result


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unreviewed_authority"),
        ("instruction_origin", "generated_guess"),
        ("user_instruction_text", " "),
        ("user_instruction_sha256", "0" * 64),
        ("delegated_original_default_review_authorized", False),
        ("delegated_original_default_review_authorized", 1),
        ("review_actor", "human_user"),
        ("reviewed_by_user", True),
        ("reviewed_by_user", 0),
        ("candidate_config_sha256", "0" * 64),
        ("scientific_gates_unchanged", False),
        ("failed_successors_remain_ineligible", False),
        ("confirmation_ids_known_at_review", True),
    ],
)
def test_changed_delegation_scope_or_fabricated_human_review_is_rejected(inputs, field, value):
    import evicastkv_original_complete_admission_v36 as original

    args, _ = inputs
    args["delegation"]["record"][field] = value
    with pytest.raises(ValueError):
        original.validate_authorization(args["delegation"]["record"])
    with pytest.raises(ValueError, match="delegated review"):
        candidate.resolve_candidate_dependency(**args)


@pytest.mark.parametrize(
    "field,value", [("path", "/synthetic/different.json"), ("sha256", "f" * 64)]
)
def test_valid_but_differently_bound_delegation_cannot_qualify(inputs, field, value):
    args, _ = inputs
    args["delegation"][field] = value
    with pytest.raises(ValueError, match="delegation differs"):
        candidate.resolve_candidate_dependency(**args)


def test_instruction_text_hash_must_bind_review_and_freeze(inputs):
    args, _ = inputs
    text = "A different synthetic instruction; not the reviewed authorization."
    args["delegation"]["record"]["user_instruction_text"] = text
    args["delegation"]["record"]["user_instruction_sha256"] = hashlib.sha256(
        text.encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="delegation differs"):
        candidate.resolve_candidate_dependency(**args)
