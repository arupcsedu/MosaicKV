"""Positive finite identity certificates and explicit negative scope fixtures."""

import copy

import evicastkv_coco_freshness_scope as scope
import pytest


@pytest.fixture
def evidence():
    candidates = [
        dict(
            sample_id=f"COCO_val2014_{i:012d}.jpg",
            encoded_sha256=str(i) * 64,
            pixel_sha256=str(i + 3) * 64,
        )
        for i in (1, 2, 3)
    ]
    contract = dict(
        contract=scope.CONTRACT,
        dataset_id="lmms-lab/COCO-Caption",
        dataset_revision=scope.COCO_REVISION,
        split="val",
        historical_roots=["/registered/run-a", "/registered/run-b"],
        candidate_count=3,
        candidate_IDs_sha256=scope.digest(sorted(r["sample_id"] for r in candidates)),
        minimum_available_images=2,
    )
    history = dict(
        historical_roots=contract["historical_roots"],
        metadata_scope_complete=True,
        media_scope_complete=True,
        unresolved_sources=[],
        bound_artifacts={"gate": "a" * 64},
        declared_image_assets=1,
        image_records=[dict(asset_key="old", encoded_sha256="b" * 64, pixel_sha256="c" * 64)],
        canonical_coco_ids=[],
    )
    return contract, history, candidates


def test_complete_legitimate_inputs_can_certify_finite_contract(evidence):
    result = scope.evaluate(*evidence)
    assert result["scope_certified"] and result["eligible_for_preregistered_image_identity_cohort"]
    assert result["available_images"] == 3
    assert not result["universal_crop_resize_scene_independence_claimed"]
    assert not result["experiment_submission_authorized"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("metadata_scope_complete", False),
        ("media_scope_complete", False),
        ("unresolved_sources", ["unknown-packet"]),
        ("declared_image_assets", 2),
        ("bound_artifacts", {}),
    ],
)
def test_actual_historical_gaps_block_certificate(evidence, field, value):
    contract, history, candidates = copy.deepcopy(evidence)
    history[field] = value
    result = scope.evaluate(contract, history, candidates)
    assert not result["scope_certified"] and result["errors"]
    assert not result["eligible_for_preregistered_image_identity_cohort"]


def test_missing_candidate_hashes_are_not_fresh_by_absence(evidence):
    contract, history, candidates = copy.deepcopy(evidence)
    del candidates[0]["pixel_sha256"]
    result = scope.evaluate(contract, history, candidates)
    assert not result["scope_certified"]
    assert "candidate canonical identity/media hashes incomplete" in result["errors"]


def test_exact_id_encoded_and_pixel_aliases_excluded_separately(evidence):
    contract, history, candidates = copy.deepcopy(evidence)
    history["canonical_coco_ids"] = [candidates[0]["sample_id"]]
    history["image_records"][0]["encoded_sha256"] = candidates[1]["encoded_sha256"]
    history["image_records"][0]["pixel_sha256"] = candidates[2]["pixel_sha256"]
    result = scope.evaluate(contract, history, candidates)
    assert result["scope_certified"] and result["available_images"] == 0
    assert result["exclusion_counts"] == dict(
        canonical_COCO_ID_overlap=1,
        historical_encoded_sha256_overlap=1,
        historical_pixel_sha256_overlap=1,
    )
    assert not result["eligible_for_preregistered_image_identity_cohort"]


def test_pixel_duplicate_classes_are_fully_excluded(evidence):
    contract, history, candidates = copy.deepcopy(evidence)
    candidates[1]["pixel_sha256"] = candidates[0]["pixel_sha256"]
    result = scope.evaluate(contract, history, candidates)
    assert result["scope_certified"] and result["available_images"] == 1
    assert result["exclusion_counts"]["within_pool_pixel_duplicate_exclude_all_members"] == 2


def test_omitted_root_or_candidate_identity_cannot_expand_freshness(evidence):
    contract, history, candidates = copy.deepcopy(evidence)
    history["historical_roots"] = history["historical_roots"][:1]
    result = scope.evaluate(contract, history, candidates)
    assert not result["scope_certified"]
    history["historical_roots"] = contract["historical_roots"]
    result = scope.evaluate(contract, history, candidates[:2])
    assert not result["scope_certified"]
