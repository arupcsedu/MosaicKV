"""No observed overlap is not a certificate of untouched source content."""

import json

import evicastkv_provenance_resolution as module
import pytest

from mosaickv.evaluation.fox_campaign import sha, write_once


def test_count_only_extraction_root_quarantines_entire_pool(tmp_path):
    root = tmp_path / "milebench-actionprediction-extract-20261018"
    root.mkdir()
    (root / "stdout").write_text("5819\n")
    ids = {"ActionPrediction-0001", "ActionPrediction-0199", "CharacterOrder-0001"}
    result = module.classify_root(root, ids)
    assert not result["purpose_verified"]
    assert result["excluded_ids"] == ["ActionPrediction-0001", "ActionPrediction-0199"]
    assert "quarantined" in result["classification"]


def test_unknown_root_stays_unresolved(tmp_path):
    result = module.classify_root(tmp_path / "unknown", set())
    assert result["classification"] == "unresolved"


def test_wrapper_requires_original_plan_and_ledger_hashes(tmp_path):
    root = tmp_path / "coco-cublas-retry-edb6edf-20261014"
    plan, ledger = tmp_path / "plan.json", tmp_path / "ledger.json"
    write_once(plan, dict(dataset_id="lmms-lab/COCO-Caption"))
    write_once(ledger, dict(samples=[dict(sample_id="COCO_fixture.jpg")]))
    write_once(
        root / "seal.json", dict(bound_files={str(plan): sha(plan), str(ledger): sha(ledger)})
    )
    result = module.classify_root(root, set())
    assert result["purpose_verified"] and result["excluded_ids"] == ["COCO_fixture.jpg"]
    ledger.write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        module.classify_root(root, set())


def test_media_audit_ids_are_not_inferred_consumed_model_predictions(tmp_path):
    root = tmp_path / "fox-review-20261028-milebench-media-content-v1"
    write_once(root / "media_identities.json", dict(identities={}))
    write_once(
        root / "report.json",
        dict(
            observed_media_audit_completed=True,
            no_images_extracted_or_downloaded=True,
            confirmation_authorized=False,
            media_identity_content_sha256=module.digest({}),
        ),
    )
    result = module.classify_root(root, {"IEdit-0001"})
    assert result["purpose_verified"] and not result["excluded_ids"]
    report = json.loads((root / "report.json").read_text())
    report["confirmation_authorized"] = True
    (root / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        module.classify_root(root, set())


def test_same_video_locator_exposes_overlap_even_when_frame_hashes_differ():
    old = dict(
        task="ActionPrediction",
        sample_id="old",
        source_image_paths=["clip_a/0.jpg"],
        combined_1_images="old.jpg",
    )
    new = dict(
        task="ActionLocalization",
        sample_id="new",
        source_image_paths=["clip_a/9.jpg"],
        combined_1_images="new.jpg",
    )
    identities = {
        name: dict(encoded_sha256=name, pixel_sha256=name)
        for row in (old, new)
        for name in module.media_names(row)
    }
    result = module.compare_media([new], [old], identities)[0]
    assert not result["observed_content_overlap"]
    assert result["observed_source_locator_overlap"] == ["clip_a"]
    assert not result["fresh_eligible"] and not result["source_video_document_identity_complete"]


def test_flat_numeric_files_never_fabricate_episode_or_document_groups():
    row = dict(
        task="ALFRED", sample_id="new", source_image_paths=["12.jpg"], combined_1_images="0.jpg"
    )
    identities = {
        name: dict(encoded_sha256=name, pixel_sha256=name) for name in module.media_names(row)
    }
    result = module.compare_media([row], [], identities)[0]
    assert not result["observed_source_locators"]
    assert not result["missing_assets"] and not result["observed_content_overlap"]
    assert not result["fresh_eligible"]
