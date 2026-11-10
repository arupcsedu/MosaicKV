"""Metadata-scope completion must not become a false media freshness claim."""

import json

import evicastkv_review_freshness_audit as audit
import pytest


def roots(tmp_path):
    result = (tmp_path / "mosaickv", tmp_path / "evicastkv")
    for path in result:
        path.mkdir()
    return result


def manifest(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_projection_never_constructs_forbidden_benchmark_values(monkeypatch):
    original = audit.json.loads
    seen = []

    def recording(text):
        # String keys may be decoded individually. Values must be replaced.
        if text.startswith("{"):
            assert "secret-answer" not in text
            assert "private-prompt" not in text
            assert "unseen-ref" not in text
        seen.append(text)
        return original(text)

    raw = json.dumps(
        dict(
            sample_id="same",
            prompt="private-prompt",
            answer="secret-answer",
            references=["unseen-ref"],
            media=[dict(sha256="a" * 64)],
        )
    ).encode()
    monkeypatch.setattr(audit.json, "loads", recording)
    value = audit.metadata_projection(raw)
    assert value["answer"] is None and value["references"] is None and value["prompt"] is None
    assert value["sample_id"] == "same" and seen


def test_identity_extraction_does_not_confuse_config_hash_with_image_hash():
    value = dict(
        sha256="b" * 64,
        sample_ids=["coco_caption:COCO_val2014_000000000001.jpg"],
        media=[dict(sha256="a" * 64)],
        media_group_ids=["input-media:" + "c" * 64],
    )
    coco, identities, hashes = audit.identities(value)
    assert coco == {"COCO_val2014_000000000001.jpg"}
    assert identities == {"coco_caption:COCO_val2014_000000000001.jpg"}
    assert hashes == {"a" * 64, "c" * 64}


def test_nested_canonical_inventory_stable_but_not_global_freshness(tmp_path):
    allowed = roots(tmp_path)
    path = manifest(
        allowed[0] / "nested" / "ledger.json",
        dict(samples=[dict(sample_id="COCO_val2014_000000000001.jpg", answer="not-read")]),
    )
    manifest(allowed[1] / "registration.json", dict(sample_ids=["ActionPrediction-0001"]))
    # Raw results are not parsed; malformed raw JSONL is a documented scope gap.
    (allowed[1] / "raw.jsonl").write_bytes(b"not JSON at all")
    report = audit.audit(allowed)
    assert report["canonical_manifest_scope_complete"]
    assert report["canonical_manifest_set_stable"]
    assert report["canonical_manifests_read"] == 2
    assert report["bound_files"][str(path)] == audit.sha(path)
    assert report["observed_coco_ids"] == ["COCO_val2014_000000000001.jpg"]
    assert report["unparsed_structured_payloads"] == 1
    assert not report["complete_historical_id_inventory"]
    assert not report["cross_dataset_media_certified"]
    assert not report["fresh_confirmation_authorized"]
    assert not report["cohort_selected"]


def test_symlink_alias_inside_scope_deduplicates_canonical_target(tmp_path):
    allowed = roots(tmp_path)
    path = manifest(allowed[0] / "real" / "ledger.json", dict(sample_ids=["same"]))
    (allowed[1] / "alias").symlink_to(path.parent, target_is_directory=True)
    report = audit.audit(allowed)
    assert report["canonical_manifest_scope_complete"]
    assert report["canonical_manifests_read"] == 1
    assert report["observed_sample_ids"] == ["same"]
    assert len(report["symlink_aliases"]) == 1


@pytest.mark.parametrize("kind", ["broken", "outside"])
def test_bad_symlinks_block_scope_completion(tmp_path, kind):
    allowed = roots(tmp_path)
    target = tmp_path / "outside.json"
    if kind == "outside":
        target.write_text("{}")
    (allowed[0] / "ledger.json").symlink_to(target)
    report = audit.audit(allowed)
    assert not report["canonical_manifest_scope_complete"]
    assert report["errors"]
    assert not report["fresh_confirmation_authorized"]


def test_malformed_canonical_file_is_not_ignored(tmp_path):
    allowed = roots(tmp_path)
    (allowed[0] / "ledger.json").write_bytes(b"malformed JSON")
    report = audit.audit(allowed)
    assert not report["canonical_manifest_scope_complete"]
    assert any("JSON" in row["error"] or "ValueError" in row["error"] for row in report["errors"])


def test_bounded_metadata_scan_reports_incompletion_instead_of_hanging(tmp_path):
    allowed = roots(tmp_path)
    for index in range(8):
        manifest(allowed[0] / str(index) / "ledger.json", {})
    report = audit.audit(allowed, budget=audit.Budget(maximum_entries=3))
    assert not report["canonical_manifest_scope_complete"]
    assert any("bound reached" in row["error"] for row in report["errors"])


@pytest.mark.parametrize("seconds", [0, -1, 901])
def test_cpu_wall_bound_cannot_exceed_fifteen_minutes(seconds):
    with pytest.raises(ValueError, match="fifteen minutes"):
        audit.Budget(seconds=seconds)


def test_recorded_hash_changes_are_not_silently_tolerated(tmp_path, monkeypatch):
    allowed = roots(tmp_path)
    path = manifest(allowed[0] / "ledger.json", dict(sample_id="initial"))
    original = audit.enumerate_canonical
    calls = []

    def mutate_on_second(roots, budget):
        calls.append(True)
        if len(calls) == 2:
            manifest(path, dict(sample_id="changed"))
        return original(roots, budget)

    monkeypatch.setattr(audit, "enumerate_canonical", mutate_on_second)
    report = audit.audit(allowed)
    assert not report["canonical_manifest_scope_complete"]
    assert any("changed after identity projection" in row["error"] for row in report["errors"])


def test_new_manifest_during_scan_is_not_an_immutable_inventory(tmp_path, monkeypatch):
    allowed = roots(tmp_path)
    manifest(allowed[0] / "ledger.json", {})
    original = audit.enumerate_canonical
    calls = []

    def addition_on_second(roots, budget):
        calls.append(True)
        if len(calls) == 2:
            manifest(allowed[1] / "protocol.json", {})
        return original(roots, budget)

    monkeypatch.setattr(audit, "enumerate_canonical", addition_on_second)
    report = audit.audit(allowed)
    assert not report["canonical_manifest_set_stable"]
    assert not report["canonical_manifest_scope_complete"]


def test_prior_exclusions_and_resolved_classifications_are_retained(tmp_path):
    allowed = roots(tmp_path)
    previous = dict(
        classifications=[dict(root="previously-resolved", classification="extraction-only")],
        prior_excluded_ids=["coco_caption:COCO_val2014_000000000001.jpg"],
        bindings={},
    )
    report = audit.audit(allowed, previous=previous)
    assert report["prior_resolved_roots_not_reintroduced_as_ambiguities"]
    assert report["recovered_classifications"] == previous["classifications"]
    assert report["observed_coco_ids"] == ["COCO_val2014_000000000001.jpg"]
    assert not report["fresh_confirmation_authorized"]


def test_canonical_name_rules_do_not_expand_to_all_raw_results():
    assert audit.canonical_name("ledger.json")
    assert audit.canonical_name("docvqa-ledger.json")
    assert not audit.canonical_name("raw.jsonl")
    assert not audit.canonical_name("answers.json")


def test_per_manifest_bound_prevents_large_read(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_bytes(b"x" * 100)
    with pytest.raises(ValueError, match="per-file byte bound"):
        audit.read_stable(path, audit.Budget(), maximum_file_bytes=10)
