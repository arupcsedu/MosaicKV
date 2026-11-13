"""Synthetic exact scoped-ID factoring; no historical source access."""

import hashlib
import json
import random
from copy import deepcopy

import evicastkv_identity_layout_v47 as codec
import pytest


def make_report(samples=None, rows=None):
    samples = [f"id-{i:04d}" for i in range(300)] if samples is None else samples
    rows = [["", "", "", value] for value in samples] if rows is None else rows
    return dict(
        identities=dict(
            sample_ids=sorted(samples),
            scoped_identities=sorted(rows),
            scopes=sorted([list(scope) for scope in {tuple(row[:3]) for row in rows}]),
            coco_ids=["COCO_val2014_000000000001.jpg"],
            media_hashes=["a" * 64],
            source_group_ids=["source-a"],
        ),
        counts=dict(raw_error_count=1, net_unresolved_error_count=1),
        raw_error_records=[dict(path="/scratch/unchanged", error="preserved")],
        source_bindings={"/scratch/evidence.json": "b" * 64},
        metadata_recovery_v41_receipts=[dict(original="untouched")],
        fresh_confirmation_authorized=False,
        materialization_authorized=False,
        paper_campaign_authorized=False,
    )


def encoded_report():
    value = codec.pack_identities(make_report())
    assert isinstance(value["identities"]["scoped_identities"], dict)
    return value


def test_full_roundtrip_preserves_all_fields_flags_and_original_input():
    original = make_report()
    snapshot = deepcopy(original)
    packed = codec.pack_identities(original)
    assert original == snapshot
    restored = codec.unpack_identities(packed)
    assert restored == original
    for name in original.keys() - {"identities"}:
        assert packed[name] is original[name]
    for name in codec.IDENTITY_FIELDS - {"scoped_identities"}:
        assert packed["identities"][name] is original["identities"][name]
    assert codec.canonical_json_size(packed) < codec.canonical_json_size(original)
    assert codec.pack_identities(packed) == packed


@pytest.mark.parametrize("seed", range(12))
def test_random_scope_subsets_extra_ids_and_unicode_roundtrip(seed):
    rng = random.Random(seed)
    samples = sorted({f"id-{i:04d}" for i in range(1000)} | {"évidence", "文書", ""})
    rows = [["", "", "", sid] for sid in samples if rng.random() < 0.85]
    rows.extend([["", "", "", "base-outside-table"], ["", "", "", "外部"]])
    for scope in [("doc", "a", "train"), ("世界", "", "検証"), ("text", "b", "test")]:
        rows.extend([*scope, sid] for sid in rng.sample(samples, 15))
    rows.extend([["other", "", "", "outside-all-samples"]])
    original = make_report(samples, rows)
    packed = codec.pack_identities(original)
    encoded = packed["identities"]["scoped_identities"]
    assert isinstance(encoded, dict)
    assert encoded["scope"] == ["", "", ""]
    assert encoded["omitted_sample_ids"]
    assert ["", "", "", "base-outside-table"] in encoded["additional_rows"]
    assert codec.unpack_identities(packed) == original


def test_small_or_empty_tables_keep_the_plain_format():
    for report in (make_report([], []), make_report(["a"], [["", "", "", "a"]])):
        packed = codec.pack_identities(report)
        assert isinstance(packed["identities"]["scoped_identities"], list)
        assert codec.unpack_identities(packed) == report


def test_dominant_scope_tie_breaking_is_deterministic():
    samples = [f"x{i:04d}" for i in range(300)]
    rows = [[*scope, sid] for scope in [("b", "", ""), ("a", "", "")] for sid in samples]
    report = make_report(samples, rows)
    packed = codec.pack_identities(report)
    assert packed["identities"]["scoped_identities"]["scope"] == ["a", "", ""]
    assert codec.unpack_identities(packed) == report


def test_streamed_size_and_hash_match_the_actual_canonical_json():
    value = [["世界", "", "", "évidence"], ["x", "a", "b", "\u0001"]]
    raw = json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    ).encode()
    assert codec.canonical_json_size(value) == len(raw)
    assert codec._measurement(value) == (len(raw), hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize(
    "change",
    ["sample_count", "sample_hash", "rows_hash", "rows_count", "kind", "extra_key", "scope"],
)
def test_malformed_or_tampered_layouts_are_rejected(change):
    report = encoded_report()
    encoded = report["identities"]["scoped_identities"]
    changes = {
        "sample_count": ("sample_ids_count", True),
        "sample_hash": ("sample_ids_sha256", "0" * 64),
        "rows_hash": ("scoped_identities_sha256", "0" * 64),
        "rows_count": ("scoped_identities_count", encoded["scoped_identities_count"] + 1),
        "kind": ("kind", "unknown"),
        "extra_key": ("ignored", "must-not-be-ignored"),
        "scope": ("scope", ["too", "short"]),
    }
    key, value = changes[change]
    encoded[key] = value
    with pytest.raises(ValueError):
        codec.unpack_identities(report)


def test_stale_or_reordered_sample_table_is_rejected():
    stale = encoded_report()
    stale["identities"]["sample_ids"].append("new-id")
    with pytest.raises(ValueError, match="stale"):
        codec.unpack_identities(stale)
    reordered = encoded_report()
    reordered["identities"]["sample_ids"].reverse()
    with pytest.raises(ValueError, match="ordered"):
        codec.unpack_identities(reordered)


@pytest.mark.parametrize("omitted", [["outside"], ["id-0001", "id-0001"], ["id-0002", "id-0001"]])
def test_omissions_must_be_unique_ordered_sample_members(omitted):
    report = encoded_report()
    report["identities"]["scoped_identities"]["omitted_sample_ids"] = omitted
    with pytest.raises(ValueError):
        codec.unpack_identities(report)


@pytest.mark.parametrize(
    "rows",
    [
        [["", "", "", "id-0001"]],
        [["x", "", "", "b"], ["x", "", "", "b"]],
        [["z", "", "", "b"], ["x", "", "", "a"]],
    ],
)
def test_additional_rows_cannot_overlap_duplicate_or_reorder(rows):
    report = encoded_report()
    report["identities"]["scoped_identities"]["additional_rows"] = rows
    with pytest.raises(ValueError):
        codec.unpack_identities(report)


def test_redundant_omit_then_restore_is_rejected_even_if_final_ids_would_match():
    report = encoded_report()
    encoded = report["identities"]["scoped_identities"]
    encoded["omitted_sample_ids"] = ["id-0001"]
    encoded["additional_rows"] = [["", "", "", "id-0001"]]
    with pytest.raises(ValueError, match=r"overlap|redundantly"):
        codec.unpack_identities(report)


def test_old_plain_readers_reject_encoded_identity_lists():
    from mosaickv.evaluation import history_occurrence_reconciliation as contract

    with pytest.raises(ValueError, match="explicit identity list"):
        contract._identities(encoded_report()["identities"])


def test_duplicate_or_unsorted_plain_scoped_rows_never_get_normalized_away():
    report = make_report(["a"], [["", "", "", "a"], ["", "", "", "a"]])
    with pytest.raises(ValueError, match="duplicate-free"):
        codec.pack_identities(report)
    with pytest.raises(ValueError, match="duplicate-free"):
        codec.unpack_identities(report)
