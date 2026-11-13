"""Exact synthetic receipt compaction; no historical sources or scheduler."""

import copy
import hashlib
import json
import time

import evicastkv_checkpoint_receipts_v47 as codec
import evicastkv_history_closure_v16 as writer
import pytest


def make_receipt(number, identities=None):
    value = dict(
        registration_sha256=f"{number + 1:064x}",
        operation="json_stream",
        path=f"/synthetic/original-{number}.json",
        signature=[1, number + 1, 30, 4],
        proof_path=f"/synthetic/proofs/{number}.json",
        proof_sha256=f"{number + 20:064x}",
        whole_file_sha256_before=f"{number + 100:064x}",
        whole_file_sha256_after=f"{number + 100:064x}",
        source_hash_kind="file_bytes",
        old_unbound_historical_version_byte_identity_claimed=False,
        **dict.fromkeys(codec.core.contract.FALSE_FIELDS, False),
    )
    if identities is not None:
        value["identities"] = identities
    return value


@pytest.fixture
def report():
    identities = dict(
        sample_ids=["one", "two", "é\ud800"],
        coco_ids=["COCO_val2014_000000000001.jpg"],
        media_hashes=["a" * 64],
        source_group_ids=["group"],
        scopes=[["dataset", "config", "train"]],
        scoped_identities=[["dataset", "config", "train", "one"]],
    )
    one = {key: list(values) for key, values in identities.items()}
    one["sample_ids"] = ["one"]
    two = dict.fromkeys(identities)
    two.update({key: [] for key in identities}, sample_ids=["two", "é\ud800"])
    receipts = [make_receipt(0, one), make_receipt(1, two), make_receipt(2)]
    return dict(
        identities=identities,
        raw_error_records=[dict(original_error_sha256="b" * 64, message="preserved synthetic")],
        resolved_occurrences=[dict(ordinal=0, marker="preserved")],
        unresolved_occurrences=[dict(ordinal=1, marker="preserved")],
        source_parent_raw_sha256="c" * 64,
        source_bindings={
            **{r["proof_path"]: r["proof_sha256"] for r in receipts},
            "/synthetic/immutable-parent.json": "d" * 64,
        },
        additive_proof_sha256s=[codec.digest(r) for r in receipts],
        identity_proof_sha256s=[codec.digest(r) for r in receipts if "identities" in r],
        metadata_recovery_v41_receipts=receipts,
        **dict.fromkeys(codec.core.contract.FALSE_FIELDS, False),
    )


def compact(report, tmp_path, **kwargs):
    output = tmp_path / "receipts"
    output.mkdir()
    return codec.compact_report(report, output, **kwargs)


def test_dropping_receipt_and_recomputing_manifest_cannot_hide_a_resolved_proof(report, tmp_path):
    report["resolved_occurrences"] = [
        dict(
            resolution_kind="V41_occurrence_specific_json_stream",
            additive_proof_sha256=codec.digest(receipt),
        )
        for receipt in report[codec.INLINE]
    ]
    result = compact(report, tmp_path)
    result[codec.REFERENCES].pop()
    result[codec.MANIFEST]["receipt_count"] -= 1
    result[codec.MANIFEST]["ordered_semantic_sha256"] = codec.digest(
        [ref["semantic_sha256"] for ref in result[codec.REFERENCES]]
    )
    with pytest.raises(ValueError, match="every resolved metadata proof"):
        codec.validate_external_receipts(result)


def test_exact_ordered_roundtrip_preserves_every_global_identity_error_and_digest(report, tmp_path):
    before = copy.deepcopy(report)
    result = compact(report, tmp_path)
    assert report == before
    assert result[codec.INLINE] == []
    assert list(codec.iter_receipts(result)) == before[codec.INLINE]
    assert codec.validate_external_receipts(result) == dict(
        receipt_count=3,
        external_receipt_count=3,
        inline_receipt_count=0,
        all_global_identities_inline=True,
    )
    for key, value in before.items():
        if key not in {codec.INLINE, "source_bindings"}:
            assert result[key] == value
    for path, expected in before["source_bindings"].items():
        assert result["source_bindings"][path] == expected
    for ordinal, reference in enumerate(result[codec.REFERENCES]):
        assert reference["ordinal"] == ordinal
        assert reference["semantic_sha256"] == codec.digest(before[codec.INLINE][ordinal])
        assert reference["bytes"] <= codec.MAXIMUM_BYTES


@pytest.mark.parametrize(
    "change",
    [
        "missing_binding",
        "wrong_binding",
        "semantic",
        "registration",
        "counts",
        "manifest_count",
        "order",
    ],
)
def test_changed_reference_binding_digest_count_or_order_fails_closed(report, tmp_path, change):
    result = compact(report, tmp_path)
    reference = result[codec.REFERENCES][0]
    if change == "missing_binding":
        result["source_bindings"].pop(reference["path"])
    elif change == "wrong_binding":
        result["source_bindings"][reference["path"]] = "e" * 64
    elif change == "semantic":
        reference["semantic_sha256"] = result[codec.REFERENCES][1]["semantic_sha256"]
        result[codec.MANIFEST]["ordered_semantic_sha256"] = codec.digest(
            [ref["semantic_sha256"] for ref in result[codec.REFERENCES]]
        )
    elif change == "registration":
        reference["registration_sha256"] = "e" * 64
    elif change == "counts":
        reference["identity_counts"]["sample_ids"] += 1
    elif change == "manifest_count":
        result[codec.MANIFEST]["receipt_count"] = 4
    else:
        result[codec.REFERENCES].reverse()
    with pytest.raises(ValueError):
        codec.validate_external_receipts(result)


def test_tampered_full_receipt_file_is_not_repaired_or_reinterpreted(report, tmp_path):
    result = compact(report, tmp_path)
    path = tmp_path / "receipts" / result[codec.REFERENCES][0]["path"].rsplit("/", 1)[-1]
    original = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match=r"size|hash"):
        codec.validate_external_receipts(result)
    assert path.read_bytes() == original + b" "


def test_sidecar_is_an_exact_full_receipt_not_only_a_proof_pointer(report, tmp_path):
    result = compact(report, tmp_path)
    reference = result[codec.REFERENCES][0]
    path = tmp_path / "receipts" / reference["path"].rsplit("/", 1)[-1]
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == reference["sha256"]
    assert hashlib.sha256(raw[:-1]).hexdigest() == reference["semantic_sha256"]
    assert json.loads(raw) == report[codec.INLINE][0]


def test_future_compaction_keeps_old_archives_and_appends_only_new_receipts(report, tmp_path):
    first = compact(report, tmp_path)
    old_refs = copy.deepcopy(first[codec.REFERENCES])
    new = make_receipt(3, {name: [] for name in report["identities"]})
    first[codec.INLINE].append(new)
    first["source_bindings"][new["proof_path"]] = new["proof_sha256"]
    first["additive_proof_sha256s"].append(codec.digest(new))
    first["identity_proof_sha256s"].append(codec.digest(new))
    output = tmp_path / "next-receipts"
    output.mkdir()
    second = codec.compact_report(first, output)
    assert second[codec.REFERENCES][:3] == old_refs
    assert len(list(output.iterdir())) == 1
    assert list(codec.iter_receipts(second)) == [*report[codec.INLINE], new]


def test_unknown_identity_cannot_be_hidden_by_receipt_externalization(report, tmp_path):
    report["identities"]["sample_ids"].remove("one")
    with pytest.raises(ValueError, match="global exclusions"):
        compact(report, tmp_path)


def test_unregistered_receipt_never_publishes_a_sidecar(report, tmp_path):
    report["additive_proof_sha256s"] = []
    with pytest.raises(ValueError, match="not registered"):
        compact(report, tmp_path)
    assert list((tmp_path / "receipts").iterdir()) == []


def test_bounded_receipt_write_is_atomic_and_does_not_modify_input(report, tmp_path):
    before = copy.deepcopy(report)
    with pytest.raises(ValueError, match="bounded full receipt output"):
        compact(report, tmp_path, maximum_bytes=128)
    assert report == before
    assert list((tmp_path / "receipts").iterdir()) == []


def test_global_identity_growth_still_hits_the_original_checkpoint_cap(report, tmp_path):
    result = compact(report, tmp_path)
    result["identities"]["sample_ids"].extend(["retained-global-ID" * 100] * 100)
    target = tmp_path / "reconciled.json"
    with pytest.raises(ValueError, match="checkpoint/export size"):
        writer.bounded_write(target, result, 4096, compact=True)
    assert not target.exists()
    assert len(result["identities"]["sample_ids"]) == 103


@pytest.mark.parametrize("limit", [True, 0, codec.MAXIMUM_BYTES + 1])
def test_original_512_MiB_bound_cannot_be_widened_or_disabled(report, tmp_path, limit):
    with pytest.raises(ValueError, match="512 MiB"):
        compact(report, tmp_path, maximum_bytes=limit)


def test_expired_analysis_deadline_fails_without_archiving(report, tmp_path):
    with pytest.raises(TimeoutError, match="deadline"):
        compact(report, tmp_path, deadline=time.monotonic() - 1)
    assert list((tmp_path / "receipts").iterdir()) == []
