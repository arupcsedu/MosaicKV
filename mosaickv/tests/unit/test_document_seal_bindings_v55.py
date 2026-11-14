"""Literal reviewed source text and synthetic transactions; no historical I/O."""

from copy import deepcopy
from pathlib import Path

import pytest
from test_parquet_binding_recovery_v49 import fixture

from mosaickv.evaluation import document_seal_bindings_v55 as document
from mosaickv.evaluation.fox_campaign import digest
from mosaickv.evaluation.parquet_binding_recovery_v49 import CLAIM_FIELDS, classify_binding

RAW = b"""# Output-preserving draft-gate v1 development seal

This file seals the 16-example COCO-caption development result inspected on
2026-10-15. It is **development-only**, is not held-out evidence, and must not
be used to tune another candidate from its answers, reference scores, or sample
failure identities.

## Immutable evidence

- Run root: `/scratch/djy8hg/runs/mosaickv/output-preserving-draft-edb6edf-20261015-v1`
- Slurm jobs: `21418258_[0-6]` and analysis job `21418259`
- Source base: `edb6edfaf50f4539c9c1969371fefecf0eabccbf`
- Source snapshot: Git-backed development overlay, recorded dirty and content-hashed
- Plan SHA-256: `d97c4b163071e69a40ff566d170d450adc58e0798f30175b66d26b216589337c`
- Input ledger SHA-256: `d389cea234f66b7226f2b1dcc3d380afa1767f3f8bccdd80f446fab11149cb09`
- Analysis report SHA-256: `aab77570386c86366781447335446e0c7d93b9e05db1d1984ced6a29857905ce`
- Aggregates SHA-256: `7d910324d263b4a13297919325dba87f55b050a78feeb0ed15f20a25c6be53b2`
- Cross-curve tests SHA-256: `78a6c9ab2fef43ecbb062a43bc21bb6bf20c7f4db1364fb5df9cf1e4c10ea07f`
- Paired samples SHA-256: `83071d46846d5019b68f269fc893b91322a4c9672db932ee2140ff78683bea22`

All seven GPU arms completed 16/16 unique identities without failed rows. The
retention-one check had token agreement 1.0 and maximum logit difference 0.0.
Every candidate trace recorded isolated four-token draft evidence and exact
initial/final/peak storage accounting.

## Sealed interpretation

At a 25.5 MiB initial cap, the draft-gated candidate and same-budget Prompt
Top-K produced identical answers on all 16 examples. Relative to Prompt Top-K
at 40 MiB, it used 36.25% less initial storage and 30.287% less final/peak
storage. Its CIDEr point difference was +2.285, but the paired 95% confidence
interval was [-2.034, 12.237]. The zero-margin quality gate therefore did not
pass. The result is classified as
`retrospective_development_only_not_paper_evidence` and authorizes no paper
claim or paper-scale campaign.

The draft gate was mechanically active: it rejected hundreds of swaps per
sample and reduced both captured prompt and draft reconstruction error. It did
not change any generated answer relative to the prompt-only two-swap candidate.
No further candidate choice may use these 16 answers or their failure identities.

The separate 128-example protocol in
`configs/experiments/output_preserving_draft_frozen_validation_v1.yaml` freezes
the candidate before execution and excludes this ledger plus the preceding
inspected capacity-sweep ledger.
"""


def git_facts():
    return dict(
        original_commit=document.DOCUMENT_COMMIT,
        repository_path=document.DOCUMENT_PATH,
        original_blob_oid=document.DOCUMENT_BLOB_OID,
        current_blob_oid=document.DOCUMENT_BLOB_OID,
        metadata_path=document.DOCUMENT_METADATA_PATH,
    )


def arguments(index=0):
    row, transaction, _, writers = fixture()
    path, expected = list(document.DOCUMENT_ARTIFACTS.items())[index]
    row["path"] = transaction["path"] = path
    transaction["before_sha256"] = transaction["after_sha256"] = expected
    row["original_registration_sha256"] = digest(
        {
            name: row[name]
            for name in document.contract.OCCURRENCE_FIELDS | {"path", "signature", "operation"}
        }
    )
    previous = classify_binding(row, transaction, [], reviewed_writers=writers)
    return dict(
        previous=previous,
        original_row=row,
        transaction=transaction,
        original_claims=[],
        claim=document.artifacts_from_seal(RAW, git_facts())[index],
        reviewed_writers=writers,
    )


def test_exact_original_document_produces_only_the_two_reviewed_artifact_claims():
    facts = git_facts()
    before = deepcopy(facts)
    claims = document.artifacts_from_seal(RAW, facts)
    assert len(RAW) == document.DOCUMENT_SIZE == 2534
    assert facts == before
    assert len(claims) == 2
    assert [claim["artifact_sha256"] for claim in claims] == list(
        document.DOCUMENT_ARTIFACTS.values()
    )
    for claim in claims:
        assert set(claim) == CLAIM_FIELDS
        assert claim["metadata_path"] == document.DOCUMENT_METADATA_PATH
        assert claim["metadata_sha256"] == document.DOCUMENT_SHA256
        assert claim["role"] == "artifact_only"
        assert claim["source_evidence_verified"] is False
        assert all(
            claim[name] is None
            for name in ("recorded_source_sha", "source_blob_sha256", "writer_path")
        )
    assert claims[0] is not claims[1]


def test_only_the_explicit_original_main_repository_scratch_alias_is_normalized():
    facts = git_facts()
    facts["metadata_path"] = document.DOCUMENT_METADATA_PATH.removeprefix("/sfs/weka")
    assert document.artifacts_from_seal(RAW, facts) == document.artifacts_from_seal(
        RAW, git_facts()
    )


@pytest.mark.parametrize("field", sorted(document.GIT_FIELDS))
def test_every_git_identity_field_is_required_before_document_parsing(monkeypatch, field):
    facts = git_facts()
    del facts[field]
    monkeypatch.setattr(
        document, "_document_fields", lambda *_: pytest.fail("unbound source parsed")
    )
    with pytest.raises(ValueError):
        document.artifacts_from_seal(RAW, facts)


@pytest.mark.parametrize(
    "field,value",
    [
        ("original_commit", "f" * 40),
        ("repository_path", document.DOCUMENT_PATH.replace("draft_v1", "draft_v2")),
        ("original_blob_oid", "f" * 40),
        ("current_blob_oid", "f" * 40),
        ("original_blob_oid", True),
        ("verified", True),
        ("metadata_path", "/home/user/" + document.DOCUMENT_PATH),
        ("metadata_path", "/scratch/djy8hg/workdir/auditor/" + document.DOCUMENT_PATH),
        ("metadata_path", "/scratch/djy8hg/workdir/MosaicKV/.codex/auth.json"),
        ("metadata_path", document.DOCUMENT_METADATA_PATH + "/../credentials.json"),
        ("metadata_path", "//" + document.DOCUMENT_METADATA_PATH.lstrip("/")),
    ],
)
def test_changed_or_redirected_git_document_identity_is_not_a_source_witness(field, value):
    facts = git_facts()
    facts[field] = value
    with pytest.raises(ValueError):
        document.artifacts_from_seal(RAW, facts)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "not bytes",
        bytearray(RAW),
        RAW[:-1],
        RAW + b"\n",
        RAW.replace(b"\n", b"\r\n"),
        RAW.replace(b"development-only", b"development-good"),
        b"x" * (8 * 1024**2 + 1),
    ],
    ids=["none", "text", "mutable", "truncated", "appended", "crlf", "changed", "huge"],
)
def test_the_whole_original_document_is_pinned_not_just_matching_hash_substrings(raw):
    with pytest.raises(ValueError):
        document.artifacts_from_seal(raw, git_facts())


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "missing",
        "unknown",
        "indented",
        "quoted",
        "bad_width",
        "uppercase",
        "missing_ticks",
        "wrong_root",
        "retry_root",
        "swapped_hashes",
        "paired_as_artifact",
        "crlf",
        "unicode",
        "unterminated",
        "malformed",
    ],
)
def test_closed_label_grammar_and_full_byte_pin_both_reject_widening(change):
    lines = RAW.splitlines(keepends=True)
    label = next(line for line in lines if line.startswith(b"- Aggregates SHA-256:"))
    expected = next(iter(document.DOCUMENT_ARTIFACTS.values())).encode()
    raw = RAW
    if change == "duplicate":
        raw += label
    elif change == "missing":
        raw = raw.replace(label, b"")
    elif change == "unknown":
        raw += b"- New artifact SHA-256: `" + b"a" * 64 + b"`\n"
    elif change == "indented":
        raw = raw.replace(label, b" " + label)
    elif change == "quoted":
        raw = raw.replace(label, b"> " + label)
    elif change == "bad_width":
        raw = raw.replace(expected, expected[:-1])
    elif change == "uppercase":
        raw = raw.replace(expected, expected.upper())
    elif change == "missing_ticks":
        raw = raw.replace(b"`" + expected + b"`", expected)
    elif change == "wrong_root":
        raw = raw.replace(b"output-preserving-draft-edb6edf", b"output-preserving-other-edb6edf")
    elif change == "retry_root":
        raw = raw.replace(b"20261015-v1`", b"20261015-v1-retry1`")
    elif change == "swapped_hashes":
        raw = raw.replace(expected, list(document.DOCUMENT_ARTIFACTS.values())[1].encode())
    elif change == "paired_as_artifact":
        raw = raw.replace(b"Paired samples SHA-256:", b"Paired samples.parquet SHA-256:")
    elif change == "crlf":
        raw = raw.replace(b"\n", b"\r\n")
    elif change == "unicode":
        raw = raw.replace(b"- Aggregates", b"- \xc0\xafAggregates")
    elif change == "unterminated":
        raw = raw.rstrip(b"\n")
    else:
        raw = raw.replace(label, label.replace(b": ", b" "))
    with pytest.raises(ValueError):
        document._document_fields(raw)
    with pytest.raises(ValueError):
        document.artifacts_from_seal(raw, git_facts())


@pytest.mark.parametrize("index", [0, 1])
def test_reuses_exact_v54_parent_without_runtime_or_scientific_promotion(index):
    values = arguments(index)
    saved = deepcopy(values)
    result = document.augment_document_binding(**values)
    assert values == saved
    assert result["schema"] == "historical_document_seal_binding_v55"
    assert result["status"] == "artifact_binding_only"
    assert result["parent_binding_sha256"] == digest(values["previous"])
    assert result["original_document_seal_claim_sha256"] == digest(values["claim"])
    assert result["original_document_commit"] == document.DOCUMENT_COMMIT
    assert result["original_document_git_blob_oid"] == document.DOCUMENT_BLOB_OID
    assert result["original_document_repository_path"] == document.DOCUMENT_PATH
    assert result["original_document_sha256"] == document.DOCUMENT_SHA256
    assert result["development_classification"] == document.DEVELOPMENT_CLASSIFICATION
    for name in ("path", "signature", "artifact_sha256", "original_registration_sha256"):
        assert result[name] == values["previous"][name]
    for name in (
        "original_producer_commit",
        "original_producer_writer_path",
        "original_producer_source_blob_sha256",
    ):
        assert result[name] is None
    assert result["reused_hash_transaction"] is True
    assert result["negative_result_preserved"] is True
    assert result["historical_development_overlay_recorded_dirty"] is True
    assert result["newly_resolved_occurrences"] == result["GPU_jobs"] == 0
    for name in (
        *document.contract.FALSE_FIELDS,
        "canonical_eligible",
        "clean_runtime_execution_claimed",
        "quality_gate_pass",
        "scientific_dependency_validation",
        "occurrence_resolved",
    ):
        assert result[name] is False


def test_both_inputs_reuse_the_existing_v49_classifier(monkeypatch):
    values, calls = arguments(), []

    def observed(*args, **kwargs):
        calls.append(deepcopy((args, kwargs)))
        return classify_binding(*args, **kwargs)

    monkeypatch.setattr(document, "classify_binding", observed)
    document.augment_document_binding(**values)
    assert len(calls) == 2
    for args, kwargs in calls:
        assert args[:2] == (values["original_row"], values["transaction"])
        assert kwargs == {"reviewed_writers": values["reviewed_writers"]}
    assert calls[0][0][2] == values["original_claims"]
    assert calls[1][0][2] == [*values["original_claims"], values["claim"]]


@pytest.mark.parametrize("index", [0, 1])
def test_exact_expected_view_is_the_same_complete_row_used_by_promotion(index):
    values = arguments(index)
    saved = deepcopy(values)
    expected = document.expected_document_binding(values["previous"], values["claim"])
    assert expected == document.augment_document_binding(**values)
    assert values == saved
    changed = {"schema", "status", "original_metadata_claim_sha256s"}
    for name, value in values["previous"].items():
        if name not in changed:
            assert expected[name] == value
    assert expected["original_metadata_claim_sha256s"] == sorted(
        [*values["previous"]["original_metadata_claim_sha256s"], digest(values["claim"])]
    )
    expected["signature"][0] += 1
    assert values == saved


def test_expected_view_requires_no_classifier_or_original_proof_replay(monkeypatch):
    values = arguments()
    expected = document.augment_document_binding(**values)

    def forbidden(*_args, **_kwargs):
        pytest.fail("expected-row view replayed a classifier, proof or external I/O")

    monkeypatch.setattr(document, "classify_binding", forbidden)
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr("os.open", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr("socket.socket", forbidden)
    for method in ("open", "stat", "lstat", "resolve", "read_bytes", "read_text"):
        monkeypatch.setattr(Path, method, forbidden)
    assert document.expected_document_binding(values["previous"], values["claim"]) == expected


@pytest.mark.parametrize("field", [*sorted(document._PARENT_FIELDS), "unexpected_field"])
def test_promotion_rejects_every_changed_inherited_classifier_field(monkeypatch, field):
    values = arguments()
    calls = 0

    def changed(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = classify_binding(*args, **kwargs)
        if calls == 2:
            result[field] = "edited"
        return result

    monkeypatch.setattr(document, "classify_binding", changed)
    with pytest.raises(ValueError, match="inherited"):
        document.augment_document_binding(**values)
    assert calls == 2


@pytest.mark.parametrize("field", sorted(document._PARENT_FIELDS))
def test_expected_view_does_not_accept_an_incomplete_parent_schema(field):
    values = arguments()
    del values["previous"][field]
    with pytest.raises(ValueError):
        document.expected_document_binding(values["previous"], values["claim"])


def test_expected_view_does_not_inherit_unregistered_parent_extensions():
    values = arguments()
    values["previous"]["unreviewed_extension"] = True
    with pytest.raises(ValueError):
        document.expected_document_binding(values["previous"], values["claim"])


@pytest.mark.parametrize(
    "digests",
    [
        "not a list",
        ["not a digest"],
        ["f" * 64, "e" * 64],
        ["e" * 64, "e" * 64],
        ["e" * 64] * document.MAXIMUM_CLAIMS,
    ],
)
def test_expected_view_rejects_malformed_duplicate_or_unbounded_original_claim_digests(digests):
    values = arguments()
    values["previous"]["original_metadata_claim_sha256s"] = digests
    with pytest.raises(ValueError):
        document.expected_document_binding(values["previous"], values["claim"])


def test_expected_view_cannot_add_the_same_document_claim_twice():
    values = arguments()
    values["previous"]["original_metadata_claim_sha256s"] = [digest(values["claim"])]
    with pytest.raises(ValueError):
        document.expected_document_binding(values["previous"], values["claim"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "runtime_writer"),
        ("recorded_source_sha", "edb6edfaf50f4539c9c1969371fefecf0eabccbf"),
        ("recorded_source_sha", document.DOCUMENT_COMMIT),
        ("source_blob_sha256", document.DOCUMENT_SHA256),
        ("writer_path", document.DOCUMENT_PATH),
        ("source_evidence_verified", True),
        ("source_evidence_verified", 0),
        ("metadata_sha256", "f" * 64),
        ("metadata_path", document.DOCUMENT_METADATA_PATH.replace("MosaicKV", "auditor")),
        ("artifact_sha256", "f" * 64),
        ("artifact_path", next(iter(document.DOCUMENT_ARTIFACTS))),
    ],
)
def test_claim_cannot_smuggle_an_extra_output_or_a_source_base_as_writer(field, value):
    values = arguments()
    values["claim"][field] = value
    with pytest.raises(ValueError):
        document.augment_document_binding(**values)
    with pytest.raises(ValueError):
        document.expected_document_binding(values["previous"], values["claim"])


@pytest.mark.parametrize(
    "schema",
    [
        "historical_negative_seal_binding_v53",
        "historical_freeze_writer_binding_v54",
        "historical_document_seal_binding_v55",
        None,
    ],
)
def test_no_other_parent_schema_is_silently_reinterpreted(schema):
    values = arguments()
    values["previous"]["schema"] = schema
    with pytest.raises(ValueError):
        document.augment_document_binding(**values)


@pytest.mark.parametrize(
    "field",
    [
        "artifact_sha256",
        "signature",
        "original_registration_sha256",
        "status",
        "original_producer_commit",
        "original_metadata_claim_sha256s",
        "occurrence_resolved",
        *document.contract.FALSE_FIELDS,
    ],
)
def test_saved_parent_digest_and_every_scientific_boundary_remain_exact(field):
    values = arguments()
    values["previous"][field] = "edited"
    with pytest.raises(ValueError):
        document.augment_document_binding(**values)


@pytest.mark.parametrize("name", ["original_row", "transaction", "original_claims"])
def test_parent_classifier_inputs_are_not_accepted_on_trust(name):
    values = arguments()
    if name == "original_row":
        values[name]["original_registration_sha256"] = "f" * 64
    elif name == "transaction":
        values[name]["after_sha256"] = "f" * 64
    else:
        values[name].append(dict(values["claim"], metadata_sha256="e" * 64))
    with pytest.raises(ValueError):
        document.augment_document_binding(**values)


@pytest.mark.parametrize(
    "name",
    [
        "paired_samples.parquet",
        "draft_gate_telemetry.parquet",
        "quality_comparisons.parquet",
        "../aggregates.parquet",
        "other/aggregates.parquet",
    ],
)
def test_other_artifacts_cannot_inherit_a_hash_from_a_similar_document_label(name):
    values = arguments()
    path = document.RUN_ROOT + "/analysis/" + name
    for key in ("previous", "original_row", "transaction"):
        values[key]["path"] = path
    with pytest.raises(ValueError):
        document.augment_document_binding(**values)


def test_the_two_recorded_hashes_cannot_be_exchanged_between_targets():
    values = arguments()
    values["claim"]["artifact_sha256"] = list(document.DOCUMENT_ARTIFACTS.values())[1]
    with pytest.raises(ValueError):
        document.augment_document_binding(**values)


@pytest.mark.parametrize("index", [0, 1])
def test_a_stable_saved_transaction_with_different_bytes_cannot_borrow_the_document_hash(index):
    values = arguments(index)
    values["transaction"].update(before_sha256="f" * 64, after_sha256="f" * 64)
    values["previous"] = classify_binding(
        values["original_row"],
        values["transaction"],
        values["original_claims"],
        reviewed_writers=values["reviewed_writers"],
    )
    with pytest.raises(ValueError, match="conflicts"):
        document.augment_document_binding(**values)


@pytest.mark.parametrize("limit", ["bytes", "nodes", "depth"])
def test_projected_git_metadata_retains_all_unchanged_bounds(limit):
    facts = git_facts()
    if limit == "bytes":
        facts["metadata_path"] = "x" * (8 * 1024**2 + 1)
    elif limit == "nodes":
        facts["metadata_path"] = [0] * 100001
    else:
        value = "x"
        for _ in range(66):
            value = [value]
        facts["metadata_path"] = value
    with pytest.raises(ValueError, match="bound"):
        document.artifacts_from_seal(RAW, facts)


def test_no_filesystem_git_network_or_scheduler_access(monkeypatch):
    facts, values = git_facts(), arguments()

    def forbidden(*_args, **_kwargs):
        pytest.fail("pure committed-document validator performed I/O")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr("os.open", forbidden)
    monkeypatch.setattr("subprocess.run", forbidden)
    monkeypatch.setattr("subprocess.Popen", forbidden)
    monkeypatch.setattr("socket.socket", forbidden)
    for method in ("open", "stat", "lstat", "resolve", "read_bytes", "read_text"):
        monkeypatch.setattr(Path, method, forbidden)
    assert len(document.artifacts_from_seal(RAW, facts)) == 2
    assert document.augment_document_binding(**values)["reused_hash_transaction"] is True
