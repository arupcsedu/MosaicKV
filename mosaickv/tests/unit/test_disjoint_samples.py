from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mosaickv.evaluation.disjoint_samples import (
    DatasetContract,
    SampleIdentity,
    SampleProtocolError,
    canonical_bytes,
    check_disjointness,
    construct_id_only_catalog,
    construct_id_only_catalog_from_dataset,
    extract_id_only_column,
    materialize_deterministic_ledger,
    require_protocol_authorization,
    scan_exclusion_sources,
    selection_digest,
    sha256_file,
    validate_exclusion_manifest,
    validate_ledger,
    write_once,
)

GIT_SHA = "a" * 40
PROTOCOL_SHA = "b" * 64
SALT = "choice-c-v101-v100-fresh-disjoint-calibration-v1"
DOC = DatasetContract("docvqa", "synthetic/DocVQA", "doc-rev", "validation")
TEXT = DatasetContract("textvqa", "synthetic/TextVQA", "text-rev", "validation")
INFO = DatasetContract(
    "infovqa",
    "synthetic/DocVQA",
    "shared-rev",
    "validation",
    dataset_config="InfographicVQA",
)
DOCUMENT = DatasetContract(
    "documentvqa",
    "synthetic/DocVQA",
    "shared-rev",
    "validation",
    dataset_config="DocumentVQA",
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _manifest(tmp_path: Path, identities: tuple[SampleIdentity, ...]) -> Path:
    prior = tmp_path / "ledgers"
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for identity in identities:
        grouped.setdefault(
            (identity.dataset_id, identity.dataset_revision, identity.split), []
        ).append(identity.sample_id)
    for index, (metadata, sample_ids) in enumerate(sorted(grouped.items())):
        _write_json(
            prior / f"prior-{index}.json",
            {
                "schema_version": 1,
                "dataset_id": metadata[0],
                "dataset_revision": metadata[1],
                "split": metadata[2],
                "sample_ids": sample_ids,
            },
        )
    payload = scan_exclusion_sources(
        roots=(prior,),
        contracts=(DOC, TEXT),
        source_git_sha=GIT_SHA,
        source_git_dirty=False,
    )
    output = tmp_path / "exclusion_manifest.json"
    write_once(output, payload)
    return output


def _catalog(path: Path, contract: DatasetContract, sample_ids: list[str]) -> None:
    payload = {
        "schema_version": 2 if contract.dataset_config is not None else 1,
        "dataset_id": contract.dataset_id,
        "dataset_revision": contract.dataset_revision,
        "split": contract.split,
        "sample_ids": sample_ids,
    }
    if contract.dataset_config is not None:
        payload["dataset_config"] = contract.dataset_config
    _write_json(path, payload)


def _ledger(
    *,
    catalog: Path,
    manifest: Path,
    contract: DatasetContract,
    sample_count: int,
) -> dict:
    return materialize_deterministic_ledger(
        catalog_path=catalog,
        exclusion_manifest_path=manifest,
        contract=contract,
        selection_seed=20310821,
        selection_salt=SALT,
        sample_count=sample_count,
        candidate_implementation_git_sha="c" * 40,
        materialization_git_sha=GIT_SHA,
        materialization_git_dirty=False,
        protocol_sha256=PROTOCOL_SHA,
    )


class _AuditedDataset:
    def __init__(self, rows: dict[str, list[object]], accesses: list[object]) -> None:
        self._rows = rows
        self.column_names = list(rows)
        self._accesses = accesses

    def __len__(self) -> int:
        return len(next(iter(self._rows.values()), ()))

    def select_columns(self, columns: list[str]) -> _AuditedDataset:
        self._accesses.append(("select_columns", tuple(columns)))
        return _AuditedDataset({column: self._rows[column] for column in columns}, self._accesses)

    def __getitem__(self, key: str) -> list[object]:
        self._accesses.append(("column", key))
        return self._rows[key]


def test_id_only_catalog_is_canonical_deterministic_and_content_free() -> None:
    accesses: list[object] = []
    dataset = _AuditedDataset(
        {
            "question_id": [30, 10, 20],
            "question": ["must", "never", "be read"],
            "answers": [["forbidden"], ["forbidden"], ["forbidden"]],
            "image": [object(), object(), object()],
        },
        accesses,
    )

    payload = construct_id_only_catalog_from_dataset(
        dataset=dataset,
        contract=INFO,
        id_field="question_id",
        expected_count=3,
        construction_git_sha=GIT_SHA,
        construction_git_dirty=False,
    )

    assert payload == {
        "schema_version": 2,
        "dataset_id": INFO.dataset_id,
        "dataset_config": INFO.dataset_config,
        "dataset_revision": INFO.dataset_revision,
        "split": INFO.split,
        "sample_ids": ["infovqa:10", "infovqa:20", "infovqa:30"],
    }
    assert accesses == [("select_columns", ("question_id",)), ("column", "question_id")]


@pytest.mark.parametrize(
    ("raw_ids", "expected_count", "dirty", "match"),
    (
        (["a", "a"], 2, False, "duplicated"),
        (["a", " b"], 2, False, "whitespace-normalized"),
        (["a", None], 2, False, "strings or integers"),
        (["a", True], 2, False, "never booleans"),
        (["a"], 2, False, "expected exactly 2"),
        (["a"], 1, True, "clean full Git SHA"),
    ),
)
def test_id_only_catalog_rejects_ambiguous_or_incomplete_identifiers(
    raw_ids: list[object], expected_count: int, dirty: bool, match: str
) -> None:
    with pytest.raises(SampleProtocolError, match=match):
        construct_id_only_catalog(
            contract=INFO,
            raw_ids=raw_ids,
            id_field="questionId",
            expected_count=expected_count,
            construction_git_sha=GIT_SHA,
            construction_git_dirty=dirty,
        )


def test_id_only_column_projection_fails_before_unapproved_access() -> None:
    accesses: list[object] = []
    dataset = _AuditedDataset({"question": ["secret"]}, accesses)

    with pytest.raises(SampleProtocolError, match=r"questionId.*absent"):
        extract_id_only_column(dataset, id_field="questionId", expected_count=1)

    assert accesses == []


def test_exclusion_scanner_deduplicates_ledgers_inputs_and_results(tmp_path: Path) -> None:
    ledgers = tmp_path / "ledgers"
    runs = tmp_path / "runs"
    runs.mkdir()
    runs.joinpath("unrelated-console-output.json").write_text(
        "not a structured sample ledger\n", encoding="utf-8"
    )
    _write_json(
        ledgers / "doc.json",
        {
            "dataset_id": DOC.dataset_id,
            "dataset_revision": DOC.dataset_revision,
            "split": DOC.split,
            "sample_ids": ["doc-1", "doc-2"],
        },
    )
    _write_json(
        runs / "old" / "inputs" / "text.json",
        {
            "dataset_id": TEXT.dataset_id,
            "dataset_revision": TEXT.dataset_revision,
            "split": TEXT.split,
            "sample_ids": ["text-1"],
        },
    )
    run = runs / "old" / "doc-run"
    _write_json(
        run / "manifest.json",
        {
            "dataset": {
                "id": DOC.dataset_id,
                "revision": DOC.dataset_revision,
                "split": DOC.split,
            }
        },
    )
    run.joinpath("results.jsonl").write_text(
        "\n".join(
            json.dumps({"task": "docvqa", "sample_id": sample_id})
            for sample_id in ("doc-2", "doc-3")
        )
        + "\n",
        encoding="utf-8",
    )

    payload = scan_exclusion_sources(
        roots=(ledgers, runs),
        contracts=(DOC, TEXT),
        source_git_sha=GIT_SHA,
        source_git_dirty=False,
    )
    identities = validate_exclusion_manifest(payload)

    assert payload["source_count"] == 3
    assert payload["excluded_unique_sample_count"] == 4
    assert {identity.sample_id for identity in identities} == {
        "doc-1",
        "doc-2",
        "doc-3",
        "text-1",
    }
    observed = next(source for source in payload["sources"] if source["kind"] == "observed_results")
    assert observed["manifest_sha256"] == sha256_file(run / "manifest.json")


def test_configuration_is_part_of_canonical_identity_and_selection_digest() -> None:
    info = INFO.identity("same-id")
    document = DOCUMENT.identity("same-id")

    assert info != document
    assert info.canonical_key() != document.canonical_key()
    assert selection_digest(info, seed=17, salt=SALT) != selection_digest(
        document, seed=17, salt=SALT
    )
    assert info.as_dict()["dataset_config"] == "InfographicVQA"


def test_configuration_aware_scan_isolates_configs_and_fails_on_ambiguous_source(
    tmp_path: Path,
) -> None:
    ledgers = tmp_path / "ledgers"
    _write_json(
        ledgers / "info.json",
        {
            "dataset_id": INFO.dataset_id,
            "dataset_config": INFO.dataset_config,
            "dataset_revision": INFO.dataset_revision,
            "split": INFO.split,
            "sample_ids": ["same-id"],
        },
    )
    _write_json(
        ledgers / "document.json",
        {
            "dataset_id": DOCUMENT.dataset_id,
            "dataset_config": DOCUMENT.dataset_config,
            "dataset_revision": DOCUMENT.dataset_revision,
            "split": DOCUMENT.split,
            "sample_ids": ["same-id"],
        },
    )

    payload = scan_exclusion_sources(
        roots=(ledgers,),
        contracts=(INFO, DOCUMENT),
        source_git_sha=GIT_SHA,
        source_git_dirty=False,
    )
    identities = validate_exclusion_manifest(payload)
    assert payload["schema_version"] == 2
    assert identities == tuple(sorted((INFO.identity("same-id"), DOCUMENT.identity("same-id"))))

    (ledgers / "ambiguous.json").write_text(
        json.dumps(
            {
                "dataset_id": INFO.dataset_id,
                "dataset_revision": INFO.dataset_revision,
                "split": INFO.split,
                "sample_ids": ["unknown-config"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SampleProtocolError, match="requires dataset_config"):
        scan_exclusion_sources(
            roots=(ledgers,),
            contracts=(INFO, DOCUMENT),
            source_git_sha=GIT_SHA,
            source_git_dirty=False,
        )


def test_exclusion_scanner_accepts_manifestless_results_only_with_exact_ledger_coverage(
    tmp_path: Path,
) -> None:
    ledgers = tmp_path / "ledgers"
    runs = tmp_path / "runs"
    _write_json(
        ledgers / "doc.json",
        {
            "dataset_id": DOC.dataset_id,
            "dataset_revision": DOC.dataset_revision,
            "split": DOC.split,
            "sample_ids": ["doc-1"],
        },
    )
    result_path = runs / "old" / "results.jsonl"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps({"task": DOC.task, "sample_id": "doc-1"}) + "\n",
        encoding="utf-8",
    )

    payload = scan_exclusion_sources(
        roots=(ledgers, runs),
        contracts=(DOC, TEXT),
        source_git_sha=GIT_SHA,
        source_git_dirty=False,
    )

    observed = next(source for source in payload["sources"] if source["kind"] == "observed_results")
    assert observed["identity_provenance"] == "exact_prior_source_coverage"
    assert "manifest_path" not in observed

    result_path.write_text(
        json.dumps({"task": DOC.task, "sample_id": "doc-uncovered"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SampleProtocolError, match="exact ledger coverage"):
        scan_exclusion_sources(
            roots=(ledgers, runs),
            contracts=(DOC, TEXT),
            source_git_sha=GIT_SHA,
            source_git_dirty=False,
        )


def test_exclusion_scanner_uses_row_referenced_manifests_for_aggregate_results(
    tmp_path: Path,
) -> None:
    ledgers = tmp_path / "ledgers"
    runs = tmp_path / "runs"
    ledgers.mkdir()
    run_manifest = runs / "worker-1" / "manifest.json"
    _write_json(
        run_manifest,
        {
            "dataset": {
                "id": DOC.dataset_id,
                "revision": DOC.dataset_revision,
                "split": DOC.split,
            }
        },
    )
    aggregate = runs / "aggregate" / "raw.jsonl"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_text(
        json.dumps(
            {
                "task": DOC.task,
                "sample_id": "doc-row-manifest",
                "manifest_path": str(run_manifest),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = scan_exclusion_sources(
        roots=(ledgers, runs),
        contracts=(DOC, TEXT),
        source_git_sha=GIT_SHA,
        source_git_dirty=False,
    )

    identities = validate_exclusion_manifest(payload)
    observed = next(source for source in payload["sources"] if source["kind"] == "observed_results")
    assert identities == (DOC.identity("doc-row-manifest"),)
    assert observed["identity_provenance"] == "row_manifests"
    assert observed["manifest_sha256"] == sha256_file(run_manifest)


def test_exclusion_scanner_requires_prior_exact_identity_for_partial_row_manifest(
    tmp_path: Path,
) -> None:
    ledgers = tmp_path / "ledgers"
    runs = tmp_path / "runs"
    ledgers.mkdir()
    exact = runs / "z-exact"
    _write_json(
        exact / "manifest.json",
        {
            "dataset": {
                "id": TEXT.dataset_id,
                "revision": TEXT.dataset_revision,
                "split": TEXT.split,
            }
        },
    )
    exact.joinpath("results.jsonl").write_text(
        json.dumps({"task": TEXT.task, "sample_id": "text-legacy"}) + "\n",
        encoding="utf-8",
    )
    partial_manifest = runs / "legacy-provenance.json"
    _write_json(
        partial_manifest,
        {
            "dataset_id": TEXT.dataset_id,
            "dataset_revision": TEXT.dataset_revision,
        },
    )
    aggregate = runs / "a-legacy" / "raw.jsonl"
    aggregate.parent.mkdir(parents=True)
    aggregate.write_text(
        json.dumps(
            {
                "task": TEXT.task,
                "sample_id": "text-legacy",
                "manifest_path": str(partial_manifest),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = scan_exclusion_sources(
        roots=(ledgers, runs),
        contracts=(DOC, TEXT),
        source_git_sha=GIT_SHA,
        source_git_dirty=False,
    )

    legacy = next(source for source in payload["sources"] if source["path"] == str(aggregate))
    assert legacy["identity_provenance"] == "exact_prior_source_coverage"
    assert legacy["manifest_sha256"] == sha256_file(partial_manifest)


def test_materializer_is_deterministic_label_blind_and_immutable(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, (DOC.identity("doc-2"),))
    catalog = tmp_path / "catalog.json"
    _catalog(catalog, DOC, ["doc-5", "doc-2", "doc-1", "doc-4", "doc-3"])

    payload = _ledger(catalog=catalog, manifest=manifest, contract=DOC, sample_count=3)
    output = tmp_path / "selected.json"
    write_once(output, payload)
    identities = validate_ledger(payload)
    eligible = [DOC.identity(value) for value in ("doc-5", "doc-1", "doc-4", "doc-3")]
    expected = sorted(
        eligible,
        key=lambda identity: (
            selection_digest(identity, seed=20310821, salt=SALT),
            identity.sample_id,
        ),
    )[:3]

    assert identities == tuple(expected)
    assert payload["excluded_matching_candidates"] == 1
    assert payload["labels_references_scores_or_media_inspected_for_selection"] is False
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_once(output, payload)


def test_materializer_rejects_content_fields_and_insufficient_fresh_ids(
    tmp_path: Path,
) -> None:
    manifest = _manifest(tmp_path, (DOC.identity("doc-1"), DOC.identity("doc-2")))
    catalog = tmp_path / "catalog.json"
    _write_json(
        catalog,
        {
            "schema_version": 1,
            "dataset_id": DOC.dataset_id,
            "dataset_revision": DOC.dataset_revision,
            "split": DOC.split,
            "sample_ids": ["doc-1", "doc-2", "doc-3"],
            "answers": ["forbidden"],
        },
    )
    with pytest.raises(SampleProtocolError, match="IDs only"):
        _ledger(catalog=catalog, manifest=manifest, contract=DOC, sample_count=2)

    catalog_payload = json.loads(catalog.read_text(encoding="utf-8"))
    del catalog_payload["answers"]
    _write_json(catalog, catalog_payload)
    with pytest.raises(SampleProtocolError, match="only 1 untouched"):
        _ledger(catalog=catalog, manifest=manifest, contract=DOC, sample_count=2)


def test_configuration_aware_materializer_rejects_wrong_catalog_config(
    tmp_path: Path,
) -> None:
    prior = tmp_path / "prior"
    prior.mkdir()
    manifest_payload = scan_exclusion_sources(
        roots=(prior,),
        contracts=(INFO, DOCUMENT),
        source_git_sha=GIT_SHA,
        source_git_dirty=False,
    )
    manifest = tmp_path / "manifest.json"
    write_once(manifest, manifest_payload)
    catalog = tmp_path / "catalog.json"
    _catalog(catalog, DOCUMENT, ["candidate"])

    with pytest.raises(SampleProtocolError, match="pinned dataset contract"):
        _ledger(catalog=catalog, manifest=manifest, contract=INFO, sample_count=1)


def test_materializer_rejects_changed_exclusion_source(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, (DOC.identity("doc-old"),))
    catalog = tmp_path / "catalog.json"
    _catalog(catalog, DOC, ["doc-new-1", "doc-new-2"])
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    source_path = Path(manifest_value["sources"][0]["path"])
    source_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(SampleProtocolError, match="missing or changed"):
        _ledger(catalog=catalog, manifest=manifest, contract=DOC, sample_count=1)


def test_disjointness_checker_passes_two_independent_task_ledgers(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        (DOC.identity("doc-old"), TEXT.identity("text-old")),
    )
    doc_catalog = tmp_path / "doc_catalog.json"
    text_catalog = tmp_path / "text_catalog.json"
    _catalog(doc_catalog, DOC, ["doc-old", "doc-new-1", "doc-new-2"])
    _catalog(text_catalog, TEXT, ["text-old", "text-new-1", "text-new-2"])
    doc_path = tmp_path / "doc_ledger.json"
    text_path = tmp_path / "text_ledger.json"
    write_once(
        doc_path,
        _ledger(catalog=doc_catalog, manifest=manifest, contract=DOC, sample_count=2),
    )
    write_once(
        text_path,
        _ledger(catalog=text_catalog, manifest=manifest, contract=TEXT, sample_count=2),
    )

    report = check_disjointness(
        exclusion_manifest_path=manifest,
        ledger_paths=(doc_path, text_path),
        validation_git_sha=GIT_SHA,
        validation_git_dirty=False,
    )

    assert report["status"] == "passed"
    assert report["overlap_count"] == 0
    assert report["cross_ledger_duplicate_count"] == 0
    assert report["selected_unique_sample_count"] == 4


def test_disjointness_checker_records_overlap_failure(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, (DOC.identity("doc-old"),))
    ledger: dict[str, object] = {
        "schema_version": 1,
        "classification": "fresh_disjoint_label_blind_calibration",
        **DOC.as_dict(),
        "sample_count": 1,
        "sample_ids": ["doc-old"],
        "exclusion_manifest_sha256": sha256_file(manifest),
    }
    ledger["ledger_content_sha256"] = hashlib.sha256(canonical_bytes(ledger)).hexdigest()
    ledger_path = tmp_path / "invalid_overlap_ledger.json"
    write_once(ledger_path, ledger)

    report = check_disjointness(
        exclusion_manifest_path=manifest,
        ledger_paths=(ledger_path,),
        validation_git_sha=GIT_SHA,
        validation_git_dirty=False,
    )

    assert report["status"] == "failed"
    assert report["overlap_count"] == 1
    assert report["overlaps"] == [DOC.identity("doc-old").as_dict()]


def test_real_v101_blocks_all_actions_after_negative_feasibility_seal() -> None:
    package = Path(__file__).parents[2]
    protocol = yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v101_v100_fresh_disjoint_calibration.yaml"
        ).read_text(encoding="utf-8")
    )

    with pytest.raises(PermissionError, match="exclusion_scan_authorized is false"):
        require_protocol_authorization(protocol, "scan")
    with pytest.raises(PermissionError, match="materialization_authorized is false"):
        require_protocol_authorization(protocol, "materialize")
    with pytest.raises(PermissionError, match="disjointness_check_authorized is false"):
        require_protocol_authorization(protocol, "check")


def test_real_v101_cli_fails_before_creating_artifacts(tmp_path: Path) -> None:
    package = Path(__file__).parents[2]
    output = tmp_path / "must_not_exist.json"
    result = subprocess.run(
        [
            sys.executable,
            str(package / "scripts/choice_c_v101_sample_tooling.py"),
            "--spec",
            str(package / "configs/experiments/choice_c_v101_v100_fresh_disjoint_calibration.yaml"),
            "materialize",
            "--task",
            "docvqa",
            "--catalog",
            str(tmp_path / "missing_catalog.json"),
            "--exclusion-manifest",
            str(tmp_path / "missing_manifest.json"),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "materialization_authorized is false" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("action", ("scan",))
def test_real_v102_cli_blocks_unauthorized_actions_before_filesystem_work(
    tmp_path: Path, action: str
) -> None:
    package = Path(__file__).parents[2]
    output = tmp_path / f"{action}-must-not-exist.json"
    command = [
        sys.executable,
        str(package / "scripts/choice_c_v102_sample_tooling.py"),
        "--spec",
        str(
            package / "configs/experiments/choice_c_v102_v100_fresh_cross_dataset_calibration.yaml"
        ),
        action,
    ]
    if action == "scan":
        command.extend(("--output", str(output)))
    elif action == "materialize":
        command.extend(
            (
                "--task",
                "infovqa",
                "--catalog",
                str(tmp_path / "missing-catalog.json"),
                "--exclusion-manifest",
                str(tmp_path / "missing-manifest.json"),
                "--output",
                str(output),
            )
        )
    else:
        command.extend(
            (
                "--exclusion-manifest",
                str(tmp_path / "missing-manifest.json"),
                "--ledger",
                str(tmp_path / "missing-ledger.json"),
                "--output",
                str(output),
            )
        )

    result = subprocess.run(command, check=False, capture_output=True, text=True)

    assert result.returncode != 0
    assert f"{action}" in result.stderr
    assert "authorized is false" in result.stderr
    assert not output.exists()


def test_real_v102_protocol_blocks_scan_reexecution_and_downstream_actions() -> None:
    package = Path(__file__).parents[2]
    protocol = yaml.safe_load(
        (
            package / "configs/experiments/choice_c_v102_v100_fresh_cross_dataset_calibration.yaml"
        ).read_text(encoding="utf-8")
    )

    with pytest.raises(PermissionError, match="exclusion_scan_authorized is false"):
        require_protocol_authorization(protocol, "scan")
    with pytest.raises(PermissionError, match="materialization_authorized is false"):
        require_protocol_authorization(protocol, "materialize")
    with pytest.raises(PermissionError, match="disjointness_check_authorized is false"):
        require_protocol_authorization(protocol, "check")
    with pytest.raises(PermissionError, match="candidate_catalog_construction_authorized is false"):
        require_protocol_authorization(protocol, "catalog")
