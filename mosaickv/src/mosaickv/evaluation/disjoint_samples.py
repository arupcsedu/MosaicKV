"""Deterministic, label-blind sample exclusion and ledger construction."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml


class SampleProtocolError(ValueError):
    """Raised when a sample-protocol artifact fails closed validation."""


@dataclass(frozen=True, order=True)
class DatasetContract:
    task: str
    dataset_id: str
    dataset_revision: str
    split: str
    dataset_config: str | None = None

    def __post_init__(self) -> None:
        if self.dataset_config is not None and not self.dataset_config.strip():
            raise SampleProtocolError("dataset_config must be nonempty when specified")

    def identity(self, sample_id: str) -> SampleIdentity:
        return SampleIdentity(
            dataset_id=self.dataset_id,
            dataset_revision=self.dataset_revision,
            split=self.split,
            sample_id=sample_id,
            dataset_config=self.dataset_config,
        )

    def as_dict(self) -> dict[str, str]:
        payload = {
            "task": self.task,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
        }
        if self.dataset_config is not None:
            payload["dataset_config"] = self.dataset_config
        return payload


@dataclass(frozen=True, order=True)
class SampleIdentity:
    dataset_id: str
    dataset_revision: str
    split: str
    sample_id: str
    dataset_config: str | None = None

    def __post_init__(self) -> None:
        if self.dataset_config is not None and not self.dataset_config.strip():
            raise SampleProtocolError("dataset_config must be nonempty when specified")

    def canonical_key(self) -> str:
        return "\0".join(
            (
                self.dataset_id,
                self.dataset_config or "",
                self.dataset_revision,
                self.split,
                self.sample_id,
            )
        )

    def as_dict(self) -> dict[str, str]:
        payload = {
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "split": self.split,
            "sample_id": self.sample_id,
        }
        if self.dataset_config is not None:
            payload["dataset_config"] = self.dataset_config
        return payload

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SampleIdentity:
        required = ("dataset_id", "dataset_revision", "split", "sample_id")
        if any(not isinstance(value.get(field), str) or not value[field] for field in required):
            raise SampleProtocolError("sample identity has a missing or non-string field")
        dataset_config = value.get("dataset_config")
        if dataset_config is not None and (
            not isinstance(dataset_config, str) or not dataset_config.strip()
        ):
            raise SampleProtocolError("sample identity has an invalid dataset_config")
        return cls(
            **{field: str(value[field]) for field in required},
            dataset_config=None if dataset_config is None else str(dataset_config),
        )


def _canonical_identities(values: Iterable[SampleIdentity]) -> tuple[SampleIdentity, ...]:
    return tuple(sorted(set(values), key=SampleIdentity.canonical_key))


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, payload: Mapping[str, Any]) -> str:
    """Atomically create one read-only JSON artifact and return its file hash."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_bytes(payload)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise
    return hashlib.sha256(encoded).hexdigest()


def extract_id_only_column(
    dataset: Any,
    *,
    id_field: str,
    expected_count: int,
) -> tuple[object, ...]:
    """Read exactly one audited ID column from a dataset-like split.

    The constructor deliberately requires ``select_columns``.  It therefore
    fails closed instead of indexing rows that may decode images or expose
    prompts, answers, references, scores, or other benchmark content.
    """

    if not isinstance(id_field, str) or not id_field.strip():
        raise SampleProtocolError("candidate catalog requires an audited ID field")
    if expected_count < 1:
        raise SampleProtocolError("candidate catalog expected count must be positive")
    columns = getattr(dataset, "column_names", None)
    if (
        not isinstance(columns, Sequence)
        or isinstance(columns, (str, bytes))
        or any(not isinstance(column, str) for column in columns)
    ):
        raise SampleProtocolError("dataset does not expose auditable column names")
    if id_field not in columns:
        raise SampleProtocolError(f"audited ID field {id_field!r} is absent from the split")
    if len(dataset) != expected_count:
        raise SampleProtocolError(
            f"pinned split has {len(dataset)} rows; expected {expected_count}"
        )
    select_columns = getattr(dataset, "select_columns", None)
    if not callable(select_columns):
        raise SampleProtocolError("dataset cannot isolate the audited ID column")
    selected = select_columns([id_field])
    selected_columns = getattr(selected, "column_names", None)
    if list(selected_columns or ()) != [id_field] or len(selected) != expected_count:
        raise SampleProtocolError("dataset ID-column projection changed schema or cardinality")
    try:
        raw_ids = selected[id_field]
    except (KeyError, TypeError, ValueError) as error:
        raise SampleProtocolError("dataset ID-column projection could not be read") from error
    values = tuple(raw_ids)
    if len(values) != expected_count:
        raise SampleProtocolError("dataset ID column differs from the pinned cardinality")
    return values


def construct_id_only_catalog(
    *,
    contract: DatasetContract,
    raw_ids: Sequence[object],
    id_field: str,
    expected_count: int,
    construction_git_sha: str,
    construction_git_dirty: bool,
) -> dict[str, Any]:
    """Construct a deterministic catalog containing canonical IDs and no content."""

    if construction_git_dirty or re.fullmatch(r"[0-9a-f]{40}", construction_git_sha) is None:
        raise SampleProtocolError("canonical catalog construction requires a clean full Git SHA")
    if not isinstance(id_field, str) or not id_field.strip():
        raise SampleProtocolError("candidate catalog requires an audited ID field")
    if expected_count < 1 or len(raw_ids) != expected_count:
        raise SampleProtocolError(
            f"ID column has {len(raw_ids)} values; expected exactly {expected_count}"
        )
    source_ids: list[str] = []
    seen_raw: set[str] = set()
    for value in raw_ids:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise SampleProtocolError("candidate IDs must be strings or integers, never booleans")
        normalized = str(value)
        if not normalized or normalized != normalized.strip():
            raise SampleProtocolError("candidate IDs must be nonempty and whitespace-normalized")
        if normalized in seen_raw:
            raise SampleProtocolError(f"candidate ID is duplicated: {normalized!r}")
        seen_raw.add(normalized)
        source_ids.append(f"{contract.task}:{normalized}")
    sample_ids = sorted(source_ids)
    if len(sample_ids) != expected_count or len(set(sample_ids)) != expected_count:
        raise SampleProtocolError("candidate catalog does not contain every unique split ID")
    payload: dict[str, Any] = {
        "schema_version": 2 if contract.dataset_config is not None else 1,
        "dataset_id": contract.dataset_id,
        "dataset_revision": contract.dataset_revision,
        "split": contract.split,
        "sample_ids": sample_ids,
    }
    if contract.dataset_config is not None:
        payload["dataset_config"] = contract.dataset_config
    return payload


def construct_id_only_catalog_from_dataset(
    *,
    dataset: Any,
    contract: DatasetContract,
    id_field: str,
    expected_count: int,
    construction_git_sha: str,
    construction_git_dirty: bool,
) -> dict[str, Any]:
    """Project a dataset split to its ID column and build a fail-closed catalog."""

    raw_ids = extract_id_only_column(
        dataset,
        id_field=id_field,
        expected_count=expected_count,
    )
    return construct_id_only_catalog(
        contract=contract,
        raw_ids=raw_ids,
        id_field=id_field,
        expected_count=expected_count,
        construction_git_sha=construction_git_sha,
        construction_git_dirty=construction_git_dirty,
    )


def require_protocol_authorization(protocol: Mapping[str, Any], action: str) -> None:
    """Prevent command-line tooling from acting before protocol authorization."""

    authorization_fields = {
        "scan": "exclusion_scan_authorized",
        "catalog": "candidate_catalog_construction_authorized",
        "materialize": "materialization_authorized",
        "check": "disjointness_check_authorized",
    }
    if action not in authorization_fields:
        raise SampleProtocolError(f"unknown sample-protocol action: {action}")
    field = authorization_fields[action]
    if protocol.get(field) is not True:
        experiment_id = str(protocol.get("experiment_id", "sample protocol"))
        raise PermissionError(f"{experiment_id} {action} is blocked while {field} is false")


def _contracts_by_identity(
    contracts: Sequence[DatasetContract],
) -> dict[tuple[str, str | None, str, str], DatasetContract]:
    result: dict[tuple[str, str | None, str, str], DatasetContract] = {}
    for contract in contracts:
        key = (
            contract.dataset_id,
            contract.dataset_config,
            contract.dataset_revision,
            contract.split,
        )
        required = (contract.dataset_id, contract.dataset_revision, contract.split)
        if key in result or not all(required) or not contract.task:
            raise SampleProtocolError("dataset contracts must be unique and nonempty")
        result[key] = contract
    if not result:
        raise SampleProtocolError("at least one dataset contract is required")
    return result


def _load_structured(path: Path) -> dict[str, Any]:
    try:
        if path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
        else:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, yaml.YAMLError) as error:
        raise SampleProtocolError(f"invalid structured exclusion source {path}: {error}") from error
    if not isinstance(value, dict):
        raise SampleProtocolError(f"exclusion source must contain an object: {path}")
    return cast("dict[str, Any]", value)


def _dataset_metadata(
    value: Mapping[str, Any],
) -> tuple[str, str | None, str, str] | None:
    dataset = value.get("dataset")
    if isinstance(dataset, dict):
        dataset_id = dataset.get("id")
        dataset_config = dataset.get("config", dataset.get("name", dataset.get("dataset_config")))
        revision = dataset.get("revision")
        split = dataset.get("split")
    else:
        dataset_id = value.get("dataset_id")
        dataset_config = value.get("dataset_config", value.get("dataset_name"))
        revision = value.get("dataset_revision")
        split = value.get("split")
    if all(isinstance(item, str) and item for item in (dataset_id, revision, split)):
        if dataset_config is not None and (
            not isinstance(dataset_config, str) or not dataset_config.strip()
        ):
            raise SampleProtocolError("dataset metadata has an invalid dataset_config")
        return (
            str(dataset_id),
            None if dataset_config is None else str(dataset_config),
            str(revision),
            str(split),
        )
    resolved = value.get("resolved_config")
    if isinstance(resolved, dict) and isinstance(resolved.get("dataset"), dict):
        return _dataset_metadata({"dataset": resolved["dataset"]})
    return None


def _dataset_id_config_revision(
    value: Mapping[str, Any],
) -> tuple[str, str | None, str] | None:
    dataset = value.get("dataset")
    if isinstance(dataset, dict):
        dataset_id = dataset.get("id")
        dataset_config = dataset.get("config", dataset.get("name", dataset.get("dataset_config")))
        revision = dataset.get("revision")
    else:
        dataset_id = value.get("dataset_id")
        dataset_config = value.get("dataset_config", value.get("dataset_name"))
        revision = value.get("dataset_revision")
    if all(isinstance(item, str) and item for item in (dataset_id, revision)):
        if dataset_config is not None and (
            not isinstance(dataset_config, str) or not dataset_config.strip()
        ):
            raise SampleProtocolError("dataset metadata has an invalid dataset_config")
        return (
            str(dataset_id),
            None if dataset_config is None else str(dataset_config),
            str(revision),
        )
    resolved = value.get("resolved_config")
    if isinstance(resolved, dict) and isinstance(resolved.get("dataset"), dict):
        return _dataset_id_config_revision({"dataset": resolved["dataset"]})
    return None


def _reject_ambiguous_missing_config(
    metadata: tuple[str, str | None, str, str] | None,
    contracts: Mapping[tuple[str, str | None, str, str], DatasetContract],
    *,
    path: Path,
) -> None:
    """Fail closed when a multi-config source omits a relevant configuration."""

    if metadata is None or metadata[1] is not None:
        return
    dataset_id, _, revision, split = metadata
    matches = [
        contract
        for (
            candidate_id,
            config,
            candidate_revision,
            candidate_split,
        ), contract in contracts.items()
        if candidate_id == dataset_id
        and config is not None
        and candidate_revision == revision
        and candidate_split == split
    ]
    if matches:
        raise SampleProtocolError(
            f"configuration-aware provenance requires dataset_config for a relevant source: {path}"
        )


def _identities_from_ledger(
    path: Path,
    contracts: Mapping[tuple[str, str | None, str, str], DatasetContract],
) -> tuple[SampleIdentity, ...]:
    value = _load_structured(path)
    metadata = _dataset_metadata(value)
    _reject_ambiguous_missing_config(metadata, contracts, path=path)
    sample_ids = value.get("sample_ids")
    if metadata not in contracts or not isinstance(sample_ids, list):
        return ()
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise SampleProtocolError(f"ledger has an invalid sample ID: {path}")
    contract = contracts[cast("tuple[str, str | None, str, str]", metadata)]
    return _canonical_identities(contract.identity(str(sample_id)) for sample_id in sample_ids)


def _result_metadata(path: Path) -> tuple[tuple[str, str | None, str, str], Path]:
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise SampleProtocolError(f"observed results lack a companion manifest: {path}")
    metadata = _dataset_metadata(_load_structured(manifest_path))
    if metadata is None:
        raise SampleProtocolError(
            f"result manifest lacks pinned dataset provenance: {manifest_path}"
        )
    return metadata, manifest_path


def _identities_from_results(
    path: Path,
    contracts: Mapping[tuple[str, str | None, str, str], DatasetContract],
    known_identities: set[SampleIdentity],
) -> tuple[tuple[SampleIdentity, ...], tuple[Path, ...], str]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise SampleProtocolError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise SampleProtocolError(f"result row is not an object at {path}:{line_number}")
        rows.append(cast("dict[str, Any]", value))
    if not any(row.get("sample_id") is not None for row in rows):
        return (), (), "not_applicable"
    identities: set[SampleIdentity] = set()
    manifest_path = path.parent / "manifest.json"
    if manifest_path.is_file():
        metadata, manifest_path = _result_metadata(path)
        _reject_ambiguous_missing_config(metadata, contracts, path=manifest_path)
        if metadata not in contracts:
            return (), (), "not_applicable"
        contract = contracts[metadata]
        for row in rows:
            sample_id = row.get("sample_id")
            if sample_id is None:
                continue
            if not isinstance(sample_id, str) or not sample_id:
                raise SampleProtocolError(f"result row has an invalid sample ID: {path}")
            if row.get("task") not in (None, contract.task):
                continue
            identities.add(contract.identity(sample_id))
        return _canonical_identities(identities), (manifest_path.resolve(),), "companion_manifest"

    contracts_by_task: dict[str, DatasetContract] = {}
    for candidate in contracts.values():
        if candidate.task in contracts_by_task:
            raise SampleProtocolError(
                "manifest-less result validation requires unique task contracts"
            )
        contracts_by_task[candidate.task] = candidate
    companion_paths: set[Path] = set()
    known_source_covered = False
    row_manifest_covered = False
    uncovered: set[SampleIdentity] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        task = row.get("task")
        if sample_id is None or task not in contracts_by_task:
            continue
        if not isinstance(sample_id, str) or not sample_id:
            raise SampleProtocolError(f"result row has an invalid sample ID: {path}")
        contract = contracts_by_task[str(task)]
        identity = contract.identity(sample_id)
        raw_manifest = row.get("manifest_path")
        row_manifest = Path(raw_manifest).resolve() if isinstance(raw_manifest, str) else None
        if row_manifest is not None and row_manifest.is_file():
            manifest_value = _load_structured(row_manifest)
            row_metadata = _dataset_metadata(manifest_value)
            expected = (
                contract.dataset_id,
                contract.dataset_config,
                contract.dataset_revision,
                contract.split,
            )
            if row_metadata == expected:
                row_manifest_covered = True
            elif (
                row_metadata is None
                and _dataset_id_config_revision(manifest_value) == expected[:3]
                and identity in known_identities
            ):
                known_source_covered = True
            else:
                raise SampleProtocolError(
                    f"row result manifest has conflicting dataset provenance: {row_manifest}"
                )
            companion_paths.add(row_manifest)
        elif identity in known_identities:
            known_source_covered = True
        else:
            uncovered.add(identity)
        identities.add(identity)
    if not identities:
        return (), (), "not_applicable"
    if uncovered:
        preview = [identity.as_dict() for identity in _canonical_identities(uncovered)[:5]]
        raise SampleProtocolError(
            "observed results lack a companion manifest and exact ledger coverage: "
            f"{path}; uncovered={preview}"
        )
    if row_manifest_covered and known_source_covered:
        provenance_basis = "row_manifests_and_exact_prior_source_coverage"
    elif row_manifest_covered:
        provenance_basis = "row_manifests"
    else:
        provenance_basis = "exact_prior_source_coverage"
    return (
        _canonical_identities(identities),
        tuple(sorted(companion_paths, key=str)),
        provenance_basis,
    )


def _discover_sources(roots: Sequence[Path], ignored: set[Path]) -> tuple[Path, ...]:
    discovered: set[Path] = set()
    for raw_root in roots:
        root = raw_root.resolve()
        if not root.is_dir():
            raise SampleProtocolError(f"exclusion scan root does not exist: {root}")
        for path in root.rglob("*"):
            resolved = path.resolve()
            if not path.is_file() or resolved in ignored:
                continue
            if path.name in {"results.jsonl", "raw.jsonl"}:
                discovered.add(resolved)
                continue
            if path.suffix.lower() not in {".json", ".yaml", ".yml"}:
                continue
            relative_parts = path.relative_to(root).parts
            if root.name == "ledgers" or "inputs" in relative_parts:
                discovered.add(resolved)
    return tuple(sorted(discovered, key=str))


def scan_exclusion_sources(
    *,
    roots: Sequence[Path],
    contracts: Sequence[DatasetContract],
    source_git_sha: str,
    source_git_dirty: bool,
    ignored_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    """Inventory all recognizable prior ledgers/results for pinned datasets."""

    if source_git_dirty or len(source_git_sha) != 40:
        raise SampleProtocolError("canonical exclusion scanning requires a clean full Git SHA")
    by_dataset = _contracts_by_identity(contracts)
    ignored = {path.resolve() for path in ignored_paths}
    sources: list[dict[str, Any]] = []
    all_identities: set[SampleIdentity] = set()
    discovered = _discover_sources(roots, ignored)
    ledger_paths = tuple(path for path in discovered if path.suffix.lower() != ".jsonl")
    result_paths = tuple(
        sorted(
            (path for path in discovered if path.suffix.lower() == ".jsonl"),
            key=lambda path: (not path.with_name("manifest.json").is_file(), str(path)),
        )
    )
    for path in ledger_paths:
        identities = _identities_from_ledger(path, by_dataset)
        if not identities:
            continue
        identity_payload = [identity.as_dict() for identity in identities]
        sources.append(
            {
                "kind": "sample_ledger",
                "path": str(path),
                "sha256": sha256_file(path),
                "sample_count": len(identities),
                "identities_sha256": hashlib.sha256(canonical_bytes(identity_payload)).hexdigest(),
            }
        )
        all_identities.update(identities)
    for path in result_paths:
        identities, companions, provenance_basis = _identities_from_results(
            path, by_dataset, all_identities
        )
        if not identities:
            continue
        identity_payload = [identity.as_dict() for identity in identities]
        record: dict[str, Any] = {
            "kind": "observed_results",
            "path": str(path),
            "sha256": sha256_file(path),
            "sample_count": len(identities),
            "identities_sha256": hashlib.sha256(canonical_bytes(identity_payload)).hexdigest(),
            "identity_provenance": provenance_basis,
        }
        if len(companions) == 1:
            record["manifest_path"] = str(companions[0])
            record["manifest_sha256"] = sha256_file(companions[0])
        elif companions:
            record["manifests"] = [
                {"path": str(companion), "sha256": sha256_file(companion)}
                for companion in companions
            ]
        sources.append(record)
        all_identities.update(identities)
    for source in sources:
        if sha256_file(Path(str(source["path"]))) != source["sha256"]:
            raise SampleProtocolError("exclusion source changed during the scan")
        if (
            "manifest_path" in source
            and sha256_file(Path(str(source["manifest_path"]))) != source["manifest_sha256"]
        ):
            raise SampleProtocolError("result manifest changed during the scan")
        for manifest in cast("list[dict[str, Any]]", source.get("manifests", [])):
            if sha256_file(Path(str(manifest["path"]))) != manifest["sha256"]:
                raise SampleProtocolError("result manifest changed during the scan")
    identities_payload = [identity.as_dict() for identity in _canonical_identities(all_identities)]
    payload: dict[str, Any] = {
        "schema_version": 2 if any(contract.dataset_config for contract in contracts) else 1,
        "classification": "label_blind_sample_exclusion_manifest",
        "source_git_sha": source_git_sha,
        "source_git_dirty": False,
        "scan_roots": [str(path.resolve()) for path in sorted(roots, key=str)],
        "dataset_contracts": [contract.as_dict() for contract in sorted(contracts)],
        "source_count": len(sources),
        "sources": sources,
        "excluded_unique_sample_count": len(identities_payload),
        "excluded_identities": identities_payload,
    }
    payload["manifest_content_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def validate_exclusion_manifest(value: Mapping[str, Any]) -> tuple[SampleIdentity, ...]:
    recorded_sha = value.get("manifest_content_sha256")
    without_sha = {key: item for key, item in value.items() if key != "manifest_content_sha256"}
    observed_sha = hashlib.sha256(canonical_bytes(without_sha)).hexdigest()
    if recorded_sha != observed_sha:
        raise SampleProtocolError("exclusion manifest content hash differs")
    raw_identities = value.get("excluded_identities")
    if not isinstance(raw_identities, list):
        raise SampleProtocolError("exclusion manifest lacks identities")
    identities = tuple(
        SampleIdentity.from_mapping(cast("Mapping[str, Any]", item))
        for item in raw_identities
        if isinstance(item, dict)
    )
    if len(identities) != len(raw_identities) or _canonical_identities(identities) != identities:
        raise SampleProtocolError("exclusion identities are not unique and canonically sorted")
    if value.get("excluded_unique_sample_count") != len(identities):
        raise SampleProtocolError("exclusion manifest sample count differs")
    return identities


def verify_exclusion_source_files(value: Mapping[str, Any]) -> None:
    """Verify that every source bound into an exclusion snapshot is unchanged."""

    sources = value.get("sources")
    if not isinstance(sources, list):
        raise SampleProtocolError("exclusion manifest lacks source records")
    for raw_source in sources:
        if not isinstance(raw_source, dict):
            raise SampleProtocolError("exclusion source record is not an object")
        source = cast("dict[str, Any]", raw_source)
        path = Path(str(source.get("path", "")))
        if not path.is_file() or sha256_file(path) != source.get("sha256"):
            raise SampleProtocolError(f"exclusion source is missing or changed: {path}")
        if "manifest_path" in source:
            manifest_path = Path(str(source["manifest_path"]))
            if not manifest_path.is_file() or sha256_file(manifest_path) != source.get(
                "manifest_sha256"
            ):
                raise SampleProtocolError(
                    f"observed-result manifest is missing or changed: {manifest_path}"
                )
        manifests = source.get("manifests", [])
        if not isinstance(manifests, list):
            raise SampleProtocolError("observed-result manifests are not a list")
        for raw_manifest in manifests:
            if not isinstance(raw_manifest, dict):
                raise SampleProtocolError("observed-result manifest record is not an object")
            manifest_path = Path(str(raw_manifest.get("path", "")))
            if not manifest_path.is_file() or sha256_file(manifest_path) != raw_manifest.get(
                "sha256"
            ):
                raise SampleProtocolError(
                    f"observed-result manifest is missing or changed: {manifest_path}"
                )


def selection_digest(identity: SampleIdentity, *, seed: int, salt: str) -> str:
    preimage = "_null_".join(
        (
            salt,
            identity.dataset_id,
            identity.dataset_config or "",
            identity.dataset_revision,
            identity.split,
            str(seed),
            identity.sample_id,
        )
    )
    return hashlib.sha256(preimage.encode()).hexdigest()


def materialize_deterministic_ledger(
    *,
    catalog_path: Path,
    exclusion_manifest_path: Path,
    contract: DatasetContract,
    selection_seed: int,
    selection_salt: str,
    sample_count: int,
    candidate_implementation_git_sha: str,
    materialization_git_sha: str,
    materialization_git_dirty: bool,
    protocol_sha256: str,
) -> dict[str, Any]:
    """Build an in-memory immutable ledger payload from ID-only inputs."""

    if sample_count < 1 or selection_seed < 0:
        raise SampleProtocolError("sample count must be positive and seed nonnegative")
    if materialization_git_dirty or len(materialization_git_sha) != 40:
        raise SampleProtocolError("canonical ledger materialization requires a clean full Git SHA")
    if len(candidate_implementation_git_sha) != 40 or len(protocol_sha256) != 64:
        raise SampleProtocolError("candidate Git SHA or protocol hash has invalid length")
    catalog_file_sha = sha256_file(catalog_path)
    manifest_file_sha = sha256_file(exclusion_manifest_path)
    catalog = _load_structured(catalog_path)
    expected_catalog = {
        "dataset_id": contract.dataset_id,
        "dataset_revision": contract.dataset_revision,
        "split": contract.split,
    }
    if contract.dataset_config is not None:
        expected_catalog["dataset_config"] = contract.dataset_config
    if any(catalog.get(key) != value for key, value in expected_catalog.items()):
        raise SampleProtocolError("candidate catalog differs from the pinned dataset contract")
    if set(catalog) != {"schema_version", *expected_catalog, "sample_ids"}:
        raise SampleProtocolError("candidate catalog must contain IDs only")
    raw_ids = catalog.get("sample_ids")
    if not isinstance(raw_ids, list) or any(
        not isinstance(sample_id, str) or not sample_id for sample_id in raw_ids
    ):
        raise SampleProtocolError("candidate catalog has invalid sample IDs")
    identities = tuple(contract.identity(str(sample_id)) for sample_id in raw_ids)
    if len(set(identities)) != len(identities):
        raise SampleProtocolError("candidate catalog contains duplicate canonical identities")
    manifest = _load_structured(exclusion_manifest_path)
    excluded = set(validate_exclusion_manifest(manifest))
    verify_exclusion_source_files(manifest)
    eligible = tuple(identity for identity in identities if identity not in excluded)
    ranked = sorted(
        eligible,
        key=lambda identity: (
            selection_digest(identity, seed=selection_seed, salt=selection_salt),
            identity.sample_id,
        ),
    )
    if len(ranked) < sample_count:
        raise SampleProtocolError(
            f"only {len(ranked)} untouched samples remain; requested {sample_count}"
        )
    selected = ranked[:sample_count]
    if (
        sha256_file(catalog_path) != catalog_file_sha
        or sha256_file(exclusion_manifest_path) != manifest_file_sha
    ):
        raise SampleProtocolError("catalog or exclusion manifest changed during selection")
    payload: dict[str, Any] = {
        "schema_version": 2 if contract.dataset_config is not None else 1,
        "classification": "fresh_disjoint_label_blind_calibration",
        **contract.as_dict(),
        "candidate_implementation_git_sha": candidate_implementation_git_sha,
        "materialization_git_sha": materialization_git_sha,
        "materialization_git_dirty": False,
        "protocol_sha256": protocol_sha256,
        "catalog_path": str(catalog_path.resolve()),
        "catalog_sha256": catalog_file_sha,
        "exclusion_manifest_path": str(exclusion_manifest_path.resolve()),
        "exclusion_manifest_sha256": manifest_file_sha,
        "exclusion_manifest_content_sha256": manifest["manifest_content_sha256"],
        "selection_seed": selection_seed,
        "selection_salt": selection_salt,
        "selection_rule": "sha256_digest_ascending_then_canonical_sample_id",
        "selection_preimage_separator": "_null_",
        "eligible_before_exclusion": len(identities),
        "excluded_matching_candidates": len(identities) - len(eligible),
        "eligible_after_exclusion": len(eligible),
        "sample_count": len(selected),
        "sample_ids": [identity.sample_id for identity in selected],
        "selection_digests": [
            selection_digest(identity, seed=selection_seed, salt=selection_salt)
            for identity in selected
        ],
        "labels_references_scores_or_media_inspected_for_selection": False,
    }
    payload["ledger_content_sha256"] = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    return payload


def validate_ledger(value: Mapping[str, Any]) -> tuple[SampleIdentity, ...]:
    recorded_sha = value.get("ledger_content_sha256")
    without_sha = {key: item for key, item in value.items() if key != "ledger_content_sha256"}
    if recorded_sha != hashlib.sha256(canonical_bytes(without_sha)).hexdigest():
        raise SampleProtocolError("ledger content hash differs")
    sample_ids = value.get("sample_ids")
    if not isinstance(sample_ids, list):
        raise SampleProtocolError("ledger lacks sample IDs")
    contract = DatasetContract(
        task=str(value.get("task", "")),
        dataset_id=str(value.get("dataset_id", "")),
        dataset_revision=str(value.get("dataset_revision", "")),
        split=str(value.get("split", "")),
        dataset_config=(
            None if value.get("dataset_config") is None else str(value.get("dataset_config"))
        ),
    )
    if not all((contract.task, contract.dataset_id, contract.dataset_revision, contract.split)):
        raise SampleProtocolError("ledger lacks its dataset contract")
    identities = tuple(contract.identity(str(sample_id)) for sample_id in sample_ids)
    if len(identities) != len(set(identities)) or value.get("sample_count") != len(identities):
        raise SampleProtocolError("ledger identities are duplicated or miscounted")
    return identities


def check_disjointness(
    *,
    exclusion_manifest_path: Path,
    ledger_paths: Sequence[Path],
    validation_git_sha: str,
    validation_git_dirty: bool,
) -> dict[str, Any]:
    """Independently verify immutable ledgers against one exclusion snapshot."""

    if not ledger_paths:
        raise SampleProtocolError("at least one materialized ledger is required")
    if validation_git_dirty or len(validation_git_sha) != 40:
        raise SampleProtocolError("canonical disjointness checking requires a clean full Git SHA")
    manifest = _load_structured(exclusion_manifest_path)
    excluded = set(validate_exclusion_manifest(manifest))
    verify_exclusion_source_files(manifest)
    observed: set[SampleIdentity] = set()
    overlaps: set[SampleIdentity] = set()
    cross_ledger_duplicates: set[SampleIdentity] = set()
    ledger_records: list[dict[str, Any]] = []
    manifest_file_sha = sha256_file(exclusion_manifest_path)
    for path in sorted((item.resolve() for item in ledger_paths), key=str):
        ledger = _load_structured(path)
        identities = validate_ledger(ledger)
        if ledger.get("exclusion_manifest_sha256") != manifest_file_sha:
            raise SampleProtocolError(f"ledger binds a different exclusion manifest: {path}")
        identity_set = set(identities)
        overlaps.update(identity_set & excluded)
        cross_ledger_duplicates.update(identity_set & observed)
        observed.update(identity_set)
        ledger_records.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "sample_count": len(identities),
            }
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "classification": "sample_disjointness_validation",
        "validation_git_sha": validation_git_sha,
        "validation_git_dirty": False,
        "status": "passed" if not overlaps and not cross_ledger_duplicates else "failed",
        "exclusion_manifest_path": str(exclusion_manifest_path.resolve()),
        "exclusion_manifest_sha256": manifest_file_sha,
        "excluded_unique_sample_count": len(excluded),
        "ledger_count": len(ledger_records),
        "ledgers": ledger_records,
        "selected_unique_sample_count": len(observed),
        "overlap_count": len(overlaps),
        "overlaps": [identity.as_dict() for identity in _canonical_identities(overlaps)],
        "cross_ledger_duplicate_count": len(cross_ledger_duplicates),
        "cross_ledger_duplicates": [
            identity.as_dict() for identity in _canonical_identities(cross_ledger_duplicates)
        ],
        "zero_overlap_required": True,
    }
    report["report_content_sha256"] = hashlib.sha256(canonical_bytes(report)).hexdigest()
    return report
